"""
Face detection and recognition. Uses face_recognition (dlib) when available, else DeepFace.
Assigns persistent IDs (face_1, face_2, ...) and recalls known faces across frames.
"""
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import face_recognition as fr
    USE_FACE_RECOGNITION = True
except ImportError:
    USE_FACE_RECOGNITION = False
    from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent
FACES_JSON = BASE_DIR / "faces.json"
FACE_MATCH_TOLERANCE = 0.5
DEEPFACE_MATCH_THRESHOLD = 1.0
MAX_EMBEDDINGS_PER_PERSON = 20

_last_detected_face_ids: list[str] = []


class FaceRegistry:
    """Stores face encodings. Uses compare_faces + face_distance for recall."""

    def __init__(self, path: Path = FACES_JSON):
        self.path = path
        self.entries: dict[str, dict] = {}
        self._next_num = 1
        self._load()

    def _to_encoding(self, emb) -> list[float] | None:
        if emb is None:
            return None
        try:
            arr = np.array(emb, dtype=float).ravel()
            if arr.size != 128:
                return None
            return arr.tolist()
        except (ValueError, TypeError):
            return None

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                raw = data.get("entries", {})
                for pid, ent in raw.items():
                    emb = ent.get("embedding")
                    embs = ent.get("embeddings")
                    if emb is not None and (not embs or len(embs) == 0):
                        embs = [emb]
                    if not embs:
                        embs = []
                    loaded = []
                    for e in embs:
                        v = self._to_encoding(e)
                        if v is not None:
                            loaded.append(v)
                    ent["embeddings"] = loaded
                    self.entries[pid] = ent
                for pid in self.entries:
                    if pid.startswith("face_") and pid[5:].isdigit():
                        n = int(pid[5:])
                        if n >= self._next_num:
                            self._next_num = n + 1
            except Exception as e:
                logger.warning("Could not load face registry: %s", e)

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({"entries": self.entries}, f, indent=2)
        except Exception as e:
            logger.warning("Could not save face registry: %s", e)

    def _get_known_encodings_and_ids(self) -> tuple[list, list[str]]:
        encodings, ids = [], []
        for pid, ent in self.entries.items():
            embs = ent.get("embeddings") or []
            for emb in embs:
                if (e := self._to_encoding(emb)) is not None:
                    encodings.append(e)
                    ids.append(pid)
        return encodings, ids

    def find_or_create(self, encoding: list[float]) -> str:
        enc = self._to_encoding(encoding)
        if enc is None:
            return self.create_new_id()
        known_encodings, known_ids = self._get_known_encodings_and_ids()
        if not known_encodings:
            person_id = f"face_{self._next_num}"
            self._next_num += 1
            self.entries[person_id] = {"embeddings": [enc], "first_seen": time.time()}
            self._save()
            return person_id
        if USE_FACE_RECOGNITION:
            enc_np = np.array(enc, dtype=float)
            matches = fr.compare_faces(known_encodings, enc_np, tolerance=FACE_MATCH_TOLERANCE)
            face_distances = fr.face_distance(known_encodings, enc_np)
            best_match_index = int(np.argmin(face_distances))
            if matches[best_match_index]:
                person_id = known_ids[best_match_index]
                embs = self.entries[person_id].setdefault("embeddings", [])
                embs.append(enc)
                if len(embs) > MAX_EMBEDDINGS_PER_PERSON:
                    self.entries[person_id]["embeddings"] = embs[-MAX_EMBEDDINGS_PER_PERSON:]
                self._save()
                return person_id
        else:
            enc_arr = np.array(enc, dtype=float)
            best_id, best_dist = None, float("inf")
            for i, k in enumerate(known_encodings):
                d = float(np.linalg.norm(np.array(k, dtype=float) - enc_arr))
                if d < best_dist:
                    best_dist, best_id = d, known_ids[i]
            if best_id is not None and best_dist <= DEEPFACE_MATCH_THRESHOLD:
                embs = self.entries[best_id].setdefault("embeddings", [])
                embs.append(enc)
                if len(embs) > MAX_EMBEDDINGS_PER_PERSON:
                    self.entries[best_id]["embeddings"] = embs[-MAX_EMBEDDINGS_PER_PERSON:]
                self._save()
                return best_id
        person_id = f"face_{self._next_num}"
        self._next_num += 1
        self.entries[person_id] = {"embeddings": [enc], "first_seen": time.time()}
        self._save()
        return person_id

    def create_new_id(self) -> str:
        person_id = f"face_{self._next_num}"
        self._next_num += 1
        self.entries[person_id] = {"embeddings": [np.random.randn(128).tolist()], "first_seen": time.time()}
        self._save()
        return person_id

    def get(self, person_id: str) -> dict | None:
        return self.entries.get(person_id)

    def update_metadata(self, person_id: str, **kwargs) -> bool:
        if person_id not in self.entries:
            return False
        for k, v in kwargs.items():
            if k not in ("embedding", "embeddings"):
                self.entries[person_id][k] = v
        self._save()
        return True

    def list_all(self) -> list[dict]:
        return [
            {"person_id": pid, **{k: v for k, v in ent.items() if k not in ("embedding", "embeddings")}}
            for pid, ent in self.entries.items()
        ]


_registry: FaceRegistry | None = None


def get_registry() -> FaceRegistry:
    global _registry
    if _registry is None:
        _registry = FaceRegistry()
    return _registry


def get_last_detected_ids() -> list[str]:
    return _last_detected_face_ids


def _draw_face_box_and_name(img, left: int, top: int, right: int, bottom: int, name: str, emotion: str = ""):
    label = emotion if emotion else "?"
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(img, (left, bottom - 28), (right, bottom), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, label, (left + 4, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


_OPENCV_CASCADE = None


def _get_opencv_cascade():
    global _OPENCV_CASCADE
    if _OPENCV_CASCADE is None:
        _OPENCV_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _OPENCV_CASCADE


def _emotion_for_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Get dominant emotion for a face crop using DeepFace; return '' on failure."""
    try:
        from deepface import DeepFace as _DF
        crop = img[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            return ""
        analyses = _DF.analyze(crop, actions=["emotion"], enforce_detection=False, detector_backend="skip", silent=True)
        entry = analyses[0] if isinstance(analyses, list) and analyses else analyses
        return (entry.get("dominant_emotion") or "").strip()
    except Exception:
        return ""


def _process_with_face_recognition(img: np.ndarray) -> list[tuple[int, int, int, int, list[float]]]:
    # dlib/face_recognition require 8-bit RGB; ensure contiguous
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    face_locations = fr.face_locations(rgb, model="hog")
    face_encodings = fr.face_encodings(rgb, face_locations, num_jitters=1)
    return [(left, top, right, bottom, enc.tolist()) for (top, right, bottom, left), enc in zip(face_locations, face_encodings)]


def _process_with_deepface(img: np.ndarray) -> list[tuple[int, int, int, int, list[float] | None]]:
    out = []
    try:
        reps = DeepFace.represent(
            img, model_name="Facenet", enforce_detection=False,
            detector_backend="opencv", align=True, silent=True,
        )
        if not isinstance(reps, list):
            reps = [reps] if reps else []
        for entry in reps:
            emb = entry.get("embedding")
            region = entry.get("facial_area") or {}
            x = int(region.get("x") or region.get("left", 0))
            y = int(region.get("y") or region.get("top", 0))
            w = int(region.get("w") or region.get("width", 0))
            h = int(region.get("h") or region.get("height", 0))
            if emb and w > 0 and h > 0:
                out.append((x, y, x + w, y + h, emb))
    except Exception as e:
        logger.debug("DeepFace.represent failed: %s", e)
    if not out:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        for (x, y, w, h) in face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
            crop = img[max(0, y) : y + h, max(0, x) : x + w]
            if crop.size == 0:
                continue
            try:
                crop_reps = DeepFace.represent(
                    crop, model_name="Facenet", enforce_detection=False,
                    detector_backend="skip", align=True, silent=True,
                )
                ent = crop_reps[0] if isinstance(crop_reps, list) and crop_reps else crop_reps
                emb = ent.get("embedding") if isinstance(ent, dict) else None
                out.append((int(x), int(y), int(x + w), int(y + h), emb))
            except Exception:
                out.append((int(x), int(y), int(x + w), int(y + h), None))
    return out


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray | None:
    """Ensure image is 3-channel uint8 BGR for OpenCV; face_recognition gets RGB from caller."""
    if img is None or img.size == 0:
        return None
    if img.dtype != np.uint8:
        try:
            img = np.clip(img, 0, 255).astype(np.uint8)
        except Exception:
            return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        return None
    return np.ascontiguousarray(img)


def process_frame(jpeg_bytes: bytes) -> bytes:
    """Detect faces with OpenCV Haar cascade; draw boxes, face ID and emotion. OpenCV-only detection for reliability."""
    global _last_detected_face_ids
    try:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jpeg_bytes
        h_img, w_img = img.shape[:2]
        if h_img < 30 or w_img < 30:
            return jpeg_bytes
        img = np.ascontiguousarray(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = _get_opencv_cascade()
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if rects is None or len(rects) == 0:
            _last_detected_face_ids = []
            _, out_buf = cv2.imencode(".jpg", img)
            return out_buf.tobytes()
        _last_detected_face_ids = []
        for i, (x, y, w, h) in enumerate(rects):
            x, y, w, h = int(x), int(y), int(w), int(h)
            left, top, right, bottom = x, y, x + w, y + h
            _last_detected_face_ids.append(f"face_{i + 1}")
            emotion = _emotion_for_crop(img, x, y, w, h)
            _draw_face_box_and_name(img, left, top, right, bottom, "", emotion)
        _, out_buf = cv2.imencode(".jpg", img)
        return out_buf.tobytes()
    except Exception as e:
        logger.warning("Face processing failed: %s", e)
        return jpeg_bytes
