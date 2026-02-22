"""
Face database. Uses face_recognition (dlib 128D) for detection and recall when available.
Pattern from https://github.com/ageitgey/face_recognition - compare_faces + face_distance.
Falls back to DeepFace if face_recognition not installed (dlib needs CMake).
"""
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try face_recognition first (dlib 128D, best recall - requires cmake for dlib)
try:
    import face_recognition as fr
    USE_FACE_RECOGNITION = True
except ImportError:
    USE_FACE_RECOGNITION = False
    from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent
FACES_JSON = BASE_DIR / "faces.json"
# face_recognition: tolerance = max distance to count as same person. Lower = stricter, more new IDs
FACE_MATCH_TOLERANCE = 0.5
MAX_EMBEDDINGS_PER_PERSON = 20

_last_detected_face_ids: list[str] = []


class FaceRegistry:
    """Stores face encodings. Uses compare_faces + face_distance for recall (face_recognition pattern)."""

    def __init__(self, path: Path = FACES_JSON):
        self.path = path
        self.entries: dict[str, dict] = {}
        self._next_num = 1
        self._load()

    def _to_encoding(self, emb) -> list[float] | None:
        """Flatten to 128D list for dlib-style encodings."""
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
        """Return (encodings, ids) for face_recognition compare_faces."""
        encodings = []
        ids = []
        for pid, ent in self.entries.items():
            embs = ent.get("embeddings") or []
            for emb in embs:
                e = self._to_encoding(emb)
                if e is not None:
                    encodings.append(e)
                    ids.append(pid)
        return encodings, ids

    def find_or_create(self, encoding: list[float]) -> str:
        """Use face_recognition-style compare_faces + face_distance. Match → existing ID; else new ID."""
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
            best_id = None
            best_dist = float("inf")
            for i, k in enumerate(known_encodings):
                d = float(np.linalg.norm(np.array(k, dtype=float) - enc_arr))
                if d < best_dist:
                    best_dist = d
                    best_id = known_ids[i]
            # DeepFace L2: only match when very close; else create new ID (different person)
            DEEPFACE_MATCH_THRESHOLD = 1.0
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
        return [{"person_id": pid, **{k: v for k, v in ent.items() if k not in ("embedding", "embeddings")}} for pid, ent in self.entries.items()]


_registry: FaceRegistry | None = None


def get_registry() -> FaceRegistry:
    global _registry
    if _registry is None:
        _registry = FaceRegistry()
    return _registry


def get_last_detected_ids() -> list[str]:
    return _last_detected_face_ids


def _draw_face_box_and_name(img, left: int, top: int, right: int, bottom: int, name: str):
    """Draw red box and label like face_recognition example (Barack, Adam, Rebecca style)."""
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


def _process_with_face_recognition(img: np.ndarray) -> list[tuple[int, int, int, int, list[float]]]:
    """face_recognition path: face_locations + face_encodings (dlib 128D)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb, model="hog")
    face_encodings = fr.face_encodings(rgb, face_locations, num_jitters=1)
    out = []
    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        out.append((left, top, right, bottom, enc.tolist()))
    return out


def _process_with_deepface(img: np.ndarray) -> list[tuple[int, int, int, int, list[float] | None]]:
    """DeepFace fallback path."""
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
        rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
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


def process_frame_faces(jpeg_bytes: bytes) -> bytes:
    """
    Real-time face detection + recall (face_recognition pattern).
    Uses face_recognition (dlib) when available, else DeepFace.
    """
    global _last_detected_face_ids
    try:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jpeg_bytes

        registry = get_registry()
        _last_detected_face_ids = []

        if USE_FACE_RECOGNITION:
            faces = _process_with_face_recognition(img)
        else:
            faces = _process_with_deepface(img)

        for item in faces:
            if len(item) == 5:
                left, top, right, bottom, enc = item
            else:
                continue
            if enc is not None:
                person_id = registry.find_or_create(enc)
            else:
                person_id = registry.create_new_id()
            _last_detected_face_ids.append(person_id)
            _draw_face_box_and_name(img, left, top, right, bottom, person_id)

        _, out_buf = cv2.imencode(".jpg", img)
        return out_buf.tobytes()
    except Exception as e:
        logger.warning("Face processing failed: %s", e)
        return jpeg_bytes
