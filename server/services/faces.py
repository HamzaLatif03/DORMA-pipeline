"""Face detection using OpenCV Haar cascade; draw boxes, face IDs and emotion."""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CASCADE = None


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def _draw_face_box_and_label(img: np.ndarray, x: int, y: int, w: int, h: int, label: str) -> None:
    """Draw a red box and label (face ID and optional emotion) using OpenCV."""
    left, top, right, bottom = x, y, x + w, y + h
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(img, (left, bottom - 28), (right, bottom), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, label, (left + 4, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def _emotion_for_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Run DeepFace emotion on a face crop; return emotion string or '?' on failure."""
    try:
        from deepface import DeepFace
        crop = img[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            return "?"
        analyses = DeepFace.analyze(crop, actions=["emotion"], enforce_detection=False, detector_backend="skip", silent=True)
        entry = analyses[0] if isinstance(analyses, list) and analyses else analyses
        return (entry.get("dominant_emotion") or "?").strip()
    except Exception:
        return "?"


def process_frame_faces(jpeg_bytes: bytes) -> bytes:
    """Detect faces with OpenCV Haar cascade; draw boxes + face ID + emotion. OpenCV-only detection."""
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
        cascade = _get_cascade()
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if rects is None or len(rects) == 0:
            _, out_buf = cv2.imencode(".jpg", img)
            return out_buf.tobytes()
        for i, (x, y, w, h) in enumerate(rects):
            x, y, w, h = int(x), int(y), int(w), int(h)
            emotion = _emotion_for_crop(img, x, y, w, h)
            label = emotion if emotion else "?"
            _draw_face_box_and_label(img, x, y, w, h, label)
        _, out_buf = cv2.imencode(".jpg", img)
        return out_buf.tobytes()
    except Exception as e:
        logger.debug("Face processing failed: %s", e)
        return jpeg_bytes
