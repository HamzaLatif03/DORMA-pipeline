"""Face detection + emotion analysis. Pure processing, no FastAPI."""
import logging

import cv2
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)


def process_frame_faces(jpeg_bytes: bytes) -> bytes:
    """Run face detection + emotion on a JPEG frame; draw boxes and labels; return annotated JPEG."""
    try:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jpeg_bytes
        analyses = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False, silent=True)
        if not isinstance(analyses, list):
            analyses = [analyses]
        for entry in analyses:
            region = entry.get("region") or entry.get("facial_area")
            if not region:
                continue
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)
            emotion = entry.get("dominant_emotion", "?")
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _, out_buf = cv2.imencode(".jpg", img)
        return out_buf.tobytes()
    except Exception as e:
        logger.debug("Face/emotion processing failed: %s", e)
        return jpeg_bytes
