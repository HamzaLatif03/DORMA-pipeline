"""
Presage Technologies vitals integration.
Buffers incoming JPEG frames into a short video clip, processes via the
SmartSpectra C++ SDK binary, and stores the latest HR / RR results.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PRESAGE_API_KEY = os.environ.get("PRESAGE_API_KEY", "")
PRESAGE_BINARY = os.environ.get("PRESAGE_BINARY", "presage_processor")
CLIP_DURATION_SEC = 30
TARGET_FPS = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _sdk_available() -> bool:
    """Check whether the presage_processor binary is on PATH."""
    return shutil.which(PRESAGE_BINARY) is not None


def _process_with_sdk(api_key: str, video_path: str, timeout: float = 120) -> dict | None:
    """Run the C++ SmartSpectra binary on a video file and return parsed JSON."""
    cmd = [
        PRESAGE_BINARY,
        f"--api_key={api_key}",
        f"--video={video_path}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr_msg = result.stderr.strip()[-200:] if result.stderr else ""
            logger.warning("presage_processor exited %d: %s", result.returncode, stderr_msg)

        stdout = result.stdout.strip()
        if not stdout:
            return None

        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                if "error" in data:
                    logger.warning("presage_processor error: %s", data["error"])
                    return None
                return data
        return None

    except subprocess.TimeoutExpired:
        logger.warning("presage_processor timed out after %ds", timeout)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("presage_processor call failed: %s", exc)
        return None


class VitalsCollector:
    """Accumulates JPEG frames into a ring buffer.  Every CLIP_DURATION_SEC
    seconds the buffer is flushed to an MP4 file which is processed by
    the SmartSpectra C++ SDK.  Results are stored in .latest_result.
    """

    def __init__(self, api_key: str = PRESAGE_API_KEY):
        self.api_key = api_key
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._clip_start: float = 0.0
        self._frame_count: int = 0
        self._receiving: bool = False
        self.latest_result: dict | None = None
        self.processing: bool = False
        self._last_error: str | None = None
        self._sdk_ok = _sdk_available()

        if not self.api_key:
            logger.warning("PRESAGE_API_KEY not set — vitals disabled")
        elif not self._sdk_ok:
            logger.warning(
                "presage_processor binary not found on PATH — "
                "vitals will be unavailable until running inside the Docker container"
            )

    def add_frame(self, jpeg_bytes: bytes) -> None:
        """Decode a JPEG, resize to target dims, and append to the buffer."""
        if not self.api_key:
            return
        try:
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return
            img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
            with self._lock:
                if not self._receiving:
                    self._receiving = True
                    self._clip_start = time.time()
                self._frames.append(img)
                self._frame_count = len(self._frames)
                elapsed = time.time() - self._clip_start
                if elapsed >= CLIP_DURATION_SEC and not self.processing:
                    self._flush()
        except Exception as e:
            logger.debug("vitals add_frame error: %s", e)

    def _flush(self) -> None:
        """Write buffered frames to a temp MP4 and kick off processing."""
        frames = list(self._frames)
        self._frames.clear()
        self._clip_start = time.time()
        self._receiving = False
        if len(frames) < TARGET_FPS * 5:
            self._last_error = f"Not enough frames ({len(frames)}), need {TARGET_FPS * 5}+"
            logger.info("Vitals: %s", self._last_error)
            return
        self._last_error = None
        self.processing = True
        t = threading.Thread(target=self._write_and_process, args=(frames,), daemon=True)
        t.start()

    def _write_and_process(self, frames: list[np.ndarray]) -> None:
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, TARGET_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            for f in frames:
                writer.write(f)
            writer.release()

            if not self._sdk_ok:
                self._last_error = (
                    "presage_processor binary not available — "
                    "deploy inside the Docker container to enable vitals"
                )
                logger.warning("Vitals: %s", self._last_error)
                return

            logger.info("Vitals: processing %d-frame clip via SDK (%s)", len(frames), tmp_path)
            data = _process_with_sdk(self.api_key, tmp_path)

            if data:
                self.latest_result = {
                    "hr": data.get("hr"),
                    "rr": data.get("rr"),
                    "hrv_sdnn": data.get("hrv_sdnn"),
                    "hrv_rmssd": data.get("hrv_rmssd"),
                    "timestamp": time.time(),
                }
                self._last_error = None
                logger.info(
                    "Vitals result: HR=%s  RR=%s  HRV_SDNN=%s  HRV_RMSSD=%s",
                    data.get("hr"), data.get("rr"),
                    data.get("hrv_sdnn"), data.get("hrv_rmssd"),
                )
            else:
                self._last_error = "No vitals detected from clip"
                logger.warning("Vitals: %s", self._last_error)
        except Exception as e:
            self._last_error = str(e)
            logger.warning("Vitals processing failed: %s", e)
        finally:
            self.processing = False
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def get_latest(self) -> dict:
        """Return the most recent vitals reading or status info."""
        if not self.api_key:
            return {"status": "no_api_key", "message": "Set PRESAGE_API_KEY in .env"}
        if not self._sdk_ok:
            return {
                "status": "sdk_unavailable",
                "message": "presage_processor not found — deploy in Docker to enable",
                "frames": self._frame_count,
                "processing": self.processing,
            }
        if self.latest_result:
            age = time.time() - self.latest_result.get("timestamp", 0)
            result = {
                "status": "ok",
                "hr": self.latest_result.get("hr"),
                "rr": self.latest_result.get("rr"),
                "age_seconds": round(age, 1),
                "processing": self.processing,
                "frames": self._frame_count,
            }
            if self.latest_result.get("hrv_sdnn") is not None:
                result["hrv_sdnn"] = self.latest_result["hrv_sdnn"]
            if self.latest_result.get("hrv_rmssd") is not None:
                result["hrv_rmssd"] = self.latest_result["hrv_rmssd"]
            return result
        if self.processing:
            return {"status": "processing", "frames": self._frame_count}
        if self._last_error:
            return {"status": "error", "message": self._last_error, "frames": self._frame_count}
        if not self._receiving:
            return {"status": "idle", "frames": 0}
        elapsed = time.time() - self._clip_start
        remaining = max(0, CLIP_DURATION_SEC - elapsed)
        return {
            "status": "buffering",
            "frames": self._frame_count,
            "seconds_left": round(remaining),
        }


_collector: VitalsCollector | None = None


def get_vitals_collector() -> VitalsCollector:
    global _collector
    if _collector is None:
        _collector = VitalsCollector()
    return _collector
