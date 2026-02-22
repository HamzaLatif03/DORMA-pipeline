"""
FastAPI server: WebRTC signaling over WebSocket, receive A/V from browser,
push play commands to browser. Relays video as MJPEG for viewers. Serves the web app and audio files.
"""
import asyncio
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="iPhone peripherals bridge")

# Directory for uploaded/streamed media and for serving audio
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Connected WebSocket clients (for pushing play commands)
connected_clients: set[WebSocket] = set()
# One peer connection per client (keyed by WebSocket)
peer_connections: dict[WebSocket, RTCPeerConnection] = {}

# Video relay: latest JPEG from broadcaster for MJPEG /stream
_latest_jpeg: bytes | None = None
_frame_event = asyncio.Event()

# HTTP signaling fallback (when WebSocket is blocked)
_http_broadcaster_pc: RTCPeerConnection | None = None
_http_broadcaster_session_id: str | None = None
# HTTP frame upload (iPhone sends JPEGs via POST - works when WebRTC is blocked)
_http_frame_session_id: str | None = None
_http_play_commands: list[dict] = []


def _process_frame_faces(jpeg_bytes: bytes) -> bytes:
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


async def _consume_video_track(track):
    """Read video frames from the WebRTC track, encode as JPEG, update global for /stream."""
    global _latest_jpeg
    try:
        while True:
            frame = await track.recv()
            try:
                try:
                    nd = frame.to_ndarray(format="rgb24")
                except Exception:
                    nd = frame.to_ndarray()
                img = Image.fromarray(nd)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=85)
                _latest_jpeg = buf.getvalue()
                _frame_event.set()
            except Exception as e:
                logger.debug("Frame encode: %s", e)
    except Exception as e:
        logger.info("Video track ended: %s", e)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    pc: RTCPeerConnection | None = None

    def on_track(track):
        logger.info("Received %s track", track.kind)
        if track.kind == "video":
            asyncio.ensure_future(_consume_video_track(track))
        else:
            blackhole = MediaBlackhole()
            blackhole.addTrack(track)
            asyncio.ensure_future(blackhole.start())

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "offer":
                sdp = msg.get("sdp")
                if not sdp:
                    logger.warning("Offer missing sdp")
                    continue
                logger.info("Received offer, creating answer...")
                try:
                    offer = RTCSessionDescription(sdp=sdp, type="offer")
                    pc = RTCPeerConnection()
                    peer_connections[websocket] = pc
                    pc.on("track", on_track)

                    @pc.on("connectionstatechange")
                    async def on_connectionstatechange():
                        if pc.connectionState == "failed" or pc.connectionState == "closed":
                            await pc.close()
                            peer_connections.pop(websocket, None)

                    await pc.setRemoteDescription(offer)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    deadline = time.monotonic() + 5.0
                    while pc.iceGatheringState != "complete":
                        if time.monotonic() > deadline:
                            logger.warning("ICE gathering timeout, sending answer anyway")
                            break
                        await asyncio.sleep(0.1)
                    await websocket.send_json({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp,
                    })
                    logger.info("Sent WebRTC answer to client")
                except Exception as e:
                    logger.exception("WebRTC offer handling failed: %s", e)
                    try:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    except Exception:
                        pass
                    if pc and websocket in peer_connections:
                        await pc.close()
                        peer_connections.pop(websocket, None)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        connected_clients.discard(websocket)
        if websocket in peer_connections:
            await peer_connections[websocket].close()
            peer_connections.pop(websocket, None)


@app.get("/stream")
async def stream_mjpeg():
    """Stream the latest video from the broadcaster as MJPEG. Viewers use this in an img or video tag."""
    boundary = "frame"

    async def generate():
        global _latest_jpeg
        while True:
            _frame_event.clear()
            await _frame_event.wait()
            jpeg = _latest_jpeg
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(0)  # yield control

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.get("/api/ping")
async def api_ping():
    """Simple connectivity check. If the iPhone can load this URL, it can reach the server."""
    return {"ok": True}


@app.post("/api/frame")
async def api_frame(request: Request):
    """Accept a JPEG frame from the broadcaster (e.g. iPhone). Runs face detection + emotion, draws boxes; updates MJPEG /stream. Returns session_id for polling play commands."""
    global _latest_jpeg, _frame_event, _http_frame_session_id
    body = await request.body()
    if not body or len(body) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Empty or too large")
    _latest_jpeg = await asyncio.to_thread(_process_frame_faces, body)
    _frame_event.set()

    nparr = np.frombuffer(body, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        pass

    if _http_frame_session_id is None:
        _http_frame_session_id = str(uuid.uuid4())
    return {"session_id": _http_frame_session_id}


def _on_track_http(track):
    """Same as on_track for WebSocket: relay video, blackhole audio."""
    logger.info("Received %s track (HTTP broadcaster)", track.kind)
    if track.kind == "video":
        asyncio.ensure_future(_consume_video_track(track))
    else:
        blackhole = MediaBlackhole()
        blackhole.addTrack(track)
        asyncio.ensure_future(blackhole.start())


@app.post("/api/signal")
async def api_signal(body: dict = Body(default={})):
    """HTTP fallback for signaling when WebSocket is blocked. Body: { \"type\": \"offer\", \"sdp\": \"...\" }. Returns answer and session_id for polling play commands."""
    global _http_broadcaster_pc, _http_broadcaster_session_id
    sdp = body.get("sdp")
    if not sdp or body.get("type") != "offer":
        raise HTTPException(status_code=400, detail="Need { \"type\": \"offer\", \"sdp\": \"...\" }")
    # Close previous HTTP broadcaster if any
    if _http_broadcaster_pc:
        try:
            await _http_broadcaster_pc.close()
        except Exception:
            pass
        _http_broadcaster_pc = None
        _http_broadcaster_session_id = None
    _http_play_commands.clear()
    offer = RTCSessionDescription(sdp=sdp, type="offer")
    pc = RTCPeerConnection()
    _http_broadcaster_pc = pc
    _http_broadcaster_session_id = str(uuid.uuid4())
    pc.on("track", _on_track_http)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    deadline = time.monotonic() + 5.0
    while pc.iceGatheringState != "complete":
        if time.monotonic() > deadline:
            break
        await asyncio.sleep(0.1)
    return {"type": "answer", "sdp": pc.localDescription.sdp, "session_id": _http_broadcaster_session_id}


@app.get("/api/poll")
async def api_poll(session_id: str = ""):
    """Poll for play commands (HTTP broadcaster or frame uploader). Returns next command or { \"pending\": true }."""
    global _http_play_commands
    if session_id not in (_http_broadcaster_session_id, _http_frame_session_id):
        return {"pending": True}
    if not _http_play_commands:
        return {"pending": True}
    cmd = _http_play_commands.pop(0)
    return cmd


@app.post("/api/play")
async def api_play(body: dict = Body(default={})):
    """Request that all connected clients play an audio file or URL. Body: { \"file\": \"name.mp3\" } or { \"url\": \"https://...\" }."""
    file = body.get("file")
    url = body.get("url")
    if not file and not url:
        return {"ok": False, "error": "Provide 'file' or 'url'"}
    base = os.environ.get("SERVER_BASE", "http://localhost:8000")
    if file:
        play_url = f"{base}/audio/{file}"
    else:
        play_url = url
    payload = {"action": "play", "url": play_url}
    dead = set()
    for ws in connected_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        connected_clients.discard(ws)
    _http_play_commands.append(payload)
    return {"ok": True, "url": play_url}


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve an audio file from the audio directory."""
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = (AUDIO_DIR / filename).resolve()
    if not path.is_file() or not str(path).startswith(str(AUDIO_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="audio/mpeg" if filename.endswith(".mp3") else "audio/wav")


@app.get("/api/audio")
async def list_audio():
    """List available audio files."""
    files = [f.name for f in AUDIO_DIR.iterdir() if f.suffix.lower() in (".mp3", ".wav", ".m4a")]
    return {"files": sorted(files)}


# Serve static files (web app)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the main web app (iPhone client)."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Put index.html in the static/ directory"}
    return FileResponse(index_path)
