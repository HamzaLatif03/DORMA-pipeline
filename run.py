"""
FastAPI server: WebRTC signaling, video relay, face recognition, audio/transcript.
"""
import asyncio
import base64
import io
import json
import logging
import os
import struct
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

from face import get_last_detected_ids, get_registry, process_frame
from transcript import add_encoded, add_pcm, get_sse_queue_count, get_sse_queues, push_text
from vitals import get_vitals_collector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="iPhone peripherals bridge")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
connected_clients: set[WebSocket] = set()
peer_connections: dict[WebSocket, RTCPeerConnection] = {}
_audio_connections: set[WebSocket] = set()
_latest_jpeg: bytes | None = None
_frame_event = asyncio.Event()
_http_broadcaster_pc: RTCPeerConnection | None = None
_http_broadcaster_session_id: str | None = None
_http_frame_session_id: str | None = None
_http_play_commands: list[dict] = []
_audio_sse_queues: list[asyncio.Queue] = []
_diag_audio_encoded_count: int = 0
_diag_audio_encoded_last: float = 0
_diag_audio_pcm_count: int = 0
_diag_audio_pcm_last: float = 0

# -----------------------------------------------------------------------------
# Video
# -----------------------------------------------------------------------------

def _on_video_track(track):
    if track.kind == "video":
        asyncio.ensure_future(_consume_video_track(track))
    else:
        bh = MediaBlackhole()
        bh.addTrack(track)
        asyncio.ensure_future(bh.start())

async def _consume_video_track(track):
    global _latest_jpeg
    try:
        while True:
            frame = await track.recv()
            try:
                nd = frame.to_ndarray(format="rgb24") if hasattr(frame, "to_ndarray") else frame.to_ndarray()
                img = Image.fromarray(nd)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=85)
                _latest_jpeg = buf.getvalue()
                _frame_event.set()
            except Exception as e:
                logger.debug("Frame encode: %s", e)
    except Exception as e:
        logger.info("Video track ended: %s", e)


# -----------------------------------------------------------------------------
# WebSocket / Signaling
# -----------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    pc: RTCPeerConnection | None = None

    def on_track(track):
        logger.info("Received %s track", track.kind)
        _on_video_track(track)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")
            if msg_type != "offer":
                continue
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
                    if pc.connectionState in ("failed", "closed"):
                        await pc.close()
                        peer_connections.pop(websocket, None)

                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                deadline = time.monotonic() + 5.0
                while pc.iceGatheringState != "complete":
                    if time.monotonic() > deadline:
                        logger.warning("ICE gathering timeout")
                        break
                    await asyncio.sleep(0.1)
                await websocket.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
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
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------

async def _broadcast_audio(body: bytes):
    dead = set()
    for ws in _audio_connections:
        try:
            await ws.send_bytes(body)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _audio_connections.discard(ws)
    for q in _audio_sse_queues:
        try:
            q.put_nowait(body)
        except asyncio.QueueFull:
            pass

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    _audio_connections.add(websocket)
    try:
        while True:
            msg = await websocket.receive()
            data = msg.get("bytes")
            if data and isinstance(data, bytes):
                if len(data) >= 8 and os.environ.get("ELEVENLABS_API_KEY"):
                    sr = struct.unpack_from("<I", data, 0)[0]
                    ch = struct.unpack_from("<I", data, 4)[0] or 1
                    add_pcm(data[8:], sr, ch)
                dead = set()
                for ws in _audio_connections:
                    if ws is websocket:
                        continue
                    try:
                        await ws.send_bytes(data)
                    except Exception:
                        dead.add(ws)
                for ws in dead:
                    _audio_connections.discard(ws)
    except WebSocketDisconnect:
        pass
    finally:
        _audio_connections.discard(websocket)


@app.post("/api/audio/encoded")
async def api_audio_encoded(request: Request):
    global _diag_audio_encoded_count, _diag_audio_encoded_last
    body = await request.body()
    if not body or len(body) < 4 or len(body) > 2 * 1024 * 1024:
        return {"ok": False}
    is_mp4 = len(body) >= 8 and body[4:8] == b"ftyp"
    is_webm = len(body) >= 4 and body[0:4] == bytes([0x1A, 0x45, 0xDF, 0xA3])
    if not (is_mp4 or is_webm):
        return {"ok": False}
    _diag_audio_encoded_count += 1
    _diag_audio_encoded_last = time.time()
    await _broadcast_audio(body)
    if os.environ.get("ELEVENLABS_API_KEY"):
        add_encoded(body)
    return {"ok": True}


@app.post("/api/audio")
async def api_audio(request: Request):
    global _diag_audio_pcm_count, _diag_audio_pcm_last
    body = await request.body()
    if not body or len(body) < 8 or len(body) > 256 * 1024:
        return {"ok": False}
    _diag_audio_pcm_count += 1
    _diag_audio_pcm_last = time.time()
    if os.environ.get("ELEVENLABS_API_KEY"):
        sr = struct.unpack_from("<I", body, 0)[0]
        ch = struct.unpack_from("<I", body, 4)[0] or 1
        add_pcm(body[8:], sr, ch)
    await _broadcast_audio(body)
    return {"ok": True}


@app.get("/api/audio/stream")
async def api_audio_stream():
    async def generate():
        q = asyncio.Queue(maxsize=30)
        _audio_sse_queues.append(q)
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {base64.b64encode(chunk).decode('ascii')}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            try:
                _audio_sse_queues.remove(q)
            except ValueError:
                pass

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


# -----------------------------------------------------------------------------
# Transcript
# -----------------------------------------------------------------------------


@app.post("/api/transcript")
async def api_transcript(body: dict = Body(default={})):
    text = (body.get("text") or body.get("transcript") or "").strip()
    if not text:
        return {"ok": False}
    push_text(text, body.get("interim", False))
    return {"ok": True}


@app.get("/api/transcript/stream")
async def api_transcript_stream():
    async def generate():
        q = asyncio.Queue(maxsize=100)
        get_sse_queues().append(q)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=60.0)
                    payload = json.dumps(msg if isinstance(msg, dict) else {"text": msg, "interim": False})
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            try:
                get_sse_queues().remove(q)
            except ValueError:
                pass

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


# -----------------------------------------------------------------------------
# Face API
# -----------------------------------------------------------------------------


@app.get("/api/faces")
async def api_list_faces():
    return {"faces": get_registry().list_all()}


@app.get("/api/faces/detected")
async def api_faces_detected():
    return {"person_ids": get_last_detected_ids()}


@app.get("/api/faces/{person_id}")
async def api_get_face(person_id: str):
    ent = get_registry().get(person_id)
    if not ent:
        raise HTTPException(status_code=404, detail="Person not found")
    out = {k: v for k, v in ent.items() if k not in ("embedding", "embeddings")}
    out["person_id"] = person_id
    return out


@app.patch("/api/faces/{person_id}")
async def api_update_face(person_id: str, body: dict = Body(default={})):
    if not get_registry().update_metadata(person_id, **body):
        raise HTTPException(status_code=404, detail="Person not found")
    return {"ok": True, "person_id": person_id}


@app.get("/api/diagnostic")
async def api_diagnostic():
    return {
        "audio_encoded_received": _diag_audio_encoded_count,
        "audio_encoded_last_sec_ago": round(time.time() - _diag_audio_encoded_last, 1) if _diag_audio_encoded_last else None,
        "audio_pcm_received": _diag_audio_pcm_count,
        "audio_pcm_last_sec_ago": round(time.time() - _diag_audio_pcm_last, 1) if _diag_audio_pcm_last else None,
        "audio_ws_connections": len(_audio_connections),
        "transcript_sse_viewers": get_sse_queue_count(),
        "audio_sse_viewers": len(_audio_sse_queues),
    }


# -----------------------------------------------------------------------------
# Frame upload & HTTP signaling
# -----------------------------------------------------------------------------


@app.post("/api/frame")
async def api_frame(request: Request):
    global _latest_jpeg, _frame_event, _http_frame_session_id
    body = await request.body()
    if not body or len(body) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Empty or too large")
    _latest_jpeg = await asyncio.to_thread(process_frame, body)
    _frame_event.set()
    get_vitals_collector().add_frame(body)
    if _http_frame_session_id is None:
        _http_frame_session_id = str(uuid.uuid4())
    return {"session_id": _http_frame_session_id}


@app.get("/api/vitals")
async def api_vitals():
    return get_vitals_collector().get_latest()


@app.post("/api/signal")
async def api_signal(body: dict = Body(default={})):
    global _http_broadcaster_pc, _http_broadcaster_session_id
    if body.get("type") != "offer" or not body.get("sdp"):
        raise HTTPException(status_code=400, detail="Need { \"type\": \"offer\", \"sdp\": \"...\" }")
    if _http_broadcaster_pc:
        try:
            await _http_broadcaster_pc.close()
        except Exception:
            pass
        _http_broadcaster_pc = None
        _http_broadcaster_session_id = None
    _http_play_commands.clear()
    offer = RTCSessionDescription(sdp=body["sdp"], type="offer")
    pc = RTCPeerConnection()
    _http_broadcaster_pc = pc
    _http_broadcaster_session_id = str(uuid.uuid4())
    def on_track_http(track):
        logger.info("Received %s track (HTTP)", track.kind)
        _on_video_track(track)
    pc.on("track", on_track_http)
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
    if session_id not in (_http_broadcaster_session_id, _http_frame_session_id):
        return {"pending": True}
    if not _http_play_commands:
        return {"pending": True}
    return _http_play_commands.pop(0)


# -----------------------------------------------------------------------------
# Play & static
# -----------------------------------------------------------------------------


@app.post("/api/play")
async def api_play(body: dict = Body(default={})):
    file, url = body.get("file"), body.get("url")
    if not file and not url:
        return {"ok": False, "error": "Provide 'file' or 'url'"}
    base = os.environ.get("SERVER_BASE", "http://localhost:8000")
    play_url = f"{base}/audio/{file}" if file else url
    payload = {"action": "play", "url": play_url}
    for ws in list(connected_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            connected_clients.discard(ws)
    _http_play_commands.append(payload)
    return {"ok": True, "url": play_url}


@app.get("/api/ping")
async def api_ping():
    return {"ok": True}


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = (AUDIO_DIR / filename).resolve()
    if not path.is_file() or not str(path).startswith(str(AUDIO_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="audio/mpeg" if filename.endswith(".mp3") else "audio/wav")


@app.get("/api/audio")
async def list_audio():
    files = [f.name for f in AUDIO_DIR.iterdir() if f.suffix.lower() in (".mp3", ".wav", ".m4a")]
    return {"files": sorted(files)}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Put index.html in the static/ directory"}
    return FileResponse(index_path)
