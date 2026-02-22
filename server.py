"""
FastAPI server: WebRTC signaling over WebSocket, receive A/V from browser,
push play commands to browser. Relays video as MJPEG for viewers. Serves the web app and audio files.
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
import wave
from pathlib import Path

import cv2
import httpx
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

# Audio relay: broadcast mic audio from broadcaster to viewers
_audio_connections: set[WebSocket] = set()
# HTTP fallback for audio (when WebSocket blocked by ngrok): POST chunks, SSE for viewers
_audio_sse_queues: list[asyncio.Queue[bytes]] = []
# Transcription: buffer raw PCM, send to ElevenLabs, push transcripts to viewers
_transcript_pcm_buffer: bytearray = bytearray()
_transcript_sample_rate: int = 48000
_transcript_channels: int = 1
_transcript_sse_queues: list[asyncio.Queue[str]] = []
_transcript_task: asyncio.Task | None = None
_transcript_encoded_chunks: list[bytes] = []
_transcript_encoded_task: asyncio.Task | None = None
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
# Diagnostics
_diag_audio_encoded_count: int = 0
_diag_audio_encoded_last: float = 0
_diag_audio_pcm_count: int = 0
_diag_audio_pcm_last: float = 0


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int) -> bytes:
    """Convert raw Int16 PCM to WAV format."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return buf.getvalue()


async def _transcribe_audio_with_elevenlabs(audio_bytes: bytes, content_type: str = "audio/wav") -> str | None:
    """Send WAV audio to ElevenLabs Speech-to-Text API. Returns transcript text or None."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        logger.debug("ELEVENLABS_API_KEY not set, skipping transcription")
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            ext = "wav" if "wav" in content_type else ("mp4" if "mp4" in content_type else "webm")
            files = {"file": (f"audio.{ext}", audio_bytes, content_type)}
            data = {"model_id": "scribe_v2"}
            headers = {"xi-api-key": api_key}
            resp = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                files=files,
                data=data,
                headers=headers,
            )
            if resp.status_code != 200:
                logger.warning("ElevenLabs STT error: %s %s", resp.status_code, resp.text[:200])
                return None
            out = resp.json()
            text = ""
            if isinstance(out, str):
                text = out
            elif isinstance(out, dict):
                text = out.get("text") or out.get("transcript") or out.get("transcription") or ""
            if text and isinstance(text, str):
                return text.strip() or None
            return None
    except Exception as e:
        logger.debug("ElevenLabs transcription failed: %s", e)
        return None


async def _transcript_loop():
    """Background task: buffer PCM, periodically send to ElevenLabs, push transcripts to viewers."""
    global _transcript_pcm_buffer, _transcript_sample_rate, _transcript_channels
    # Need ~1.5 seconds of audio for lower latency
    min_bytes = int(_transcript_sample_rate * 2 * _transcript_channels * 1.5)
    while True:
        await asyncio.sleep(1.2)
        buf = bytes(_transcript_pcm_buffer)
        if len(buf) < min_bytes:
            continue
        _transcript_pcm_buffer.clear()
        wav_bytes = _pcm_to_wav(buf, _transcript_sample_rate, _transcript_channels)
        text = await _transcribe_audio_with_elevenlabs(wav_bytes, "audio/wav")
        if not text:
            continue
        for q in _transcript_sse_queues:
            try:
                q.put_nowait({"text": text, "interim": False})
            except asyncio.QueueFull:
                pass


async def _transcript_encoded_loop():
    """Background task: buffer MediaRecorder chunks (MP4/WebM), periodically send to ElevenLabs."""
    global _transcript_encoded_chunks
    while True:
        await asyncio.sleep(1.2)
        chunks = list(_transcript_encoded_chunks)
        if not chunks or sum(len(c) for c in chunks) < 8000:  # min ~0.4 sec
            continue
        combined = b"".join(chunks)
        # Keep first chunk (init segment) for next batch; remove the rest
        if len(chunks) > 1:
            _transcript_encoded_chunks[:] = [chunks[0]]
        else:
            _transcript_encoded_chunks.clear()
        content_type = "audio/mp4" if b"ftyp" in combined[:32] else "audio/webm"
        text = await _transcribe_audio_with_elevenlabs(combined, content_type)
        if not text:
            continue
        for q in _transcript_sse_queues:
            try:
                q.put_nowait({"text": text, "interim": False})
            except asyncio.QueueFull:
                pass


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


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """Audio relay: broadcaster sends binary chunks, viewers receive. Broadcasts from sender to all other connections."""
    await websocket.accept()
    _audio_connections.add(websocket)
    logger.info("Audio connection; total=%d", len(_audio_connections))
    try:
        while True:
            msg = await websocket.receive()
            data = msg.get("bytes")
            if data and isinstance(data, bytes):
                # Add to transcript buffer for ElevenLabs
                if len(data) >= 8 and os.environ.get("ELEVENLABS_API_KEY"):
                    global _transcript_pcm_buffer, _transcript_sample_rate, _transcript_channels, _transcript_task
                    sr = struct.unpack_from("<I", data, 0)[0]
                    ch = struct.unpack_from("<I", data, 4)[0] or 1
                    if sr and 8000 <= sr <= 96000:
                        _transcript_sample_rate = sr
                        _transcript_channels = ch
                    _transcript_pcm_buffer.extend(data[8:])
                    if len(_transcript_pcm_buffer) > 600000:
                        _transcript_pcm_buffer = _transcript_pcm_buffer[-300000:]
                    if _transcript_task is None:
                        _transcript_task = asyncio.create_task(_transcript_loop())
                dead = set()
                for ws in _audio_connections:
                    if ws is not websocket:
                        try:
                            await ws.send_bytes(data)
                        except Exception:
                            dead.add(ws)
                for ws in dead:
                    _audio_connections.discard(ws)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("Audio WebSocket error: %s", e)
    finally:
        _audio_connections.discard(websocket)
        logger.info("Audio connection closed; total=%d", len(_audio_connections))


@app.get("/api/diagnostic")
async def api_diagnostic():
    """Diagnostic: check if server is receiving audio."""
    return {
        "audio_encoded_received": _diag_audio_encoded_count,
        "audio_encoded_last_sec_ago": round(time.time() - _diag_audio_encoded_last, 1) if _diag_audio_encoded_last else None,
        "audio_pcm_received": _diag_audio_pcm_count,
        "audio_pcm_last_sec_ago": round(time.time() - _diag_audio_pcm_last, 1) if _diag_audio_pcm_last else None,
        "audio_ws_connections": len(_audio_connections),
        "audio_sse_viewers": len(_audio_sse_queues),
        "transcript_sse_viewers": len(_transcript_sse_queues),
    }


@app.post("/api/audio/encoded")
@app.post("/api/audio-encoded")  # Alias in case path/proxy strips trailing segment
async def api_audio_encoded(request: Request):
    """Accept MediaRecorder chunks (MP4/WebM) from broadcaster. Used on iOS Safari where ScriptProcessorNode fails."""
    global _transcript_encoded_chunks, _transcript_encoded_task, _diag_audio_encoded_count, _diag_audio_encoded_last
    body = await request.body()
    if not body or len(body) < 4 or len(body) > 2 * 1024 * 1024:
        return {"ok": False}
    # Detect MP4 (ftyp at offset 4) or WebM (EBML 0x1A45DFA3)
    is_mp4 = len(body) >= 8 and body[4:8] == b"ftyp"
    is_webm = len(body) >= 4 and body[0:4] == bytes([0x1A, 0x45, 0xDF, 0xA3])
    if not (is_mp4 or is_webm):
        if _diag_audio_encoded_count < 3:  # Log first few rejections
            logger.info("Audio encoded: rejected (not mp4/webm), len=%d, first16=%s", len(body), body[:16].hex() if len(body) >= 16 else (body.hex() if body else ""))
        return {"ok": False}
    _diag_audio_encoded_count += 1
    _diag_audio_encoded_last = time.time()
    # Relay to Mac viewers for playback (WebSocket + SSE)
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
    if os.environ.get("ELEVENLABS_API_KEY"):
        _transcript_encoded_chunks.append(body)
        while sum(len(c) for c in _transcript_encoded_chunks) > 500000:  # ~10 sec max
            _transcript_encoded_chunks.pop(0)
        if _transcript_encoded_task is None:
            _transcript_encoded_task = asyncio.create_task(_transcript_encoded_loop())
    return {"ok": True}


@app.post("/api/audio")
async def api_audio(request: Request):
    """HTTP fallback: accept raw PCM chunks from broadcaster. Used when WebSocket is blocked (e.g. ngrok)."""
    global _transcript_pcm_buffer, _transcript_sample_rate, _transcript_channels, _transcript_task, _diag_audio_pcm_count, _diag_audio_pcm_last
    body = await request.body()
    if not body or len(body) < 8 or len(body) > 256 * 1024:
        return {"ok": False}
    _diag_audio_pcm_count += 1
    _diag_audio_pcm_last = time.time()
    # Add to transcript buffer (8-byte header: sample_rate uint32, channels uint32; rest is PCM)
    if len(body) >= 8 and os.environ.get("ELEVENLABS_API_KEY"):
        sr = struct.unpack_from("<I", body, 0)[0]
        ch = struct.unpack_from("<I", body, 4)[0] or 1
        if sr and 8000 <= sr <= 96000:
            _transcript_sample_rate = sr
            _transcript_channels = ch
        _transcript_pcm_buffer.extend(body[8:])
        if len(_transcript_pcm_buffer) > 600000:  # cap at ~6 sec
            _transcript_pcm_buffer = _transcript_pcm_buffer[-300000:]
        if _transcript_task is None:
            _transcript_task = asyncio.create_task(_transcript_loop())
    # Broadcast to WebSocket viewers
    dead = set()
    for ws in _audio_connections:
        try:
            await ws.send_bytes(body)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _audio_connections.discard(ws)
    # Push to SSE viewers
    for q in _audio_sse_queues:
        try:
            q.put_nowait(body)
        except asyncio.QueueFull:
            pass
    return {"ok": True}


@app.get("/api/audio/stream")
async def api_audio_stream():
    """SSE fallback: stream audio chunks to viewer. Used when WebSocket is blocked (e.g. ngrok)."""

    async def generate():
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
        _audio_sse_queues.append(q)
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=30.0)
                    b64 = base64.b64encode(chunk).decode("ascii")
                    yield f"data: {b64}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _audio_sse_queues.remove(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/api/transcript")
async def api_transcript(body: dict = Body(default={})):
    """Accept transcript text from broadcaster (e.g. Web Speech API on iPhone). Pushes to transcript SSE viewers."""
    text = (body.get("text") or body.get("transcript") or "").strip()
    if not text:
        return {"ok": False}
    interim = body.get("interim", False)
    msg = {"text": text, "interim": interim}
    for q in _transcript_sse_queues:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass
    return {"ok": True}


@app.get("/api/transcript/stream")
async def api_transcript_stream():
    """SSE stream of live transcripts (from Web Speech API or ElevenLabs)."""

    async def generate():
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        _transcript_sse_queues.append(q)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=60.0)
                    payload = json.dumps(msg) if isinstance(msg, dict) else json.dumps({"text": msg, "interim": False})
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _transcript_sse_queues.remove(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
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
