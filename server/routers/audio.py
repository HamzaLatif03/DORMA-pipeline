"""Audio relay: /ws/audio WebSocket, POST /api/audio, POST /api/audio/encoded, GET /api/audio/stream (SSE)."""
import asyncio
import base64
import logging
import os
import struct
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse

from .. import state
from ..services.transcribe import transcript_pcm_loop, transcript_encoded_loop

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """Broadcaster sends binary audio chunks; relay to all other connections and SSE queues."""
    await websocket.accept()
    state.audio_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()

            dead = set()
            for ws in state.audio_connections:
                if ws is websocket:
                    continue
                try:
                    await ws.send_bytes(data)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                state.audio_connections.discard(ws)

            for q in list(state.audio_sse_queues):
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass

            if os.environ.get("ELEVENLABS_API_KEY") and len(data) > 8:
                sample_rate, channels = struct.unpack("<II", data[:8])
                state.transcript_sample_rate = sample_rate
                state.transcript_channels = channels
                state.transcript_pcm_buffer.extend(data[8:])
                if state.transcript_task is None or state.transcript_task.done():
                    state.transcript_task = asyncio.create_task(transcript_pcm_loop())
    except WebSocketDisconnect:
        pass
    finally:
        state.audio_connections.discard(websocket)


async def _handle_encoded_audio(request: Request):
    """Shared handler for encoded audio (MediaRecorder chunks)."""
    body = await request.body()
    state.diag_audio_encoded_count += 1
    state.diag_audio_encoded_last = time.time()
    state.transcript_encoded_chunks.append(body)

    if os.environ.get("ELEVENLABS_API_KEY"):
        if state.transcript_encoded_task is None or state.transcript_encoded_task.done():
            state.transcript_encoded_task = asyncio.create_task(transcript_encoded_loop())

    for q in list(state.audio_sse_queues):
        try:
            q.put_nowait(body)
        except asyncio.QueueFull:
            pass
    return {"ok": True}


@router.post("/api/audio/encoded")
async def api_audio_encoded(request: Request):
    """Accept encoded audio from MediaRecorder (MP4/WebM)."""
    return await _handle_encoded_audio(request)


@router.post("/api/audio-encoded")
async def api_audio_encoded_alt(request: Request):
    """Alias path for encoded audio."""
    return await _handle_encoded_audio(request)


@router.post("/api/audio")
async def api_audio(request: Request):
    """Accept raw PCM chunk via HTTP (8-byte header: uint32 sample_rate + uint32 channels + PCM data)."""
    body = await request.body()
    state.diag_audio_pcm_count += 1
    state.diag_audio_pcm_last = time.time()

    if len(body) > 8:
        sample_rate, channels = struct.unpack("<II", body[:8])
        state.transcript_sample_rate = sample_rate
        state.transcript_channels = channels
        state.transcript_pcm_buffer.extend(body[8:])
        if os.environ.get("ELEVENLABS_API_KEY"):
            if state.transcript_task is None or state.transcript_task.done():
                state.transcript_task = asyncio.create_task(transcript_pcm_loop())

    dead = set()
    for ws in state.audio_connections:
        try:
            await ws.send_bytes(body)
        except Exception:
            dead.add(ws)
    for ws in dead:
        state.audio_connections.discard(ws)

    for q in list(state.audio_sse_queues):
        try:
            q.put_nowait(body)
        except asyncio.QueueFull:
            pass
    return {"ok": True}


@router.get("/api/audio/stream")
async def audio_stream():
    """SSE stream of audio chunks (base64-encoded) for viewers."""
    q: asyncio.Queue[bytes] = asyncio.Queue()
    state.audio_sse_queues.append(q)

    async def generate():
        try:
            while True:
                data = await q.get()
                encoded = base64.b64encode(data).decode()
                yield f"data: {encoded}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in state.audio_sse_queues:
                state.audio_sse_queues.remove(q)

    return StreamingResponse(generate(), media_type="text/event-stream")
