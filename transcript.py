"""
Live transcription: buffer audio, send to ElevenLabs STT, push to SSE viewers.
"""
import asyncio
import io
import logging
import os
import wave

import httpx

logger = logging.getLogger(__name__)

_transcript_pcm_buffer: bytearray = bytearray()
_transcript_sample_rate: int = 48000
_transcript_channels: int = 1
_transcript_sse_queues: list[asyncio.Queue] = []
_transcript_task: asyncio.Task | None = None
_transcript_encoded_chunks: list[bytes] = []
_transcript_encoded_task: asyncio.Task | None = None


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return buf.getvalue()


async def _transcribe(audio_bytes: bytes, content_type: str = "audio/wav") -> str | None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            ext = "wav" if "wav" in content_type else ("mp4" if "mp4" in content_type else "webm")
            files = {"file": (f"audio.{ext}", audio_bytes, content_type)}
            data = {"model_id": "scribe_v2"}
            resp = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                files=files, data=data, headers={"xi-api-key": api_key},
            )
            if resp.status_code != 200:
                return None
            out = resp.json()
            text = out.get("text") or out.get("transcript") or out.get("transcription") or "" if isinstance(out, dict) else str(out)
            return text.strip() or None
    except Exception as e:
        logger.debug("Transcription failed: %s", e)
        return None


async def _pcm_loop():
    global _transcript_pcm_buffer
    min_bytes = int(_transcript_sample_rate * 2 * _transcript_channels * 1.5)
    while True:
        await asyncio.sleep(1.2)
        buf = bytes(_transcript_pcm_buffer)
        if len(buf) < min_bytes:
            continue
        _transcript_pcm_buffer.clear()
        wav = _pcm_to_wav(buf, _transcript_sample_rate, _transcript_channels)
        text = await _transcribe(wav, "audio/wav")
        if not text:
            continue
        for q in _transcript_sse_queues:
            try:
                q.put_nowait({"text": text, "interim": False})
            except asyncio.QueueFull:
                pass


async def _encoded_loop():
    global _transcript_encoded_chunks
    while True:
        await asyncio.sleep(2)
        chunks = list(_transcript_encoded_chunks)
        if not chunks:
            continue
        _transcript_encoded_chunks.clear()
        # Prefer a chunk that looks like media (WebM Cluster) or take largest; need enough bytes for STT
        def _is_webm_cluster(b: bytes) -> bool:
            return len(b) >= 4 and b[0:4] == bytes([0x1F, 0x43, 0xB6, 0x75])
        media_like = [c for c in chunks if _is_webm_cluster(c) and len(c) >= 300]
        audio_data = max(media_like, key=len) if media_like else max(chunks, key=len)
        if len(audio_data) < 300:
            continue
        content_type = "audio/mp4" if len(audio_data) >= 8 and audio_data[4:8] == b"ftyp" else "audio/webm"
        text = await _transcribe(audio_data, content_type)
        if not text:
            continue
        for q in _transcript_sse_queues:
            try:
                q.put_nowait({"text": text, "interim": False})
            except asyncio.QueueFull:
                pass
        # Trigger reply + TTS
        try:
            from mongodb.reply_api import get_conversational_reply
            from mongodb.voice_11labs import speak_text
            reply_text = get_conversational_reply(text)
            print("[TTS] Reply for TTS (run.py transcript):", repr((reply_text or "")[:200]))
            if reply_text:
                server_base = os.environ.get("SERVER_BASE", "http://localhost:8000")
                speak_text(reply_text, play_after=True, server_base=server_base)
        except Exception as e:
            logger.warning("Reply+TTS failed: %s", e)


def add_pcm(data: bytes, sample_rate: int, channels: int) -> None:
    global _transcript_pcm_buffer, _transcript_sample_rate, _transcript_channels, _transcript_task
    if sample_rate and 8000 <= sample_rate <= 96000:
        _transcript_sample_rate = int(sample_rate)
        _transcript_channels = int(channels or 1)
    _transcript_pcm_buffer.extend(data)
    if len(_transcript_pcm_buffer) > 600000:
        _transcript_pcm_buffer[:] = _transcript_pcm_buffer[-300000:]
    if _transcript_task is None:
        _transcript_task = asyncio.create_task(_pcm_loop())


def add_encoded(chunk: bytes) -> None:
    global _transcript_encoded_chunks, _transcript_encoded_task
    _transcript_encoded_chunks.append(chunk)
    while sum(len(c) for c in _transcript_encoded_chunks) > 500000:
        _transcript_encoded_chunks.pop(0)
    if _transcript_encoded_task is None:
        _transcript_encoded_task = asyncio.create_task(_encoded_loop())


def push_text(text: str, interim: bool = False) -> None:
    for q in _transcript_sse_queues:
        try:
            q.put_nowait({"text": text, "interim": interim})
        except asyncio.QueueFull:
            pass


def get_sse_queues() -> list:
    return _transcript_sse_queues


def get_sse_queue_count() -> int:
    return len(_transcript_sse_queues)
