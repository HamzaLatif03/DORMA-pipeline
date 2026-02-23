"""ElevenLabs STT client, PCM-to-WAV conversion, and background transcription loops."""
import asyncio
import io
import logging
import os
import wave

import requests

from .. import state

logger = logging.getLogger(__name__)

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 48000, channels: int = 1) -> bytes:
    """Convert raw PCM-16 LE bytes into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def _guess_audio_mime(audio_bytes: bytes) -> tuple[str, str]:
    """Return (filename, mime_type) for ElevenLabs. They accept webm/mp4/wav."""
    if len(audio_bytes) >= 4:
        # WebM EBML header: 0x1A 0x45 0xDF 0xA3
        if audio_bytes[:4] == b"\x1aE\xdf\xa3":
            return "audio.webm", "audio/webm"
        # WebM Cluster (media chunks from MediaRecorder)
        if audio_bytes[:4] == bytes([0x1F, 0x43, 0xB6, 0x75]):
            return "audio.webm", "audio/webm"
        # MP4/ftyp
        if len(audio_bytes) >= 8 and audio_bytes[4:8] == b"ftyp":
            return "audio.mp4", "audio/mp4"
    return "audio.wav", "audio/wav"


async def transcribe_with_elevenlabs(audio_bytes: bytes, *, is_wav: bool = False) -> str | None:
    """Send audio to ElevenLabs STT and return the transcribed text."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    fname, mime = ("audio.wav", "audio/wav") if is_wav else _guess_audio_mime(audio_bytes)
    try:
        def _call():
            resp = requests.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": api_key},
                files={"audio": (fname, audio_bytes, mime)},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.warning("ElevenLabs STT failed: %s", e)
        return None


def _trigger_reply_tts(text: str) -> None:
    """Run Gemini → Eleven Labs TTS in a thread so the assistant speaks back."""
    text = (text or "").strip()
    if not text:
        return
    try:
        from ..routers.transcript import _reply_with_tts
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _reply_with_tts, text)
    except Exception as e:
        logger.warning("Reply+TTS trigger failed: %s", e)


async def transcript_pcm_loop():
    """Background loop: periodically drain the PCM buffer, convert to WAV, transcribe."""
    while True:
        await asyncio.sleep(5)
        if not state.transcript_pcm_buffer:
            continue
        pcm_data = bytes(state.transcript_pcm_buffer)
        state.transcript_pcm_buffer.clear()
        wav_data = pcm_to_wav(pcm_data, state.transcript_sample_rate, state.transcript_channels)
        text = await transcribe_with_elevenlabs(wav_data, is_wav=True)
        if text:
            text = (text or "").strip()
            for q in list(state.transcript_sse_queues):
                try:
                    await q.put(text)
                except Exception:
                    pass
            _trigger_reply_tts(text)


async def transcript_encoded_loop():
    """Background loop: periodically drain encoded audio chunks, transcribe."""
    while True:
        await asyncio.sleep(2.5)
        if not state.transcript_encoded_chunks:
            continue
        chunks = state.transcript_encoded_chunks[:]
        state.transcript_encoded_chunks.clear()
        # Prefer WebM Cluster (actual speech); else largest chunk
        def _is_webm_cluster(b: bytes) -> bool:
            return len(b) >= 4 and b[:4] == bytes([0x1F, 0x43, 0xB6, 0x75])
        media = [c for c in chunks if _is_webm_cluster(c) and len(c) >= 300]
        audio_data = max(media, key=len) if media else (max(chunks, key=len) if chunks else b"")
        if not audio_data or len(audio_data) < 300:
            continue
        text = await transcribe_with_elevenlabs(audio_data)
        if text:
            text = (text or "").strip()
            for q in list(state.transcript_sse_queues):
                try:
                    await q.put(text)
                except Exception:
                    pass
            _trigger_reply_tts(text)
