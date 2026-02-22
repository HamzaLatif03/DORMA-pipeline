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


async def transcribe_with_elevenlabs(audio_bytes: bytes) -> str | None:
    """Send audio to ElevenLabs STT and return the transcribed text."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    try:
        def _call():
            resp = requests.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": api_key},
                files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.warning("ElevenLabs STT failed: %s", e)
        return None


async def transcript_pcm_loop():
    """Background loop: periodically drain the PCM buffer, convert to WAV, transcribe."""
    while True:
        await asyncio.sleep(5)
        if not state.transcript_pcm_buffer:
            continue
        pcm_data = bytes(state.transcript_pcm_buffer)
        state.transcript_pcm_buffer.clear()
        wav_data = pcm_to_wav(pcm_data, state.transcript_sample_rate, state.transcript_channels)
        text = await transcribe_with_elevenlabs(wav_data)
        if text:
            for q in list(state.transcript_sse_queues):
                try:
                    await q.put(text)
                except Exception:
                    pass


async def transcript_encoded_loop():
    """Background loop: periodically drain encoded audio chunks, transcribe."""
    while True:
        await asyncio.sleep(5)
        if not state.transcript_encoded_chunks:
            continue
        chunks = state.transcript_encoded_chunks[:]
        state.transcript_encoded_chunks.clear()
        audio_data = b"".join(chunks)
        text = await transcribe_with_elevenlabs(audio_data)
        if text:
            for q in list(state.transcript_sse_queues):
                try:
                    await q.put(text)
                except Exception:
                    pass
