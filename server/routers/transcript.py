"""Transcript endpoints: POST /api/transcript + GET /api/transcript/stream (SSE)."""
import asyncio
import json
import logging
import os

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from .. import state

logger = logging.getLogger(__name__)
router = APIRouter()


def _reply_with_tts(transcript_text: str) -> None:
    """Run in thread: OpenAI reply → reply text → Eleven Labs TTS → play."""
    try:
        from mongodb.reply_api import get_conversational_reply
        from mongodb.voice_11labs import speak_text
        logger.info("Transcript (final) → reply + Eleven Labs TTS: %s", transcript_text[:60])
        text_to_speak = get_conversational_reply(transcript_text)
        if not text_to_speak:
            text_to_speak = "I didn't catch that."
        print("[TTS] Speaking via Eleven Labs:", repr(text_to_speak[:200]))
        logger.info("Speaking via Eleven Labs: %s", text_to_speak[:60])
        server_base = os.environ.get("SERVER_BASE", "http://localhost:8000")
        speak_text(text_to_speak, play_after=True, server_base=server_base)
    except Exception as e:
        logger.exception("TTS reply failed: %s", e)


@router.post("/api/transcript")
async def api_transcript(body: dict = Body(default={})):
    """Accept transcript text from the broadcaster (Web Speech API). On final, reply via API + TTS."""
    text = (body.get("text") or body.get("transcript") or "").strip()
    interim = body.get("interim", False)
    if text:
        for q in list(state.transcript_sse_queues):
            try:
                await q.put({"text": text, "interim": interim})
            except Exception:
                pass
    # When user finishes a phrase (final), get conversational reply and speak it
    if not interim and text:
        asyncio.get_event_loop().run_in_executor(None, _reply_with_tts, text)
    return {"ok": True}


@router.get("/api/transcript/stream")
async def transcript_stream():
    """SSE stream of transcript messages."""
    q: asyncio.Queue[str] = asyncio.Queue()
    state.transcript_sse_queues.append(q)

    async def generate():
        try:
            while True:
                msg = await q.get()
                payload = msg if isinstance(msg, dict) else {"text": msg, "interim": False}
                yield f"data: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in state.transcript_sse_queues:
                state.transcript_sse_queues.remove(q)

    return StreamingResponse(generate(), media_type="text/event-stream")
