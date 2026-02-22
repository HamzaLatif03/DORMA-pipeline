"""Transcript endpoints: POST /api/transcript + GET /api/transcript/stream (SSE)."""
import asyncio
import json
import logging

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from .. import state

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/transcript")
async def api_transcript(body: dict = Body(default={})):
    """Accept transcript text from the broadcaster (Web Speech API)."""
    text = body.get("text", "")
    if text:
        for q in list(state.transcript_sse_queues):
            try:
                await q.put(text)
            except Exception:
                pass
    return {"ok": True}


@router.get("/api/transcript/stream")
async def transcript_stream():
    """SSE stream of transcript messages."""
    q: asyncio.Queue[str] = asyncio.Queue()
    state.transcript_sse_queues.append(q)

    async def generate():
        try:
            while True:
                text = await q.get()
                yield f"data: {json.dumps({'text': text})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in state.transcript_sse_queues:
                state.transcript_sse_queues.remove(q)

    return StreamingResponse(generate(), media_type="text/event-stream")
