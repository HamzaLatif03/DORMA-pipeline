"""Video pipeline: POST /api/frame + GET /stream (MJPEG)."""
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import state
from ..services.faces import process_frame_faces

router = APIRouter()


@router.post("/api/frame")
async def api_frame(request: Request):
    """Accept a JPEG frame, run face detection + emotion, update MJPEG /stream."""
    body = await request.body()
    if not body or len(body) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Empty or too large")
    state.latest_jpeg = await asyncio.to_thread(process_frame_faces, body)
    state.frame_event.set()
    if state.http_frame_session_id is None:
        state.http_frame_session_id = str(uuid.uuid4())
    return {"session_id": state.http_frame_session_id}


@router.get("/stream")
async def stream_mjpeg():
    """Stream the latest video from the broadcaster as MJPEG."""
    boundary = "frame"

    async def generate():
        while True:
            state.frame_event.clear()
            await state.frame_event.wait()
            jpeg = state.latest_jpeg
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )
