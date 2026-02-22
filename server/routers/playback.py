"""Play commands: POST /api/play, GET /api/poll, GET /audio/{filename}, GET /api/audio-list."""
import os
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

from .. import state

router = APIRouter()

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


@router.post("/api/play")
async def api_play(body: dict = Body(default={})):
    """Push a play command to all connected clients."""
    file = body.get("file")
    url = body.get("url")
    if not file and not url:
        return {"ok": False, "error": "Provide 'file' or 'url'"}
    base = os.environ.get("SERVER_BASE", "http://localhost:8000")
    play_url = f"{base}/audio/{file}" if file else url
    payload = {"action": "play", "url": play_url}
    dead = set()
    for ws in state.connected_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        state.connected_clients.discard(ws)
    state.http_play_commands.append(payload)
    return {"ok": True, "url": play_url}


@router.get("/api/poll")
async def api_poll(session_id: str = ""):
    """Poll for the next play command (HTTP broadcaster or frame uploader)."""
    if session_id not in (state.http_broadcaster_session_id, state.http_frame_session_id):
        return {"pending": True}
    if not state.http_play_commands:
        return {"pending": True}
    cmd = state.http_play_commands.pop(0)
    return cmd


@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve an audio file from the audio directory."""
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = (AUDIO_DIR / filename).resolve()
    if not path.is_file() or not str(path).startswith(str(AUDIO_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="audio/mpeg" if filename.endswith(".mp3") else "audio/wav")


@router.get("/api/audio-list")
async def list_audio():
    """List available audio files."""
    files = [f.name for f in AUDIO_DIR.iterdir() if f.suffix.lower() in (".mp3", ".wav", ".m4a")]
    return {"files": sorted(files)}
