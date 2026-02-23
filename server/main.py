"""FastAPI app entry point — wires routers, mounts static files, and has /api/ping + /api/diagnostic."""
import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
# Load .env from project root (parent of server/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import state
from .routers import signaling, video, audio, transcript, playback

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
for _noisy in ("deepface", "tensorflow", "keras", "absl", "h5py", "PIL", "aiortc",
               "aioice", "urllib3", "multipart"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = FastAPI(title="iPhone peripherals bridge")

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR.mkdir(exist_ok=True)
# So TTS (voice_11labs) writes to the same dir the server serves from
os.environ["AUDIO_OUTPUT_DIR"] = str(AUDIO_DIR.resolve())
STATIC_DIR.mkdir(exist_ok=True)

app.include_router(signaling.router)
app.include_router(video.router)
app.include_router(audio.router)
app.include_router(transcript.router)
app.include_router(playback.router)


@app.get("/api/ping")
async def api_ping():
    """Simple connectivity check."""
    return {"ok": True}


@app.get("/api/diagnostic")
async def api_diagnostic():
    """Cross-cutting diagnostic snapshot."""
    return {
        "audio_encoded_count": state.diag_audio_encoded_count,
        "audio_encoded_last": state.diag_audio_encoded_last,
        "audio_pcm_count": state.diag_audio_pcm_count,
        "audio_pcm_last": state.diag_audio_pcm_last,
        "transcript_pcm_buffer_len": len(state.transcript_pcm_buffer),
        "transcript_encoded_chunks": len(state.transcript_encoded_chunks),
        "transcript_task_running": state.transcript_task is not None and not state.transcript_task.done(),
        "transcript_encoded_task_running": state.transcript_encoded_task is not None and not state.transcript_encoded_task.done(),
        "connected_clients": len(state.connected_clients),
        "audio_connections": len(state.audio_connections),
        "audio_sse_queues": len(state.audio_sse_queues),
        "transcript_sse_queues": len(state.transcript_sse_queues),
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the main web app."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Put index.html in the static/ directory"}
    return FileResponse(index_path)
