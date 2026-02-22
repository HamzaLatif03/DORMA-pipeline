"""Centralized shared mutable state — no logic, just variable declarations."""
import asyncio
from typing import Any

connected_clients: set = set()

audio_connections: set = set()
audio_sse_queues: list[asyncio.Queue[bytes]] = []

transcript_pcm_buffer: bytearray = bytearray()
transcript_sample_rate: int = 48000
transcript_channels: int = 1
transcript_sse_queues: list[asyncio.Queue[str]] = []
transcript_task: asyncio.Task | None = None
transcript_encoded_chunks: list[bytes] = []
transcript_encoded_task: asyncio.Task | None = None

latest_jpeg: bytes | None = None
frame_event: asyncio.Event = asyncio.Event()

http_broadcaster_pc: Any = None
http_broadcaster_session_id: str | None = None
http_frame_session_id: str | None = None
http_play_commands: list[dict] = []

diag_audio_encoded_count: int = 0
diag_audio_encoded_last: float = 0.0
diag_audio_pcm_count: int = 0
diag_audio_pcm_last: float = 0.0
