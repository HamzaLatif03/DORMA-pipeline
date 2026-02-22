"""WebRTC signaling: /ws WebSocket + /api/signal HTTP fallback."""
import asyncio
import io
import json
import logging
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Body, HTTPException
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from PIL import Image

from .. import state

logger = logging.getLogger(__name__)
router = APIRouter()

peer_connections: dict[WebSocket, RTCPeerConnection] = {}


async def _consume_video_track(track):
    """Read video frames from a WebRTC track, encode as JPEG, update state for /stream."""
    try:
        while True:
            frame = await track.recv()
            try:
                try:
                    nd = frame.to_ndarray(format="rgb24")
                except Exception:
                    nd = frame.to_ndarray()
                img = Image.fromarray(nd)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=85)
                state.latest_jpeg = buf.getvalue()
                state.frame_event.set()
            except Exception as e:
                logger.debug("Frame encode: %s", e)
    except Exception as e:
        logger.info("Video track ended: %s", e)


def _on_track(track):
    """Handle incoming WebRTC track: relay video, blackhole audio."""
    logger.info("Received %s track", track.kind)
    if track.kind == "video":
        asyncio.ensure_future(_consume_video_track(track))
    else:
        blackhole = MediaBlackhole()
        blackhole.addTrack(track)
        asyncio.ensure_future(blackhole.start())


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.connected_clients.add(websocket)
    pc: RTCPeerConnection | None = None

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "offer":
                sdp = msg.get("sdp")
                if not sdp:
                    logger.warning("Offer missing sdp")
                    continue
                logger.info("Received offer, creating answer...")
                try:
                    offer = RTCSessionDescription(sdp=sdp, type="offer")
                    pc = RTCPeerConnection()
                    peer_connections[websocket] = pc
                    pc.on("track", _on_track)

                    @pc.on("connectionstatechange")
                    async def on_connectionstatechange():
                        if pc.connectionState in ("failed", "closed"):
                            await pc.close()
                            peer_connections.pop(websocket, None)

                    await pc.setRemoteDescription(offer)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    deadline = time.monotonic() + 5.0
                    while pc.iceGatheringState != "complete":
                        if time.monotonic() > deadline:
                            logger.warning("ICE gathering timeout, sending answer anyway")
                            break
                        await asyncio.sleep(0.1)
                    await websocket.send_json({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp,
                    })
                    logger.info("Sent WebRTC answer to client")
                except Exception as e:
                    logger.exception("WebRTC offer handling failed: %s", e)
                    try:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    except Exception:
                        pass
                    if pc and websocket in peer_connections:
                        await pc.close()
                        peer_connections.pop(websocket, None)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        state.connected_clients.discard(websocket)
        if websocket in peer_connections:
            await peer_connections[websocket].close()
            peer_connections.pop(websocket, None)


@router.post("/api/signal")
async def api_signal(body: dict = Body(default={})):
    """HTTP fallback for signaling when WebSocket is blocked."""
    sdp = body.get("sdp")
    if not sdp or body.get("type") != "offer":
        raise HTTPException(status_code=400, detail='Need { "type": "offer", "sdp": "..." }')
    if state.http_broadcaster_pc:
        try:
            await state.http_broadcaster_pc.close()
        except Exception:
            pass
        state.http_broadcaster_pc = None
        state.http_broadcaster_session_id = None
    state.http_play_commands.clear()
    offer = RTCSessionDescription(sdp=sdp, type="offer")
    pc = RTCPeerConnection()
    state.http_broadcaster_pc = pc
    state.http_broadcaster_session_id = str(uuid.uuid4())
    pc.on("track", _on_track)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    deadline = time.monotonic() + 5.0
    while pc.iceGatheringState != "complete":
        if time.monotonic() > deadline:
            break
        await asyncio.sleep(0.1)
    return {
        "type": "answer",
        "sdp": pc.localDescription.sdp,
        "session_id": state.http_broadcaster_session_id,
    }
