"""
One sentence: load from Eleven Labs → save MP3 in audio folder → play it.

Focus: get sentence 1 out (generate, save, play). Sentence 2 later.
"""
import os
import shutil
from pathlib import Path

import requests
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import subprocess

load_dotenv()

client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

DEFAULT_VOICE = "auq43ws1oslv0tO4BDa7"  # safe default

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = _PROJECT_ROOT / "audio"

# Single file for now: one sentence → save here → play
OUTPUT_FILENAME = "output.mp3"

# Where to send play command so the website loads and plays the audio.
# Override with SERVER_BASE in .env or --server.
DEFAULT_SERVER_BASE = os.environ.get(
    "SERVER_BASE",
    "https://digitinervate-noneligible-hue.ngrok-free.dev",
)


def get_audio_dir_path() -> str:
    """Where playback audio files are stored (for debugging)."""
    return str(AUDIO_DIR.resolve())


def play_local(path: Path) -> bool:
    """Play the MP3 on this machine (no server, no button). Uses afplay on macOS."""
    if not path.is_file():
        return False
    try:
        subprocess.run(["afplay", str(path)], check=True, timeout=60)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def play_audio_file(
    file_path: str,
    server_base: str = DEFAULT_SERVER_BASE,
    play_locally: bool = True,
) -> bool:
    """
    Play the file: (1) on this machine so you hear it without pressing anything,
    (2) via server so connected clients (e.g. phone) hear it too.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        print(f"⚠️ Audio file not found: {path}")
        print(f"   Playback audio dir: {get_audio_dir_path()}")
        return False
    size = path.stat().st_size
    print(f"✅ Audio file exists: {path} ({size} bytes)")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest = AUDIO_DIR / path.name
    if path != dest:
        shutil.copy2(path, dest)
    if not dest.is_file():
        print(f"⚠️ Copy failed: {dest} not found")
        return False
    ok = False
    if play_locally:
        print("   Playing here (no button press)...")
        if play_local(dest):
            print("   ✅ Played on this machine.")
            ok = True
        else:
            print("   ⚠️ Local play failed (afplay not found or error).")
    try:
        r = requests.post(
            f"{server_base.rstrip('/')}/api/play",
            json={"file": dest.name},
            timeout=5,
        )
        if r.ok and r.json().get("ok"):
            print(f"   ✅ Play sent to server → connected clients")
            ok = True
        else:
            print(f"   ⚠️ Server play failed: {r.status_code} (server may be off)")
    except requests.RequestException as e:
        print(f"   ⚠️ Server play error: {e}")
    return ok


def speak_text(
    text: str,
    play_after: bool = True,
    server_base: str = DEFAULT_SERVER_BASE,
) -> str | None:
    """
    One sentence: get audio from Eleven Labs → save to audio/output.mp3 → play here (no button)
    and, if server is on, send play to connected clients too.
    Returns OUTPUT_FILENAME on success, else None.
    """
    print("[VOICE_11LABS] speak_text called with:", repr((text or "")[:250]))
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("⚠️ ELEVENLABS_API_KEY is not set. Set it in .env or environment.")
        subprocess.run(["say", text])
        return None
    # Use server's audio dir when set (so /audio/output.mp3 is served correctly)
    output_dir = Path(os.environ.get("AUDIO_OUTPUT_DIR", str(AUDIO_DIR)))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / OUTPUT_FILENAME
        print(f"1. Loading from Eleven Labs (sentence): \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        audio = client.text_to_speech.convert(
            voice_id=DEFAULT_VOICE,
            model_id="eleven_turbo_v2",
            text=text,
        )
        total = 0
        with open(out_path, "wb") as f:
            if isinstance(audio, bytes):
                f.write(audio)
                total = len(audio)
            else:
                for chunk in audio:
                    if chunk:
                        n = f.write(chunk) if isinstance(chunk, bytes) else f.write(bytes([chunk]))
                        total += n
        if total == 0:
            print("⚠️ Eleven Labs returned 0 bytes. Check API key / quota.")
            return None
        print(f"2. Saved MP3: {out_path} ({total} bytes)")
        print(f"   Folder: {get_audio_dir_path()}")
        if play_after:
            print("3. Playing back (here + server if on)...")
            play_audio_file(str(out_path), server_base=server_base)
        return OUTPUT_FILENAME
    except Exception as e:
        print("⚠️ ElevenLabs failed:", e)
        subprocess.run(["say", text])
        return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="One sentence: load from Eleven Labs → save MP3 → play.")
    p.add_argument("text", nargs="?", help="One sentence to speak (saved to audio/output.mp3, then play)")
    p.add_argument("--play", metavar="FILE", help="Only play an existing file (e.g. output.mp3)")
    p.add_argument("--server", default=DEFAULT_SERVER_BASE, help=f"Server/website URL for play (default: {DEFAULT_SERVER_BASE})")
    p.add_argument("--no-play", action="store_true", help="Save MP3 only, do not send play")
    args = p.parse_args()
    if args.play:
        play_audio_file(args.play, server_base=args.server)
    elif args.text:
        speak_text(args.text, play_after=not args.no_play, server_base=args.server)
    else:
        p.print_help()
        print("\nFlow: 1) Load from Eleven Labs  2) Save to audio/output.mp3  3) Play")
        print("Audio folder:", get_audio_dir_path())
        print("\n  python -m mongodb.voice_11labs 'Hello world'   # one sentence, save + play")
        print("  python -m mongodb.voice_11labs 'Hi' --no-play  # save only")
        print("  python -m mongodb.voice_11labs --play output.mp3")
