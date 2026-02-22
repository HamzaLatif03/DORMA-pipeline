import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import subprocess
load_dotenv()

client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

DEFAULT_VOICE = "auq43ws1oslv0tO4BDa7"  # safe default


def speak_text(text: str, output_file: str = "output.mp3"):
    try:
        audio = client.text_to_speech.convert(
            voice_id=DEFAULT_VOICE,
            model_id="eleven_turbo_v2",
            text=text,
        )

        with open(output_file, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        print(f"✅ Audio saved to {output_file}")
    except Exception as e:
        print("⚠️ ElevenLabs failed, using macOS voice:", e)
        subprocess.run(["say", text])
