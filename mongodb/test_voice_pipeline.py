from mongodb.voice_11labs import speak_text


def main():
    spoken_text = "Hello. Testing ElevenLabs."
    print("Chars:", len(spoken_text))
    speak_text(spoken_text, output_file="output.mp3")


if __name__ == "__main__":
    main()
