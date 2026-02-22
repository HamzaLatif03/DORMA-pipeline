"""
Conversation agent: combines user story from DB2 with live audio transcription
and generates contextual responses using ElevenLabs.
"""
import asyncio
from datetime import datetime
from typing import Optional
from mongodb.agent_story import build_user_story
from mongodb.db import get_client
from mongodb.log_event import log_event
from mongodb.voice_11labs import speak_text
import os
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))

DB2 = "user_history_db"
DEFAULT_VOICE = "auq43ws1oslv0tO4BDa7"


class ConversationAgent:
    """Manages context-aware conversations with users."""

    def __init__(self, user_id: str, voice_id: str = DEFAULT_VOICE):
        self.user_id = user_id
        self.voice_id = voice_id
        self.conversation_history = []
        self.context = None
        self._load_context()

    def _load_context(self):
        """Load user story and context from MongoDB."""
        story_data = build_user_story(self.user_id)
        self.context = story_data["story"]
        print(f"📖 Loaded context for user: {self.user_id}")

    def _build_prompt(self, user_input: str) -> str:
        """Combine user context with conversation history and current input."""
        prompt_parts = [
            "=== USER CONTEXT ===",
            self.context,
            "",
            "=== CONVERSATION HISTORY ===",
        ]

        if not self.conversation_history:
            prompt_parts.append("(No previous messages)")
        else:
            for msg in self.conversation_history[-5:]:  # Last 5 exchanges
                prompt_parts.append(f"User: {msg['user']}")
                prompt_parts.append(f"Assistant: {msg['assistant']}")

        prompt_parts.extend([
            "",
            "=== CURRENT INPUT ===",
            f"User: {user_input}",
            "",
            "=== INSTRUCTIONS ===",
            "Respond naturally based on the user context and conversation history.",
            "Keep responses concise and relevant to the user's preferences.",
            "",
            "Assistant:"
        ])

        return "\n".join(prompt_parts)

    async def process_audio_input(self, audio_file_path: str) -> str:
        """Transcribe audio file using ElevenLabs STT."""
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = client.speech_to_text.convert(
                    audio=audio_file.read(),
                    model_id="scribe_v2"
                )
                transcription = response.get("text", "")
                print(f"🎤 Transcribed: {transcription}")
                return transcription
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return ""

    async def process_text_input(self, user_input: str) -> str:
        """Process text input (from transcription or direct text)."""
        if not user_input.strip():
            return "I didn't catch that. Could you repeat?"

        # Build prompt with full context
        full_prompt = self._build_prompt(user_input)

        # For now, create a simple response
        # In production, you'd call an LLM API here
        response = self._generate_response(user_input)

        # Store in conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.utcnow()
        })

        # Log event to MongoDB
        log_event(
            self.user_id,
            "conversation",
            {
                "user_input": user_input,
                "agent_response": response,
                "context_used": True
            }
        )

        return response

    def _generate_response(self, user_input: str) -> str:
        """
        Generate response based on context.
        TODO: Replace with actual LLM API call (OpenAI, Anthropic, etc.)
        """
        # Simple pattern matching for demo
        user_input_lower = user_input.lower()

        db = get_client()[DB2]
        profile = db["profiles"].find_one({"_id": self.user_id}) or {}
        name = profile.get("name", "there")

        if any(word in user_input_lower for word in ["hello", "hi", "hey"]):
            return f"Hello {name}! How can I help you today?"
        elif any(word in user_input_lower for word in ["how are you", "how's it going"]):
            return f"I'm doing well, {name}. How about you?"
        elif any(word in user_input_lower for word in ["help", "assist"]):
            return f"I'm here to help you, {name}. What do you need assistance with?"
        else:
            return f"I understand you said: {user_input}. Let me help you with that."

    async def speak_response(self, response_text: str, output_file: str = "response.mp3"):
        """Convert response text to speech using ElevenLabs TTS."""
        try:
            audio = client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2",
                text=response_text,
            )

            with open(output_file, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            print(f"🔊 Audio saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ TTS failed: {e}")
            return None

    async def handle_conversation(self, audio_file_path: Optional[str] = None, text_input: Optional[str] = None):
        """
        Main conversation handler.
        Accepts either audio file or text input.
        """
        # Step 1: Get user input (transcribe audio or use text)
        if audio_file_path:
            user_input = await self.process_audio_input(audio_file_path)
        elif text_input:
            user_input = text_input
        else:
            print("❌ No input provided")
            return None

        if not user_input:
            return None

        # Step 2: Process input with context and generate response
        response = await self.process_text_input(user_input)
        print(f"🤖 Response: {response}")

        # Step 3: Convert response to speech
        audio_file = await self.speak_response(response)

        return {
            "user_input": user_input,
            "response_text": response,
            "response_audio": audio_file
        }


async def main():
    """Demo usage of the conversation agent."""
    agent = ConversationAgent(user_id="user_123")

    # Example 1: Text input
    print("\n=== Text Input Example ===")
    result = await agent.handle_conversation(text_input="Hello, how are you?")

    # Example 2: Audio input (if you have an audio file)
    # print("\n=== Audio Input Example ===")
    # result = await agent.handle_conversation(audio_file_path="user_audio.mp3")

    if result:
        print(f"\n✅ Conversation processed:")
        print(f"   User: {result['user_input']}")
        print(f"   Agent: {result['response_text']}")
        print(f"   Audio: {result['response_audio']}")


if __name__ == "__main__":
    asyncio.run(main())
