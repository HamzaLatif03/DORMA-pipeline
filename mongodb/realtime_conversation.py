"""
Real-time conversation handler that integrates with the server's audio/transcript streams.
"""
import asyncio
from typing import Optional
from mongodb.conversation_agent import ConversationAgent
from mongodb.db import get_client
import tempfile
import os

DB2 = "user_history_db"


class RealtimeConversationHandler:
    """Handles real-time audio conversations from the web interface."""

    def __init__(self, user_id: str):
        self.agent = ConversationAgent(user_id)
        self.processing = False

    async def handle_transcript_chunk(self, transcript_text: str, is_interim: bool = False):
        """
        Handle incoming transcript chunks from the server.
        Only process final (non-interim) transcripts.
        """
        if is_interim or not transcript_text.strip():
            return None

        if self.processing:
            print("⏳ Already processing, skipping...")
            return None

        self.processing = True
        try:
            result = await self.agent.handle_conversation(text_input=transcript_text)
            return result
        finally:
            self.processing = False

    async def handle_audio_chunk(self, audio_data: bytes, sample_rate: int = 48000):
        """
        Handle raw audio data from the server.
        Saves to temp file and processes.
        """
        if self.processing:
            return None

        self.processing = True
        try:
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            result = await self.agent.handle_conversation(audio_file_path=tmp_path)

            # Cleanup
            os.unlink(tmp_path)
            return result
        finally:
            self.processing = False
