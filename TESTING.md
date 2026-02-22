# Testing Guide: Conversation Agent System

This guide will walk you through testing all components of the conversation agent.

## Prerequisites

1. **MongoDB running** (local or Atlas)
2. **Environment variables set** in `.env`:
   ```
   MONGO_URI=mongodb://localhost:27017  # or your Atlas URI
   ELEVENLABS_API_KEY=your_api_key_here
   ```

## Step 1: Setup Database

```bash
# Initialize databases and collections
python mongodb/init_dbs.py

# Seed demo user data
python mongodb/seed_demo_user.py
```

Expected output:
```
✅ Connected
✅ biometrics_db created
✅ user_history_db created
✅ Indexes created
```

## Step 2: Run Complete Test Suite

```bash
python mongodb/test_conversation_agent.py
```

This runs 7 automated tests:
1. ✅ MongoDB Connection
2. ✅ User Data (profile, events, sessions)
3. ✅ Combined Input Builder
4. ✅ ElevenLabs API Configuration
5. ✅ Event Logging
6. ✅ Conversation Agent (text input)
7. ✅ Realtime Handler

## Step 3: Test Individual Components

### Test Combined Input Builder
```bash
python mongodb/combined_input_builder.py
```

Expected: Shows formatted prompt with user context, history, and metrics.

### Test Conversation Agent
```bash
python mongodb/example_usage.py
```

Expected: 
- Processes text conversations
- Generates responses
- Creates audio files (response.mp3)

### Test with Your Own Input

Create a quick test script:

```python
import asyncio
from mongodb.conversation_agent import ConversationAgent

async def my_test():
    agent = ConversationAgent(user_id="user_123")
    
    # Test with your own text
    result = await agent.handle_conversation(
        text_input="Tell me about my recent health data"
    )
    
    if result:
        print(f"You: {result['user_input']}")
        print(f"Agent: {result['response_text']}")
        print(f"Audio: {result['response_audio']}")

asyncio.run(my_test())
```

## Step 4: Test with Audio (Optional)

If you have an audio file (MP3, WAV):

```python
import asyncio
from mongodb.conversation_agent import ConversationAgent

async def test_audio():
    agent = ConversationAgent(user_id="user_123")
    
    result = await agent.handle_conversation(
        audio_file_path="your_audio.mp3"
    )
    
    if result:
        print(f"Transcribed: {result['user_input']}")
        print(f"Response: {result['response_text']}")

asyncio.run(test_audio())
```

## Step 5: Verify Data in MongoDB

Check the database directly:

```python
from mongodb.db import get_client

client = get_client()
db = client["user_history_db"]

# Check profile
profile = db["profiles"].find_one({"_id": "user_123"})
print("Profile:", profile)

# Check recent events
events = list(db["events"].find({"userId": "user_123"}).sort("ts", -1).limit(5))
print(f"\nRecent events: {len(events)}")
for event in events:
    print(f"  - {event['type']}: {event.get('meta')}")

# Check conversation events
convos = list(db["events"].find({"userId": "user_123", "type": "conversation"}))
print(f"\nConversations logged: {len(convos)}")
```

## Step 6: Integration Testing

Test the realtime handler:

```python
import asyncio
from mongodb.realtime_conversation import RealtimeConversationHandler

async def test_realtime():
    handler = RealtimeConversationHandler(user_id="user_123")
    
    # Simulate transcript chunks from your server
    transcripts = [
        ("Hello", True),   # interim
        ("Hello there", False),  # final
    ]
    
    for text, is_interim in transcripts:
        result = await handler.handle_transcript_chunk(text, is_interim)
        if result:
            print(f"Response: {result['response_text']}")

asyncio.run(test_realtime())
```

## Troubleshooting

### Test fails: "MongoDB connection failed"
- Check `MONGO_URI` in `.env`
- Ensure MongoDB is running
- Test connection: `mongosh <your_uri>`

### Test fails: "User profile not found"
- Run: `python mongodb/seed_demo_user.py`

### Test fails: "ELEVENLABS_API_KEY not set"
- Add to `.env`: `ELEVENLABS_API_KEY=your_key`
- Get key from: https://elevenlabs.io/

### TTS/STT not working
- Verify API key is valid
- Check ElevenLabs account has credits
- Test API: 
  ```python
  from elevenlabs.client import ElevenLabs
  client = ElevenLabs(api_key="your_key")
  voices = client.voices.get_all()
  print(voices)
  ```

### Audio files not playing
- Check file was created: `ls -la response.mp3`
- Play manually: `afplay response.mp3` (macOS)
- Check file size: should be > 0 bytes

## Expected Files After Testing

After running tests, you should see:
```
response.mp3          # Generated TTS response
mongodb/
  __pycache__/       # Python cache
  *.pyc              # Compiled Python files
```

## Success Criteria

✅ All 7 tests pass  
✅ User data exists in MongoDB  
✅ Conversations logged to `events` collection  
✅ Audio files generated successfully  
✅ Context properly loaded from DB2  

## Next Steps

1. **Replace mock responses** in `conversation_agent.py` with real LLM API calls (OpenAI, Anthropic, etc.)
2. **Integrate with server** - add handler to `server/routers/transcript.py`
3. **Add more users** - modify `seed_demo_user.py` or create users dynamically
4. **Customize voice** - change `voice_id` in ConversationAgent
5. **Add metrics** - run `python mongodb/init_metrics.py` and add session data

## Quick Verification Commands

```bash
# Full test suite
python mongodb/test_conversation_agent.py

# Quick demo
python mongodb/example_usage.py

# Check input building
python mongodb/combined_input_builder.py

# Verify DB setup
python -c "from mongodb.db import get_client; print(get_client().list_database_names())"
```
