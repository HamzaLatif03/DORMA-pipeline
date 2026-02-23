# AI Agent Integration Guide

## Overview
The transcript system now automatically processes user speech with your AI agent, which:
1. Loads user context from MongoDB
2. Maintains conversation history
3. Generates contextual responses
4. Converts responses to speech using ElevenLabs TTS

## How It Works

### Automatic Flow
```
User speaks → Audio captured → ElevenLabs STT → Transcript
                                                      ↓
                                    AI Agent (with user context)
                                                      ↓
                                    Generate response
                                                      ↓
                                    ElevenLabs TTS
                                                      ↓
                                    Audio response
```

### Setup

1. **Set environment variable** (optional):
```bash
export DEFAULT_USER_ID="your_user_id"
```

2. **Start the server**:
```bash
cd /Users/mahfuz/UCL/Hackathon/Hack_london/Code_base_software/DORMA-pipeline
export SERVER_BASE=http://10.71.71.117:8000
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

The AI agent will automatically initialize when the first transcript is received.

## API Endpoints

### Set User ID
```bash
curl -X POST http://localhost:8000/api/transcript/ai/user \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123"}'
```

### Enable/Disable AI Processing
```bash
# Enable
curl -X POST http://localhost:8000/api/transcript/ai/enable \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# Disable
curl -X POST http://localhost:8000/api/transcript/ai/enable \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

### Send Transcript (with optional user_id)
```bash
curl -X POST http://localhost:8000/api/transcript \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "user_id": "user_123"}'
```

## Features

### Context-Aware Responses
The AI agent automatically:
- Loads user profile from MongoDB
- Uses conversation history (last 5 exchanges)
- Applies user preferences and context
- Generates personalized responses

### Conversation History
All conversations are logged to MongoDB with:
- User input
- AI response
- Timestamp
- Context used flag

### Non-Blocking Processing
AI processing happens asynchronously, so:
- Transcripts are broadcast immediately to viewers
- AI processing doesn't block the transcription pipeline
- Multiple conversations can be handled concurrently

## Configuration

### In Code
```python
import transcript

# Set user for AI agent
transcript.set_user_id("user_123")

# Enable/disable AI processing
transcript.enable_ai_processing(True)  # or False

# Manually push text for processing
transcript.push_text("Hello!", interim=False)
```

### Environment Variables
```bash
# Default user ID (optional)
DEFAULT_USER_ID=demo_user

# ElevenLabs API key (required)
ELEVENLABS_API_KEY=your_api_key

# MongoDB connection (required for user context)
MONGO_URI=mongodb://localhost:27017/
```

## Response Format

When AI processes a transcript, responses are broadcast to SSE viewers with:
```json
{
  "text": "[AI]: Hello! How can I help you today?",
  "interim": false,
  "is_ai": true
}
```

## Troubleshooting

### AI Not Responding
1. Check MongoDB connection
2. Verify user exists in database: `python mongodb/seed_demo_user.py`
3. Check logs for errors
4. Verify ELEVENLABS_API_KEY is set

### No Audio Response
1. Check ElevenLabs API key
2. Verify `response.mp3` is being created
3. Check server logs for TTS errors

### Wrong User Context
```bash
# Update user ID
curl -X POST http://localhost:8000/api/transcript/ai/user \
  -H "Content-Type: application/json" \
  -d '{"user_id": "correct_user_id"}'
```

## Testing

### Test the Full Pipeline
```bash
# 1. Seed a demo user
python mongodb/seed_demo_user.py

# 2. Test conversation
python mongodb/test_conversation_agent.py

# 3. Or use the API
curl -X POST http://localhost:8000/api/transcript \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "user_id": "demo_user"}'
```

## Integration with Frontend

Your frontend can:
1. Stream transcripts via SSE: `GET /api/transcript/stream`
2. Receive both user transcripts and AI responses
3. Filter AI responses using `is_ai` flag
4. Play audio responses from `response_audio` field

```javascript
const eventSource = new EventSource('/api/transcript/stream');
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.is_ai) {
    // Display AI response
    console.log('AI:', data.text);
  } else {
    // Display user transcript
    console.log('User:', data.text);
  }
};
```
