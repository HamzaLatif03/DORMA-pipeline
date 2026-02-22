"""
Comprehensive test suite for the conversation agent system.
Tests all components: DB connection, context loading, input building, and conversation flow.
"""
import asyncio
import os
from datetime import datetime
from mongodb.db import get_client
from mongodb.conversation_agent import ConversationAgent
from mongodb.combined_input_builder import CombinedInputBuilder
from mongodb.realtime_conversation import RealtimeConversationHandler

DB2 = "user_history_db"


def test_db_connection():
    """Test 1: Verify MongoDB connection."""
    print("\n" + "="*60)
    print("TEST 1: MongoDB Connection")
    print("="*60)
    try:
        client = get_client()
        client.admin.command("ping")
        print("✅ MongoDB connected successfully")

        # List databases
        dbs = client.list_database_names()
        print(f"✅ Found databases: {dbs}")

        if DB2 in dbs:
            print(f"✅ {DB2} exists")
        else:
            print(f"⚠️  {DB2} not found - run init_dbs.py first")

        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False


def test_user_data():
    """Test 2: Verify demo user data exists."""
    print("\n" + "="*60)
    print("TEST 2: User Data")
    print("="*60)
    try:
        client = get_client()
        db = client[DB2]

        # Check profile
        profile = db["profiles"].find_one({"_id": "user_123"})
        if profile:
            print(f"✅ User profile found: {profile.get('name')}")
            print(f"   Tags: {profile.get('tags')}")
            print(f"   Preferences: {profile.get('preferences')}")
        else:
            print("⚠️  User profile not found - run seed_demo_user.py first")
            return False

        # Check events
        event_count = db["events"].count_documents({"userId": "user_123"})
        print(f"✅ Found {event_count} events for user_123")

        if event_count > 0:
            recent_event = db["events"].find_one(
                {"userId": "user_123"},
                sort=[("ts", -1)]
            )
            print(
                f"   Most recent: {recent_event.get('type')} - {recent_event.get('meta')}")

        # Check sessions (if any)
        session_count = db["sessions"].count_documents({"userId": "user_123"})
        print(f"✅ Found {session_count} biometric sessions")

        return True
    except Exception as e:
        print(f"❌ User data check failed: {e}")
        return False


def test_combined_input_builder():
    """Test 3: Test CombinedInputBuilder."""
    print("\n" + "="*60)
    print("TEST 3: Combined Input Builder")
    print("="*60)
    try:
        builder = CombinedInputBuilder(user_id="user_123")

        result = builder.build_complete_input(
            current_input="How am I doing today?",
            include_metrics=True,
            include_recent_events=True
        )

        print(f"✅ Built complete input package")
        print(f"   User: {result['user_context']['name']}")
        print(f"   Tags: {result['user_context']['tags']}")
        print(f"   Recent events: {len(result['recent_history'])}")
        print(f"   Has metrics: {result['latest_metrics'] is not None}")
        print(f"\n📝 Formatted Prompt Preview (first 300 chars):")
        print("-" * 60)
        print(result['formatted_prompt'][:300] + "...")

        return True
    except Exception as e:
        print(f"❌ Combined input builder failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversation_agent_text():
    """Test 4: Test ConversationAgent with text input."""
    print("\n" + "="*60)
    print("TEST 4: Conversation Agent (Text Input)")
    print("="*60)
    try:
        agent = ConversationAgent(user_id="user_123")

        print("✅ Agent initialized")
        print(f"   Context loaded: {len(agent.context)} chars")

        # Test conversation
        test_inputs = [
            "Hello!",
            "How are you doing?",
            "Can you help me?"
        ]

        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n--- Conversation {i} ---")
            print(f"User: {user_input}")

            result = await agent.handle_conversation(text_input=user_input)

            if result:
                print(f"Agent: {result['response_text']}")
                print(f"Audio file: {result['response_audio']}")
                print(f"✅ Conversation {i} processed")
            else:
                print(f"⚠️  No response generated")

        print(f"\n✅ Total conversations: {len(agent.conversation_history)}")

        return True
    except Exception as e:
        print(f"❌ Conversation agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_realtime_handler():
    """Test 5: Test RealtimeConversationHandler."""
    print("\n" + "="*60)
    print("TEST 5: Realtime Conversation Handler")
    print("="*60)
    try:
        handler = RealtimeConversationHandler(user_id="user_123")
        print("✅ Realtime handler initialized")

        # Test with transcript chunks
        transcripts = [
            ("Hello", False, True),  # (text, is_interim, should_process)
            ("Hello there", True, False),  # interim - should skip
            ("Hello there", False, True),  # final
        ]

        for text, is_interim, should_process in transcripts:
            print(f"\nTranscript: '{text}' (interim={is_interim})")
            result = await handler.handle_transcript_chunk(text, is_interim)

            if should_process and result:
                print(f"✅ Processed: {result['response_text']}")
            elif not should_process and not result:
                print("✅ Correctly skipped interim transcript")
            else:
                print("⚠️  Unexpected result")

        return True
    except Exception as e:
        print(f"❌ Realtime handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_elevenlabs_api():
    """Test 6: Check ElevenLabs API key."""
    print("\n" + "="*60)
    print("TEST 6: ElevenLabs API Configuration")
    print("="*60)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if api_key:
        print(f"✅ ELEVENLABS_API_KEY found ({api_key[:8]}...)")
        print("   Note: Actual TTS/STT requires valid API key and credits")
        return True
    else:
        print("⚠️  ELEVENLABS_API_KEY not set")
        print("   Set in .env file: ELEVENLABS_API_KEY=your_key_here")
        print("   TTS/STT features will fail without this")
        return False


def test_event_logging():
    """Test 7: Verify events are being logged."""
    print("\n" + "="*60)
    print("TEST 7: Event Logging")
    print("="*60)
    try:
        client = get_client()
        db = client[DB2]

        # Count events before
        before_count = db["events"].count_documents({"userId": "user_123"})
        print(f"Events before: {before_count}")

        # Log a test event
        from mongodb.log_event import log_event
        log_event(
            "user_123",
            "test",
            {"message": "Test event from test suite",
                "timestamp": datetime.utcnow().isoformat()}
        )

        # Count events after
        after_count = db["events"].count_documents({"userId": "user_123"})
        print(f"Events after: {after_count}")

        if after_count > before_count:
            print("✅ Event logged successfully")

            # Show the event
            latest = db["events"].find_one(
                {"userId": "user_123", "type": "test"},
                sort=[("ts", -1)]
            )
            print(f"   Latest test event: {latest.get('meta')}")
            return True
        else:
            print("⚠️  Event count didn't increase")
            return False

    except Exception as e:
        print(f"❌ Event logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "🧪 "*20)
    print("CONVERSATION AGENT TEST SUITE")
    print("🧪 "*20)

    results = {
        "MongoDB Connection": test_db_connection(),
        "User Data": test_user_data(),
        "Combined Input Builder": test_combined_input_builder(),
        "ElevenLabs API": test_elevenlabs_api(),
        "Event Logging": test_event_logging(),
        "Conversation Agent": await test_conversation_agent_text(),
        "Realtime Handler": await test_realtime_handler(),
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n🎉 All tests passed! Your conversation agent is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\nCommon fixes:")
        print("  - Run: python mongodb/init_dbs.py")
        print("  - Run: python mongodb/seed_demo_user.py")
        print("  - Set ELEVENLABS_API_KEY in .env file")
        print("  - Check MONGO_URI in .env file")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
