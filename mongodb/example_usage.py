"""Examples of using the conversation agent."""
import asyncio
from mongodb.conversation_agent import ConversationAgent
from mongodb.combined_input_builder import CombinedInputBuilder


async def example_text_conversation():
    """Simple text-based conversation."""
    agent = ConversationAgent(user_id="user_123")

    # Simulate a conversation
    inputs = [
        "Hello!",
        "How am I doing with my health?",
        "Can you remind me what we talked about yesterday?"
    ]

    for user_input in inputs:
        print(f"\n{'='*60}")
        result = await agent.handle_conversation(text_input=user_input)
        if result:
            print(f"User: {result['user_input']}")
            print(f"Agent: {result['response_text']}")


async def example_audio_conversation():
    """Audio-based conversation (requires audio file)."""
    agent = ConversationAgent(user_id="user_123")

    # Process audio file
    result = await agent.handle_conversation(audio_file_path="user_audio.mp3")

    if result:
        print(f"Transcribed: {result['user_input']}")
        print(f"Response: {result['response_text']}")
        print(f"Audio saved: {result['response_audio']}")


def example_combined_input():
    """Build complete input package."""
    builder = CombinedInputBuilder(user_id="user_123")

    complete_input = builder.build_complete_input(
        current_input="What should I know about my recent activity?",
        include_metrics=True,
        include_recent_events=True
    )

    # Use this with any LLM API
    prompt = complete_input["formatted_prompt"]
    print(prompt)

    # Example: Send to OpenAI (pseudo-code)
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )


if __name__ == "__main__":
    print("\n=== Text Conversation Example ===")
    asyncio.run(example_text_conversation())

    print("\n\n=== Combined Input Example ===")
    example_combined_input()
