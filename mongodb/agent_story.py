from pymongo import DESCENDING
from mongodb.db import get_client

DB2 = "user_history_db"


def build_user_story(user_id: str) -> dict:
    """
    Returns dict with:
      - story (for ElevenLabs)
      - history (debug/log)
    """

    client = get_client()
    db = client[DB2]

    profile = db["profiles"].find_one({"_id": user_id}) or {}
    events = list(
        db["events"]
        .find({"userId": user_id})
        .sort("ts", DESCENDING)
        .limit(20)
    )
    events.reverse()

    name = profile.get("name", "Unknown user")
    tags = profile.get("tags", [])
    prefs = profile.get("preferences", {})

    # ---------- history ----------
    history_lines = []
    history_lines.append(f"User: {name} (id={user_id})")

    if tags:
        history_lines.append(f"Tags: {', '.join(tags)}")

    history_lines.append("Recent events:")

    if not events:
        history_lines.append("- None")
    else:
        for e in events:
            ts = e.get("ts")
            etype = e.get("type")
            meta = e.get("meta", {})

            if etype == "seen":
                history_lines.append(f"- Seen at {meta.get('location')}")
            elif etype == "conversation":
                history_lines.append(f"- Conversation: {meta.get('summary')}")
            elif etype == "note":
                history_lines.append(f"- Note: {meta.get('text')}")
            else:
                history_lines.append(f"- {etype}: {meta}")

    history = "\n".join(history_lines)

    # ---------- story for voice ----------
    voice_context_lines = []
    voice_context_lines.append(f"IDENTITY: {name} (userId={user_id})")

    if tags:
        voice_context_lines.append(f"TAGS: {', '.join(tags)}")

    if prefs.get("tone"):
        voice_context_lines.append(f"TONE: {prefs['tone']}")

    voice_context_lines.append("RECENT HISTORY:")
    if not events:
        voice_context_lines.append("- None")
    else:
        for e in events[-5:]:  # last 5 only
            etype = e.get("type")
            meta = e.get("meta", {})
            if etype == "seen":
                voice_context_lines.append(f"- Seen at {meta.get('location')}")
            elif etype == "conversation":
                voice_context_lines.append(
                    f"- Conversation: {meta.get('summary')}")
            elif etype == "note":
                voice_context_lines.append(f"- Note: {meta.get('text')}")
            else:
                voice_context_lines.append(f"- {etype}: {meta}")

    voice_context = "\n".join(voice_context_lines)

    return {
        "story": voice_context,   # now this is your ElevenLabs context block
        "history": history,
    }


if __name__ == "__main__":
    out = build_user_story("user_123")

    print("\n=== HISTORY ===\n")
    print(out["history"])

    print("\n=== STORY FOR ELEVENLABS ===\n")
    print(out["story"])
