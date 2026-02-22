from datetime import datetime, timedelta
from mongodb.db import get_client

DB2 = "user_history_db"


def main():
    client = get_client()
    db = client[DB2]

    user_id = "user_123"

    # ---- profile ----
    db["profiles"].update_one(
        {"_id": user_id},
        {
            "$set": {
                "name": "Mahfuz",
                "tags": ["friendly", "returning"],
                "preferences": {"tone": "direct"},
                "updatedAt": datetime.utcnow(),
            },
            "$setOnInsert": {"createdAt": datetime.utcnow()},
        },
        upsert=True,
    )

    # ---- events ----
    db["events"].delete_many({"userId": user_id})

    now = datetime.utcnow()
    db["events"].insert_many(
        [
            {
                "userId": user_id,
                "ts": now - timedelta(days=2),
                "type": "seen",
                "meta": {"location": "UCL lobby"},
            },
            {
                "userId": user_id,
                "ts": now - timedelta(days=1),
                "type": "conversation",
                "meta": {
                    "summary": "Asked about MongoDB and agents; prefers concise answers."
                },
            },
            {
                "userId": user_id,
                "ts": now,
                "type": "note",
                "meta": {"text": "Prefers fast responses."},
            },
        ]
    )

    print("✅ Seeded demo user")


if __name__ == "__main__":
    main()
