from datetime import datetime
from mongodb.db import get_client

DB2 = "user_history_db"


def log_event(user_id: str, event_type: str, meta: dict):
    db = get_client()[DB2]
    db["events"].insert_one(
        {"userId": user_id, "ts": datetime.utcnow(), "type": event_type, "meta": meta})
