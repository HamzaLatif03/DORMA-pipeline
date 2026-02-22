from datetime import datetime
from pymongo import ASCENDING, DESCENDING
from mongodb.db import get_client

DB1 = "biometrics_db"
DB2 = "user_history_db"


def main():
    client = get_client()
    client.admin.command("ping")
    print("✅ Connected")

    # DB1: biometrics placeholder
    db1 = client[DB1]
    db1["__init__"].insert_one(
        {"createdAt": datetime.utcnow(), "note": "init biometrics db"})
    print(f"✅ {DB1} created")

    # DB2: history/story db
    db2 = client[DB2]
    db2["__init__"].insert_one(
        {"createdAt": datetime.utcnow(), "note": "init user history db"})
    print(f"✅ {DB2} created")

    # Useful indexes for DB2
    db2["events"].create_index([("userId", ASCENDING), ("ts", DESCENDING)])

    print("✅ Indexes created")
    print("Databases:", client.list_database_names())


if __name__ == "__main__":
    main()
