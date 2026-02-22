import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def get_client() -> MongoClient:
    uri = os.environ["MONGO_URI"]
    return MongoClient(uri, serverSelectionTimeoutMS=8000)
