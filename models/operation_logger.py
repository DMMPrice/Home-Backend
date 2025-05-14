# models/operation_logger.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from pytz import timezone

# Load environment variables
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
operation_logs = db["operation_logs"]


class OperationLogger:
    @staticmethod
    def log(message, ip=None, username=None):
        ist = timezone("Asia/Kolkata")
        log_entry = {
            "message": message,
            "ip": ip,
            "username": username,
            "timestamp": datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
        }
        operation_logs.insert_one(log_entry)
