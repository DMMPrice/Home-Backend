# routes/ip_logger.py
import os
from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
from models.ip_logger import IPLogger
from models.operation_logger import OperationLogger

# Load environment variables
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
ip_logs = db["ip_logs"]

ip_bp = Blueprint("ip_logger", __name__, url_prefix="/ip-log")


# ✅ Log IP and save to DB (POST)
@ip_bp.route("/operation-logs", methods=["POST"])
def operation_logs():
    try:
        data = request.get_json() or {}
        username = data.get("username", "Unknown")
        ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
        result = IPLogger.log_request(ip_address, username=username)
        return jsonify(result), 200 if "message" in result else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Get all IP logs (GET)
@ip_bp.route("", methods=["GET"])
def get_logs():
    try:
        username = request.args.get("username", "Unknown")
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)

        logs = list(ip_logs.find())
        for log in logs:
            log["_id"] = str(log["_id"])

        # ✅ Operation log
        OperationLogger.log("Fetched all IP logs", ip, username)

        return jsonify(logs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
