# routes/operation_logger_routes.py
from flask import Blueprint, jsonify, request
from models.operation_logger import operation_logs
from bson import ObjectId

operation_logger_bp = Blueprint("operation_logger", __name__, url_prefix="/operation-logs")


def serialize_log(log):
    log["_id"] = str(log["_id"])
    return log


@operation_logger_bp.route("", methods=["GET"])
def get_operation_logs():
    logs = [serialize_log(log) for log in operation_logs.find()]
    return jsonify(logs), 200
