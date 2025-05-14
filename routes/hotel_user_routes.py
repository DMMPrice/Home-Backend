import os
from flask import Blueprint, request, jsonify
from bson import ObjectId
from pymongo import MongoClient
from models.operation_logger import OperationLogger

hotel_user_bp = Blueprint("hotel_user_bp", __name__, url_prefix="/hotel/users")

# MongoDB setup
mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client[os.getenv("MONGO_DB")]
hotel_users = mongo_db["hotel_users"]


def serialize(doc):
    doc["_id"] = str(doc["_id"])
    doc["hotel_id"] = str(doc["hotel_id"])
    return doc


@hotel_user_bp.route("/", methods=["POST"])
def create_user():
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    data = request.json
    data["hotel_id"] = ObjectId(data["hotel_id"])
    result = hotel_users.insert_one(data)

    OperationLogger.log("Created hotel user", ip=ip, username=username)
    return jsonify({"user_id": str(result.inserted_id)})


@hotel_user_bp.route("/", methods=["GET"])
def get_users():
    username = request.headers.get("username")
    user_type = request.headers.get("user_type")
    hotel_id = request.args.get("hotel_id")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    if not username:
        return jsonify({"error": "Missing required header: username"}), 400

    try:
        if user_type and user_type.lower() in ["admin", "super-admin"]:
            users = hotel_users.find()
            OperationLogger.log("Viewed all hotel users (admin access)", ip=ip, username=username)
        else:
            if not hotel_id or not ObjectId.is_valid(hotel_id):
                return jsonify({"error": "Missing or invalid hotel_id for non-admin users"}), 400
            users = hotel_users.find({"hotel_id": ObjectId(hotel_id)})
            OperationLogger.log(f"Viewed hotel users for hotel {hotel_id}", ip=ip, username=username)

        return jsonify([serialize(u) for u in users]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@hotel_user_bp.route("/<user_id>", methods=["GET"])
def get_user(user_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    user = hotel_users.find_one({"_id": ObjectId(user_id)})
    if user:
        OperationLogger.log(f"Viewed user ID {user_id}", ip=ip, username=username)
        return jsonify(serialize(user))
    else:
        return jsonify({"error": "User not found"}), 404


@hotel_user_bp.route("/<user_id>", methods=["PUT"])
def update_user(user_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    data = request.json
    if "hotel_id" in data:
        data["hotel_id"] = ObjectId(data["hotel_id"])

    hotel_users.update_one({"_id": ObjectId(user_id)}, {"$set": data})
    OperationLogger.log(f"Updated user ID {user_id}", ip=ip, username=username)
    return jsonify({"message": "User updated"})


@hotel_user_bp.route("/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    hotel_users.delete_one({"_id": ObjectId(user_id)})
    OperationLogger.log(f"Deleted user ID {user_id}", ip=ip, username=username)
    return jsonify({"message": "User deleted"})
