import os
from flask import Blueprint, request, jsonify
from bson import ObjectId
from pymongo import MongoClient
from models.operation_logger import OperationLogger

guest_bp = Blueprint("guest_bp", __name__, url_prefix="/hotel/guests")

# MongoDB setup
mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client[os.getenv("MONGO_DB")]
hotel_guest_list = mongo_db["hotel_guest_list"]


def serialize(doc):
    doc["_id"] = str(doc["_id"])
    doc["hotel_id"] = str(doc["hotel_id"])
    return doc


@guest_bp.route("/", methods=["POST"])
def create_guest():
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    data = request.json
    data["hotel_id"] = ObjectId(data["hotel_id"])
    result = hotel_guest_list.insert_one(data)

    OperationLogger.log("Created a new guest", ip=ip, username=username)

    return jsonify({"guest_id": str(result.inserted_id)})


@guest_bp.route("/", methods=["GET"])
def get_guests():
    username = request.headers.get("username")
    user_type = request.headers.get("user_type")
    hotel_id = request.args.get("hotel_id")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    if not username:
        return jsonify({"error": "Missing required header: username"}), 400

    try:
        if user_type and user_type.lower() in ["admin", "super-admin"]:
            guests = hotel_guest_list.find()
            OperationLogger.log("Viewed all hotel guests (admin access)", ip=ip, username=username)
        else:
            if not hotel_id or not ObjectId.is_valid(hotel_id):
                return jsonify({"error": "Missing or invalid hotel_id for non-admin users"}), 400
            guests = hotel_guest_list.find({"hotel_id": ObjectId(hotel_id)})
            OperationLogger.log(f"Viewed guests for hotel {hotel_id}", ip=ip, username=username)

        return jsonify([serialize(g) for g in guests]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@guest_bp.route("/<guest_id>", methods=["GET"])
def get_guest(guest_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    guest = hotel_guest_list.find_one({"_id": ObjectId(guest_id)})
    if guest:
        OperationLogger.log(f"Viewed guest ID {guest_id}", ip=ip, username=username)
        return jsonify(serialize(guest))
    else:
        return jsonify({"error": "Guest not found"}), 404


@guest_bp.route("/<guest_id>", methods=["PUT"])
def update_guest(guest_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    data = request.json
    if "hotel_id" in data:
        data["hotel_id"] = ObjectId(data["hotel_id"])

    hotel_guest_list.update_one({"_id": ObjectId(guest_id)}, {"$set": data})
    OperationLogger.log(f"Updated guest ID {guest_id}", ip=ip, username=username)
    return jsonify({"message": "Guest updated"})


@guest_bp.route("/<guest_id>", methods=["DELETE"])
def delete_guest(guest_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    hotel_guest_list.delete_one({"_id": ObjectId(guest_id)})
    OperationLogger.log(f"Deleted guest ID {guest_id}", ip=ip, username=username)
    return jsonify({"message": "Guest deleted"})
