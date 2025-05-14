import os
from flask import Blueprint, request, jsonify
from bson import ObjectId
from pymongo import MongoClient
from models.operation_logger import OperationLogger

hotel_bp = Blueprint("hotel_bp", __name__, url_prefix="/hotel")

# MongoDB setup
mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client[os.getenv("MONGO_DB")]
hotel_master = mongo_db["hotel_master"]


def serialize(doc):
    doc["_id"] = str(doc["_id"])
    return doc


@hotel_bp.route("/", methods=["POST"])
def create_hotel():
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    data = request.json
    result = hotel_master.insert_one(data)

    OperationLogger.log("Created a new hotel", ip=ip, username=username)
    return jsonify({"hotel_id": str(result.inserted_id)})


@hotel_bp.route("/", methods=["GET"])
def get_hotels():
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    hotels = [serialize(h) for h in hotel_master.find()]
    OperationLogger.log("Viewed all hotels", ip=ip, username=username)
    return jsonify(hotels)


@hotel_bp.route("/<hotel_id>", methods=["GET"])
def get_hotel(hotel_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    hotel = hotel_master.find_one({"_id": ObjectId(hotel_id)})
    if hotel:
        OperationLogger.log(f"Viewed hotel {hotel_id}", ip=ip, username=username)
        return jsonify(serialize(hotel))
    else:
        return jsonify({"error": "Hotel not found"}), 404


@hotel_bp.route("/<hotel_id>", methods=["PUT"])
def update_hotel(hotel_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    hotel_master.update_one({"_id": ObjectId(hotel_id)}, {"$set": request.json})
    OperationLogger.log(f"Updated hotel {hotel_id}", ip=ip, username=username)
    return jsonify({"message": "Hotel updated"})


@hotel_bp.route("/<hotel_id>", methods=["DELETE"])
def delete_hotel(hotel_id):
    username = request.headers.get("username")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    hotel_master.delete_one({"_id": ObjectId(hotel_id)})
    OperationLogger.log(f"Deleted hotel {hotel_id}", ip=ip, username=username)
    return jsonify({"message": "Hotel deleted"})
