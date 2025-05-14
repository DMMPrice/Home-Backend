import os
from flask import request, jsonify, Blueprint, abort
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import cloudinary
import cloudinary.uploader
from models.operation_logger import OperationLogger

# Load environment variables
load_dotenv()

# MongoDB setup
mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client[os.getenv("MONGO_DB")]
master_collection = mongo_db[os.getenv("MONGO_MASTER_COLLECTION", "master_database")]

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Blueprint
criminal_bp = Blueprint("criminals", __name__, url_prefix="/criminals")


# ✅ Helper to enforce required header
def get_username_or_abort():
    username = request.headers.get("username")
    if not username:
        abort(400, description="Missing required header: username")
    return username


# ✅ Add new criminal
@criminal_bp.route("/add", methods=["POST"])
def add_criminal():
    username = get_username_or_abort()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    data = request.form.to_dict()
    image_files = request.files.getlist("images")

    required_fields = ["Family name", "Forename", "Folder Name", "Gender", "Date of birth",
                       "Place of birth", "Nationality", "Distinguishing marks and characteristics", "Charges"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    image_urls = []

    if image_files:
        folder = f"master_database/{data['Folder Name']}"
        try:
            for image in image_files:
                img = Image.open(image.stream)
                img.thumbnail((500, 500))
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=70, optimize=True)
                buffer.seek(0)

                while buffer.getbuffer().nbytes > 50 * 1024 and img.size[0] > 100 and img.size[1] > 100:
                    new_size = (int(img.size[0] * 0.9), int(img.size[1] * 0.9))
                    img = img.resize(new_size)
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=60, optimize=True)
                    buffer.seek(0)

                upload_result = cloudinary.uploader.upload(buffer, folder=folder)
                image_urls.append(upload_result["secure_url"])
        except Exception as e:
            return jsonify({"error": f"Image upload failed: {str(e)}"}), 500

    data["images"] = image_urls

    try:
        result = master_collection.insert_one(data)
        OperationLogger.log("Criminal record added", ip, username)

        return jsonify({
            "message": "Criminal record added successfully",
            "id": str(result.inserted_id),
            "image_urls": image_urls
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Retrieve criminals
@criminal_bp.route("", methods=["GET"])
def get_criminals():
    username = get_username_or_abort()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    family_name = request.args.get("family_name")
    forename = request.args.get("forename")

    query = {}
    if family_name:
        query["Family name"] = {"$regex": family_name, "$options": "i"}
    if forename:
        query["Forename"] = {"$regex": forename, "$options": "i"}

    try:
        records = list(master_collection.find(query))
        for record in records:
            record["_id"] = str(record["_id"])

        OperationLogger.log("Retrieved list of criminals", ip, username)
        return jsonify(records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Delete criminal by ID
@criminal_bp.route("/<id>", methods=["DELETE"])
def delete_criminal(id):
    username = get_username_or_abort()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    if not ObjectId.is_valid(id):
        return jsonify({"error": "Invalid ID format"}), 400

    try:
        result = master_collection.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Criminal record not found"}), 404

        OperationLogger.log(f"Deleted criminal record with ID {id}", ip, username)
        return jsonify({"message": "Criminal record deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
