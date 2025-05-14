import os
import pickle
from datetime import datetime
from io import BytesIO
from pytz import timezone

import cloudinary
import cloudinary.uploader
import numpy as np
from PIL import Image
from deepface import DeepFace
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from models.operation_logger import OperationLogger  # ✅ Operation logger

# Load environment variables
load_dotenv()

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# MongoDB setup
mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client[os.getenv("MONGO_DB")]
mongo_collection = mongo_db[os.getenv("MONGO_COLLECTION")]

# Load face embeddings
with open("face_embeddings.pkl", "rb") as f:
    embedding_data = pickle.load(f)

# Flask blueprint
image_bp = Blueprint("image", __name__, url_prefix="/image")

# Allowed file formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@image_bp.route("/recognize", methods=["POST"])
def recognize_person():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Compress image to under 50KB
            img = Image.open(file.stream)
            img.thumbnail((500, 500))
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=70, optimize=True)
            buffer.seek(0)

            while buffer.getbuffer().nbytes > 50 * 1024 and img.size[0] > 100 and img.size[1] > 100:
                img = img.resize((int(img.size[0] * 0.9), int(img.size[1] * 0.9)))
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=60, optimize=True)
                buffer.seek(0)

            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(buffer, folder="face_uploads")
            image_url = upload_result["secure_url"]


            # Get face embedding
            test_obj = DeepFace.represent(img_path=image_url, model_name="Facenet", enforce_detection=True)
            if not test_obj:
                return jsonify({"error": "No face detected"}), 400

            test_embedding = np.array(test_obj[0]["embedding"]).reshape(1, -1)

            # Compare with known embeddings
            best_match = None
            best_score = -1
            threshold = 0.65

            for saved_embedding, saved_name in embedding_data:
                saved_embedding = np.array(saved_embedding).reshape(1, -1)
                score = cosine_similarity(test_embedding, saved_embedding)[0][0]
                if score > best_score:
                    best_score = score
                    best_match = saved_name

            # Prepare response
            response_data = {
                "person": best_match if best_score >= threshold else "not in our Criminal Database",
                "confidence": float(best_score),
                "image_url": image_url,
            }

            # Get client's IP and username from headers
            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
            if client_ip and ',' in client_ip:
                client_ip = client_ip.split(',')[0].strip()

            username = request.headers.get("username") or None

            # Get IST time
            ist = timezone("Asia/Kolkata")
            ist_time = datetime.now(ist)

            # Save result to MongoDB
            mongo_collection.insert_one({
                **response_data,
                "timestamp": ist_time.strftime("%Y-%m-%d %H:%M:%S"),
                "ip_address": client_ip,
                "username": username
            })

            # ✅ Log operation
            OperationLogger.log("Image recognition triggered", client_ip, username)

            return jsonify(response_data), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format. Only JPG, JPEG, PNG allowed."}), 400
