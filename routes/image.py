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
from utils.featureExtractor import extract_all_features
import tempfile
import pandas as pd
import os

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
image_collection = mongo_db["images"]

# Load face embeddings
with open("face_embeddings.pkl", "rb") as f:
    embedding_data = pickle.load(f)

# Load and cache facial features dataset
def load_facial_features_dataset():
    """Load the facial features dataset and cache it in memory."""
    try:
        csv_path = os.path.join("data", "facial_features_final.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded facial features dataset: {len(df)} records with {len(df.columns)-1} features")
            return df
        else:
            print(f"❌ Facial features CSV not found at: {csv_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading facial features dataset: {e}")
        return None

# Cache the dataset in memory
facial_features_df = load_facial_features_dataset()

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
            image_collection.insert_one({
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


@image_bp.route("/analyze", methods=["POST"])
def comprehensive_facial_analysis():
    """Comprehensive facial analysis: extract features + match against dataset + face recognition."""
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            # Extract all 201 features
            feature_vector = extract_all_features(tmp_path)
            
            if not feature_vector:
                os.unlink(tmp_path)
                return jsonify({"error": "Feature extraction failed. Please ensure the image contains a clear face."}), 500

            # Ensure we have exactly 201 features
            if len(feature_vector) != 201:
                if len(feature_vector) < 201:
                    feature_vector.extend([0.0] * (201 - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:201]

            # Convert numpy types to Python types for JSON serialization
            feature_vector = [float(f) for f in feature_vector]

            # Initialize response data
            response_data = {
                "success": True,
                "features": feature_vector,
                "feature_count": len(feature_vector),
                "analysis": {}
            }

            # 1. FEATURE MATCHING (if dataset is loaded)
            if facial_features_df is not None:
                try:
                    # Convert to numpy array for comparison
                    uploaded_features = np.array(feature_vector).reshape(1, -1)
                    
                    # Get feature columns (exclude 'name' column)
                    feature_columns = [col for col in facial_features_df.columns if col.startswith('feature_')]
                    dataset_features = facial_features_df[feature_columns].values
                    
                    # Calculate cosine similarity with all records
                    similarities = cosine_similarity(uploaded_features, dataset_features)[0]
                    
                    # Find best matches
                    best_match_idx = np.argmax(similarities)
                    best_similarity = similarities[best_match_idx]
                    best_match_name = facial_features_df.iloc[best_match_idx]['name']
                    
                    # Get top 5 matches
                    top_5_indices = np.argsort(similarities)[-5:][::-1]
                    top_5_matches = []
                    for idx in top_5_indices:
                        top_5_matches.append({
                            "name": facial_features_df.iloc[idx]['name'],
                            "similarity": float(similarities[idx])
                        })

                    response_data["analysis"]["feature_matching"] = {
                        "best_match": {
                            "name": best_match_name,
                            "similarity": float(best_similarity),
                            "confidence": "High" if best_similarity > 0.8 else "Medium" if best_similarity > 0.6 else "Low"
                        },
                        "top_5_matches": top_5_matches,
                        "total_records_compared": len(dataset_features)
                    }
                except Exception as e:
                    response_data["analysis"]["feature_matching"] = {"error": f"Feature matching failed: {str(e)}"}

            # 2. FACE RECOGNITION (using existing DeepFace system)
            try:
                # Compress image for face recognition
                img = Image.open(tmp_path)
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
                if test_obj:
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

                    response_data["analysis"]["face_recognition"] = {
                        "person": best_match if best_score >= threshold else "not in our Criminal Database",
                        "confidence": float(best_score),
                        "image_url": image_url,
                        "threshold_met": best_score >= threshold
                    }
                else:
                    response_data["analysis"]["face_recognition"] = {"error": "No face detected for recognition"}
            except Exception as e:
                response_data["analysis"]["face_recognition"] = {"error": f"Face recognition failed: {str(e)}"}

            # Clean up temporary file
            os.unlink(tmp_path)

            # Get client info for logging
            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
            if client_ip and ',' in client_ip:
                client_ip = client_ip.split(',')[0].strip()
            username = request.headers.get("username") or None

            # Log operation
            OperationLogger.log("Comprehensive facial analysis triggered", client_ip, username)

            response_data["message"] = "Comprehensive facial analysis completed successfully"
            return jsonify(response_data), 200

        except Exception as e:
            # Clean up temporary file if it exists
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format. Only JPG, JPEG, PNG allowed."}), 400


@image_bp.route("/match-features", methods=["POST"])
def match_facial_features():
    """Match uploaded image features against the cached dataset."""
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if facial_features_df is None:
        return jsonify({"error": "Facial features dataset not loaded"}), 500

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            # Extract features from uploaded image
            uploaded_features = extract_all_features(tmp_path)
            
            # Clean up temporary file
            os.unlink(tmp_path)

            if not uploaded_features or len(uploaded_features) != 201:
                return jsonify({"error": "Failed to extract 201 features from uploaded image"}), 500

            # Convert to numpy array for comparison
            uploaded_features = np.array(uploaded_features).reshape(1, -1)
            
            # Get feature columns (exclude 'name' column)
            feature_columns = [col for col in facial_features_df.columns if col.startswith('feature_')]
            dataset_features = facial_features_df[feature_columns].values
            
            # Calculate cosine similarity with all records
            similarities = cosine_similarity(uploaded_features, dataset_features)[0]
            
            # Find best matches
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            best_match_name = facial_features_df.iloc[best_match_idx]['name']
            
            # Get top 5 matches
            top_5_indices = np.argsort(similarities)[-5:][::-1]
            top_5_matches = []
            for idx in top_5_indices:
                top_5_matches.append({
                    "name": facial_features_df.iloc[idx]['name'],
                    "similarity": float(similarities[idx])
                })

            # Get client info for logging
            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
            if client_ip and ',' in client_ip:
                client_ip = client_ip.split(',')[0].strip()
            username = request.headers.get("username") or None

            # Log operation
            OperationLogger.log("Facial feature matching triggered", client_ip, username)

            return jsonify({
                "success": True,
                "best_match": {
                    "name": best_match_name,
                    "similarity": float(best_similarity),
                    "confidence": "High" if best_similarity > 0.8 else "Medium" if best_similarity > 0.6 else "Low"
                },
                "top_5_matches": top_5_matches,
                "total_records_compared": len(dataset_features),
                "message": "Feature matching completed successfully"
            }), 200

        except Exception as e:
            # Clean up temporary file if it exists
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format. Only JPG, JPEG, PNG allowed."}), 400


@image_bp.route("/dataset-info", methods=["GET"])
def get_dataset_info():
    """Get information about the cached facial features dataset."""
    if facial_features_df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    return jsonify({
        "success": True,
        "total_records": len(facial_features_df),
        "total_features": len([col for col in facial_features_df.columns if col.startswith('feature_')]),
        "sample_names": facial_features_df['name'].head(10).tolist(),
        "message": "Dataset information retrieved successfully"
    }), 200
