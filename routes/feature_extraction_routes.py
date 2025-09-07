from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
from utils.featureExtractor import extract_all_features
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import cloudinary
import cloudinary.uploader
from PIL import Image
from io import BytesIO
from deepface import DeepFace
import pickle

# Create router
feature_bp = APIRouter(prefix="/feature-extraction", tags=["Feature Extraction"])

# Configure logging
logger = logging.getLogger(__name__)

# Load and cache facial features dataset
def load_facial_features_dataset():
    """Load the facial features dataset and cache it in memory."""
    try:
        csv_path = os.path.join("data", "facial_features_final.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"✅ Loaded facial features dataset: {len(df)} records with {len(df.columns)-1} features")
            return df
        else:
            logger.error(f"❌ Facial features CSV not found at: {csv_path}")
            return None
    except Exception as e:
        logger.error(f"❌ Error loading facial features dataset: {e}")
        return None

# Load face embeddings
def load_face_embeddings():
    """Load face embeddings for recognition."""
    try:
        with open("face_embeddings.pkl", "rb") as f:
            embedding_data = pickle.load(f)
        logger.info(f"✅ Loaded face embeddings: {len(embedding_data)} records")
        return embedding_data
    except Exception as e:
        logger.error(f"❌ Error loading face embeddings: {e}")
        return None

# Cache the datasets in memory
facial_features_df = load_facial_features_dataset()
embedding_data = load_face_embeddings()

# Prepare scaled features for improved matching
def prepare_scaled_features():
    """Prepare scaled features for improved cosine similarity matching."""
    if facial_features_df is None:
        return None, None
    
    try:
        # Get feature columns (exclude 'name' column)
        feature_columns = [col for col in facial_features_df.columns if col.startswith('feature_')]
        features_data = facial_features_df[feature_columns].values
        
        # Create names dataframe for results
        df_names = facial_features_df[['name']].copy()
        
        # Scale the features using StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_data)
        
        logger.info(f"✅ Prepared scaled features: {features_scaled.shape}")
        return features_scaled, df_names, scaler
    except Exception as e:
        logger.error(f"❌ Error preparing scaled features: {e}")
        return None, None, None

# Prepare scaled features at startup
df_features_scaled, df_names, scaler = prepare_scaled_features()

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Pydantic models
class ImagePathRequest(BaseModel):
    image_path: str

class FeatureResponse(BaseModel):
    success: bool
    features: list
    feature_count: int
    message: str

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@feature_bp.post('/extract-features', response_model=FeatureResponse)
async def extract_features(image: UploadFile = File(...)):
    """
    Extract 201 facial features from an uploaded image
    """
    try:
        # Check if file is selected
        if not image.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Check if file type is allowed
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, tiff"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{image.filename.rsplit(".", 1)[1].lower()}') as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Extract features
            logger.info(f"Extracting features from image: {image.filename}")
            features = extract_all_features(temp_path)
            
            if not features or len(features) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract features from image. Please ensure the image contains a clear face."
                )
            
            # Ensure we have exactly 201 features
            if len(features) != 201:
                logger.warning(f"Expected 201 features, got {len(features)}")
                # Pad or truncate to 201 features
                if len(features) < 201:
                    features.extend([0.0] * (201 - len(features)))
                else:
                    features = features[:201]
            
            # Convert numpy types to Python types for JSON serialization
            features = [float(f) for f in features]
            
            return FeatureResponse(
                success=True,
                features=features,
                feature_count=len(features),
                message="Features extracted successfully"
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning(f"Could not delete temporary file: {temp_path}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@feature_bp.post('/extract-features-from-path', response_model=FeatureResponse)
async def extract_features_from_path(request: ImagePathRequest):
    """
    Extract 201 facial features from an image file path
    """
    try:
        image_path = request.image_path
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Check if file type is allowed
        if not allowed_file(image_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, tiff"
            )
        
        # Extract features
        logger.info(f"Extracting features from image path: {image_path}")
        features = extract_all_features(image_path)
        
        if not features or len(features) == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract features from image. Please ensure the image contains a clear face."
            )
        
        # Ensure we have exactly 201 features
        if len(features) != 201:
            logger.warning(f"Expected 201 features, got {len(features)}")
            # Pad or truncate to 201 features
            if len(features) < 201:
                features.extend([0.0] * (201 - len(features)))
            else:
                features = features[:201]
        
        # Convert numpy types to Python types for JSON serialization
        features = [float(f) for f in features]
        
        return FeatureResponse(
            success=True,
            features=features,
            feature_count=len(features),
            message="Features extracted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_features_from_path: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@feature_bp.post('/analyze', response_model=FeatureResponse)
async def comprehensive_facial_analysis(image: UploadFile = File(...)):
    """
    Comprehensive facial analysis: extract features + match against dataset + face recognition
    """
    try:
        # Check if file is selected
        if not image.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Check if file type is allowed
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, tiff"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{image.filename.rsplit(".", 1)[1].lower()}') as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Extract features
            logger.info(f"Extracting features from image: {image.filename}")
            features = extract_all_features(temp_path)
            
            if not features or len(features) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract features from image. Please ensure the image contains a clear face."
                )
            
            # Ensure we have exactly 201 features
            if len(features) != 201:
                logger.warning(f"Expected 201 features, got {len(features)}")
                if len(features) < 201:
                    features.extend([0.0] * (201 - len(features)))
                else:
                    features = features[:201]
            
            # Convert numpy types to Python types for JSON serialization
            features = [float(f) for f in features]

            # Initialize response data
            response_data = {
                "success": True,
                "features": features,
                "feature_count": len(features),
                "analysis": {}
            }

            # 1. IMPROVED FEATURE MATCHING (using scaled features like Jupyter notebook)
            if df_features_scaled is not None and df_names is not None and scaler is not None:
                try:
                    # Convert uploaded features to numpy array and scale them
                    uploaded_features = np.array(features).reshape(1, -1)
                    uploaded_features_scaled = scaler.transform(uploaded_features)
                    
                    # Calculate cosine similarity using scaled features (same as Jupyter notebook)
                    similarity_scores = cosine_similarity(uploaded_features_scaled, df_features_scaled)
                    
                    # Add similarity scores to names dataframe
                    df_names_with_scores = df_names.copy()
                    df_names_with_scores['similarity_score'] = similarity_scores[0]
                    
                    # Sort by similarity score (descending) to get top matches
                    top_matches = df_names_with_scores.sort_values(by='similarity_score', ascending=False)
                    
                    # Get best match
                    best_match = top_matches.iloc[0]
                    best_match_name = best_match['name']
                    best_similarity = float(best_match['similarity_score'])
                    
                    # Get top 5 matches
                    top_5_matches = []
                    for idx in range(min(5, len(top_matches))):
                        match = top_matches.iloc[idx]
                        top_5_matches.append({
                            "name": match['name'],
                            "similarity": float(match['similarity_score'])
                        })

                    # Determine if it's a criminal match based on similarity threshold
                    criminal_threshold = 0.7  # Adjust this threshold as needed
                    is_criminal_match = best_similarity >= criminal_threshold

                    response_data["analysis"]["criminal_identification"] = {
                        "is_criminal": is_criminal_match,
                        "best_match": {
                            "name": best_match_name,
                            "similarity": best_similarity,
                            "confidence": "High" if best_similarity > 0.8 else "Medium" if best_similarity > 0.6 else "Low"
                        },
                        "top_5_matches": top_5_matches,
                        "total_records_compared": len(df_features_scaled),
                        "criminal_threshold": criminal_threshold,
                        "match_status": "CRIMINAL IDENTIFIED" if is_criminal_match else "No criminal match found"
                    }
                except Exception as e:
                    response_data["analysis"]["criminal_identification"] = {"error": f"Criminal identification failed: {str(e)}"}
            else:
                response_data["analysis"]["criminal_identification"] = {"error": "Scaled features not available for criminal identification"}

            # 2. FACE RECOGNITION (if embeddings are loaded)
            if embedding_data is not None:
                try:
                    # Compress image for face recognition
                    img = Image.open(temp_path)
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

            response_data["message"] = "Comprehensive facial analysis completed successfully"
            return response_data
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning(f"Could not delete temporary file: {temp_path}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive_facial_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@feature_bp.get('/dataset-info')
async def get_dataset_info():
    """
    Get information about the cached facial features dataset
    """
    if facial_features_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    return {
        "success": True,
        "total_records": len(facial_features_df),
        "total_features": len([col for col in facial_features_df.columns if col.startswith('feature_')]),
        "columns": list(facial_features_df.columns),
        "sample_data": facial_features_df.head(5).to_dict('records'),
        "sample_names": facial_features_df['name'].head(10).tolist(),
        "message": "Dataset information retrieved successfully"
    }


@feature_bp.get('/dataset-preview')
async def get_dataset_preview(limit: int = 10):
    """
    Get a preview of the cached dataset with specified limit
    """
    if facial_features_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    if limit > 100:
        limit = 100  # Cap at 100 records for performance
    
    preview_data = facial_features_df.head(limit)
    
    return {
        "success": True,
        "limit": limit,
        "total_records": len(facial_features_df),
        "preview_data": preview_data.to_dict('records'),
        "message": f"Dataset preview with {limit} records"
    }


@feature_bp.get('/health')
async def health_check():
    """
    Health check endpoint for feature extraction service
    """
    return {
        'success': True,
        'message': 'Feature extraction service is running',
        'status': 'healthy'
    }
