# Feature Extraction API

This module provides facial feature extraction functionality that extracts 201 features from facial images using computer vision techniques.

## Features

The API extracts 201 facial features including:
- Delaunay triangulation area and centroid length
- Grayscale matrix statistics (16x16)
- Pupil distance and facial measurements
- Face area and perimeter
- Eye aspect ratios and distances
- Facial symmetry scores
- Local Binary Pattern (LBP) histograms
- Gabor filter responses
- Gray-Level Co-occurrence Matrix (GLCM) features
- Jaw curvature and symmetry
- Golden ratio analysis

## API Endpoints

### 1. Extract Features from File Upload
**POST** `/extract-features`

Upload an image file to extract features.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` (file)

**Response:**
```json
{
    "success": true,
    "features": [0.1, 0.2, ...], // 201 features
    "feature_count": 201,
    "message": "Features extracted successfully"
}
```

### 2. Extract Features from File Path
**POST** `/extract-features-from-path`

Extract features from an image file at a given path.

**Request:**
- Method: POST
- Content-Type: application/json
- Body:
```json
{
    "image_path": "/path/to/image.jpg"
}
```

**Response:**
```json
{
    "success": true,
    "features": [0.1, 0.2, ...], // 201 features
    "feature_count": 201,
    "message": "Features extracted successfully"
}
```

### 3. Health Check
**GET** `/health`

Check if the feature extraction service is running.

**Response:**
```json
{
    "success": true,
    "message": "Feature extraction service is running",
    "status": "healthy"
}
```

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP
- TIFF

## Dependencies

The feature extraction requires the following Python packages:
- opencv-python-headless
- dlib
- mediapipe
- scikit-image
- scipy
- numpy

## Usage Example

### Python
```python
import requests

# Extract features from file upload
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:6000/extract-features', files=files)
    result = response.json()
    features = result['features']

# Extract features from file path
data = {"image_path": "/path/to/image.jpg"}
response = requests.post('http://localhost:6000/extract-features-from-path', json=data)
result = response.json()
features = result['features']
```

### JavaScript/Frontend
```javascript
// Extract features from file upload
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:6000/extract-features', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Features:', data.features);
});

// Extract features from file path
fetch('http://localhost:6000/extract-features-from-path', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        image_path: '/path/to/image.jpg'
    })
})
.then(response => response.json())
.then(data => {
    console.log('Features:', data.features);
});
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- **400 Bad Request**: Invalid file type, missing file, or invalid request
- **404 Not Found**: Image file not found (for path-based extraction)
- **500 Internal Server Error**: Server-side processing error

## Notes

- The API automatically downloads the dlib shape predictor model on first use
- Feature extraction may take several seconds depending on image complexity
- Ensure images contain clear, well-lit faces for best results
- The API returns exactly 201 features as a flat array
