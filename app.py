from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn

from routes.feature_extraction_routes import feature_bp
from routes.user_routes import user_bp

# Initialize FastAPI app
app = FastAPI(
    title="Home Backend API",
    description="Backend API with facial feature extraction",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[       
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Include routers (FastAPI equivalent of Flask blueprints)
app.include_router(feature_bp)
app.include_router(user_bp)


@app.get("/")
async def home():
    return {"message": "HOME API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"message": "API is healthy", "status": "running"}


# Error handling for 404 - Not Found
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

# Error handling for 500 - Internal Server Error
@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import os
    
    port = int(os.environ.get("PORT", 6000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
