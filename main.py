from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn

from routes.user_routes import user_bp
from routes.image import router as image_router

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

app.include_router(user_bp,prefix="/users", tags=["User Management"])
app.include_router(image_router, prefix="/image", tags=["Image Recognition"])


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
    import uvicorn

    # reload and workers>1 are incompatible; pick ONE
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
