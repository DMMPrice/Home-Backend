from fastapi import APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from models.users import User
from models.operation_logger import OperationLogger
from typing import Optional

user_bp = APIRouter(prefix="/users", tags=["Users"])

# Pydantic models
class UserRegistration(BaseModel):
    username: str
    email: str
    password: str
    name: str
    role: str = "user"

class UserLogin(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None


# ✅ Helper to require username from headers
def get_username_from_header(username: str = Header(None)):
    if not username:
        raise HTTPException(status_code=400, detail="Missing required header: username")
    return username


# ✅ Register a new user (no username required)
@user_bp.post("/register")
async def register_user(user_data: UserRegistration):
    username = user_data.username
    name = user_data.name
    email = user_data.email
    password = user_data.password
    role = user_data.role

    if not all([username, name, email, password]):
        raise HTTPException(status_code=400, detail="Username, name, email, and password are required")

    response, status = User.register(username, name, email, password, role)
    
    # Log operation (simplified for FastAPI)
    OperationLogger.log(f"User '{username}' registered", "127.0.0.1", username)
    
    if status != 201:
        raise HTTPException(status_code=status, detail=response.get("error", "Registration failed"))
    return response


# ✅ Login user (no username header required)
@user_bp.post("/login")
async def login_user(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    response, status = User.login(username, password)

    OperationLogger.log(f"User '{username}' attempted login", "127.0.0.1", username)

    if status != 200:
        raise HTTPException(status_code=status, detail=response.get("error", "Login failed"))
    return response


# ✅ Get all users (requires username in header)
@user_bp.get("/")
async def get_all_users(username: str = Header(None)):
    if not username:
        raise HTTPException(status_code=401, detail="Username header required")
    users = User.get_all_users()
    OperationLogger.log("Fetched all users", "127.0.0.1", username)

    return users


# ✅ Get single user by ID (requires username in header)
@user_bp.get("/{user_id}")
async def get_user(user_id: str, username: str = Header(None)):
    if not username:
        raise HTTPException(status_code=401, detail="Username header required")
    
    response = User.get_user_by_id(user_id)

    if isinstance(response, dict) and "username" in response:
        OperationLogger.log(f"Fetched user by ID {user_id}", "127.0.0.1", username)
    else:
        OperationLogger.log(f"Tried to fetch user by ID {user_id}", "127.0.0.1", username)

    if isinstance(response, dict) and response.get("error"):
        if response["error"] == "Invalid user ID":
            raise HTTPException(status_code=400, detail=response["error"])
        else:
            raise HTTPException(status_code=404, detail=response["error"])
    return response


# ✅ Update user by ID (requires username in header)
@user_bp.put("/{user_id}")
async def update_user(user_id: str, updates: dict, username: str = Header(None)):
    if not username:
        raise HTTPException(status_code=401, detail="Username header required")
    
    response = User.update_user(user_id, updates)

    OperationLogger.log(f"Updated user {user_id}", "127.0.0.1", username)

    if "Invalid" in response.get("error", ""):
        raise HTTPException(status_code=400, detail=response["error"])
    elif "not found" in response.get("error", ""):
        raise HTTPException(status_code=404, detail=response["error"])
    return response


# ✅ Delete user by ID (requires username in header)
@user_bp.delete("/{user_id}")
async def delete_user(user_id: str, username: str = Header(None)):
    if not username:
        raise HTTPException(status_code=401, detail="Username header required")
    
    response = User.delete_user(user_id)

    OperationLogger.log(f"Deleted user with ID {user_id}", "127.0.0.1", username)

    if "Invalid" in response.get("error", ""):
        raise HTTPException(status_code=400, detail=response["error"])
    elif "not found" in response.get("error", ""):
        raise HTTPException(status_code=404, detail=response["error"])
    return response


# ✅ Forgot password route
@user_bp.post("/forgot-password")
async def forgot_password(request: Request):
    data = await request.json()
    username = data.get("username")  # ✅ only username allowed now
    new_password = data.get("new_password")

    if not username or not new_password:
        raise HTTPException(status_code=400, detail="Username and new password are required")

    response, status = User.forgot_password_by_username(username, new_password)

    OperationLogger.log(f"Forgot password attempt for '{username}'", "127.0.0.1", username)

    if status != 200:
        raise HTTPException(status_code=status, detail=response.get("error", "Password reset failed"))
    return response
