import os
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# MongoDB setup using local .env config
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
user_collection = db["users"]

VALID_ROLES = ["super-admin", "admin", "user"]


# User Model
class User:
    @staticmethod
    def register(username, name, email, password, role="guest"):
        if user_collection.find_one({"username": username}):
            return {"error": "User already exists"}, 409

        if role not in VALID_ROLES:
            return {"error": "Invalid role specified"}, 400

        hashed_password = generate_password_hash(password)
        user_id = user_collection.insert_one({
            "username": username,
            "name": name,
            "email": email,
            "password": hashed_password,
            "role": role
        }).inserted_id

        return {"message": "User registered successfully", "user_id": str(user_id)}, 201

    @staticmethod
    def login(username, password):
        user = user_collection.find_one({"username": username})
        if not user or not check_password_hash(user["password"], password):
            return {"error": "Invalid username or password"}, 401

        return {
            "message": "Login successful",
            "user_id": str(user["_id"]),
            "role": user.get("role", "guest"),
            "name": user.get("name"),
            "email": user.get("email"),
            "username": user.get("username")
        }, 200

    @staticmethod
    def get_all_users():
        users = list(user_collection.find())
        for user in users:
            user["_id"] = str(user["_id"])
            del user["password"]  # optional: hide password hash in output
        return users

    @staticmethod
    def get_user_by_id(user_id):
        if not ObjectId.is_valid(user_id):
            return {"error": "Invalid user ID"}, 400

        user = user_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return {"error": "User not found"}, 404

        user["_id"] = str(user["_id"])
        del user["password"]
        return user

    @staticmethod
    def update_user(user_id, updates):
        if not ObjectId.is_valid(user_id):
            return {"error": "Invalid user ID"}, 400

        update_fields = {}
        if "username" in updates:
            update_fields["username"] = updates["username"]
        if "name" in updates:
            update_fields["name"] = updates["name"]
        if "email" in updates:
            update_fields["email"] = updates["email"]
        if "password" in updates:
            update_fields["password"] = generate_password_hash(updates["password"])
        if "role" in updates:
            if updates["role"] not in VALID_ROLES:
                return {"error": "Invalid role"}, 400
            update_fields["role"] = updates["role"]

        result = user_collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_fields})
        if result.matched_count == 0:
            return {"error": "User not found"}, 404

        return {"message": "User updated successfully"}, 200

    @staticmethod
    def delete_user(user_id):
        if not ObjectId.is_valid(user_id):
            return {"error": "Invalid user ID"}, 400

        result = user_collection.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            return {"error": "User not found"}, 404

        return {"message": "User deleted successfully"}, 200

    @staticmethod
    def forgot_password_by_username(username, new_password):
        user = user_collection.find_one({"username": username})

        if not user:
            return {"error": "User not found"}, 404

        if check_password_hash(user["password"], new_password):
            return {"error": "New password cannot be the same as the old password"}, 400

        hashed_password = generate_password_hash(new_password)
        result = user_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"password": hashed_password}}
        )

        if result.modified_count == 1:
            return {"message": "Password updated successfully"}, 200
        else:
            return {"error": "Password reset failed"}, 500
