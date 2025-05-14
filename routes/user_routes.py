from flask import Blueprint, request, jsonify, abort
from models.users import User
from models.operation_logger import OperationLogger

user_bp = Blueprint("user", __name__, url_prefix="/users")


# ✅ Helper to require username from headers
def get_username_or_abort():
    username = request.headers.get("username")
    if not username:
        abort(400, description="Missing required header: username")
    return username


# ✅ Register a new user (no username required)
@user_bp.route("/register", methods=["POST"])
def register_user():
    data = request.get_json()
    username = data.get("username")
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "guest")

    if not all([username, name, email, password]):
        return jsonify({"error": "Username, name, email, and password are required"}), 400

    response, status = User.register(username, name, email, password, role)

    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    OperationLogger.log(f"User '{username}' registered", ip, username)

    return jsonify(response), status


# ✅ Login user (no username header required)
@user_bp.route("/login", methods=["POST"])
def login_user():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    response, status = User.login(username, password)

    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    OperationLogger.log(f"User '{username}' attempted login", ip, username)

    return jsonify(response), status


# ✅ Get all users (requires username in header)
@user_bp.route("/", methods=["GET"])
def get_all_users():
    username = get_username_or_abort()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    users = User.get_all_users()
    OperationLogger.log("Fetched all users", ip, username)

    return jsonify(users), 200


# ✅ Get single user by ID (requires username in header)
@user_bp.route("/<user_id>", methods=["GET"])
def get_user(user_id):
    username = get_username_or_abort()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    response = User.get_user_by_id(user_id)

    if isinstance(response, dict) and "username" in response:
        OperationLogger.log(f"Fetched user by ID {user_id}", ip, username)
    else:
        OperationLogger.log(f"Tried to fetch user by ID {user_id}", ip, username)

    if isinstance(response, dict) and response.get("error"):
        return jsonify(response), 400 if response["error"] == "Invalid user ID" else 404
    return jsonify(response), 200


# ✅ Update user by ID (requires username in header)
@user_bp.route("/<user_id>", methods=["PUT"])
def update_user(user_id):
    username = get_username_or_abort()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    updates = request.get_json()
    response = User.update_user(user_id, updates)

    OperationLogger.log(f"Updated user {user_id}", ip, username)

    status = 400 if "Invalid" in response.get("error", "") else 404 if "not found" in response.get("error", "") else 200
    return jsonify(response), status


# ✅ Delete user by ID (requires username in header)
@user_bp.route("/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    username = get_username_or_abort()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    response = User.delete_user(user_id)

    OperationLogger.log(f"Deleted user with ID {user_id}", ip, username)

    status = 400 if "Invalid" in response.get("error", "") else 404 if "not found" in response.get("error", "") else 200
    return jsonify(response), status


# ✅ Forgot password route
@user_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json()
    username = data.get("username")  # ✅ only username allowed now
    new_password = data.get("new_password")

    if not username or not new_password:
        return jsonify({"error": "Username and new password are required"}), 400

    response, status = User.forgot_password_by_username(username, new_password)

    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    OperationLogger.log(f"Forgot password attempt for '{username}'", ip, username)

    return jsonify(response), status
