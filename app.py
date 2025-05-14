from flask import Flask
from flask_cors import CORS
import logging

from routes.user_routes import user_bp
from routes.image import image_bp
from routes.master_database_routes import criminal_bp
from routes.ip_logger_routes import ip_bp
from routes.operational_logger_routes import operation_logger_bp
from routes.hotel_master_routes import hotel_bp
from routes.hotel_user_routes import hotel_user_bp
from routes.guest_routes import guest_bp

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO)

# Register Blueprints
app.register_blueprint(user_bp)
app.register_blueprint(image_bp)
app.register_blueprint(criminal_bp)
app.register_blueprint(ip_bp)
app.register_blueprint(operation_logger_bp)
app.register_blueprint(hotel_bp)
app.register_blueprint(hotel_user_bp)
app.register_blueprint(guest_bp)


@app.route("/")
def home():
    return "HOME API is running!"


# Error handling for 404 - Not Found
@app.errorhandler(404)
def not_found(error):
    return {"error": "Endpoint not found"}, 404


# Error handling for 500 - Internal Server Error
@app.errorhandler(500)
def server_error(error):
    return {"error": "Internal server error"}, 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
