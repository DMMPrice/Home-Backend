# models/ip_logger.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import requests

# Load env variables
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
ip_logs = db["ip_logs"]


class IPLogger:
    @staticmethod
    def log_request(ip_address, username=None):
        try:
            # Get location info from IP using ipinfo.io
            location_data = {}
            coords = {}
            try:
                response = requests.get(f"https://ipinfo.io/{ip_address}/json")
                if response.status_code == 200:
                    info = response.json()
                    location_data["city"] = info.get("city")
                    location_data["region"] = info.get("region")
                    location_data["country"] = info.get("country")
                    loc = info.get("loc", "")  # "lat,lon"
                    if loc:
                        lat, lon = loc.split(",")
                        coords = {"latitude": lat, "longitude": lon}
            except Exception as e:
                location_data["error"] = f"Location lookup failed: {str(e)}"

            log_entry = {
                "ip": ip_address,
                "location": location_data,
                "coordinates": coords,
                "username": username,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            ip_logs.insert_one(log_entry)
            return {"message": "IP logged successfully"}
        except Exception as e:
            return {"error": f"Failed to log IP: {str(e)}"}
