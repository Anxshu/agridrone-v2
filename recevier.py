
from flask import Flask, request, jsonify
from datetime import datetime
from pathlib import Path
import json

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "sensor_data.json"
LOG_FILE = BASE_DIR / "sensor_traffic.log"

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "AgriDrone Sensor Receiver",
        "endpoint": "/sensor",
        "methods_supported": ["POST", "GET"],
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }), 200

@app.route("/sensor", methods=["POST", "GET"])
def receive_sensor():
    # 1. Try to read JSON body
    data = request.get_json(silent=True) or {}
    
    # 2. If not JSON, try URL-encoded form data
    if not data and request.form:
        data = request.form.to_dict()
        
    # 3. If GET or query string parameters exist, use them
    if not data and request.args:
        data = request.args.to_dict()

    # Log incoming request for diagnostics
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {request.method} from {request.remote_addr} - Raw: {request.get_data(as_text=True)} - Parsed: {data}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as lf:
        lf.write(log_entry)

    # 4. Extract temperature with fallback key names
    temp = None
    for k in ["temperature", "temp", "t", "Temperature", "TEMP"]:
        if k in data:
            try:
                temp = float(data[k])
                break
            except (ValueError, TypeError):
                pass

    # 5. Extract humidity with fallback key names
    hum = None
    for k in ["humidity", "hum", "h", "Humidity", "HUM"]:
        if k in data:
            try:
                hum = float(data[k])
                break
            except (ValueError, TypeError):
                pass

    # If it was a browser GET request with no params, show current data
    if request.method == "GET" and temp is None and hum is None:
        if DATA_FILE.exists():
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return jsonify(json.load(f)), 200
        return jsonify({"message": "Sensor endpoint ready. Send POST with {'temperature': ..., 'humidity': ...} or GET with ?temp=...&hum=..."}), 200

    if temp is None or hum is None:
        return jsonify({
            "error": "Missing temperature or humidity",
            "received": data,
            "expected_format": {"temperature": 27.5, "humidity": 55.0}
        }), 400

    record = {
        "temperature": round(temp, 1),
        "humidity": round(hum, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sender_ip": request.remote_addr
    }

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"📡 [{record['timestamp']}] Sensor telemetry from {request.remote_addr}: Temp={temp}°C, Humidity={hum}%", flush=True)
    return jsonify({"status": "ok", "saved": record}), 200

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  🌱 AgriDrone Sensor Receiver v2.0", flush=True)
    print("  Listening on http://0.0.0.0:5000", flush=True)
    print("  Endpoints: POST/GET /sensor", flush=True)
    print("=" * 60, flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
