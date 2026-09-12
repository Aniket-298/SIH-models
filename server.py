# server.py
# Simple Flask backend that receives an audio file from the frontend
# and forwards it to a prediction service running at PREDICT_URL.

import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the frontend (different origin) to call this server

PREDICT_URL = os.environ.get("PREDICT_URL", "http://localhost:8080/predict")
PORT = int(os.environ.get("PORT", 3000))

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB cap, adjust as needed
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    # Frontend should send a multipart/form-data request with the audio
    # file under the field name "audio".
    if "audio" not in request.files:
        return jsonify(
            {"error": "No audio file uploaded. Use form field name 'audio'."}
        ), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        files = {
            "file": (
                audio_file.filename,
                audio_file.stream,
                audio_file.mimetype or "application/octet-stream",
            )
        }

        response = requests.post(PREDICT_URL, files=files, timeout=60)

        # Pass the prediction service's response straight back to the frontend
        try:
            return jsonify(response.json()), response.status_code
        except ValueError:
            # predict service didn't return JSON
            return response.text, response.status_code

    except requests.exceptions.ConnectionError:
        return jsonify(
            {"error": f"Could not reach prediction service at {PREDICT_URL}"}
        ), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Prediction service timed out"}), 504
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    print(f"Audio forward server running on http://localhost:{PORT}")
    print(f"Forwarding uploads to {PREDICT_URL}")
    app.run(host="0.0.0.0", port=PORT, debug=True)