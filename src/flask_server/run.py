import os
import sys
import logging
import traceback # Added for detailed error reporting
from pathlib import Path

# Force TensorFlow 2.15 to use Legacy Keras behavior
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# --- Logging Configuration ---
# This prints errors to your terminal with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add parent directory to path for direct execution support
if __package__:
    from .utility import save_image_from_base64
else:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from flask_server.utility import save_image_from_base64

from lstm.services import analyze_text as lstm_analyze_text
from cnn.services import cnn_analyze_image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Global Error Handler ---
@app.errorhandler(Exception)
def handle_exception(e):
    """Logs the actual traceback when a 500 error occurs."""
    msg = traceback.format_exc()
    logger.error(f"!!! SERVER ERROR !!!\n{msg}")
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e),
        "traceback": msg.split('\n')[-2] # Gives you the last line of the error
    }), 500

@app.get("/")
def index():
    return jsonify({"message": "Welcome to the server homepage"})

@app.get("/ping")
def ping():
    return jsonify({"message": "pong"})

@app.post("/analyze_image")
def analyze_image():
    payload = request.get_json(silent=True) or {}
    b64 = payload.get("image") or payload.get("image_base64")
    model_preferred = payload.get("model")
    extension = payload.get("ext", "jpg")

    if not b64:
        return jsonify({"error": "Missing image data"}), 400

    image_path = save_image_from_base64(base64_str=b64, ext=extension)
    
    # We removed the local try/except here so the global handler 
    # can catch and log the full traceback for us.
    confidence = cnn_analyze_image(image_path, model_name=model_preferred) if model_preferred else cnn_analyze_image(image_path)
    
    return jsonify({
        "confidence": confidence, 
        "details": {"model_used": model_preferred or "default"}
    })

@app.post("/analyze_text")
def analyze_text():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing/Invalid text"}), 400

    confidence = lstm_analyze_text(text=text)
    if confidence == -1.0:
        return jsonify({"error": "Internal LSTM error"}), 500
    return jsonify({"confidence": confidence})

if __name__ == "__main__":
    # debug=True is critical here; it reloads the server when you change code
    app.run(host="0.0.0.0", port=1337, debug=True)