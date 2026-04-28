import os
import sys
import logging
import traceback
from pathlib import Path

# Force TensorFlow 2.15 to use Legacy Keras behavior
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# --- Logging Configuration ---
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
from utility import get_str_message_from_confidence_score

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
        "traceback": msg.split('\n')[-2] 
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

    try:
        image_path = save_image_from_base64(base64_str=b64, ext=extension)
        confidence = cnn_analyze_image(image_path, model_name=model_preferred) if model_preferred else cnn_analyze_image(image_path)
        # return jsonify({"confidence": confidence, "details": {"model_used": model_preferred or "default"}})
    except Exception as e:
        logging.error(f"Failed to save image: {e}", exc_info=True)
        return jsonify({"error": f"Failed on saving image! {e}"}), 400

    # Run inference
    try:
        if model_preferred:
            logging.info(f"Running inference with model: {model_preferred}")
            confidence = cnn_analyze_image(image_path, model_name=model_preferred)
        else:
            logging.info("Running inference with default model")
            confidence = cnn_analyze_image(image_path)

        logging.info(f"Inference completed. Confidence: {confidence}")

    except Exception as e:
        logging.error(f"Inference failed: {e}", exc_info=True)
        return jsonify({"error": f"Inference failed: {e}"}), 500

    response = {
        "confidence": confidence,
        "message": get_str_message_from_confidence_score(confidence),
        "details": {"model_used": model_preferred if model_preferred else "default_model_set"},
    }

    logging.debug(f"Response payload: {response}")
    return jsonify(response)


@app.post("/analyze_text")
def analyze_text():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing/Invalid text"}), 400

    try:
        confidence = lstm_analyze_text(text=text)
	
        if confidence == -1.0:
            return jsonify({"error": "Internal LSTM error"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error during text analysis: {e}", exc_info=True)
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    response = {"confidence": confidence, "message": get_str_message_from_confidence_score(confidence)}
    logging.debug(f"Response payload: {response}")
																	
	 
	
    logging.info("Successfully returning /analyze_text response")
    return jsonify(response)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1337, debug=True)