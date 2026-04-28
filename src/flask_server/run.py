import sys
from pathlib import Path
import logging

# Add parent directory to path for direct execution support
if __package__:
    from .utility import save_image_from_base64
else:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from flask_server.utility import save_image_from_base64

from flask import Flask, jsonify, request
from flask_cors import CORS
from lstm.services import analyze_text as lstm_analyze_text
from lstm.services import ping_text_analysis_side as ping_text
from cnn.services import cnn_analyze_image

app = Flask(__name__)

# Minimal CORS fix
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/")
def index():
    return jsonify({"message": "Welcome to the server homepage"})


@app.get("/ping")
def ping():
    return jsonify({"message": "pong"})


@app.get("/ping_text_side")
def ping_text_side():
    return lstm_analyze_text.ping_text()


# ANALYZE IMAGE !!
@app.post("/analyze_image")
def analyze_image():
    logging.info("Received request for /analyze_image endpoint")

    payload = request.get_json(silent=True) or {}
    logging.debug(f"Raw payload keys: {list(payload.keys())}")

    # Support multiple frontend keys
    b64 = payload.get("image") or payload.get("image_base64") or payload.get("base64") or payload.get("b64")
    model_preferred = payload.get("model")
    extension = payload.get("ext", "jpg")

    if not b64:
        logging.warning("No base64 image data found in request payload")
        return jsonify({"error": "Missing image data"}), 400

    logging.info(f"Image extension: {extension}")
    logging.info(f"Model requested: {model_preferred if model_preferred else 'default'}")

    # Save image
    try:
        image_path = save_image_from_base64(base64_str=b64, ext=extension)
        logging.info(f"Image successfully saved at: {image_path}")
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

    response = {"confidence": confidence, "details": {"model_used": model_preferred if model_preferred else "default_model_set"}}

    logging.debug(f"Response payload: {response}")
    return jsonify(response)


# ANALYZE TEXT !!
@app.post("/analyze_text")
def analyze_text():
    logging.info("Received request for /analyze_text endpoint")

    payload = request.get_json(silent=True) or {}
    logging.debug(f"Raw payload keys: {list(payload.keys())}")

    text = payload.get("text")
    logging.debug(f"Text parameter received (length: {len(text) if text else 0} characters)")

    if not isinstance(text, str) or not text.strip():
        logging.warning("Invalid or missing 'text' parameter in request")
        return jsonify({"error": "Missing/Invalid parameter_key: 'text'"}), 400

    logging.info(f"Text validation passed. First 50 chars: {text[:50]}...")

    try:
        logging.info("Calling LSTM analyze_text service")
        confidence = lstm_analyze_text(text=text)
        logging.info(f"LSTM analysis completed. Confidence score: {confidence}")
    except RuntimeError as e:
        logging.error(f"RuntimeError during text analysis: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error during text analysis: {e}", exc_info=True)
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    if confidence == -1.0:
        logging.error("LSTM service returned error code (-1.0). Check model loading and input validity.")
        return jsonify({"error": "Text analysis failed due to internal error."}), 500

    response = {"confidence": confidence}
    logging.debug(f"Response payload: {response}")
    logging.info("Successfully returning /analyze_text response")
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1337, debug=True)
