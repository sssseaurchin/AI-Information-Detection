import sys
from pathlib import Path

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
    payload = request.get_json(silent=True) or {}

    # Support your current frontend key too
    b64 = payload.get("image") or payload.get("image_base64") or payload.get("base64") or payload.get("b64")

    payload = request.get_json() or {}
    b64 = payload.get("image_base64") or payload.get("base64") or payload.get("b64")
    model_preffered = payload.get("model")

    extension = payload.get("ext", "jpg")  # !!!!

    try:
        image_path = save_image_from_base64(base64_str=b64, ext=extension)
        print(f"Image saved at: {image_path}")
    except Exception as e:
        return jsonify({"error": f"Failed on saving image! {e}"}), 400

    if model_preffered:
        confidence = cnn_analyze_image(image_path, model_name=model_preffered)
    else:
        confidence = cnn_analyze_image(image_path)
    # return jsonify({"label": "Likeness to be Generated", "confidence": confidence, "details": "Someone should write a text classifier for this later"})
    return jsonify({"confidence": confidence, "details": {"model_used": model_preffered if model_preffered else "default_model_set"}})


# ANALYZE TEXT !!
@app.post("/analyze_text")
def analyze_text():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing/Invalid parameter_key: 'text'"}), 400

    try:
        confidence = lstm_analyze_text(text=text)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Text analysis failed: {e}"}), 500

    print(f"Text analyzed with confidence: {confidence}")
    return jsonify({"label": "Likeness to be Generated", "confidence": confidence, "details": "Someone should write a text classifier for this later"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1337, debug=True)
