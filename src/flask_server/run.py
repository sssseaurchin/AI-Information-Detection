from utility import save_image_from_base64
from flask import Flask, jsonify, request
from flask_cors import CORS
from lstm.services import analyze_text as lstm_analyze_text
from cnn.services import cnn_analyze_image

app = Flask(__name__)

# Minimal CORS fix
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/")
def index():
    return jsonify({"message": "Welcome to homepage!"})


@app.get("/ping")
def ping():
    return jsonify({"message": "pong"})


@app.post("/analyze_image")
def analyze_image():
    payload = request.get_json(silent=True) or {}

    # Support your current frontend key too
    b64 = payload.get("image") or payload.get("image_base64") or payload.get("base64") or payload.get("b64")

    if not isinstance(b64, str) or not b64.strip():
        return jsonify({"error": "Missing/Invalid parameter_key: 'image'"}), 400

    ext = payload.get("ext", "jpg")

    try:
        image_path = save_image_from_base64(base64_str=b64, ext=ext)
        print(f"Image saved at: {image_path}")
    except Exception as e:
        return jsonify({"error": f"Failed on saving image! {e}"}), 400

    try:
        confidence = cnn_analyze_image(image_path)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Image analysis failed: {e}"}), 500

    return jsonify({"label": "Likeness to be Generated", "confidence": confidence, "details": "Someone should write a text classifier for this later"})


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
    app.run(host="0.0.0.0", port=5000, debug=True)
