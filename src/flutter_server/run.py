from .utility import save_image_from_base64
from flask import Flask, jsonify, request
from lstm.services import analyze_text as lstm_analyze_text
from cnn.services import cnn_analyze_image


app = Flask(__name__)

@app.get("/")
def index():
    return jsonify({"message": "Welcome to homepage!"})

@app.get("/ping")
def ping():
    return jsonify({"message": "pong"})




@app.post("/analyze_image")
def analyze_image():

    payload = request.get_json() or {}
    b64 = payload.get("image_base64") or payload.get("base64") or payload.get("b64")

    ext = payload.get("ext", "jpg") # !!!!

    try:
        image_path = save_image_from_base64(base64_str=b64, ext=ext)
    except Exception as e:
        return jsonify({"error": f"Failed on saving image! {e}"}), 400
    
    confidence = cnn_analyze_image(image_path=image_path)

    return jsonify({
        "label": "Likeness to be Generated",
        "confidence": confidence,
        "details": "Someone should write a text classifier for this later"
    })




@app.post("/analyze_text")
def analyze_text():

    payload = request.get_json() or {}
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing/Invalid parameter_key: 'text'"}), 400

    try:
        confidence = lstm_analyze_text(text=text) # lstm.services'ten alınan func
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({
    "label": "Likeness to be Generated",
    "confidence": confidence,
    "details": "Someone should write a text classifier for this later"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
