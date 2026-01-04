from flask import Flask, jsonify, request
from lstm.services import analyze_text as lstm_analyze_text

app = Flask(__name__)


@app.get("/analyze_image")
def analyze_image():
    confidence = 0.95  # Örnek confidence score
    return jsonify({"message": "ok"})

@app.post("/analyze_text")
def analyze_text():

    payload = request.get_json() or {}
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing/Invalid parameter_key: 'text'"}), 400

    confidence = lstm_analyze_text(text=text) # lstm.services'ten alınan func

    return jsonify({
        "message": "ok",
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
