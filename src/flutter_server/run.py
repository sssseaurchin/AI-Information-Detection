from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/CNN")
def health():
    confidence = 0.95  # Örnek confidence score
    return jsonify({"status": "ok"})

@app.get("/LSTM")
def greet_endpoint():
    confidence = 0.95  # Örnek confidence score
    return jsonify({"message": message})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
