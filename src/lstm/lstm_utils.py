import os
import pickle

# TensorFlow 2.10 typically uses tf.keras (standalone keras may mismatch)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(BASE_DIR, "model", "lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")

model = None
tokenizer = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Missing tokenizer file: {TOKENIZER_PATH}")

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)

except Exception as e:
    print(f"ERROR: Model/tokenizer not loaded: {e}")


def get_ai_score(text: str) -> float:
    """Returns a score between 0.0 and 1.0. Returns -1.0 on error."""
    if model is None or tokenizer is None:
        print("ERROR: Model or tokenizer not loaded.")
        return -1.0
    if not isinstance(text, str) or not text.strip():
        print("ERROR: Model or tokenizer not loaded.")
        return -1.0

    try:
        text_clean = text.lower()
        seq = tokenizer.texts_to_sequences([text_clean])
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(pad, verbose=0)

        # pred shape typically (1, 1) for binary sigmoid
        return float(pred[0][0])
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return -1.0


