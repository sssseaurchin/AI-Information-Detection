import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ai_detector_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pickle')

print(">>> Model resources loading...")
try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"ERROR: Model yüklenmedi! {e}")
    model, tokenizer = None, None

def get_ai_score(text: str) -> float:
    """0.0 between 1.0."""
    if model is None or tokenizer is None or not text:
        return 0.0
    
    try:
        text_clean = str(text).lower()
        seq = tokenizer.texts_to_sequences([text_clean])
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        prediction = model.predict(pad, verbose=0)
        return float(prediction[0][0])
    except:
        return 0.0