import os
import pickle
import logging

# TensorFlow 2.10 typically uses tf.keras (standalone keras may mismatch)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(BASE_DIR, "model", "bilstm_att_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")

model = None
tokenizer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.info("Trying to load LSTM model and tokenizer...")
try:
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Missing model file: {MODEL_PATH} !!!!!!!!!!!!!!!!!")
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not os.path.exists(TOKENIZER_PATH):
        logging.error(f"Missing tokenizer file: {TOKENIZER_PATH} !!!!!!!!!!!!!!!!!")
        raise FileNotFoundError(f"Missing tokenizer file: {TOKENIZER_PATH}")

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as handle:
        logging.info("Model and tokenizer loaded successfully.")
        tokenizer = pickle.load(handle)

except Exception as e:
    print(f"ERROR: Model/tokenizer not loaded: {e}")


def get_ai_score(text: str) -> float:
    """Returns a score between 0.0 and 1.0. Returns -1.0 on error."""
    logger.info("[LSTM Utils] get_ai_score called")

    if model is None or tokenizer is None:
        logger.error("[LSTM Utils] Model or tokenizer is None. Cannot proceed with prediction.")
        logger.debug(f"[LSTM Utils] Model loaded: {model is not None}, Tokenizer loaded: {tokenizer is not None}")
        return -1.0

    if not isinstance(text, str) or not text.strip():
        logger.warning("[LSTM Utils] Invalid input: text is not a non-empty string")
        return -1.0

    try:
        logger.debug(f"[LSTM Utils] Input text length: {len(text)} characters")

        logger.debug("[LSTM Utils] Converting text to lowercase")
        text_clean = text.lower()

        logger.debug("[LSTM Utils] Tokenizing text")
        seq = tokenizer.texts_to_sequences([text_clean])
        logger.debug(f"[LSTM Utils] Tokenized sequence length: {len(seq[0]) if seq and seq[0] else 0}")

        logger.debug(f"[LSTM Utils] Padding sequences to max length: {MAX_SEQUENCE_LENGTH}")
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        logger.debug(f"[LSTM Utils] Padded shape: {pad.shape}")

        logger.info("[LSTM Utils] Running model prediction")
        pred = model.predict(pad, verbose=0)
        logger.debug(f"[LSTM Utils] Raw prediction output shape: {pred.shape}")

        # pred shape typically (1, 1) for binary sigmoid
        score = float(pred[0][0])
        logger.info(f"[LSTM Utils] Prediction score: {score}")
        return score
    except Exception as e:
        logger.error(f"[LSTM Utils] Exception during prediction: {e}", exc_info=True)
        print(f"ERROR during prediction: {e}")
        return -1.0
