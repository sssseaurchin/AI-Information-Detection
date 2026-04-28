import os
import pickle
import logging
import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import custom_object_scope
from keras import backend as K

# --- Custom Attention Layer ---
# This class provides the blueprint Keras needs to load your .h5 file.
class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Alignment score calculation
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

# --- Setup Paths ---
MAX_SEQUENCE_LENGTH = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "bilstm_att_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")

model = None
tokenizer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Loading Logic ---
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        # We wrap the load_model call in a scope so Keras knows 
        # to map the string 'AttentionLayer' to our class above.
        with custom_object_scope({'AttentionLayer': AttentionLayer}):
            model = load_model(MODEL_PATH)
            
        with open(TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
        logger.info("LSTM Model and tokenizer loaded successfully.")
    else:
        logger.error(f"Files not found: Model={os.path.exists(MODEL_PATH)}, Tokenizer={os.path.exists(TOKENIZER_PATH)}")
except Exception as e:
    logger.error(f"Failed to load LSTM components: {e}")

def get_ai_score(text: str) -> float:
    """Returns a score between 0.0 and 1.0. Returns -1.0 on error."""
    if model is None or tokenizer is None:
        return -1.0
    try:
        # Preprocessing matching the training pipeline
        seq = tokenizer.texts_to_sequences([text.lower()])
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        
        # Inference
        pred = model.predict(pad, verbose=0)
        
        # Extract scalar value from the prediction tensor
        return float(pred[0][0])
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return -1.0