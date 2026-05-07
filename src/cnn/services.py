from collections.abc import Callable
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras  # Import directly for Keras 2 stability
from keras.models import load_model
from keras.utils import custom_object_scope
from keras.layers import Layer
import logging
from cnn.preprocessing import get_preprocess_fn
from cnn.cnnModel import SobelMagnitudeLayer, HaarWaveletLayer

path = Path(__file__).resolve().parent
DEF_MODEL_NAME = "model.h5"
MODELS_FOLDER = path / "models"

# --- GLOBAL MODEL CACHE ---
# This dictionary stays in RAM as long as the server is running.
_MODEL_CACHE = {}

class LayerScale(Layer):
    """
    Updated LayerScale to handle 'projection_dim' and extra kwargs 
    to avoid deserialization errors.
    """
    def __init__(self, init_values=1e-5, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )

    def call(self, x):
        return x * self.gamma
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim,
        })
        return config

def get_cnn_model(model_name):
    """
    Ensures the model is loaded exactly once into the cache.
    """
    global _MODEL_CACHE
    
    if model_name not in _MODEL_CACHE:
        MODEL_PATH = MODELS_FOLDER / model_name
        print(f"--- [INITIALIZING] Loading CNN: {MODEL_PATH} ---")
        
        with custom_object_scope({
            "LayerScale": LayerScale,
            "SobelMagnitudeLayer": SobelMagnitudeLayer,
            "HaarWaveletLayer": HaarWaveletLayer,
            "aid>SobelMagnitudeLayer": SobelMagnitudeLayer,
            "aid>HaarWaveletLayer": HaarWaveletLayer,
        }):
            model = load_model(MODEL_PATH, compile=False)
            
            # WARM-UP: Run a dummy prediction so the first user 
            # request doesn't suffer from 'cold start' latency.
            dummy_input = tf.zeros((1, 224, 224, 3))
            model.predict(dummy_input, verbose=0)
            
            _MODEL_CACHE[model_name] = model
            print(f"--- [READY] {model_name} is cached and warmed up ---")
            
    return _MODEL_CACHE[model_name]

def _predict_image(model, image_path: str, image_size=(224, 224), preprocessing_func=None, preprocess_mode="rgb") -> float:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    preprocess_callable = preprocessing_func or get_preprocess_fn(preprocess_mode)
    img = preprocess_callable(image_path, label=0, image_size=image_size)[0]

    if len(img.shape) == 3:
        img = tf.expand_dims(img, 0)

    # Use the pre-loaded model here
    predictions = model.predict(img, verbose=0)
    try:
        return float(predictions[0][1])
    except:
        return float(np.max(predictions))

def cnn_analyze_image(image_path, model_name=DEF_MODEL_NAME):
    # Fetch the model from cache (or load it if this is the first call)
    model = get_cnn_model(model_name)
    return _predict_image(model=model, image_path=str(image_path))