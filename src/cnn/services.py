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

class LayerScale(Layer):
    def __init__(self, init_values=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )

    def call(self, x):
        return x * self.gamma

def _predict_image(model, image_path: str, image_size=(224, 224), preprocessing_func=None, preprocess_mode="rgb") -> float:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    preprocess_callable = preprocessing_func or get_preprocess_fn(preprocess_mode)
    img = preprocess_callable(image_path, label=0, image_size=image_size)[0]

    if len(img.shape) == 3:
        img = tf.expand_dims(img, 0)

    predictions = model.predict(img, verbose=0)
    try:
        return float(predictions[0][1])
    except:
        return float(np.max(predictions))

def cnn_analyze_image(image_path, model_name=DEF_MODEL_NAME):
    MODEL_PATH = MODELS_FOLDER / model_name
    with custom_object_scope({
        "LayerScale": LayerScale,
        "SobelMagnitudeLayer": SobelMagnitudeLayer,
        "HaarWaveletLayer": HaarWaveletLayer,
        "aid>SobelMagnitudeLayer": SobelMagnitudeLayer, # Add mapping if necessary
        "aid>HaarWaveletLayer": HaarWaveletLayer,
    }):
        model = load_model(MODEL_PATH, compile=False)
    
    return _predict_image(model=model, image_path=str(image_path))