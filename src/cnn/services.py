from collections.abc import Callable
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from cnn.preprocessing import get_preprocess_fn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.utils import custom_object_scope
from keras.layers import Layer
import logging
from cnn.cnnModel import SobelMagnitudeLayer, HaarWaveletLayer  # new stuff

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
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )

    def call(self, x):
        return x * self.gamma


def _predict_image(model: tf.keras.Model, image_path: str, image_size: tuple = (224, 224), preprocessing_func: Callable | None = None, preprocess_mode: str = "rgb") -> float:
    # Predict if an image is AI-generated or real using TensorFlow ops (GPU-accelerated) - returns confidence score 0.0 to 1.0
    # Check if file exists
    logging.info(f"Predicting image: {image_path} with model: {model.name}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    logging.info(f"Image file found: {image_path}")
    preprocess_callable = preprocessing_func or get_preprocess_fn(preprocess_mode)
    img = preprocess_callable(image_path, label=0, image_size=image_size)[0]  # Get preprocessed image tensor

    # Ensure we have a batch dimension: model expects (batch, h, w, c)
    if len(img.shape) == 3:
        img = tf.expand_dims(img, 0)

    # Convert to numpy if possible (eager mode), otherwise pass the tensor
    try:
        img_input = img.numpy()
    except Exception:
        img_input = img

    # Make prediction (use integer verbose)
    logging.info(f"Running model prediction on image: {image_path}")
    predictions = model.predict(img_input, verbose=0)
    logging.info(f"Raw model predictions: {predictions}")
    # Extract confidence for AI-generated class (assumes 2-class softmax)
    try:
        confidence = float(predictions[0][1])
    except Exception:
        # Fallback: if predictions shape unexpected, try a safe conversion
        preds = np.asarray(predictions)
        if preds.ndim == 1 and preds.size >= 2:
            confidence = float(preds[1])
        else:
            # As a last resort, return the max class probability
            confidence = float(np.max(preds))

    return confidence


def cnn_analyze_image(image_path, model_name=DEF_MODEL_NAME):
    MODEL_PATH = MODELS_FOLDER / model_name
    logging.info(f"Loading CNN model froms: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        logging.error(f"\n!!!!!!!! CNN Model file not found: {MODEL_PATH} !!!!!!!!!!!!!!!!!")
        logging.error(f"!!!!!!!! CNN Model file not found: {MODEL_PATH} !!!!!!!!!!!!!!!!!")
        logging.error(f"!!!!!!!! CNN Model file not found: {MODEL_PATH} !!!!!!!!!!!!!!!!!\n")

        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        with custom_object_scope(
            {
                "LayerScale": LayerScale,
                "SobelMagnitudeLayer": SobelMagnitudeLayer,
                "aid>SobelMagnitudeLayer": HaarWaveletLayer,
                "HaarWaveletLayer": HaarWaveletLayer,
                "aid>HaarWaveletLayer": HaarWaveletLayer,
            }
        ):
            model = load_model(MODEL_PATH, compile=False)

    except Exception as e:
        # logging.exception(f"Error loading CNN model: {e}")
        raise e

    logging.info(f"stringified image_path: {str(image_path)}")
    logging.info(f"Model {model_name} loaded successfully. Starting analysis...")
    score = _predict_image(
        model=model,
        image_path=str(image_path),
        image_size=(224, 224),
        preprocessing_func=None,
        preprocess_mode="rgb",
    )

    return score


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pred = cnn_analyze_image(image_path=path.parent / "uploaded_images" / "53aee66c20d04fbf9ac83904648b1305.jpg")
    logging.info(f"Predicted confidence: {pred}")
