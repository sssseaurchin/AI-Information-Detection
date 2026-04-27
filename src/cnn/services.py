from collections.abc import Callable
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from cnn.preprocessing import get_preprocess_fn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging

path = Path(__file__).resolve().parent

DEF_MODEL_NAME = "efficientnet_v2b0_rgb_20260325.h5"
MODELS_FOLDER = path / "models"


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
    try:
        model = load_model(MODEL_PATH, compile=False)  # LOAD MODEL
    except Exception as e:
        logging.error(f"Error loading model: {e} AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        raise
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


def verify_model_existence(model_name=None) -> dict:
    """if we want to change the model used for image detect. on runtime
    mode_name: name of the model with .h5 extension, should be inside model folder
    Return: # dict:{"status":200|404, "errorMsg":}
    """

    pass


def get_model_summary(model_name=DEF_MODEL_NAME) -> dict:  # dict:{"status":200|404, "errorMsg":}
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pred = cnn_analyze_image(image_path=path.parent / "uploaded_images" / "53aee66c20d04fbf9ac83904648b1305.jpg")
    logging.info(f"Predicted confidence: {pred}")
