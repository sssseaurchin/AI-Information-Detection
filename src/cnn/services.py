from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging

# Support both module and direct execution
if __package__:
    from .cnnModel import predict_image
else:
    from cnnModel import predict_image

path = Path(__file__).resolve().parent

DEF_MODEL_NAME = "efficientnet_v2b0_rgb_20260325.h5"
MODELS_FOLDER = path / "models"


def cnn_analyze_image(image_path, model_name=DEF_MODEL_NAME):
    MODEL_PATH = MODELS_FOLDER / model_name

    model = load_model(MODEL_PATH, compile=False)  # LOAD MODEL
    logging.info(f"stringified image_path: {str(image_path)}")
    score = predict_image(
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
