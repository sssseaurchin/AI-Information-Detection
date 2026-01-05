import cnnModel
import tensorflow as tf
import os

def cnn_analyze_image(image_path:str):
    path = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(path, "model", "ai_detection_model.h5")
    model = tf.keras.models.load_model(MODEL_PATH)
    prediction = cnnModel.predict_image(model, image_path)

    print(f"Prediction for image {image_path}: {prediction}")

if __name__ == "__main__": 
    cnn_analyze_image(r"\model\kirkified_ruzgar.jpeg")
            