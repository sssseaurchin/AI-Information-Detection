import cnnModel
import tensorflow as tf
import os

def cnn_analyze_image(image_name:str):
    path = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(path, "model", "ai_detection_model.h5")
    image_name = path + "\\Model\\" + image_name
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    prediction = cnnModel.predict_image(model, image_name)

    print(f"Prediction for image {image_name}: {prediction}")

if __name__ == "__main__": 
    cnn_analyze_image(r"aaa.jpg")
            