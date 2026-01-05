import cnnModel

def cnn_analyze_image(image_path:str):
    MODEL_PATH = r"\model\ai_detection_model.h5"


    model = cnnModel.load_model(MODEL_PATH)
    prediction = cnnModel.predict_image(model, image_path)

    print(f"Prediction for image {image_path}: {prediction}")

if __name__ == "__main__": 
    cnn_analyze_image(r"\model\kirkified_ruzgar.jpeg")
            