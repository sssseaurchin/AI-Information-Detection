import os
# --- DİKKAT: BU SATIRLAR EN TEPEDE OLMAK ZORUNDA ---
# Başka hiçbir import yapmadan önce bunu ayarlamalıyız.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# ---------------------------------------------------

# ŞİMDİ diğerlerini çağırabiliriz
from cnn import cnnModel
import tensorflow as tf

"""def cnn_analyze_image(image_name:str):
    path = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(path, "model", "ai_detection_model.h5")
    # path'i str() içine alıyoruz
    image_name = str(path) + "\\Model\\" + image_name    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    prediction = cnnModel.predict_image(model, image_name)

    print(f"Prediction for image {image_name}: {prediction}")"""

MODEL_NAME = "model.h5"

def cnn_analyze_image(image_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    
    MODEL_PATH = os.path.join(current_dir, "model", MODEL_NAME) 
    
    final_image_path = os.path.join(current_dir, image_name)

    print(f"DEBUG: Model Yolu: {MODEL_PATH}")
    print(f"DEBUG: Resim Yolu: {final_image_path}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("MODEL YUKLENDI")
        prediction = cnnModel.predict_image(model, final_image_path)
        
        print(f"Prediction result: {prediction}")
        
        return prediction
    except Exception as e:
        print(f"HATA (CNN Analiz): {e}")
        # Hata durumunda -1 veya hata mesajı dönebilirsin
        return -1
    

if __name__ == "__main__": 
    cnn_analyze_image(r"aaa.jpg")
    # prayers
            