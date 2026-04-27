from .lstm_utils import get_ai_score
import logging


def ping_text_analysis_side(body_json: dict) -> str:
    return {"message": "send successfull", "status": 200, "body_recieved": body_json}


def analyze_text(text: str) -> float:
    logging.info(f"[LSTM Service] Starting text analysis")
    logging.debug(f"[LSTM Service] Text length: {len(text)} characters")
    logging.debug(f"[LSTM Service] Text preview: {text[:50]}...")

    try:
        prediction = get_ai_score(text)
        logging.info(f"[LSTM Service] Prediction score obtained: {prediction}")

        if prediction == -1.0:
            logging.warning(f"[LSTM Service] Model returned error code (-1.0). Model may not be loaded properly.")
        else:
            confidence_percent = prediction * 100
            label = "AI" if prediction > 0.5 else "HUMAN"
            logging.info(f"[LSTM Service] Classification: {label} (confidence: {confidence_percent:.2f}%)")

        print(f"Input: {text[:30]}... -> Score: {prediction}")
        return prediction
    except Exception as e:
        logging.error(f"[LSTM Service] Exception during analyze_text: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    ai_text = "As an AI language model, I do not have personal opinions or feelings. However, I can provide information based on the data I was trained on."

    result = analyze_text(ai_text)

    print(f"Text: {ai_text[:50]}...")
    print(f"Result: {result}")

    if result > 0.5:  # CLASSIFICATION DEMOO
        print(f"Result: AI (accuracy: %{result*100:.2f})")
    else:
        print(f"Result: HUMAN (accuracy: %{(1-result)*100:.2f})")


""" demo.py:


# Artık model yükleme yok, sadece servisi çağırıyoruz!
from services import analyze_text
import time 

while True:
    try:
        user_input = input("Enter Demo Text: ")
         
        if not user_input.strip():
            continue

        print("Analiz ediliyor...")
        
        score = analyze_text(user_input)

        guven = score * 100 if score > 0.5 else (1 - score) * 100
        label = "AI" if score > 0.5 else "HUMAN"
        

        print(f"SONUÇ: {label}")
        print(f"Güven Skoru: %{guven:.2f} (Ham Skor: {score:.4f})")

    except KeyboardInterrupt:
        break
"""
