from .lstm_utils import get_ai_score
import logging


def ping_text_analysis_side(body_json: dict) -> str:
    return {"message": "send successfull", "status": 200, "body_recieved": body_json}


def analyze_text(text: str) -> float:
    logging.info(f"[LSTM Service] Starting text analysis")

    if not os.path.exists(MODEL_PATH):
        logging.error(f"[LSTM Service] Missing model file: {MODEL_PATH} !!!!!!!!!!!!!!!!!")
        logging.error(f"[LSTM Service] Missing model file: {MODEL_PATH} !!!!!!!!!!!!!!!!!")
        logging.error(f"[LSTM Service] Missing model file: {MODEL_PATH} !!!!!!!!!!!!!!!!!")
        logging.error(f"[LSTM Service] Missing model file: {MODEL_PATH} !!!!!!!!!!!!!!!!!")
        return -1.0

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
