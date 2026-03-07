from .lstm_utils import get_ai_score

def analyze_text(text: str) -> float:
    prediction = get_ai_score(text)
    
    print(f"Input: {text[:30]}... -> Score: {prediction}")
    return prediction






if __name__ == "__main__": 
    ai_text = "As an AI language model, I do not have personal opinions or feelings. However, I can provide information based on the data I was trained on."
    
    result = analyze_text(ai_text)
    
    print(f"Text: {ai_text[:50]}...")
    print(f"Result: {result}")
    
    if result > 0.5: # CLASSIFICATION DEMOO
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
