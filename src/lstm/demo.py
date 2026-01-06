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