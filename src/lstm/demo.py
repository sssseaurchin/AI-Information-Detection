# Artık model yükleme yok, sadece servisi çağırıyoruz!
from services import analyze_text
import time

print("\n" + "="*50)
print("   AI vs HUMAN - ETKİLEŞİMLİ DEMO   ")
print("="*50)
print("Çıkış yapmak için 'q' yazıp Enter'a basın.\n")

while True:
    try:
        user_input = input("Metni Girin: ")
        
        if user_input.lower() == 'q':
            print("Güle güle! 👋")
            break
            
        if not user_input.strip():
            continue

        print("Analiz ediliyor...")
        time.sleep(0.5) # Biraz heyecan katalım :)
        
        # --- BÜTÜN SİHİR BURADA ---
        skor = analyze_text(user_input)
        # --------------------------

        guven = skor * 100 if skor > 0.5 else (1 - skor) * 100
        etiket = "🤖 YAPAY ZEKA (AI)" if skor > 0.5 else "👤 İNSAN (HUMAN)"
        
        print("-" * 40)
        print(f"SONUÇ: {etiket}")
        print(f"Güven Skoru: %{guven:.2f} (Ham Skor: {skor:.4f})")
        print("-" * 40 + "\n")

    except KeyboardInterrupt:
        print("\nProgram kapatıldı.")
        break