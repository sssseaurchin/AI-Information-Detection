"""
generateAIData.py
Human chunk'ları Claude API'ye gönderir, AI tarzında yeniden yazar.

Kullanım:
  1. ANTHROPIC_API_KEY ortam değişkenini set et
     Windows: set ANTHROPIC_API_KEY=sk-ant-...
  2. python generateAIData.py --test     (50 chunk test)
     python generateAIData.py            (tümü)

Maliyet tahmini (claude-haiku-3 ile):
  - 2597 chunk × ~600 token/istek = ~1.56M token
  - Input: $0.25/1M, Output: $1.25/1M → toplam ~$2.5
"""

import os
import sys
import time
import json
import random
import argparse
import requests
import pandas as pd
from pathlib import Path

# ─── AYARLAR ─────────────────────────────────────────────────────────────────

INPUT_PATH  = "ourDatas/cleaned_datas/cleaned_human_all.csv"
OUTPUT_DIR  = "ourDatas/raw_datas"
TEST_OUTPUT = "ourDatas/raw_datas/ai_text/ai_test.csv"
FULL_OUTPUT = "ourDatas/raw_datas/ai_text/ai_all.csv"
PROGRESS_FILE = "ourDatas/raw_datas/ai_text/ai_progress.json"  # checkpoint

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"   # En ucuz, test için ideal

SYSTEM_PROMPT = "You are a text rewriter. Output ONLY the rewritten text, no explanations, no quotes, no labels."

# Farklı prompt varyasyonları — her chunk için rastgele seçilir
PROMPTS = [
    "Rewrite this text in a highly academic and formal tone, using transitional phrases and complex vocabulary. Keep the response between 350-450 words.",
    "Rewrite this text in a casual, conversational, simple tone as if explaining to a friend. Keep the response between 350-450 words.",
    "Paraphrase this text carefully, preserving all details but using completely different wording. Keep the response between 350-450 words.",
    "Summarize and expand this text into a cohesive essay paragraph with clear structure. Keep the response between 350-450 words.",
    "Rewrite this text as a college student would, in your own words, naturally and straightforwardly. Keep the response between 350-450 words.",
]

RETRY_LIMIT = 3
RETRY_DELAY = 5   # saniye
REQUEST_DELAY = 0.5  # istek arası bekleme

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── API ÇAĞRISI ──────────────────────────────────────────────────────────────

def call_claude(human_text: str, api_key: str) -> str | None:
    """Bir human chunk'ı Claude'a gönder, AI versiyonunu al."""
    # Her çağrıda rastgele prompt ve temperature seç
    prompt = random.choice(PROMPTS)
    temperature = round(random.uniform(0.3, 0.9), 2)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "temperature": temperature,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}\n\nText:\n{human_text}"
            }
        ]
    }

    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=30)

            if r.status_code == 200:
                data = r.json()
                return data["content"][0]["text"].strip(), prompt, temperature

            elif r.status_code == 429:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    Rate limit, {wait}s bekleniyor...")
                time.sleep(wait)

            elif r.status_code == 401:
                print("    HATA: Geçersiz API key!")
                return None

            else:
                print(f"    HTTP {r.status_code}: {r.text[:100]}")
                time.sleep(RETRY_DELAY)

        except requests.Timeout:
            print(f"    Timeout (deneme {attempt+1}/{RETRY_LIMIT})")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"    Hata: {e}")
            time.sleep(RETRY_DELAY)

    return None, None, None


# ─── PROGRESS (CHECKPOINT) ───────────────────────────────────────────────────

def load_progress() -> set:
    """Daha önce işlenmiş index'leri yükle."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(done_indices: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(done_indices), f)


# ─── ANA FONKSİYON ───────────────────────────────────────────────────────────

def generate(test_mode: bool = False):
    # API key kontrolü
    api_key = "YOUR_API_KEY_HERE"
    if not api_key:
        print("HATA: ANTHROPIC_API_KEY ortam değişkeni set edilmemiş!")
        print("Windows: set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Input yükle
    df = pd.read_csv(INPUT_PATH)
    print(f"Human data yüklendi: {len(df)} chunk")

    if test_mode:
        df = df.head(50)
        output_path = TEST_OUTPUT
        print(f"TEST MODU: İlk 50 chunk işlenecek")
    else:
        output_path = FULL_OUTPUT
        print(f"TAM MOD: {len(df)} chunk işlenecek")

    # Progress kontrolü
    done = load_progress()
    print(f"Daha önce tamamlanan: {len(done)} chunk")

    # Sonuçları yükle (varsa)
    results = []
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        results = existing.to_dict("records")
        print(f"Mevcut output: {len(results)} kayıt")

    # Maliyet tahmini
    remaining = len(df) - len(done)
    est_cost = remaining * 600 * 0.00000025 + remaining * 600 * 0.00000125
    print(f"\nTahmini maliyet ({MODEL}): ${est_cost:.3f}")
    print(f"İşlenecek chunk: {remaining}")

    if not test_mode and remaining > 100:
        confirm = input("\nDevam etmek için 'evet' yaz: ")
        if confirm.lower() != "evet":
            print("İptal edildi.")
            return

    print(f"\n{'='*40}")
    print("Üretim başlıyor...\n")

    errors = 0
    for i, row in df.iterrows():
        if i in done:
            continue

        human_text = str(row["text"])
        original_source = str(row.get("source", "unknown"))

        ai_text, used_prompt, used_temp = call_claude(human_text, api_key)

        if ai_text is None:
            errors += 1
            print(f"  [{i}] HATA (toplam hata: {errors})")
            if errors > 10:
                print("Çok fazla hata, durduruluyor.")
                break
            continue

        results.append({
            "text": ai_text,
            "isGenerated": 1,
            "source": f"{original_source}_ai",
            "prompt_used": used_prompt,
            "temperature": used_temp,
        })

        done.add(i)

        # Her 10 chunk'ta bir kaydet
        if len(done) % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False, escapechar="\\")
            save_progress(done)
            print(f"  [{i}] Kaydedildi ({len(results)}/{len(df)} chunk)")
        else:
            print(f"  [{i}] OK ({len(ai_text.split())} kelime)")

        time.sleep(REQUEST_DELAY)

    # Final kayıt
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_path, index=False, escapechar="\\")
        save_progress(done)

    print(f"\n{'='*40}")
    print(f"Tamamlandı: {len(results)} AI chunk üretildi")
    print(f"Hatalar: {errors}")
    print(f"Kaydedildi: {output_path}")

    # Özet istatistik
    if results:
        wc = [len(r["text"].split()) for r in results]
        print(f"Ort. kelime/chunk: {sum(wc)/len(wc):.0f}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Sadece 50 chunk test et")
    args = parser.parse_args()

    generate(test_mode=args.test)