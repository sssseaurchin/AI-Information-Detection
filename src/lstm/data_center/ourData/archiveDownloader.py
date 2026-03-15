import requests
import pandas as pd
import time
import os
import re

OUTPUT_RAW_DIR = "ourDatas/raw_datas/archive"
OUTPUT_CLEANED_DIR = "ourDatas/cleaned_datas"
os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLEANED_DIR, exist_ok=True)

SEARCH_URL = "https://archive.org/advancedsearch.php"

# Her kategoriden kaç tane çekilsin
TARGETS = {
    "academic": 30,
    "news":     25,
    "magazine": 25,
    "books":    20,
}

# Archive.org koleksiyon + medya tipi sorguları
QUERIES = {
    "academic": {
        "q": 'mediatype:texts subject:"science" date:[1990-01-01 TO 2014-12-31] language:English',
        "label": "academic"
    },
    "news": {
        "q": 'mediatype:texts subject:"news" date:[1990-01-01 TO 2014-12-31] language:English',
        "label": "news"
    },
    "magazine": {
        "q": 'mediatype:texts subject:"magazine" date:[1990-01-01 TO 2014-12-31] language:English',
        "label": "magazine"
    },
    "books": {
        "q": 'mediatype:texts subject:"history OR literature OR politics" date:[1990-01-01 TO 2014-12-31] language:English',
        "label": "books"
    },
}

MIN_WORDS = 100
MAX_WORDS = 600
CHUNK_SIZE = 400  # hedef chunk kelime sayısı


def clean_text(text):
    """Temel temizlik: fazla boşluk, özel karakter, vs."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # non-ASCII kaldır
    text = re.sub(r'\b\w{1,2}\b', '', text)      # 1-2 harfli kelimeleri kaldır (gürültü)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Metni chunk_size kelimelik parçalara böl."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if MIN_WORDS <= len(chunk.split()) <= MAX_WORDS:
            chunks.append(chunk)
    return chunks


def fetch_item_text(identifier):
    """Bir Archive.org item'ının ilk text dosyasını indir."""
    try:
        meta_url = f"https://archive.org/metadata/{identifier}"
        meta = requests.get(meta_url, timeout=10).json()
        files = meta.get("files", [])

        # .txt dosyasını bul
        txt_files = [f for f in files if f.get("name", "").endswith(".txt")]
        if not txt_files:
            # _djvu.txt veya diğer text formatları
            txt_files = [f for f in files if "_djvu.txt" in f.get("name", "")]
        if not txt_files:
            return None

        filename = txt_files[0]["name"]
        text_url = f"https://archive.org/download/{identifier}/{filename}"
        response = requests.get(text_url, timeout=15)
        if response.status_code == 200:
            return response.text[:5000]  # ilk 5000 karakter yeterli
        return None
    except Exception:
        return None


def search_and_fetch(category, query, target):
    """Archive.org'dan arama yap ve text çek."""
    print(f"\n[{category.upper()}] {target} adet hedefleniyor...")

    params = {
        "q": query,
        "fl[]": ["identifier", "title", "date", "subject"],
        "rows": target * 3,  # fazla al, filtreleyeceğiz
        "page": 1,
        "output": "json",
        "sort[]": "date desc",
    }

    try:
        response = requests.get(SEARCH_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  Search failed: {e}")
        return []

    docs = data.get("response", {}).get("docs", [])
    print(f"  {len(docs)} item bulundu, text çekiliyor...")

    results = []
    for doc in docs:
        if len(results) >= target:
            break

        identifier = doc.get("identifier", "")
        title = doc.get("title", "")
        date = doc.get("date", "")

        text = fetch_item_text(identifier)
        if not text:
            continue

        text = clean_text(text)
        chunks = chunk_text(text)

        if not chunks:
            continue

        # İlk geçerli chunk'ı al
        chunk = chunks[0]
        results.append({
            "text": chunk,
            "isGenerated": 0,
            "source": f"archive_{category}",
            "title": title,
            "date": date,
            "identifier": identifier,
            "word_count": len(chunk.split()),
        })
        print(f"  ✓ [{len(results)}/{target}] {title[:60]}")
        time.sleep(0.5)  # nazik ol

    return results


def main():
    print("=== Archive.org Downloader ===")
    print(f"Hedef: ~100 chunk (akademik, haber, dergi, kitap)\n")

    all_results = []

    for category, info in QUERIES.items():
        target = TARGETS[category]
        results = search_and_fetch(category, info["q"], target)
        all_results.extend(results)
        print(f"  → {len(results)} chunk toplandı.")
        time.sleep(1)

    if not all_results:
        print("\nHiç veri çekilemedi.")
        return

    df = pd.DataFrame(all_results)
    print(f"\n{'='*40}")
    print(f"Toplam: {len(df)} chunk")
    print(f"\nKategorilere göre dağılım:")
    print(df["source"].value_counts())
    print(f"\nKelime sayısı istatistikleri:")
    print(df["word_count"].describe())

    # Raw kaydet
    raw_path = os.path.join(OUTPUT_RAW_DIR, "archive_raw_100.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw data: {raw_path}")

    # Cleaned kaydet (pipeline formatı)
    cleaned_df = df[["text", "isGenerated"]]
    cleaned_path = os.path.join(OUTPUT_CLEANED_DIR, "cleaned_archive_100.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data: {cleaned_path}")

    print("\n--- İlk 3 örnek ---")
    for i, row in df.head(3).iterrows():
        print(f"\n[{row['source']}] {row['title'][:50]} ({row['date'][:4]})")
        print(f"  {row['text'][:200]}...")


if __name__ == "__main__":
    main()
