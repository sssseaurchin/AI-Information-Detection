import requests
import pandas as pd
import time
import os
import re
import xml.etree.ElementTree as ET

OUTPUT_RAW_DIR = "ourDatas/raw_datas/arxiv"
OUTPUT_CLEANED_DIR = "ourDatas/cleaned_datas"
os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLEANED_DIR, exist_ok=True)

ARXIV_API = "https://export.arxiv.org/api/query"

# Çeşitli kategoriler — hepsi 2014 öncesi
# cs.CL = doğal dil işleme, cs.AI = yapay zeka, econ = ekonomi,
# q-bio = biyoloji, physics = fizik, eess = elektrik müh.
CATEGORIES = [
    ("cs.CL", 15),    # Computational Linguistics
    ("cs.AI", 10),    # Artificial Intelligence
    ("econ.GN", 5),   # Economics
    ("q-bio.NC", 5),  # Neuroscience
    ("physics.soc-ph", 5),  # Social Physics
    ("cs.CV", 10),    # Computer Vision
]

MIN_WORDS = 100
MAX_WORDS = 350


def clean_abstract(text):
    """Abstract'ı temizle — sadece düz İngilizce metin kalsın."""
    # Latex komutlarını kaldır
    text = re.sub(r'\$[^\$]+\$', '', text)          # inline math $...$
    text = re.sub(r'\\\w+\{[^}]*\}', '', text)      # \command{...}
    text = re.sub(r'\\\w+', '', text)                # \command
    text = re.sub(r'\{[^}]*\}', '', text)            # {gruplar}
    # Referansları kaldır
    text = re.sub(r'\[\d+\]', '', text)              # [1], [2]
    text = re.sub(r'\(cf\..*?\)', '', text)          # (cf. ...)
    # Fazla boşluk ve newline
    text = re.sub(r'\s+', ' ', text).strip()
    # Çok kısa kelimeler (OCR artığı)
    text = re.sub(r'\b\w{1}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_good_abstract(text):
    """Abstract kalite kontrolü."""
    words = text.split()
    word_count = len(words)

    if word_count < MIN_WORDS or word_count > MAX_WORDS:
        return False

    # Çok fazla sayı/sembol varsa at (formül ağırlıklı)
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    if non_alpha / len(text) > 0.15:  # %15'ten fazla sembol varsa at
        return False

    # Çok kısa kelime oranı yüksekse at (OCR bozukluğu)
    short_words = sum(1 for w in words if len(w) <= 2)
    if short_words / word_count > 0.25:
        return False

    return True


def fetch_arxiv(category, count):
    """ArXiv API'den belirli kategoride 2014 öncesi abstract çek."""
    print(f"\n[{category}] {count} abstract hedefleniyor...")

    results = []
    start = 0
    batch_size = 50  # her seferinde 50 çek, filtreden geçenler hedefi doldursun

    while len(results) < count:
        params = {
            "search_query": f"cat:{category} AND submittedDate:[19900101 TO 20141231]",
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(ARXIV_API, params=params, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"  Request failed: {e}")
            break

        # XML parse
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        if not entries:
            print("  Daha fazla sonuç yok.")
            break

        for entry in entries:
            if len(results) >= count:
                break

            abstract_el = entry.find("atom:summary", ns)
            title_el = entry.find("atom:title", ns)
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)

            if abstract_el is None:
                continue

            raw_abstract = abstract_el.text or ""
            title = title_el.text.strip() if title_el is not None else ""
            arxiv_id = id_el.text.strip() if id_el is not None else ""
            published = published_el.text[:10] if published_el is not None else ""

            # Temizle
            cleaned = clean_abstract(raw_abstract)

            # Kalite kontrolü
            if not is_good_abstract(cleaned):
                continue

            results.append({
                "text": cleaned,
                "isGenerated": 0,
                "source": f"arxiv_{category.replace('.', '_')}",
                "title": title,
                "date": published,
                "arxiv_id": arxiv_id,
                "word_count": len(cleaned.split()),
            })
            print(f"  ✓ [{len(results)}/{count}] {title[:60]}")

        start += batch_size
        time.sleep(3)  # ArXiv rate limit: 3 saniye bekleme zorunlu

    return results


def main():
    print("=== ArXiv Abstract Downloader ===")
    print("Kaynak: ArXiv API (2014 öncesi, temiz abstract)\n")

    all_results = []

    for category, count in CATEGORIES:
        results = fetch_arxiv(category, count)
        all_results.extend(results)
        print(f"  → {len(results)} abstract toplandı.")

    if not all_results:
        print("\nHiç veri çekilemedi.")
        return

    df = pd.DataFrame(all_results)

    print(f"\n{'='*40}")
    print(f"Toplam: {len(df)} abstract")
    print(f"\nKategorilere göre dağılım:")
    print(df["source"].value_counts())
    print(f"\nKelime sayısı istatistikleri:")
    print(df["word_count"].describe())

    # Raw kaydet
    raw_path = os.path.join(OUTPUT_RAW_DIR, "arxiv_raw.csv")
    df.to_csv(raw_path, index=False, escapechar='\\')
    print(f"\nRaw data: {raw_path}")

    # Cleaned kaydet
    cleaned_df = df[["text", "isGenerated"]]
    cleaned_path = os.path.join(OUTPUT_CLEANED_DIR, "cleaned_arxiv.csv")
    cleaned_df.to_csv(cleaned_path, index=False, escapechar='\\')
    print(f"Cleaned data: {cleaned_path}")

    print("\n--- İlk 3 örnek ---")
    for i, row in df.head(3).iterrows():
        print(f"\n[{row['source']}] {row['title'][:60]}")
        print(f"  Tarih: {row['date']} | Kelime: {row['word_count']}")
        print(f"  {row['text'][:300]}...")


if __name__ == "__main__":
    main()