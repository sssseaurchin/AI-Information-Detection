"""
arxivDownloader.py — v2
ArXiv API'den temiz akademik abstract'lar çeker.

Geliştirmeler:
  - 200 chunk hedef (50'den artırıldı)
  - Daha fazla kategori: biyoloji, fizik, ekonomi, matematik vb.
  - Gutenberg filtrelerine benzer kalite kontrolleri
  - 1-2 harfli kelime silme kaldırıldı (function words gidiyordu)
  - Cümle uzunluğu, diyalog, büyük harf filtreleri eklendi
  - ourDatas/ path düzeltildi
"""

import requests
import pandas as pd
import time
import os
import re
from collections import Counter
import xml.etree.ElementTree as ET
from pathlib import Path

_DATA_CENTER = Path(__file__).resolve().parent.parent
OUTPUT_RAW_DIR = str(_DATA_CENTER / "ourDatas" / "raw_datas" / "arxiv")
OUTPUT_CLEANED_DIR = str(_DATA_CENTER / "ourDatas" / "cleaned_datas")
os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLEANED_DIR, exist_ok=True)

ARXIV_API = "https://export.arxiv.org/api/query"

# Geniş kategori listesi — çeşitli domain
CATEGORIES = [
    ("cs.CL",        20),   # Computational Linguistics / NLP
    ("cs.AI",        20),   # Artificial Intelligence
    ("cs.CV",        15),   # Computer Vision
    ("cs.LG",        15),   # Machine Learning
    ("econ.GN",      15),   # General Economics
    ("q-bio.NC",     15),   # Neuroscience
    ("physics.soc-ph", 15), # Social Physics
    ("math.HO",      15),   # Mathematics (History/Overview — readable text)
    ("q-bio.PE",     15),   # Populations & Evolution
    ("cs.HC",        15),   # Human-Computer Interaction
]

MIN_WORDS = 100
MAX_WORDS = 350

FUNCTION_WORDS = {
    'that', 'with', 'they', 'have', 'were', 'would', 'will', 'which',
    'what', 'their', 'from', 'this', 'been', 'there', 'very', 'into',
    'could', 'then', 'them', 'him', 'her', 'she', 'his', 'you', 'the',
    'and', 'but', 'for', 'not', 'are', 'was', 'had', 'has', 'its', 'our',
    'more', 'some', 'when', 'than', 'also', 'just', 'like', 'about',
    'over', 'after', 'before', 'being', 'other', 'much', 'well', 'only',
    'even', 'back', 'know', 'here', 'time', 'down', 'still', 'again',
    'never', 'same', 'away', 'make', 'made', 'take', 'took', 'look',
    'because', 'though', 'without', 'through', 'every', 'might', 'must',
    'each', 'where', 'while', 'always', 'already', 'between',
}


def clean_abstract(text):
    """Abstract'ı temizle — LaTeX ve referansları kaldır."""
    # LaTeX komutları
    text = re.sub(r'\$[^\$]+\$', '', text)           # inline math $...$
    text = re.sub(r'\$\$[^\$]+\$\$', '', text)       # display math $$...$$
    text = re.sub(r'\\\w+\{[^}]*\}', '', text)       # \command{...}
    text = re.sub(r'\\\w+', '', text)                 # \command
    text = re.sub(r'\{[^}]*\}', '', text)             # {gruplar}

    # Referanslar
    text = re.sub(r'\[\d+\]', '', text)               # [1], [2]
    text = re.sub(r'\(cf\..*?\)', '', text)           # (cf. ...)
    text = re.sub(r'\bet al\.', 'et al.', text)       # et al. normalize

    # Non-ASCII → normalize
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Noktalama normalizasyonu
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def filter_abstract(text):
    """
    Abstract kalite filtresi. Gutenberg'deki filtrelere paralel.
    True = geçerli, False = at.
    """
    words = text.split()
    word_count = len(words)

    if word_count < MIN_WORDS or word_count > MAX_WORDS:
        return False, "kelime sayisi"

    # Sembol/rakam yoğunluğu — formül ağırlıklı abstract'ları at
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    if non_alpha / len(text) > 0.15:
        return False, "sembol yogunlugu yuksek"

    # Küçük harfle başlama
    if text and text[0].islower():
        return False, "kucuk harfle basliyor"

    # Büyük harf oranı > %20 (akronim ağırlıklı)
    alpha = [c for c in text if c.isalpha()]
    if alpha and sum(c.isupper() for c in alpha) / len(alpha) > 0.20:
        return False, "buyuk harf orani yuksek"

    # Ortalama cümle uzunluğu
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len < 8 or avg_len > 60:
            return False, f"ort cumle uzunlugu: {avg_len:.1f}"

    # Tekrarlayan içerik kelimesi > %5
    content_words = [w.lower().strip('.,!?";:()') for w in words
                     if len(w) > 3 and w.lower() not in FUNCTION_WORDS]
    if content_words:
        counter = Counter(content_words)
        top_word, top_count = counter.most_common(1)[0]
        if top_count / len(content_words) > 0.05:
            return False, f"tekrarlayan kelime: '{top_word}'"

    return True, "ok"


def fetch_arxiv(category, count):
    """ArXiv API'den belirli kategoride 2014 öncesi abstract çek."""
    print(f"\n[{category}] {count} abstract hedefleniyor...")
    results = []
    start = 0
    batch_size = 50

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

            cleaned = clean_abstract(raw_abstract)
            passed, reason = filter_abstract(cleaned)

            if not passed:
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
            print(f"  + [{len(results)}/{count}] {title[:55]}")

        start += batch_size
        time.sleep(3)  # ArXiv rate limit zorunlu

    return results


def validate_quality(df):
    print(f"\n{'='*40}")
    print("KALİTE DOGRULAMA")
    issues = {
        "Kucuk harfle baslayan": sum(1 for t in df['text'] if str(t)[0].islower()),
        "Non-ASCII": sum(1 for t in df['text'] if any(ord(c) > 127 for c in str(t))),
        "Sembol yogunlugu > %15": sum(
            1 for t in df['text']
            if sum(1 for c in str(t) if not c.isalpha() and not c.isspace()) / max(len(str(t)), 1) > 0.15
        ),
    }
    for check, count in issues.items():
        status = "OK" if count == 0 else "SORUN"
        print(f"  [{status}] {check}: {count}")


def main():
    print("=== ArXiv Abstract Downloader v2 ===")
    print(f"Hedef: {sum(c for _, c in CATEGORIES)} abstract | {len(CATEGORIES)} kategori\n")

    all_results = []

    for category, count in CATEGORIES:
        results = fetch_arxiv(category, count)
        all_results.extend(results)
        print(f"  → {len(results)} abstract toplandi.")

    if not all_results:
        print("\nHic veri cekilemedi.")
        return

    df = pd.DataFrame(all_results)

    print(f"\n{'='*40}")
    print(f"Toplam: {len(df)} abstract")
    print(f"\nKategorilere gore dagilim:")
    print(df["source"].value_counts().to_string())
    print(f"\nKelime sayisi istatistikleri:")
    print(df["word_count"].describe())

    validate_quality(df)

    raw_path = os.path.join(OUTPUT_RAW_DIR, "arxiv_raw.csv")
    df.to_csv(raw_path, index=False, escapechar='\\')
    print(f"\nRaw data: {raw_path}")

    cleaned_df = df[["text", "isGenerated"]]
    cleaned_path = os.path.join(OUTPUT_CLEANED_DIR, "cleaned_arxiv.csv")
    cleaned_df.to_csv(cleaned_path, index=False, escapechar='\\')
    print(f"Cleaned data: {cleaned_path}")

    print("\n--- Ilk 3 ornek ---")
    for i, row in df.head(3).iterrows():
        print(f"\n[{row['source']}] {row['title'][:55]}")
        print(f"  {row['date']} | {row['word_count']} kelime")
        print(f"  {row['text'][:280]}...")


if __name__ == "__main__":
    main()