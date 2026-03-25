"""
archiveDownloader.py — v2
Archive.org'dan temiz human text çeker.

Geliştirmeler:
  - _djvu.txt (OCR) dosyaları tamamen reddedildi
  - Sadece .txt uzantılı ve makul boyutlu dosyalar alınıyor
  - Gutenberg'deki tüm filtreler eklendi
  - Cümle sınırlarına göre chunklama (kelime bazlı değil)
  - Dosyanın ortasından chunk alınıyor (başlık/kapanış kısmı atlanıyor)
  - Tek chunk değil, birden fazla chunk alınıyor
  - Non-ASCII normalizasyon eklendi
  - escapechar CSV fix
"""

import requests
import pandas as pd
import time
import os
import re
from collections import Counter
from pathlib import Path

_DATA_CENTER = Path(__file__).resolve().parent.parent
OUTPUT_RAW_DIR = str(_DATA_CENTER / "ourDatas" / "raw_datas" / "archive")
OUTPUT_CLEANED_DIR = str(_DATA_CENTER / "ourDatas" / "cleaned_datas")
os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLEANED_DIR, exist_ok=True)

SEARCH_URL = "https://archive.org/advancedsearch.php"

CHUNK_SIZE = 400
MIN_WORDS = 200
MAX_WORDS = 500
MAX_CHUNKS_PER_ITEM = 3   # Her kayıttan max chunk

TARGETS = {
    "news":      60,
    "magazine":  60,
}

QUERIES = {
    "news": {
        "q": 'mediatype:texts subject:"newspaper" date:[1990-01-01 TO 2010-12-31] language:English',
    },
    "magazine": {
        "q": 'mediatype:texts subject:"magazine" date:[1990-01-01 TO 2010-12-31] language:English',
    },
}

NON_ASCII_MAP = {
    '\u2014': '-', '\u2013': '-',
    '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"',
    '\u00e6': 'ae', '\u0153': 'oe',
    '\u00e9': 'e', '\u00e8': 'e',
    '\u00e0': 'a', '\u00e7': 'c',
}

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

SENTENCE_START_STOPWORDS = {
    'the', 'a', 'an', 'in', 'it', 'he', 'she', 'this', 'that',
    'they', 'we', 'i', 'its', 'his', 'her', 'their', 'our',
    'at', 'on', 'by', 'as', 'of', 'to', 'for', 'and', 'but',
}


def normalize_text(text):
    for char, replacement in NON_ASCII_MAP.items():
        text = text.replace(char, replacement)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'-{3,}', '-', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_archive_text(text):
    """Archive.org text'ini temizle."""
    # HTML tag kalıntıları
    text = re.sub(r'<[^>]+>', '', text)

    # Sayfa numaraları / başlıklar (tamamen büyük harf satırlar)
    lines = text.split('\n')
    good_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isupper() and len(line) < 60:
            continue  # Başlık/bölüm numarası
        if len(line.split()) < 4:
            continue  # Çok kısa satır
        good_lines.append(line)

    text = ' '.join(good_lines)
    return normalize_text(text)


def split_sentences(text):
    """Basit cümle bölücü."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [p.strip() for p in parts if p.strip()]


def filter_chunk(chunk):
    words = chunk.split()
    word_count = len(words)

    if word_count < MIN_WORDS:
        return False, "cok kisa"
    if not chunk or chunk[0].islower():
        return False, "kucuk harfle basliyor"
    if re.search(r'https?://', chunk):
        return False, "url iceriyor"

    # Rakam yoğunluğu
    if sum(c.isdigit() for c in chunk) / len(chunk) > 0.08:
        return False, "rakam yogunlugu yuksek"

    # Büyük harf oranı
    alpha = [c for c in chunk if c.isalpha()]
    if alpha and sum(c.isupper() for c in alpha) / len(alpha) > 0.15:
        return False, "buyuk harf orani yuksek"

    # Diyalog oranı
    quoted_words = sum(len(q.split()) for q in re.findall(r'"([^"]*)"', chunk))
    if word_count > 0 and quoted_words / word_count > 0.30:
        return False, "diyalog orani yuksek"

    sentences = [s for s in split_sentences(chunk) if s.strip()]

    # Ortalama cümle uzunluğu
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len < 8 or avg_len > 45:
            return False, f"ort cumle uzunlugu: {avg_len:.1f}"

    # Kısa cümle oranı
    if sentences:
        short = [s for s in sentences if len(s.split()) <= 3]
        if len(short) / len(sentences) > 0.30:
            return False, "kisa cumle orani yuksek"

    # 100+ kelime tek cümle
    if sentences and any(len(s.split()) > 100 for s in sentences):
        return False, "100+ kelime tek cumle"

    # Tekrarlayan cümle başlangıcı
    if sentences and len(sentences) >= 4:
        starts = [s.split()[0].lower() for s in sentences if s.split()]
        valid_starts = [s for s in starts if s not in SENTENCE_START_STOPWORDS]
        if valid_starts:
            top_start, count = Counter(valid_starts).most_common(1)[0]
            if count / len(sentences) > 0.40:
                return False, f"tekrarlayan cumle baslangici: '{top_start}'"

    # Tekrarlayan kelime
    content_words = [w.lower().strip('.,!?";:()') for w in words
                     if len(w) > 3 and w.lower().strip('.,!?";:()') not in FUNCTION_WORDS]
    if content_words:
        counter = Counter(content_words)
        top_word, top_count = counter.most_common(1)[0]
        if top_count / len(content_words) > 0.04:
            return False, f"tekrarlayan kelime: '{top_word}'"

    return True, "ok"


def make_chunks(text, meta, source):
    """Cümle sınırlarında chunk'la, filtreleri uygula."""
    sentences = split_sentences(text)

    # Baştan ve sondan %10 atla
    skip = max(10, len(sentences) // 10)
    if len(sentences) > skip * 2:
        sentences = sentences[skip:-skip]

    raw_chunks = []
    current_words = []

    for sentence in sentences:
        current_words.extend(sentence.split())
        if len(current_words) >= CHUNK_SIZE:
            chunk = ' '.join(current_words[:CHUNK_SIZE])
            last_punct = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
            if last_punct > CHUNK_SIZE // 2:
                chunk = chunk[:last_punct + 1]
            wc = len(chunk.split())
            if MIN_WORDS <= wc <= MAX_WORDS:
                raw_chunks.append(chunk)
            current_words = current_words[len(chunk.split()):]

    filtered = []
    stats = Counter()
    for chunk in raw_chunks[:MAX_CHUNKS_PER_ITEM * 3]:  # fazla işleme
        if len(filtered) >= MAX_CHUNKS_PER_ITEM:
            break
        passed, reason = filter_chunk(chunk)
        if passed:
            filtered.append({
                "text": chunk,
                "isGenerated": 0,
                "source": source,
                "word_count": len(chunk.split()),
                **meta
            })
        else:
            stats[reason] += 1

    return filtered, stats


def fetch_item_text(identifier):
    """
    Archive.org item'ından plain text çek.
    _djvu.txt (OCR) dosyalarını reddeder, sadece gerçek .txt alır.
    """
    try:
        meta_url = f"https://archive.org/metadata/{identifier}"
        meta = requests.get(meta_url, timeout=10).json()
        files = meta.get("files", [])

        # Sadece gerçek .txt dosyaları — OCR olan _djvu.txt, _abbyy.txt reddedilir
        txt_files = [
            f for f in files
            if f.get("name", "").endswith(".txt")
            and "_djvu" not in f.get("name", "")
            and "_abbyy" not in f.get("name", "")
            and "_hocr" not in f.get("name", "")
            and int(f.get("size", 0) or 0) > 5000    # çok küçük dosyaları atla
            and int(f.get("size", 0) or 0) < 2000000  # 2MB'dan büyük dosyaları atla
        ]

        if not txt_files:
            return None

        # En büyük .txt dosyasını al (en fazla içerik)
        txt_files.sort(key=lambda f: int(f.get("size", 0) or 0), reverse=True)
        filename = txt_files[0]["name"]

        text_url = f"https://archive.org/download/{identifier}/{filename}"
        response = requests.get(text_url, timeout=20)
        if response.status_code == 200:
            # Dosyanın ortasından al (başlık/kapanış atlanır)
            text = response.text
            total = len(text)
            start = total // 5       # %20'sinden başla
            end = total * 4 // 5     # %80'inde bitir
            return text[start:end][:8000]  # max 8000 karakter
        return None
    except Exception:
        return None


def search_and_fetch(category, query, target):
    print(f"\n[{category.upper()}] {target} chunk hedefleniyor...")

    params = {
        "q": query,
        "fl[]": ["identifier", "title", "date"],
        "rows": target * 5,   # fazla çek, filtreden az geçecek
        "page": 1,
        "output": "json",
        "sort[]": "date desc",
    }

    try:
        response = requests.get(SEARCH_URL, params=params, timeout=15)
        response.raise_for_status()
        docs = response.json().get("response", {}).get("docs", [])
    except Exception as e:
        print(f"  Arama basarisiz: {e}")
        return [], Counter()

    print(f"  {len(docs)} item bulundu")

    all_chunks = []
    total_stats = Counter()

    for doc in docs:
        if len(all_chunks) >= target:
            break

        identifier = doc.get("identifier", "")
        title = doc.get("title", "")
        date = doc.get("date", "")

        raw_text = fetch_item_text(identifier)
        if not raw_text:
            time.sleep(0.3)
            continue

        clean = clean_archive_text(raw_text)
        if len(clean.split()) < MIN_WORDS * 2:
            continue

        chunks, stats = make_chunks(
            clean,
            {"title": title, "date": date, "identifier": identifier},
            f"archive_{category}"
        )
        all_chunks.extend(chunks)
        total_stats.update(stats)

        if chunks:
            print(f"  + {title[:55]} → {len(chunks)} chunk (toplam: {len(all_chunks)})")

        time.sleep(0.5)

    return all_chunks, total_stats


def validate_quality(df):
    print(f"\n{'='*40}")
    print("KALİTE DOGRULAMA")
    issues = {
        "Kucuk harfle baslayan": sum(1 for t in df['text'] if str(t)[0].islower()),
        "Non-ASCII": sum(1 for t in df['text'] if any(ord(c) > 127 for c in str(t))),
        "URL iceriyor": sum(1 for t in df['text'] if re.search(r'https?://', str(t))),
        "Rakam yogunlugu > %8": sum(
            1 for t in df['text']
            if sum(c.isdigit() for c in str(t)) / max(len(str(t)), 1) > 0.08
        ),
    }
    all_ok = True
    for check, count in issues.items():
        status = "OK" if count == 0 else "SORUN"
        print(f"  [{status}] {check}: {count}")
        if count > 0:
            all_ok = False
    if all_ok:
        print("  Tum kontroller gecti.")


def main():
    print("=== Archive.org Downloader v2 ===")
    print(f"Kategoriler: {list(TARGETS.keys())}")
    print(f"Hedef: {sum(TARGETS.values())} chunk\n")

    all_chunks = []
    total_stats = Counter()

    for category, target in TARGETS.items():
        query = QUERIES[category]["q"]
        chunks, stats = search_and_fetch(category, query, target)
        all_chunks.extend(chunks)
        total_stats.update(stats)
        print(f"  → {category}: {len(chunks)} chunk toplandi")
        time.sleep(1)

    if not all_chunks:
        print("Hic chunk olusturulamadi.")
        return

    df = pd.DataFrame(all_chunks)

    print(f"\n{'='*40}")
    print(f"Toplam: {len(df)} chunk")
    print(f"\nKaynaga gore:")
    print(df["source"].value_counts().to_string())
    print(f"\nKelime sayisi istatistikleri:")
    print(df["word_count"].describe())
    print(f"\nTop filter reasons:")
    for r, c in total_stats.most_common(8):
        print(f"  {r}: {c}")

    validate_quality(df)

    raw_path = os.path.join(OUTPUT_RAW_DIR, "archive_raw.csv")
    df.to_csv(raw_path, index=False, escapechar='\\')
    print(f"\nRaw data: {raw_path}")

    cleaned_df = df[["text", "isGenerated"]]
    cleaned_path = os.path.join(OUTPUT_CLEANED_DIR, "cleaned_archive.csv")
    cleaned_df.to_csv(cleaned_path, index=False, escapechar='\\')
    print(f"Cleaned data: {cleaned_path}")

    print("\n--- Ilk 2 ornek ---")
    for i, row in df.head(2).iterrows():
        print(f"\n[{row['source']}] {str(row.get('title', ''))[:55]}")
        print(f"  {str(row['text'])[:300]}...")


if __name__ == "__main__":
    main()