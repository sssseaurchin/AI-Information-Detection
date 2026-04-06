import requests
import pandas as pd
import time
import os
import re
from collections import Counter

OUTPUT_RAW_DIR = "ourDatas/raw_datas/gutenberg"
OUTPUT_CLEANED_DIR = "ourDatas/cleaned_datas"
os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLEANED_DIR, exist_ok=True)

CHUNK_SIZE = 400
MIN_WORDS = 200
MAX_WORDS = 500
MAX_CHUNKS_PER_BOOK = 150

# Analiz sonucunda:
# CIKARILAN: War and Peace (bozuk OCR çeviri), Huckleberry Finn (diyalekt),
#             Tom Sawyer (diyalekt), Alice in Wonderland (3 chunk - yetersiz),
#             The Yellow Wallpaper (4 chunk - yetersiz)
# EKLENEN: Northanger Abbey, Sense and Sensibility, The Scarlet Letter,
#           Tess of the d'Urbervilles, North and South
BOOKS = [
    {"id": 84,    "title": "Frankenstein",                     "author": "Mary Shelley"},
    {"id": 1342,  "title": "Pride and Prejudice",              "author": "Jane Austen"},
    {"id": 2701,  "title": "Moby Dick",                        "author": "Herman Melville"},
    {"id": 98,    "title": "A Tale of Two Cities",             "author": "Charles Dickens"},
    {"id": 174,   "title": "The Picture of Dorian Gray",       "author": "Oscar Wilde"},
    {"id": 2554,  "title": "Crime and Punishment",             "author": "Fyodor Dostoevsky"},
    {"id": 345,   "title": "Dracula",                          "author": "Bram Stoker"},
    {"id": 36,    "title": "The War of the Worlds",            "author": "H.G. Wells"},
    {"id": 1400,  "title": "Great Expectations",               "author": "Charles Dickens"},
    {"id": 1661,  "title": "The Adventures of Sherlock Holmes","author": "Arthur Conan Doyle"},
    {"id": 5200,  "title": "Metamorphosis",                    "author": "Franz Kafka"},
    {"id": 768,   "title": "Wuthering Heights",                "author": "Emily Bronte"},
    {"id": 1260,  "title": "Jane Eyre",                        "author": "Charlotte Bronte"},
    {"id": 158,   "title": "Emma",                             "author": "Jane Austen"},
    {"id": 2097,  "title": "The Jungle Book",                  "author": "Rudyard Kipling"},
    # Yeni eklenenler
    {"id": 121,   "title": "Northanger Abbey",                 "author": "Jane Austen"},
    {"id": 161,   "title": "Sense and Sensibility",            "author": "Jane Austen"},
    {"id": 25344, "title": "The Scarlet Letter",               "author": "Nathaniel Hawthorne"},
    {"id": 110,   "title": "Tess of the d'Urbervilles",        "author": "Thomas Hardy"},
    {"id": 4276,  "title": "North and South",                  "author": "Elizabeth Gaskell"},
]

URL_TEMPLATES = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]

# Non-ASCII karakter dönüşüm tablosu
NON_ASCII_MAP = {
    '\u2014': '-', '\u2013': '-',
    '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"',
    '\u00e6': 'ae', '\u0153': 'oe',
    '\u00e9': 'e', '\u00e8': 'e', '\u00ea': 'e',
    '\u00e0': 'a', '\u00e2': 'a',
    '\u00f4': 'o', '\u00fb': 'u',
    '\u00ee': 'i', '\u00ef': 'i',
    '\u00e7': 'c', '\u00fc': 'u',
    '\u00f6': 'o', '\u00e4': 'a',
    '\u00df': 'ss',
}

# Analiz sonucu belirlenen diyalekt kelimeler
DIALECT_WORDS = re.compile(
    r"\b(warn't|hain't|gov'ment|sivilized|hick'ry|gwyne|wuz\b|"
    r"git\b|fer\b|yer\b|woulda|coulda|shoulda|gonna|wanna|"
    r"ain't gonna|hafta|gotta)\b",
    re.IGNORECASE
)

# Function words - tekrar filtresinde görmezden gel
FUNCTION_WORDS = {
    'that', 'with', 'they', 'have', 'said', 'were', 'would',
    'your', 'will', 'which', 'what', 'their', 'from', 'this',
    'been', 'there', 'very', 'into', 'could', 'then', 'them',
    'him', 'her', 'she', 'his', 'you', 'the', 'and', 'but',
    'for', 'not', 'are', 'was', 'had', 'has', 'its', 'our',
    'more', 'some', 'when', 'than', 'also', 'just', 'like',
    'upon', 'shall', 'whom', 'such', 'those', 'these', 'about',
    'over', 'after', 'before', 'being', 'other', 'much', 'well',
    'only', 'even', 'back', 'know', 'here', 'time', 'long',
    'down', 'still', 'again', 'never', 'ever', 'same', 'away',
    'came', 'come', 'went', 'going', 'make', 'made', 'take',
    'took', 'look', 'looked', 'tell', 'told', 'because',
    'though', 'without', 'through', 'every', 'might', 'must',
    'each', 'where', 'while', 'always', 'already', 'between',
}


def download_book(book_id):
    for template in URL_TEMPLATES:
        url = template.format(id=book_id)
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                return response.text
        except Exception:
            continue
    return None


def clean_book_text(text):
    # Gutenberg header/footer - genişletilmiş marker listesi
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
        "This etext was prepared",
        "This Etext was prepared",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
        "END OF THE PROJECT GUTENBERG",
        "Project Gutenberg-tm",
    ]

    start_pos = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_pos = text.find('\n', idx) + 1
            break

    end_pos = len(text)
    for marker in end_markers:
        idx = text.rfind(marker)
        if idx != -1:
            end_pos = idx
            break

    text = text[start_pos:end_pos]

    # Non-ASCII dönüşümü
    for char, replacement in NON_ASCII_MAP.items():
        text = text.replace(char, replacement)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Noktalama normalizasyonu
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'-{2,}', '-', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)

    # Kısa ve ALL CAPS satırları at
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line.split()) < 4:
            continue
        if line.isupper():
            continue
        cleaned_lines.append(line)

    text = ' '.join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def filter_chunk(chunk):
    """
    Chunk kalite filtresi. True = geçerli, False = at.

    Analiz bulgularına göre filtreler:
      1. Diyalog oranı        : tırnak içi kelimeler > %30 -> at  (analiz: 271 chunk etkilenir)
      2. Tekrarlayan kelime   : içerik kelimelerinde en sık > %3 -> at
      3. Ort. cümle uzunluğu  : < 10 veya > 40 kelime -> at
      4. Tırnak yoğunluğu     : '"' > %5 -> at
      5. Kısa cümle oranı     : 3 kelimeden kısa cümle > %30 -> at
      6. Kelime çeşitliliği   : en sık 10 kelime > %60 -> at
      7. Küçük harfle başlama : chunk küçük harfle başlıyorsa -> at  (analiz: 66 chunk)
      8. 100+ kelime tek cümle: chunk içinde 100+ kelimelik cümle -> at  (analiz: 109 chunk)
      9. Diyalekt             : diyalekt kelimesi varsa -> at  (analiz: 111 chunk)
    """
    words = chunk.split()
    word_count = len(words)

    # 7. Küçük harfle başlama (analiz: 66 chunk)
    if chunk and chunk[0].islower():
        return False, "kucuk harfle basliyor"

    # 1. Diyalog oranı — 0.40'tan 0.30'a düşürüldü (analiz: 271 chunk)
    quoted = re.findall(r'"([^"]*)"', chunk)
    quoted_words = sum(len(q.split()) for q in quoted)
    if word_count > 0 and quoted_words / word_count > 0.30:
        return False, "diyalog orani yuksek"

    # 9. Diyalekt (analiz: 111 chunk)
    if DIALECT_WORDS.search(chunk):
        return False, "diyalekt kelimesi"

    # 2. Tekrarlayan içerik kelimesi
    lower_words = [w.lower().strip('.,!?";:') for w in words
                   if len(w) > 3 and w.lower().strip('.,!?";:') not in FUNCTION_WORDS]
    if lower_words:
        counter = Counter(lower_words)
        most_common_word, most_common_count = counter.most_common(1)[0]
        if most_common_count / len(lower_words) > 0.03:
            return False, f"tekrarlayan kelime: '{most_common_word}'"

    # 3. Ortalama cümle uzunluğu
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    sentences = [s for s in sentences if s.strip()]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len < 10 or avg_len > 40:
            return False, f"ort. cumle uzunlugu: {avg_len:.1f}"

    # 4. Tırnak yoğunluğu
    if len(chunk) > 0 and chunk.count('"') / len(chunk) > 0.05:
        return False, "tirnak yogunlugu yuksek"

    # 5. Kısa cümle oranı
    if sentences:
        short_sentences = [s for s in sentences if len(s.split()) <= 3]
        if len(short_sentences) / len(sentences) > 0.30:
            return False, "cok kisa cumle orani yuksek"

    # 6. Kelime çeşitliliği
    if lower_words:
        top10_count = sum(count for _, count in counter.most_common(10))
        if top10_count / len(lower_words) > 0.60:
            return False, "kelime cesitliligi dusuk"

    # 8. 100+ kelime tek cümle (analiz: 109 chunk)
    if sentences:
        if any(len(s.split()) > 100 for s in sentences):
            return False, "100+ kelime tek cumle"

    return True, "ok"


def chunk_text(text, book_info):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # İlk ve son %10'u atla
    skip = max(50, len(sentences) // 10)
    sentences = sentences[skip:-skip]

    raw_chunks = []
    current_words = []

    for sentence in sentences:
        current_words.extend(sentence.split())

        if len(current_words) >= CHUNK_SIZE:
            chunk = ' '.join(current_words[:CHUNK_SIZE])
            last_period = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
            if last_period > CHUNK_SIZE // 2:
                chunk = chunk[:last_period + 1]

            word_count = len(chunk.split())
            if MIN_WORDS <= word_count <= MAX_WORDS:
                raw_chunks.append(chunk)

            current_words = current_words[len(chunk.split()):]

    # Filtrele
    filtered = []
    filter_stats = Counter()
    for chunk in raw_chunks:
        passed, reason = filter_chunk(chunk)
        if passed:
            filtered.append(chunk)
        else:
            filter_stats[reason] += 1

    # Kitap başına max chunk — ortadan al
    if len(filtered) > MAX_CHUNKS_PER_BOOK:
        start = (len(filtered) - MAX_CHUNKS_PER_BOOK) // 2
        filtered = filtered[start:start + MAX_CHUNKS_PER_BOOK]

    result = []
    for chunk in filtered:
        result.append({
            "text": chunk,
            "isGenerated": 0,
            "source": "gutenberg",
            "title": book_info["title"],
            "author": book_info["author"],
            "gutenberg_id": book_info["id"],
            "word_count": len(chunk.split()),
        })

    return result, filter_stats


def main():
    print("=== Project Gutenberg Downloader (v4 - Final) ===")
    print(f"Kitap sayisi: {len(BOOKS)} | Kitap basina max: {MAX_CHUNKS_PER_BOOK} chunk\n")

    all_chunks = []
    total_filter_stats = Counter()

    for book in BOOKS:
        print(f"[{book['id']}] {book['title']} - {book['author']}")

        text = download_book(book["id"])
        if not text:
            print(f"  x Indirilemedi, atlaniyor.\n")
            continue

        print(f"  + Indirildi ({len(text):,} karakter)")
        cleaned = clean_book_text(text)
        print(f"  + Temizlendi ({len(cleaned):,} karakter)")

        chunks, filter_stats = chunk_text(cleaned, book)
        all_chunks.extend(chunks)
        total_filter_stats.update(filter_stats)

        filtered_out = sum(filter_stats.values())
        print(f"  + {len(chunks)} chunk alindi (filtrelenen: {filtered_out})\n")
        time.sleep(1)

    if not all_chunks:
        print("Hic chunk olusturulamadi.")
        return

    df = pd.DataFrame(all_chunks)

    print(f"{'='*40}")
    print(f"Toplam: {len(df)} chunk")
    print(f"\nKitaplara gore dagilim:")
    print(df.groupby("title")["word_count"].count().to_string())
    print(f"\nKelime sayisi istatistikleri:")
    print(df["word_count"].describe())
    print(f"\nFiltre istatistikleri (en cok atilan 10):")
    for reason, count in total_filter_stats.most_common(10):
        print(f"  {reason}: {count}")

    # Kalite dogrulama
    print(f"\n=== KALİTE DOGRULAMA ===")
    starts_lower = sum(1 for t in df['text'] if str(t)[0].islower())
    high_dialog = sum(1 for t in df['text']
                      if len(re.findall(r'"([^"]*)"', str(t))) > 0
                      and sum(len(q.split()) for q in re.findall(r'"([^"]*)"', str(t))) / len(str(t).split()) > 0.30)
    dialect_found = sum(1 for t in df['text'] if DIALECT_WORDS.search(str(t)))
    long_sent = sum(1 for t in df['text']
                    if any(len(s.split()) > 100
                           for s in re.split(r'(?<=[.!?])\s+', str(t))))
    print(f"  Kucuk harfle baslayan: {starts_lower} (olmali: 0)")
    print(f"  Diyalog orani > 0.30: {high_dialog} (olmali: 0)")
    print(f"  Diyalekt kelimesi: {dialect_found} (olmali: 0)")
    print(f"  100+ kelime tek cumle: {long_sent} (olmali: 0)")

    raw_path = os.path.join(OUTPUT_RAW_DIR, "gutenberg_raw.csv")
    df.to_csv(raw_path, index=False, escapechar='\\')
    print(f"\nRaw data: {raw_path}")

    cleaned_df = df[["text", "isGenerated"]]
    cleaned_path = os.path.join(OUTPUT_CLEANED_DIR, "cleaned_gutenberg.csv")
    cleaned_df.to_csv(cleaned_path, index=False, escapechar='\\')
    print(f"Cleaned data: {cleaned_path}")

    print("\n--- Ilk 2 ornek ---")
    for i, row in df.head(2).iterrows():
        print(f"\n[{row['title']}] ({row['word_count']} kelime)")
        print(f"  {row['text'][:300]}...")


if __name__ == "__main__":
    main()