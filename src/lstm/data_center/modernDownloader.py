"""
modernDownloader.py — v6
Düzeltmeler:
  1. Reddit limit: 25 → 100, sort=top&t=all
  2. worldnews çıkarıldı, tifu + TrueOffMyChest eklendi (text-heavy)
  3. Wikipedia random loop: except pass → except continue (DNS crash fix)
  4. Terminal glitch yok — threading kullanılmıyor
"""

import re
import os
import html
import time
import random
import requests
import pandas as pd
from collections import Counter

OUTPUT_RAW_DIR = "ourDatas/raw_datas/modern"
OUTPUT_CLEANED_DIR = "ourDatas/cleaned_datas"
os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLEANED_DIR, exist_ok=True)

CHUNK_SIZE = 400
MIN_WORDS = 200
MAX_WORDS = 500
WIKIPEDIA_TARGET = 600
REDDIT_TARGET = 400
MAX_CHUNKS_PER_TOPIC = 5
REDDIT_REPEAT_THRESHOLD = 0.05

# Text-heavy subredditler — link/image değil, uzun yazı içerir
REDDIT_SUBREDDITS = [
    ("explainlikeimfive", 80),   # ELI5: uzun açıklamalar
    ("changemyview",      80),   # Argümanlı yazılar
    ("AskHistorians",     80),   # Detaylı tarih cevapları
    ("askscience",        60),   # Bilimsel açıklamalar
    ("tifu",              50),   # Today I F***ed Up: uzun kişisel hikayeler
    ("TrueOffMyChest",    50),   # Uzun kişisel yazılar
]

HEADERS = {
    "User-Agent": "python:ai_detection_research:v1.0 (academic project)"
}

WIKIPEDIA_TOPICS = [
    "Quantum mechanics", "Relativity", "Thermodynamics", "Electromagnetism",
    "Nuclear physics", "Particle physics", "Optics", "Acoustics",
    "Black hole", "Neutron star", "Supernova", "Dark matter",
    "Evolution", "Natural selection", "DNA", "Genetics", "Ecology",
    "Photosynthesis", "Cell biology", "Microbiology", "Neuroscience",
    "Immunology", "Stem cell", "Biodiversity", "Marine biology",
    "Periodic table", "Chemical reaction", "Polymer", "Metabolism",
    "Plate tectonics", "Earthquake", "Volcano", "Climate change",
    "Ozone layer", "Ocean acidification", "Glacier",
    "World War I", "World War II", "Cold War", "French Revolution",
    "American Revolution", "Industrial Revolution", "Space Race",
    "Civil rights movement", "Decolonization", "Great Depression",
    "Renaissance", "Scientific Revolution", "Age of Exploration",
    "Ottoman Empire", "Roman Empire", "Byzantine Empire",
    "Mongol Empire", "British Empire", "Silk Road",
    "United Nations", "European Union", "NATO",
    "World Trade Organization", "Globalization", "Urbanization",
    "Green Revolution", "Internet", "Artificial intelligence",
    "Computer", "Semiconductor", "Renewable energy", "Nuclear power",
    "Satellite", "Smartphone", "Blockchain", "Machine learning",
    "Robotics", "Nanotechnology", "Electric vehicle", "Solar panel",
    "Wind power", "Democracy", "Capitalism", "Socialism",
    "Economics", "Sociology", "Psychology", "Anthropology",
    "Philosophy", "Linguistics", "Journalism", "Education",
    "Public health", "Human rights", "International law", "Diplomacy",
    "Cinema", "Literature", "Architecture", "Music theory",
    "Opera", "Ballet", "Jazz", "Photography", "Theater",
    "Modernism", "Romanticism", "Surrealism",
    "Amazon rainforest", "Sahara", "Himalaya",
    "Mediterranean Sea", "Atlantic Ocean", "Pacific Ocean",
    "Arctic", "Antarctic", "Mississippi River",
    "Great Barrier Reef", "Serengeti",
    "Vaccination", "Antibiotic", "Cancer", "Diabetes",
    "Cardiovascular disease", "Infectious disease", "Surgery",
    "Epidemiology", "Nutrition", "Mental health",
    "Deforestation", "Pollution", "Sustainability",
    "Conservation biology", "Water cycle", "Nitrogen cycle",
    "Stock market", "Central bank", "Inflation",
    "International trade", "Supply and demand",
    "Ethics", "Epistemology", "Logic", "Existentialism",
    "Enlightenment", "Empiricism", "Rationalism",
    "Olympic Games", "Association football", "Basketball",
    "Scientific method", "Peer review", "Open source",
    "Photonics", "Quantum computing", "Cryptography",
    "Immunotherapy", "Gene therapy", "CRISPR",
    "Social media", "Digital divide", "Net neutrality",
    "Behavioral economics", "Game theory", "Cognitive science",
    "Comparative literature", "Semiotics", "Cultural anthropology",
    "Urban planning", "Public transport", "Infrastructure",
    "Medieval history", "Feudalism", "Crusades",
    "Agricultural revolution", "Bronze Age", "Iron Age",
    "Colonialism", "Imperialism", "Nationalism",
    "Human migration", "Diaspora", "Refugee",
    "Biodiversity loss", "Species extinction", "Coral reef",
    "Deep sea", "Atmosphere", "Hydrosphere",
]

ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'rev', 'gen', 'sgt',
    'cpl', 'pvt', 'capt', 'lt', 'col', 'brig', 'maj', 'adm', 'est',
    'dept', 'approx', 'inc', 'corp', 'ltd', 'vs', 'etc', 'e.g', 'i.e',
    'fig', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep',
    'oct', 'nov', 'dec', 'u.s', 'u.k', 'u.n', 'a.m', 'p.m', 'b.c',
    'a.d', 'ph.d', 'b.a', 'm.a', 'no', 'vol', 'pp', 'op',
}

SENTENCE_START_STOPWORDS = {
    'the', 'a', 'an', 'in', 'it', 'he', 'she', 'this', 'that',
    'they', 'we', 'i', 'its', 'his', 'her', 'their', 'our',
    'at', 'on', 'by', 'as', 'of', 'to', 'for', 'and', 'but',
}

NON_ASCII_MAP = {
    '\u2014': '-', '\u2013': '-',
    '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"',
    '\u00e6': 'ae', '\u0153': 'oe',
    '\u00e9': 'e', '\u00e8': 'e', '\u00ea': 'e',
    '\u00e0': 'a', '\u00e2': 'a', '\u00f4': 'o',
    '\u00fb': 'u', '\u00ee': 'i', '\u00ef': 'i',
    '\u00e7': 'c', '\u00fc': 'u', '\u00f6': 'o',
    '\u00e4': 'a', '\u00df': 'ss',
}

FUNCTION_WORDS = {
    'that', 'with', 'they', 'have', 'said', 'were', 'would', 'your',
    'will', 'which', 'what', 'their', 'from', 'this', 'been', 'there',
    'very', 'into', 'could', 'then', 'them', 'him', 'her', 'she', 'his',
    'you', 'the', 'and', 'but', 'for', 'not', 'are', 'was', 'had', 'has',
    'its', 'our', 'more', 'some', 'when', 'than', 'also', 'just', 'like',
    'upon', 'shall', 'whom', 'such', 'those', 'these', 'about', 'over',
    'after', 'before', 'being', 'other', 'much', 'well', 'only', 'even',
    'back', 'know', 'here', 'time', 'long', 'down', 'still', 'again',
    'never', 'ever', 'same', 'away', 'came', 'come', 'went', 'going',
    'make', 'made', 'take', 'took', 'look', 'looked', 'tell', 'told',
    'because', 'though', 'without', 'through', 'every', 'might', 'must',
    'each', 'where', 'while', 'always', 'already', 'between',
}


def split_sentences(text):
    protected = text
    for abbr in ABBREVIATIONS:
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\.', re.IGNORECASE)
        protected = pattern.sub(abbr.replace('.', '<DOT>'), protected)
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def normalize_text(text):
    for char, replacement in NON_ASCII_MAP.items():
        text = text.replace(char, replacement)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'-{3,}', '-', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_wikipedia_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[note \d+\]', '', text)
    text = re.sub(r'\[[a-z]\]', '', text)
    text = re.sub(r'\([^)]*[ˈˌ][^)]*\)', '', text)
    text = re.sub(r'\(\d+°[NS][^)]*\)', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', text)
    text = re.sub(r"'{2,3}", '', text)
    text = re.sub(r'\([^)]{1,25}\)', '', text)
    return normalize_text(text)


def clean_reddit_text(text):
    text = html.unescape(text)
    text = text.replace("[deleted]", "").replace("[removed]", "")
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'^\s*[-*>]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'u/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    text = re.sub(r'\bEDIT\s*\d*\s*:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'TL;?DR.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    return normalize_text(text)


def filter_chunk(chunk, repeat_threshold=0.03):
    words = chunk.split()
    word_count = len(words)
    if word_count < MIN_WORDS:
        return False, "cok kisa"
    if chunk[0].islower():
        return False, "kucuk harfle basliyor"
    if re.search(r'https?://', chunk):
        return False, "url iceriyor"
    if sum(c.isdigit() for c in chunk) / len(chunk) > 0.08:
        return False, "rakam yogunlugu yuksek"
    if (chunk.count('(') + chunk.count(')')) / len(chunk) > 0.06:
        return False, "parantez yogunlugu yuksek"
    alpha = [c for c in chunk if c.isalpha()]
    if alpha and sum(c.isupper() for c in alpha) / len(alpha) > 0.15:
        return False, "buyuk harf orani yuksek"
    quoted_words = sum(len(q.split()) for q in re.findall(r'"([^"]*)"', chunk))
    if word_count > 0 and quoted_words / word_count > 0.30:
        return False, "diyalog orani yuksek"
    if chunk.count('"') / len(chunk) > 0.05:
        return False, "tirnak yogunlugu yuksek"
    sentences = [s for s in split_sentences(chunk) if s.strip()]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len < 8 or avg_len > 45:
            return False, f"ort. cumle uzunlugu: {avg_len:.1f}"
    if sentences:
        short = [s for s in sentences if len(s.split()) <= 3]
        if len(short) / len(sentences) > 0.30:
            return False, "kisa cumle orani yuksek"
    if sentences and any(len(s.split()) > 100 for s in sentences):
        return False, "100+ kelime tek cumle"
    if sentences and len(sentences) >= 4:
        starts = [s.split()[0].lower() for s in sentences if s.split()]
        valid_starts = [s for s in starts if s not in SENTENCE_START_STOPWORDS]
        if valid_starts:
            top_start, count = Counter(valid_starts).most_common(1)[0]
            if count / len(sentences) > 0.40:
                return False, f"tekrarlayan cumle baslangici: '{top_start}'"
    if chunk.count(',') / word_count > 0.12:
        return False, "fazla virgul (liste)"
    content_words = [w.lower().strip('.,!?";:()') for w in words
                     if len(w) > 3 and w.lower().strip('.,!?";:()') not in FUNCTION_WORDS]
    if content_words:
        counter = Counter(content_words)
        top_word, top_count = counter.most_common(1)[0]
        if top_count / len(content_words) > repeat_threshold:
            return False, f"tekrarlayan kelime: '{top_word}'"
        top10 = sum(c for _, c in counter.most_common(10))
        if top10 / len(content_words) > 0.60:
            return False, "kelime cesitliligi dusuk"
    return True, "ok"


def make_chunks(text, meta, source, repeat_threshold=0.03):
    sentences = split_sentences(text)
    skip = max(10, len(sentences) // 20)
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
    for chunk in raw_chunks:
        passed, reason = filter_chunk(chunk, repeat_threshold)
        if passed:
            filtered.append({"text": chunk, "isGenerated": 0,
                              "source": source, "word_count": len(chunk.split()), **meta})
        else:
            stats[reason] += 1
    return filtered, stats


def fetch_wikipedia(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "titles": title, "prop": "extracts",
              "explaintext": True, "exsectionformat": "plain",
              "format": "json", "redirects": 1}
    try:
        r = requests.get(url, params=params, timeout=15, headers=HEADERS)
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return None, None
        return page.get("extract", ""), page.get("title", title)
    except Exception:
        return None, None


def fetch_wikipedia_random():
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "generator": "random", "grnnamespace": 0,
              "prop": "extracts", "explaintext": True,
              "exsectionformat": "plain", "format": "json"}
    try:
        r = requests.get(url, params=params, timeout=15, headers=HEADERS)
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract", ""), page.get("title", "Random")
    except Exception:
        return None, None


def process_wiki_text(raw_text, actual_title):
    if not raw_text or len(raw_text) < 500:
        return [], Counter()
    clean = clean_wikipedia_text(raw_text)
    lines = clean.split('\n')
    good_lines = [l.strip() for l in lines
                  if len(l.strip().split()) >= 8 and not l.strip().startswith('=')]
    clean = re.sub(r'\s+', ' ', ' '.join(good_lines)).strip()
    chunks, stats = make_chunks(clean, {"title": actual_title}, "wikipedia", repeat_threshold=0.03)
    if len(chunks) > MAX_CHUNKS_PER_TOPIC:
        chunks = chunks[:MAX_CHUNKS_PER_TOPIC]
    return chunks, stats


def get_wikipedia_chunks(target=WIKIPEDIA_TARGET):
    print(f"\n{'='*50}")
    print(f"[WIKIPEDIA] Hedef: {target} | Maks/konu: {MAX_CHUNKS_PER_TOPIC}")
    print(f"{'='*50}")
    all_chunks = []
    total_stats = Counter()

    for topic in WIKIPEDIA_TOPICS:
        if len(all_chunks) >= target:
            break
        raw_text, actual_title = fetch_wikipedia(topic)
        chunks, stats = process_wiki_text(raw_text, actual_title)
        all_chunks.extend(chunks)
        total_stats.update(stats)
        print(f"  {topic}: +{len(chunks)} (toplam: {len(all_chunks)})")
        time.sleep(0.8)

    # Gemini fix: DNS crash durumunda continue ile devam et
    if len(all_chunks) < target:
        print(f"\n  {len(all_chunks)} chunk toplandi, kalan {target - len(all_chunks)} rastgele dolduruluyor...")
        attempts = 0
        while len(all_chunks) < target and attempts < 300:
            attempts += 1
            try:
                raw_text, actual_title = fetch_wikipedia_random()
                chunks, stats = process_wiki_text(raw_text, actual_title)
                if chunks:
                    all_chunks.extend(chunks)
                    total_stats.update(stats)
                    print(f"  [Rastgele] {actual_title}: +{len(chunks)} (toplam: {len(all_chunks)})")
                time.sleep(0.5)
            except Exception:
                time.sleep(2)
                continue  # DNS crash fix: durmak yerine tekrar dene

    print(f"\n  Wikipedia toplam: {len(all_chunks)} chunk")
    print(f"  Top filter reasons:")
    for r, c in total_stats.most_common(5):
        print(f"    {r}: {c}")
    return all_chunks


def get_reddit_posts(subreddit, limit=100):
    # Gemini fix: limit=100, sort=top&t=all
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t=all"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}")
            return []
        return [p['data'] for p in r.json()['data']['children']]
    except Exception as e:
        print(f"    Error: {e}")
        return []


def get_post_comments(subreddit, post_id, limit=10):
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit={limit}&sort=top"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        comments = r.json()[1]['data']['children']
        return [c['data'].get('body', '') for c in comments
                if c['kind'] == 't1'
                and c['data'].get('score', 0) > 5
                and len(c['data'].get('body', '').split()) >= MIN_WORDS]
    except Exception:
        return []


def get_reddit_chunks(target=REDDIT_TARGET):
    print(f"\n{'='*50}")
    print(f"[REDDIT JSON API] Hedef: {target}")
    print(f"{'='*50}")
    all_chunks = []
    total_stats = Counter()

    for subreddit, sub_target in REDDIT_SUBREDDITS:
        if len(all_chunks) >= target:
            break

        print(f"\n  r/{subreddit} (hedef: {sub_target})...")
        posts = get_reddit_posts(subreddit, limit=100)
        print(f"    {len(posts)} post bulundu")

        if not posts:
            time.sleep(30)
            continue

        sub_chunks = []
        for post in posts:
            if len(sub_chunks) >= sub_target:
                break
            post_id = post.get('id', '')
            post_title = post.get('title', '')

            # Self post text
            selftext = post.get('selftext', '')
            if selftext and selftext not in ['[deleted]', '[removed]', '']:
                clean = clean_reddit_text(selftext)
                if len(clean.split()) >= MIN_WORDS:
                    chunks, stats = make_chunks(
                        clean,
                        {"title": post_title, "subreddit": subreddit},
                        f"reddit_{subreddit}",
                        repeat_threshold=REDDIT_REPEAT_THRESHOLD
                    )
                    sub_chunks.extend(chunks)
                    total_stats.update(stats)

            # Comments
            comments = get_post_comments(subreddit, post_id)
            for comment in comments:
                if len(sub_chunks) >= sub_target:
                    break
                clean = clean_reddit_text(comment)
                if len(clean.split()) < MIN_WORDS:
                    continue
                chunks, stats = make_chunks(
                    clean,
                    {"title": post_title, "subreddit": subreddit},
                    f"reddit_{subreddit}",
                    repeat_threshold=REDDIT_REPEAT_THRESHOLD
                )
                sub_chunks.extend(chunks)
                total_stats.update(stats)

            time.sleep(1.5)

        all_chunks.extend(sub_chunks)
        print(f"    +{len(sub_chunks)} chunk (toplam: {len(all_chunks)})")
        time.sleep(30)

    print(f"\n  Reddit toplam: {len(all_chunks)} chunk")
    print(f"  Top filter reasons:")
    for r, c in total_stats.most_common(5):
        print(f"    {r}: {c}")
    return all_chunks


def validate_quality(df):
    print(f"\n{'='*50}")
    print("KALİTE DOGRULAMA")
    print(f"{'='*50}")
    issues = {
        "Kucuk harfle baslayan":   sum(1 for t in df['text'] if str(t)[0].islower()),
        "URL iceriyor":            sum(1 for t in df['text'] if re.search(r'https?://', str(t))),
        "[deleted]/[removed]":     sum(1 for t in df['text'] if '[deleted]' in str(t) or '[removed]' in str(t)),
        "HTML entity":             sum(1 for t in df['text'] if '&amp;' in str(t) or '&quot;' in str(t)),
        "Rakam yogunlugu > %8":    sum(1 for t in df['text'] if sum(c.isdigit() for c in str(t)) / max(len(str(t)),1) > 0.08),
        "Diyalog orani > 0.30":    0,
        "Non-ASCII":               sum(1 for t in df['text'] if any(ord(c) > 127 for c in str(t))),
        "100+ kelime tek cumle":   sum(1 for t in df['text'] if any(len(s.split()) > 100 for s in split_sentences(str(t)))),
    }
    for t in df['text']:
        text, words = str(t), str(t).split()
        qw = sum(len(q.split()) for q in re.findall(r'"([^"]*)"', text))
        if words and qw / len(words) > 0.30:
            issues["Diyalog orani > 0.30"] += 1
    all_ok = True
    for check, count in issues.items():
        status = "OK" if count == 0 else "SORUN"
        print(f"  [{status}] {check}: {count}")
        if count > 0:
            all_ok = False
    if all_ok:
        print("\n  Tum kontroller gecti.")
    return all_ok


def main():
    print("=== Modern Human Text Downloader v6 ===")
    print(f"Wikipedia: {WIKIPEDIA_TARGET} | Reddit: {REDDIT_TARGET}\n")
    all_chunks = []

    wiki_chunks = get_wikipedia_chunks(WIKIPEDIA_TARGET)
    all_chunks.extend(wiki_chunks)

    reddit_chunks = get_reddit_chunks(REDDIT_TARGET)
    all_chunks.extend(reddit_chunks)

    if not all_chunks:
        print("Hic chunk olusturulamadi.")
        return

    df = pd.DataFrame(all_chunks)
    print(f"\n{'='*50}")
    print(f"TOPLAM: {len(df)} chunk")
    print(f"\nKaynaga gore:")
    print(df['source'].value_counts().to_string())
    print(f"\nKelime sayisi istatistikleri:")
    print(df['word_count'].describe())

    validate_quality(df)

    raw_path = os.path.join(OUTPUT_RAW_DIR, "modern_raw.csv")
    df.to_csv(raw_path, index=False, escapechar='\\')
    print(f"\nRaw data: {raw_path}")

    cleaned_df = df[["text", "isGenerated"]]
    cleaned_path = os.path.join(OUTPUT_CLEANED_DIR, "cleaned_modern.csv")
    cleaned_df.to_csv(cleaned_path, index=False, escapechar='\\')
    print(f"Cleaned data: {cleaned_path}")

    print("\n--- Her kaynaktan 1 ornek ---")
    for source in df['source'].unique()[:6]:
        row = df[df['source'] == source].iloc[0]
        print(f"\n[{source}] {str(row.get('title', ''))[:60]}")
        print(f"  {str(row['text'])[:300]}...")


if __name__ == "__main__":
    main()