import requests
import pandas as pd
import time
import os

API_KEY = "hgj75tb5ctcfgvawv6jab9an"
BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

OUTPUT_DIR = "data_center/data/raw_datas/ieee"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_ieee_abstracts(total=100):
    results = []
    max_per_request = 25  # API limiti

    # 2015 öncesi, İngilizce, abstract'ı olan makaleler
    params = {
        "apikey": API_KEY,
        "format": "json",
        "max_records": max_per_request,
        "start_record": 1,
        "sort_order": "desc",
        "sort_field": "article_number",
        "end_year": 2014,
        "start_year": 1990,
        "abstract": "yes",        # sadece abstract'ı olanlar
    }

    fetched = 0
    start_record = 1

    while fetched < total:
        params["start_record"] = start_record
        params["max_records"] = min(max_per_request, total - fetched)

        print(f"Fetching records {start_record} to {start_record + params['max_records'] - 1}...")

        try:
            response = requests.get(BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Request failed: {e}")
            break

        articles = data.get("articles", [])
        if not articles:
            print("No more articles returned.")
            break

        for article in articles:
            abstract = article.get("abstract", "").strip()
            if not abstract or len(abstract.split()) < 50:
                continue  # çok kısa abstract'ları atla

            results.append({
                "text": abstract,
                "isGenerated": 0,  # human yazılı
                "source": "ieee_xplore",
                "title": article.get("title", ""),
                "year": article.get("publication_year", ""),
                "doi": article.get("doi", ""),
            })

        fetched += len(articles)
        start_record += len(articles)

        print(f"  → {len(results)} valid abstracts collected so far.")
        time.sleep(0.2)  # rate limit koruma

    return results

def main():
    print("=== IEEE Xplore Abstract Downloader ===")
    print(f"Target: 100 abstracts (2015 öncesi)\n")

    abstracts = fetch_ieee_abstracts(total=100)

    if not abstracts:
        print("Hiç abstract çekilemedi. API key status 'waiting' olabilir, biraz bekle.")
        return

    df = pd.DataFrame(abstracts)
    print(f"\nToplamda {len(df)} abstract çekildi.")
    print(f"Kelime sayısı istatistikleri:")
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    print(df["word_count"].describe())

    # Raw olarak kaydet (tüm metadata ile)
    raw_path = os.path.join(OUTPUT_DIR, "ieee_raw_100.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw data kaydedildi: {raw_path}")

    # Cleaned format (sadece text + isGenerated)
    cleaned_df = df[["text", "isGenerated"]]
    cleaned_path = "data_center/data/cleaned_datas/ieee_cleaned_100.csv"
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data kaydedildi: {cleaned_path}")

    print("\nİlk 3 abstract:")
    for i, row in df.head(3).iterrows():
        print(f"\n[{i+1}] {row['title']} ({row['year']})")
        print(f"    {row['text'][:200]}...")

if __name__ == "__main__":
    main()
