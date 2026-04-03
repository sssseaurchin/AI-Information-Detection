"""
combineHumanData.py
Tüm human data CSV'lerini birleştir, duplicate'leri temizle, kaydet.
"""

import pandas as pd
import os

INPUT_FILES = [
    ("ourDatas/cleaned_datas/cleaned_gutenberg.csv", "gutenberg"),
    ("ourDatas/cleaned_datas/cleaned_arxiv.csv",     "arxiv"),
    ("ourDatas/cleaned_datas/cleaned_modern.csv",    "modern"),
]

OUTPUT_PATH = "ourDatas/cleaned_datas/cleaned_human_all.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def main():
    print("=== Human Data Birleştirici ===\n")

    dfs = []
    for filepath, label in INPUT_FILES:
        if not os.path.exists(filepath):
            print(f"  [ATLA] {filepath} bulunamadı")
            continue
        df = pd.read_csv(filepath)
        df["source"] = label
        print(f"  [{label}] {len(df)} chunk yüklendi")
        dfs.append(df)

    if not dfs:
        print("Hiç dosya yüklenemedi.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nBirleştirme sonrası: {len(combined)} chunk")

    # Duplicate temizle (text bazlı)
    before = len(combined)
    combined = combined.drop_duplicates(subset="text")
    after = len(combined)
    print(f"Duplicate temizlendi: {before - after} adet")

    # Sadece text + isGenerated tut
    combined = combined[["text", "isGenerated"]].reset_index(drop=True)

    print(f"\n{'='*40}")
    print(f"FINAL: {len(combined)} chunk")
    print(f"isGenerated=0 (human): {(combined['isGenerated']==0).sum()}")
    print(f"Ort. kelime/chunk: {combined['text'].apply(lambda x: len(str(x).split())).mean():.0f}")

    combined.to_csv(OUTPUT_PATH, index=False, escapechar="\\")
    print(f"\nKaydedildi: {OUTPUT_PATH}")

    print("\n--- İlk 3 örnek ---")
    for i, row in combined.head(3).iterrows():
        print(f"\n[{i}] {str(row['text'])[:200]}...")


if __name__ == "__main__":
    main()