import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_vs_AI_cleaned.csv"



def analyze_dataset(path: str):
    
    print("\n========== DATASET ANALYSIS (student_vs_AI_cleaned) ==========\n")
    
    df = pd.read_csv(path)

    print("Columns:")
    print(df.columns.tolist())

    print("\nTotal rows:", len(df))

    text_col = df.columns[0]
    label_col = df.columns[1]

    print("\nText column:", text_col)
    print("Label column:", label_col)

    print("\n---------- CLASS DISTRIBUTION ----------")
    
    counts = df[label_col].value_counts(dropna=False)
    percentages = df[label_col].value_counts(normalize=True, dropna=False) * 100

    for label in counts.index:
        print(f"{label}: {counts[label]} ({percentages[label]:.2f}%)")

    print("\n---------- MISSING VALUES ----------")
    print(df[[text_col, label_col]].isna().sum())

    print("\n---------- DUPLICATE TEXT CHECK ----------")
    total_text = len(df[text_col])
    unique_text = df[text_col].astype(str).nunique()

    print("Unique texts:", unique_text)
    print("Duplicate texts:", total_text - unique_text)

    print("\n---------- TEXT LENGTH ANALYSIS ----------")

    df["word_count"] = df[text_col].astype(str).str.split().str.len()
    df["char_count"] = df[text_col].astype(str).str.len()

    print("\nWord count statistics:")
    print(df["word_count"].describe())

    print("\nCharacter count statistics:")
    print(df["char_count"].describe())

    print("\nShortest text length (words):", df["word_count"].min())
    print("Longest text length (words):", df["word_count"].max())

    print("\nAverage text length (words):", round(df["word_count"].mean(), 2))

    print("\n========== END ANALYSIS ==========\n")


# KULLANIM
analyze_dataset(DATA_PATH)