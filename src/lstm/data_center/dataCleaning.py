import json
import pandas as pd
from pathlib import Path

# All cleaning functions normalize to: text, isGenerated (0=Human, 1=AI)

BASE_DIR = Path(__file__).resolve().parent
KAGGLE_FOLDER = BASE_DIR / "data" / "raw_datas" / "kaggle"
HF_FOLDER = BASE_DIR / "data" / "raw_datas" / "huggingface"
CLEANED_FOLDER = BASE_DIR / "data" / "cleaned_datas"
CLEANED_FOLDER.mkdir(parents=True, exist_ok=True)


MIN_WORD_COUNT = 10

def _save_to_disk(df: pd.DataFrame, filename: str):
    """Removes duplicates, drops nulls, and saves as CSV."""
    df = df[["text", "isGenerated"]].copy()
    df.drop_duplicates(subset=["text", "isGenerated"], inplace=True)
    df.dropna(subset=["text", "isGenerated"], inplace=True)
    df["isGenerated"] = df["isGenerated"].astype(int)

    
    # remove in-text labels that cause data leakage (e.g. "[AI-Generated]")

    df["text"] = df["text"].str.replace(r"\[AI[^\]]*\]", "", regex=True)

    df["text"] = df["text"].str.replace(r"\[Human[^\]]*\]", "", regex=True)



    # normalize whitespace (multiple spaces, tabs, newlines -> single space)

    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()



    # filter out short texts (less than MIN_WORD_COUNT words)

    word_counts = df["text"].str.split().str.len()

    before = len(df)

    df = df[word_counts >= MIN_WORD_COUNT]



    path_to_save = CLEANED_FOLDER / filename
    df.to_csv(path_to_save, index=False)
    print(f"Saved {len(df)} rows (filtered {before - len(df)} short texts) -> {path_to_save}")


# ========================
# KAGGLE DATASETS
# ========================

def clean_AI_Human():
    # Columns: Text, generated (0/1)
    df = pd.read_csv(KAGGLE_FOLDER / "AI_Human.csv")
    df = df.rename(columns={"Text": "text", "generated": "isGenerated"})
    _save_to_disk(df, "cleaned_AI_Human.csv")


def clean_Training_Essay_Data():
    # Columns: text, generated (0/1)
    df = pd.read_csv(KAGGLE_FOLDER / "Training_Essay_Data.csv")
    df = df.rename(columns={"generated": "isGenerated"})
    _save_to_disk(df, "cleaned_Training_Essay_Data.csv")


def clean_student_vs_AI():
    # Columns: Text, Label (student/ai)
    df = pd.read_csv(KAGGLE_FOLDER / "student_vs_AI.csv")
    df = df.rename(columns={"Text": "text", "Label": "isGenerated"})
    df["isGenerated"] = df["isGenerated"].map({"student": 0, "ai": 1})
    _save_to_disk(df, "cleaned_student_vs_AI.csv")


def clean_ai_vs_human_comparison():
    # Columns: text, label (human=0, ai=1 as string or int)
    df = pd.read_csv(KAGGLE_FOLDER / "ai-vs-human-comparison-dataset.csv")
    labels = df["label"].astype(str).str.strip().str.lower()
    mapping = {"human": 0, "ai": 1, "0": 0, "1": 1}
    df["isGenerated"] = labels.map(mapping)
    _save_to_disk(df, "cleaned_ai_vs_human_comparison.csv")


def clean_aknjit():
    # Columns: text, label (0/1)
    df = pd.read_csv(KAGGLE_FOLDER / "aknjit" / "human-vs-ai-text-classification-dataset.csv")
    df = df.rename(columns={"label": "isGenerated"})
    _save_to_disk(df, "cleaned_aknjit.csv")


def clean_algozee():
    # Columns: content_text, author_type (human/ai)
    df = pd.read_csv(KAGGLE_FOLDER / "algozee" / "ai-generated-vs-human-written-text-dataset.csv")
    df = df.rename(columns={"content_text": "text"})
    labels = df["author_type"].astype(str).str.strip().str.lower()
    df["isGenerated"] = labels.map({"human": 0, "ai": 1})
    _save_to_disk(df, "cleaned_algozee.csv")


def clean_mostafabakr():
    # Columns: text, label (0/1)
    df = pd.read_csv(KAGGLE_FOLDER / "ai-vs-human-classification-dataset.csv")
    df = df.rename(columns={"label": "isGenerated"})
    _save_to_disk(df, "cleaned_mostafabakr.csv")


def clean_khushu89():
    # Columns: text, isGenerated (0/1) - already in correct format
    df = pd.read_csv(KAGGLE_FOLDER / "khushu89_collected.csv")
    _save_to_disk(df, "cleaned_khushu89.csv")


# ========================
# HUGGING FACE DATASETS
# ========================

def clean_ai_text_detection_pile():
    # Columns: source (human/ai), id, text
    df = pd.read_csv(HF_FOLDER / "ai-text-detection-pile" / "train.csv")
    labels = df["source"].astype(str).str.strip().str.lower()
    df["isGenerated"] = labels.map({"human": 0, "ai": 1})
    _save_to_disk(df, "cleaned_ai_text_detection_pile.csv")


def clean_HC3():
    # Columns: id, question, human_answers (JSON list), chatgpt_answers (JSON list), source
    # Each row has lists of answers -> need to explode into individual text rows
    df = pd.read_csv(HF_FOLDER / "HC3__all" / "train.csv")

    rows = []
    for _, row in df.iterrows():
        # human answers
        try:
            human_answers = json.loads(row["human_answers"])
            if isinstance(human_answers, list):
                for text in human_answers:
                    if isinstance(text, str) and text.strip():
                        rows.append({"text": text.strip(), "isGenerated": 0})
        except (json.JSONDecodeError, TypeError):
            pass

        # chatgpt answers
        try:
            chatgpt_answers = json.loads(row["chatgpt_answers"])
            if isinstance(chatgpt_answers, list):
                for text in chatgpt_answers:
                    if isinstance(text, str) and text.strip():
                        rows.append({"text": text.strip(), "isGenerated": 1})
        except (json.JSONDecodeError, TypeError):
            pass

    result = pd.DataFrame(rows)
    _save_to_disk(result, "cleaned_HC3.csv")


def clean_ai_human_detection_v1():
    # Columns: text, label (human/ai as string)
    # Has train.csv, validation.csv, test.csv - merge all
    dfs = []
    folder = HF_FOLDER / "ai-human-text-detection-v1"
    for csv_file in ["train.csv", "validation.csv", "test.csv"]:
        path = folder / csv_file
        if path.exists():
            dfs.append(pd.read_csv(path))

    df = pd.concat(dfs, ignore_index=True)
    labels = df["label"].astype(str).str.strip().str.lower()
    df["isGenerated"] = labels.map({"human": 0, "ai": 1})
    _save_to_disk(df, "cleaned_ai_human_detection_v1.csv")


# ========================
# RUN ALL
# ========================

if __name__ == "__main__":
    print("=== Cleaning Kaggle datasets ===")
    clean_AI_Human()
    clean_AI_Generated_Essays()
    clean_Training_Essay_Data()
    clean_student_vs_AI()
    clean_ai_vs_human_comparison()
    clean_aknjit()
    clean_algozee()
    clean_mostafabakr()
    clean_khushu89()

    print("\n=== Cleaning HuggingFace datasets ===")
    clean_ai_text_detection_pile()
    clean_HC3()
    clean_ai_human_detection_v1()

    print("\nDone! All cleaned files saved to:", CLEANED_FOLDER)