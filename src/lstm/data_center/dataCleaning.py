import json
import re
import pandas as pd
from pathlib import Path

# All cleaning functions normalize to: text, isGenerated (0=Human, 1=AI)

BASE_DIR = Path(__file__).resolve().parent
KAGGLE_FOLDER = BASE_DIR / "external_datasets" / "raw_datas" / "kaggle"
HF_FOLDER = BASE_DIR / "external_datasets" / "raw_datas" / "huggingface"
CLEANED_FOLDER = BASE_DIR / "external_datasets" / "cleaned_datas"
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
# OUR DATA — AI CLEANING
# ========================

OUR_DATA_DIR = BASE_DIR / "ourDatas"
AI_RAW_CSV = OUR_DATA_DIR / "raw_datas" / "ai_text" / "ai_all.csv"
OUR_CLEANED_DIR = OUR_DATA_DIR / "cleaned_datas"
OUR_CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# Patterns for markdown removal
_MD_HEADING = re.compile(r"(?m)^#{1,6} .+\n?")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_MD_LIST_ITEM = re.compile(r"(?m)^[-*] ")

# Opening/closing meta sentences produced by the LLM
_META_OPENING = re.compile(
    r"(?i)^(?:here(?:'?s| is)(?: the| a| my)?(?: rewritten| revised| paraphrased| expanded| following| essay| text| passage| version)?[:\s]*\n?)"
)
_META_CLOSING = re.compile(
    r"(?i)(?:i hope (?:this|you)[^.]*\.|feel free to[^.]*\.|let me know if[^.]*\.|please note[^.]*\.)\s*$"
)
_META_PREAMBLE = re.compile(
    r"(?i)^(?:this (?:is a |is an )?(?:rewritten|revised|paraphrased|essay|text|passage)[^.]*\.\s*)"
)

# Filler openings at the very start of the text
_FILLER_OPENING = re.compile(
    r"(?i)^(?:so[,\s]+(?:basically[,\s]+|here'?s?\s+the\s+thing[,\s]*|anyway[,\s]+|yeah[,\s]+)?)"
)


def _remove_markdown(text: str) -> tuple[str, bool]:
    """Strip markdown artefacts. Returns (cleaned_text, was_changed)."""
    original = text
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_LIST_ITEM.sub("", text)
    return text, text != original


def _remove_meta_sentences(text: str) -> tuple[str, bool]:
    """Remove LLM opening/closing boilerplate. Returns (cleaned_text, was_changed)."""
    original = text
    text = _META_OPENING.sub("", text)
    text = _META_PREAMBLE.sub("", text)
    text = _META_CLOSING.sub("", text)
    return text.strip(), text.strip() != original.strip()


def _remove_filler_opening(text: str) -> tuple[str, bool]:
    """Remove filler openings like 'So, ', 'So basically, ' from the start. Returns (cleaned_text, was_changed)."""
    cleaned = _FILLER_OPENING.sub("", text).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned, cleaned != text.strip()


def clean_ai_data():
    """
    Clean AI-generated text data from ai_all.csv.

    Steps:
      1. Remove markdown headings, bold, italic, list markers
      2. Remove LLM meta opening/closing boilerplate sentences
      3. Normalize whitespace (multiple spaces/newlines -> single space)
      4. Filter rows with word count < 50 (too short after cleaning)
      5. Flag rows with word count > 600 (log but keep)
      6. Apply human-data consistency checks:
         - Strip data-leakage labels [AI-...] / [Human-...]
         - Remove encoding artifacts

    Saves to: ourDatas/cleaned_datas/cleaned_ai_all.csv
    Columns preserved: text, isGenerated, source, prompt_used, temperature
    """
    df = pd.read_csv(AI_RAW_CSV)
    total_in = len(df)
    print(f"\n=== clean_ai_data() ===")
    print(f"Gelen satir sayisi: {total_in}")

    # ---- 1. Markdown temizleme ----
    md_cleaned_count = 0
    texts_md = []
    for t in df["text"].astype(str):
        cleaned, changed = _remove_markdown(t)
        texts_md.append(cleaned)
        if changed:
            md_cleaned_count += 1
    df["text"] = texts_md

    # ---- 2. Meta cumle temizleme ----
    meta_cleaned_count = 0
    texts_meta = []
    for t in df["text"]:
        cleaned, changed = _remove_meta_sentences(t)
        texts_meta.append(cleaned)
        if changed:
            meta_cleaned_count += 1
    df["text"] = texts_meta

    # ---- 2b. Filler acilis temizleme ("So, ", "So basically, " vb.) ----
    filler_cleaned_count = 0
    texts_filler = []
    for t in df["text"]:
        cleaned, changed = _remove_filler_opening(t)
        texts_filler.append(cleaned)
        if changed:
            filler_cleaned_count += 1
    df["text"] = texts_filler

    # ---- 3. Data-leakage etiket temizleme (human verisindeki gibi) ----
    df["text"] = df["text"].str.replace(r"\[AI[^\]]*\]", "", regex=True)
    df["text"] = df["text"].str.replace(r"\[Human[^\]]*\]", "", regex=True)

    # ---- 4. Whitespace normalizasyonu ----
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # ---- 5. Bos/null dusur ----
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]

    # ---- 6. Kelime sayisi filtresi ----
    df["_wc"] = df["text"].str.split().str.len()
    before_filter = len(df)
    too_short = df["_wc"] < 50
    too_long_count = (df["_wc"] > 600).sum()
    df = df[~too_short]
    filtered_short = before_filter - len(df)

    if too_long_count > 0:
        print(f"  UYARI: {too_long_count} satir >600 kelime — korunuyor ama flaglendi.")

    df = df.drop(columns=["_wc"])

    # ---- 7. Kaydet ----
    out_path = OUR_CLEANED_DIR / "cleaned_ai_all.csv"
    df.to_csv(out_path, index=False)

    # ---- 8. Rapor ----
    total_out = len(df)
    word_counts = df["text"].str.split().str.len()
    print(f"\n--- TEMİZLEME RAPORU ---")
    print(f"  Gelen satir         : {total_in}")
    print(f"  Kalan satir         : {total_out}")
    print(f"  Markdown temizlenen : {md_cleaned_count}")
    print(f"  Meta cumle temizlenen: {meta_cleaned_count}")
    print(f"  Filler acilis temizlenen: {filler_cleaned_count}")
    print(f"  Filtrelenen (kisa)  : {filtered_short}  (<50 kelime)")
    print(f"  Temizleme sonrasi kelime sayisi:")
    print(f"    min={word_counts.min()}, max={word_counts.max()}, "
          f"mean={word_counts.mean():.1f}, median={word_counts.median():.1f}, "
          f"std={word_counts.std():.1f}")
    print(f"  Kaydedildi          : {out_path}")
    print(f"------------------------")


# ========================
# OUR DATA — FINAL DATASET
# ========================

HUMAN_CLEANED_CSV = OUR_CLEANED_DIR / "cleaned_human_all.csv"
AI_CLEANED_CSV    = OUR_CLEANED_DIR / "cleaned_ai_all.csv"
FINAL_CSV         = OUR_CLEANED_DIR / "final_dataset.csv"


def _check_dataframe(df: pd.DataFrame, label: str) -> None:
    """Print per-file quality summary."""
    wc = df["text"].str.split().str.len()
    null_count = df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum()
    dup_count  = df.duplicated(subset=["text"]).sum()
    is_gen_vals = df["isGenerated"].unique().tolist()
    print(f"  [{label}]")
    print(f"    Satir sayisi  : {len(df)}")
    print(f"    Null/bos text : {null_count}")
    print(f"    Duplicate text: {dup_count}")
    print(f"    isGenerated   : {is_gen_vals}")
    print(f"    Kelime sayisi : min={wc.min()}, max={wc.max()}, "
          f"mean={wc.mean():.1f}, median={wc.median():.1f}, std={wc.std():.1f}")


def build_final_ourDataset() -> None:
    """
    Merge cleaned human and AI data into a single shuffled final dataset.

    Steps:
      1. Load and validate both cleaned CSVs
      2. Concat — keep text, isGenerated + metadata columns where available
      3. Cross-duplicate check (same text in both classes)
      4. Shuffle (random_state=42)
      5. Save to cleaned_datas/final_dataset.csv
      6. Print summary report
    """
    print(f"\n=== build_final_ourDataset() ===")

    # ---- 1. Yükle ve kontrol et ----
    print("\n-- Ön kontrol --")
    human_df = pd.read_csv(HUMAN_CLEANED_CSV)
    ai_df    = pd.read_csv(AI_CLEANED_CSV)
    _check_dataframe(human_df, "cleaned_human_all.csv")
    _check_dataframe(ai_df,    "cleaned_ai_all.csv")

    # ---- 2. Birleştir ----
    # Her iki df'de ortak olan sütunları koru; eksik olanlar NaN kalır
    combined = pd.concat([human_df, ai_df], ignore_index=True)

    # isGenerated int olsun
    combined["isGenerated"] = combined["isGenerated"].astype(int)

    # ---- 3. Cross-duplicate kontrolü ----
    texts_human = set(human_df["text"].astype(str).str.strip())
    texts_ai    = set(ai_df["text"].astype(str).str.strip())
    cross_dups  = texts_human & texts_ai

    # ---- 4. Shuffle ----
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---- 5. Kaydet ----
    combined.to_csv(FINAL_CSV, index=False)

    # ---- 6. Özet rapor ----
    wc_human = combined.loc[combined["isGenerated"] == 0, "text"].str.split().str.len()
    wc_ai    = combined.loc[combined["isGenerated"] == 1, "text"].str.split().str.len()
    n_human  = (combined["isGenerated"] == 0).sum()
    n_ai     = (combined["isGenerated"] == 1).sum()

    print(f"\n--- FINAL DATASET RAPORU ---")
    print(f"  Toplam satir        : {len(combined)}")
    print(f"  isGenerated=0 (human): {n_human}  ({n_human/len(combined)*100:.1f}%)")
    print(f"  isGenerated=1 (AI)  : {n_ai}  ({n_ai/len(combined)*100:.1f}%)")
    print(f"  Cross-duplicate     : {len(cross_dups)}")
    print(f"  Sutunlar            : {combined.columns.tolist()}")
    print(f"  Kelime sayisi (human): min={wc_human.min()}, max={wc_human.max()}, mean={wc_human.mean():.1f}")
    print(f"  Kelime sayisi (AI)  : min={wc_ai.min()}, max={wc_ai.max()}, mean={wc_ai.mean():.1f}")
    print(f"  Kaydedildi          : {FINAL_CSV}")
    print(f"----------------------------")


# ========================
# RUN ALL
# ========================

if __name__ == "__main__":
    # print("=== Cleaning Kaggle datasets ===")
    # clean_AI_Human()
    # clean_AI_Generated_Essays()
    # clean_Training_Essay_Data()
    # clean_student_vs_AI()
    # clean_ai_vs_human_comparison()
    # clean_aknjit()
    # clean_algozee()
    # clean_mostafabakr()
    # clean_khushu89()

    # print("\n=== Cleaning HuggingFace datasets ===")
    # clean_ai_text_detection_pile()
    # clean_HC3()
    # clean_ai_human_detection_v1()

    print("=== Cleaning our AI data ===")
    clean_ai_data()

    print("\n=== Building final dataset ===")
    build_final_ourDataset()

    print("\nDone!")
