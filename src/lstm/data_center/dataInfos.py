import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CLEANED_FOLDER = BASE_DIR / "external_datasets" / "cleaned_datas"


def analyze_single(path: Path) -> dict:
    """Prints detailed analysis for a single cleaned CSV. Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(path)

    # class distribution
    total = len(df)
    ai = int((df["isGenerated"] == 1).sum())
    human = int((df["isGenerated"] == 0).sum())
    other = total - ai - human

    print(f"\nRows: {total}")
    print(f"  AI:    {ai} ({ai/total*100:.1f}%)")
    print(f"  Human: {human} ({human/total*100:.1f}%)")
    if other > 0:
        print(f"  Other/NaN: {other}")

    # missing values
    null_text = int(df["text"].isna().sum())
    null_label = int(df["isGenerated"].isna().sum())
    if null_text > 0 or null_label > 0:
        print(f"\nMissing values: text={null_text}, isGenerated={null_label}")

    # duplicates
    dup_count = int(df.duplicated(subset=["text", "isGenerated"]).sum())
    dup_text_only = int(df.duplicated(subset=["text"]).sum())
    print(f"\nDuplicates: {dup_count} (exact) | {dup_text_only} (text only)")

    # text length stats
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    df["char_count"] = df["text"].astype(str).str.len()

    print(f"\nWord count:")
    print(f"  min={int(df['word_count'].min())}  max={int(df['word_count'].max())}  "
          f"mean={df['word_count'].mean():.1f}  median={df['word_count'].median():.1f}")

    print(f"Char count:")
    print(f"  min={int(df['char_count'].min())}  max={int(df['char_count'].max())}  "
          f"mean={df['char_count'].mean():.1f}  median={df['char_count'].median():.1f}")

    # per-class text length
    for label, name in [(1, "AI"), (0, "Human")]:
        subset = df[df["isGenerated"] == label]
        if len(subset) > 0:
            print(f"  {name} avg words: {subset['word_count'].mean():.1f}")

    # free memory
    summary = {"dataset": path.stem, "total": total, "AI": ai, "Human": human, "AI%": round(ai / total * 100, 1)}
    del df
    return summary


def analyze_all():
    """Analyzes all cleaned CSVs and prints a summary table at the end."""
    csv_files = sorted(CLEANED_FOLDER.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {CLEANED_FOLDER}")
        return

    summary_rows = []

    for path in csv_files:
        summary = analyze_single(path)
        summary_rows.append(summary)

    # summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    # grand total
    total_rows = summary["total"].sum()
    total_ai = summary["AI"].sum()
    total_human = summary["Human"].sum()
    print(f"\nGrand total: {total_rows} rows | AI={total_ai} ({total_ai/total_rows*100:.1f}%) | Human={total_human} ({total_human/total_rows*100:.1f}%)")


if __name__ == "__main__":
    analyze_all()