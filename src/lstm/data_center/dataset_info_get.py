from pathlib import Path
import pandas as pd


def summarize_csvs(csv_paths: list[str]) -> pd.DataFrame:
    rows = []

    for csv_path in csv_paths:
        path = Path(csv_path)

        if not path.exists():
            rows.append({
                "csv_name": path.name,
                "size_in_gb": None,
                "row_count": None,
                "isGenerated_1_count": None,
                "isGenerated_0_count": None,
                "avg_text_length": None,
                "error": "file not found",
            })
            continue

        try:
            df = pd.read_csv(path)

            size_gb = round(path.stat().st_size / (1024 ** 3), 6)
            row_count = len(df)

            # count labels
            if "isGenerated" in df.columns:
                labels = pd.to_numeric(df["isGenerated"], errors="coerce")
                count_1 = int((labels == 1).sum())
                count_0 = int((labels == 0).sum())
            else:
                count_1 = None
                count_0 = None

            # average text length
            if "text" in df.columns:
                avg_text_length = float(df["text"].astype(str).str.len().mean())
            else:
                avg_text_length = None

            rows.append({
                "csv_name": path.name,
                "size_in_gb": size_gb,
                "row_count": row_count,
                "isGenerated_1_count": count_1,
                "isGenerated_0_count": count_0,
                "avg_text_length": avg_text_length 
            })

        except Exception as e:
            rows.append({
                "csv_name": path.name,
                "size_in_gb": round(path.stat().st_size / (1024 ** 3), 6),
                "row_count": None,
                "isGenerated_1_count": None,
                "isGenerated_0_count": None,
                "avg_text_length": None 
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    csv_files = [
        "data/AI_Human_cleaned.csv",
        "data/AI_Generated_Essays_cleaned.csv"
    ]

    summary_df = summarize_csvs(csv_files)
    print(summary_df.to_string(index=False))
