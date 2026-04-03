import pandas as pd
from pathlib import Path
import logging

DATA_FOLDER = Path("data")
CLEANED_DATA_FOLDER = DATA_FOLDER / "cleaned"


def load_cleaned_csvs(data_dir: Path = CLEANED_DATA_FOLDER) -> pd.DataFrame:
    """
    Load and combine all CSV files starting with 'cleaned_' from data/cleaned/.
    Returns a combined pandas DataFrame.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        logging.error(f"{data_path} does not exist.")
        str = f"{data_path} does not exist."
        raise FileNotFoundError(str)

    csv_files = sorted(data_path.rglob("cleaned_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No files matching cleaned_*.csv in {data_path}")

    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        print(f"Loading {file.name} to DataFrame. Row count of dataset: {len(df)}")
        df["source_file"] = file.name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined DataFrame row count: {len(combined_df)}")
    return combined_df


def _log_dataset_info(data_dir: Path = CLEANED_DATA_FOLDER):

    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"{data_path} does not exist.")
        return

    csv_files = sorted(data_path.rglob("cleaned_*.csv"))

    if not csv_files:
        print("No cleaned CSV files were found. They should start with cleaned_")
        return

    total_rows = 0
    total_size_bytes = 0

    print("\nDATASET FILES")
    for file in csv_files:
        size_bytes = file.stat().st_size
        df = pd.read_csv(file)

        size_gb = size_bytes / (1024**3)

        total_rows += len(df)
        total_size_bytes += size_bytes

        print(file.name)
        print(f"  rows: {len(df)} \nsize: {size_gb:.4f} GB\n")

    total_size_gb = total_size_bytes / (1024**3)

    print("--- THEORITICAL COMBINED DATASET ---")
    print(f"total files: {len(csv_files)}")
    print(f"total rows: {total_rows}")
    print(f"total size: {total_size_gb:.4f} GB")


if __name__ == "__main__":
    df = load_cleaned_csvs()
    # _log_dataset_info()
