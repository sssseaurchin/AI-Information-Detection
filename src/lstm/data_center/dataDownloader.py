from pathlib import Path
import shutil
import kagglehub
import logging
logging.basicConfig(level=logging.INFO)
from datasets import load_dataset, Dataset, DatasetDict
import json


# HUMAN = 0; AI = 1;    

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_FILE = "kaggle.json"
HUGGING_FACE_FILE = "hugging_face.json"


### KAGGLE
def _list_all_kaggle() -> list[dict]:
    json_path =  Path(KAGGLE_FILE)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)

def download_all_kaggle():
    list = _list_all_kaggle()
    
    for dic in list:
        _kaggle_download(handle=dic["handle"], filename=dic["file_name"])
def _kaggle_download(handle: str, filename: str) -> Path:
    out_file = DATA_DIR / filename

    # ensure destination directories exist (handles filename containing subfolders)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and out_file.stat().st_size > 0:
        logging.info(f"[Kaggle] Already exists, skipping: {handle}")
        return out_file

    dataset_dir = Path(kagglehub.dataset_download(handle))

    # 1) Exact match
    src_exact = dataset_dir / Path(filename).name  # important: match by basename, not subpath
    if src_exact.exists() and src_exact.is_file():
        shutil.copy2(src_exact, out_file)
        logging.info(f"[Kaggle] Copied exact match to: {out_file}")
        return out_file

    # 2) Auto-discover candidates
    candidates = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
    ]

    if not candidates:
        logging.warning(f"[Kaggle] No data files found in {dataset_dir}. Creating empty file: {out_file}")
        out_file.touch()
        return out_file

    def score(p: Path) -> tuple:
        name = p.name.lower()
        ext_priority = {".csv": 0, ".tsv": 1, ".parquet": 2, ".jsonl": 3, ".json": 4}.get(p.suffix.lower(), 9)
        name_priority = 0
        if name in {"train.csv", "data.csv", "dataset.csv"}:
            name_priority = -2
        elif "train" in name:
            name_priority = -1
        return (ext_priority, name_priority, -p.stat().st_size)

    candidates.sort(key=score)
    src = candidates[0]

    shutil.copy2(src, out_file)
    logging.warning(
        f"[Kaggle] Exact file not found ({Path(filename).name}). "
        f"Copied best match '{src.relative_to(dataset_dir)}' -> '{out_file}'."
    )
    return out_file

### HUGGING FACE
def _list_all_hugging_face()-> list[dict]:
    json_path =  Path(HUGGING_FACE_FILE)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def download_all_hugging_face():
    list = _list_all_hugging_face()
    logging.info(f"Downloading from list: {list}")
    
    for dic in list:
        _hugging_face_download(name=dic["name"])


def _hugging_face_download(
    name: str,
    filename: str | None = None,
    subset: str | None = None,
) -> Path:
    hf_cache = DATA_DIR / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    out_dir_name = filename or name.split("/")[-1]
    if subset:
        out_dir_name = f"{out_dir_name}__{subset}"
    out_dir = DATA_DIR / "datasets" / out_dir_name

    if out_dir.exists() and any(out_dir.iterdir()):
        logging.info(f"[HF] Already exists, skipping save_to_disk: {name}")
        return out_dir

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(name, name=subset, cache_dir=str(hf_cache)) if subset else \
             load_dataset(name, cache_dir=str(hf_cache))

    except RuntimeError as e:
        # datasets>=4.0.0 blocks script-based datasets; fall back to parquet conversion branch
        if "Dataset scripts are no longer supported" not in str(e):
            raise

        logging.warning(f"[HF] Script-based dataset blocked; falling back to refs/convert/parquet: {name}")

        # HC3 and similar datasets store configs as folders under refs/convert/parquet (e.g., all/, finance/, ...)
        data_files = (
            f"hf://datasets/{name}@refs/convert/parquet/{subset}/**/*.parquet"
            if subset else
            f"hf://datasets/{name}@refs/convert/parquet/**/*.parquet"
        )

        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            cache_dir=str(hf_cache),
        )
    def __jsonify_complex_columns(ds: Dataset) -> Dataset: # yeah a function inside of a function
        """
        CSV can't store list/dict columns well. Convert list/dict values to JSON strings.
        """
        def convert_row(row):
            for k, v in row.items():
                if isinstance(v, (list, dict)):
                    row[k] = json.dumps(v, ensure_ascii=False)
            return row

        # map is lazy-ish; this will materialize when writing
        return ds.map(convert_row)
    
     # Export to CSV
    if isinstance(dataset, DatasetDict):
        for split, split_ds in dataset.items():
            split_ds = __jsonify_complex_columns(split_ds)
            csv_path = out_dir / f"{split}.csv"
            split_ds.to_csv(str(csv_path), index=False)
            logging.info(f"[HF] Wrote CSV: {csv_path}")
    elif isinstance(dataset, Dataset):
        dataset = __jsonify_complex_columns(dataset)
        csv_path = out_dir / "data.csv"
        dataset.to_csv(str(csv_path), index=False)
        logging.info(f"[HF] Wrote CSV: {csv_path}")
    else:
        # Very rare, but keep it explicit
        raise TypeError(f"Unexpected dataset type: {type(dataset)}")

    logging.info(f"[HF] Saved CSVs to: {out_dir}")
    return out_dir

if __name__ == "__main__":
    print("Starting Downloader") 
    # _kaggle_download(handle="khushu89/human-vs-ai-text-classification-dataset", filename="human-vs-ai-text-classification-dataset.csv")
    # download_all_kaggle()
    download_all_hugging_face()