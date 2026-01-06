from pathlib import Path
import shutil
import kagglehub
import logging
logging.basicConfig(level=logging.INFO)
from datasets import load_dataset
import json


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_FILE = "kaggle.json"
HUGGING_FACE_FILE = "hugging_face.json"

def _list_all_kaggle() -> list[dict]:
    json_path =  Path(KAGGLE_FILE)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)

def _list_all_hugging_face()-> list[dict]:
    json_path =  Path(HUGGING_FACE_FILE)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)

def _download_all_kaggle():
    list = _list_all_kaggle()
    
    for dic in list:
        _kaggle_download(handle=dic["handle"], filename=dic["file_name"])

def _download_all_hugging_face():
    list = _list_all_hugging_face()
    
    for dic in list:
        _hugging_face_download(name=dic["name"])
    


def _kaggle_download(handle: str, filename: str) -> Path: 
    
    file = DATA_DIR / filename
    if file.exists() and file.stat().st_size > 0: # if file exists and larger than 1 bit
        logging.info(f"[Kaggle] Already exists, skipping: {handle}")
        return file

    dataset_dir = Path(kagglehub.dataset_download(handle))
    src_file = dataset_dir / filename # cache

    if not src_file.exists():
        raise FileNotFoundError(f"[Kaggle] File not found in cache: {src_file} ???")

    shutil.copy2(src_file, file) # does this mean we have the downloaded files twice?
    logging.info(f"[Kaggle] Copied to: {file}") 
    return file

def _hugging_face_download(
    name: str,
    save_name: str | None = None,
    subset: str | None = None,
) -> Path:
    hf_cache = DATA_DIR / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    out_dir_name = save_name or name.split("/")[-1]
    if subset:
        out_dir_name = f"{out_dir_name}__{subset}"
    out_dir = DATA_DIR / "datasets" / out_dir_name

    if out_dir.exists() and any(out_dir.iterdir()):
        logging.info(f"[HF] Already exists, skipping save_to_disk: {name}")
        return out_dir

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        # "subset" here is HF config name, passed as `name=...`
        ds = load_dataset(name, name=subset, cache_dir=str(hf_cache)) if subset else \
             load_dataset(name, cache_dir=str(hf_cache))

    except RuntimeError as e:
        # datasets>=4.0.0 blocks script-based datasets; fall back to parquet conversion branch
        if "Dataset scripts are no longer supported" not in str(e):
            raise

        logging.warning(f"[HF] Script-based dataset blocked; falling back to refs/convert/parquet: {name}")

        # HC3 and similar datasets store configs as folders under refs/convert/parquet (e.g., all/, finance/, ...)
        parquet_glob = (
            f"hf://datasets/{name}@refs/convert/parquet/{subset}/**/*.parquet"
            if subset else
            f"hf://datasets/{name}@refs/convert/parquet/**/*.parquet"
        )

        ds = load_dataset(
            "parquet",
            data_files=parquet_glob,
            cache_dir=str(hf_cache),
        )

    ds.save_to_disk(str(out_dir))
    logging.info(f"[HF] Saved to: {out_dir}")
    return out_dir

if __name__ == "__main__":
    print("Starting Downloader") 
    _download_all_kaggle()
    _download_all_hugging_face()
    # prayers : me