from pathlib import Path
import shutil
import kagglehub
import logging
from datasets import load_dataset
import json


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_FILE = "kaggle.json"
HUGGING_FACE_FILE = "hugging_face.json"

def _list_all_kaggle() -> list[dict]:
    json_path =  Path(KAGGLE_FILE)
    with json_path.open("r", encoding="utf-8") as json:
        return json.load(json)

def _list_all_hugging_face()-> list[dict]:
    json_path =  Path(HUGGING_FACE_FILE)
    with json_path.open("r", encoding="utf-8") as json:
        return json.load(json)

def _download_all_kaggle():
    list = _list_all_kaggle()
    
    for dic in list:
        _kaggle_download(handle=dic["handle"], filename=dic["file_name"])

def _download_all_hugging_face():
    list = _list_all_kaggle()
    
    for dic in list:
        _hugging_face_download(handle=dic["url"], filename=dic["file_name"])
    


def _kaggle_download(handle: str, filename: str) -> Path: 
    
    file = DATA_DIR / filename
    if file.exists() and file.stat().st_size > 0: # if file exists and larger than 1 bit
        logging.info(f"[Kaggle] Already exists, skipping: {file}")
        return file

    dataset_dir = Path(kagglehub.dataset_download(handle))
    src_file = dataset_dir / filename # cache

    if not src_file.exists():
        raise FileNotFoundError(f"[Kaggle] File not found in cache: {src_file} ???")

    shutil.copy2(src_file, file) # does this mean we have the downloaded files twice?
    logging.info(f"[Kaggle] Copied to: {file}") 
    return file

def _hugging_face_download(name: str, save_name: str | None = None, subset = None) -> Path:
    hf_cache = DATA_DIR / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    out_dir_name = save_name or name.split("/")[-1]
    out_dir = DATA_DIR / "datasets" / out_dir_name

    if out_dir.exists() and any(out_dir.iterdir()): 
        logging.info(f"[HF] Already exists, skipping save_to_disk: {out_dir}")
        return out_dir
    
    if subset:
        dataset = load_dataset(name, cache_dir=str(hf_cache), subset=subset)
    else:
        dataset = load_dataset(name, cache_dir=str(hf_cache))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))

    # print(f"[HF] Cache dir: {hf_cache}")
    logging.info(f"[HF] Saved to: {out_dir}")
    return out_dir

if __name__ == "__main__":
    print("Starting Downloader")

    # Kaggle
    _kaggle_download("shanegerami/ai-vs-human-text", "AI_Human.csv")

    # Hugging Face
    _hugging_face_download("artem9k/ai-text-detection-pile")
