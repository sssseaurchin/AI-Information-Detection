from pathlib import Path
import shutil
import kagglehub
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def kaggle_download(handle: str, filename: str) -> Path:
    """
    Downloads via kagglehub cache, then copies the requested file into ./data/.
    Skips if the destination file already exists.
    """
    dst_file = DATA_DIR / filename
    if dst_file.exists() and dst_file.stat().st_size > 0:
        print(f"[Kaggle] Already exists, skipping: {dst_file}")
        return dst_file

    dataset_dir = Path(kagglehub.dataset_download(handle))
    src_file = dataset_dir / filename
    if not src_file.exists():
        raise FileNotFoundError(f"[Kaggle] File not found in cache: {src_file}")

    shutil.copy2(src_file, dst_file)
    print(f"[Kaggle] Copied to: {dst_file}")
    return dst_file

def hugging_face_download(name: str, save_name: str | None = None) -> Path:
    """
    Downloads a HF dataset into ./data/hf_cache (cache) and saves an on-disk copy
    into ./data/datasets/<save_name or last part of repo>.
    Skips saving if the on-disk copy already exists.
    """
    hf_cache = DATA_DIR / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    out_dir_name = save_name or name.split("/")[-1]
    out_dir = DATA_DIR / "datasets" / out_dir_name

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[HF] Already exists, skipping save_to_disk: {out_dir}")
        return out_dir

    ds = load_dataset(name, cache_dir=str(hf_cache))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

    print(f"[HF] Cache dir: {hf_cache}")
    print(f"[HF] Saved to: {out_dir}")
    return out_dir

if __name__ == "__main__":
    print("Starting Downloader")

    # Kaggle
    kaggle_download("shanegerami/ai-vs-human-text", "AI_Human.csv")

    # Hugging Face
    hugging_face_download("artem9k/ai-text-detection-pile")
