from pathlib import Path
import shutil
import kagglehub

HANDLE = "denvermagtibay/ai-generated-essays-dataset"
FILENAME = "AI Generated Essays Dataset.csv"  # exact name from your output

# where kagglehub cached it
dataset_dir = Path(kagglehub.dataset_download(HANDLE))
src_file = dataset_dir / FILENAME

# where your python file is
script_dir = Path(__file__).resolve().parent
dst_file = script_dir / FILENAME

shutil.copy2(src_file, dst_file)
print("Copied to:", dst_file)
