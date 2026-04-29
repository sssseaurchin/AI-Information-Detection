import argparse
import json
import shutil
import subprocess
from pathlib import Path

import gdown
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from label_config import load_label_mapping


HF_MIRROR_REPO = "ENSTA-U2IS/GenImage"

GENERATOR_CONFIGS = {
    "adm": {
        "drive_id": "1-9o163XaC-7L8Ch9-r7dLxbY_yjN8SVj",
        "archive_name": "imagenet_ai_0508_adm.zip",
        "dataset_slug": "adm",
        "hf_folder": "ADM",
    },
    "biggan": {
        "drive_id": "1ajlTuN34gLyJWxRQ6NyUcnkfrS8QEVKt",
        "archive_name": "imagenet_ai_0419_biggan.zip",
        "dataset_slug": "biggan",
        "hf_folder": "BigGAN",
    },
    "glide": {
        "drive_id": "1H2_4VPlla4OuKYU2CbMYx-8X0nbGJ2Mc",
        "archive_name": "imagenet_glide.zip",
        "dataset_slug": "glide",
        "hf_folder": "glide",
    },
    "sd14": {
        "drive_id": "12xighYOtu-ryfYEUnNrSeZqrxT8P08Zy",
        "archive_name": "imagenet_ai_0419_sdv4.zip",
        "dataset_slug": "stable_diffusion_v1_4",
        "hf_folder": "stable_diffusion_v_1_4",
    },
    "midjourney": {
        "drive_id": "1GLZedAqYuBh0kIQCZcbY_RspPJsR7ZyE",
        "archive_name": "imagenet_midjourney.zip",
        "dataset_slug": "midjourney",
        "hf_folder": "Midjourney",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare a controlled GenImage subset for external evaluation."
    )
    parser.add_argument(
        "--generator",
        required=True,
        choices=sorted(GENERATOR_CONFIGS.keys()),
        help="Generator subset to download and prepare.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="GenImage source split to prepare. Defaults to val for external eval.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root. Defaults to repo root inferred from this script.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded multi-part archives after preparation.",
    )
    return parser.parse_args()


def find_project_root(args: argparse.Namespace) -> Path:
    if args.project_root:
        return Path(args.project_root).resolve()
    return Path(__file__).resolve().parents[2]


def find_seven_zip() -> Path:
    candidates = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("7-Zip not found. Please install 7zip.7zip first.")


def download_generator_archives(generator: str, download_root: Path) -> Path:
    config = GENERATOR_CONFIGS[generator]
    download_root.mkdir(parents=True, exist_ok=True)
    try:
        files = gdown.download_folder(
            id=config["drive_id"],
            output=str(download_root),
            quiet=False,
            remaining_ok=True,
            resume=True,
        )
    except Exception as error:
        print(f"Google Drive download failed for '{generator}': {error}")
        print("Falling back to Hugging Face mirror.")
        files = download_from_huggingface_mirror(generator, download_root)
    if not files:
        raise RuntimeError(f"Failed to download GenImage folder for generator '{generator}'.")
    archive_path = download_root / config["archive_name"]
    if not archive_path.exists():
        raise FileNotFoundError(f"Expected archive not found after download: {archive_path}")
    return archive_path


def download_from_huggingface_mirror(generator: str, download_root: Path) -> list[str]:
    config = GENERATOR_CONFIGS[generator]
    hf_folder = str(config["hf_folder"])
    repo_files = list_repo_files(HF_MIRROR_REPO, repo_type="dataset")
    generator_files = [file_name for file_name in repo_files if file_name.startswith(f"{hf_folder}/")]
    if not generator_files:
        raise FileNotFoundError(
            f"No files found for generator '{generator}' in Hugging Face mirror folder '{hf_folder}'."
        )

    downloaded_files: list[str] = []
    for repo_file in sorted(generator_files):
        local_name = repo_file.split("/", 1)[1]
        target_path = download_root / local_name
        if target_path.exists():
            downloaded_files.append(str(target_path))
            continue
        cache_path = Path(
            hf_hub_download(
                repo_id=HF_MIRROR_REPO,
                repo_type="dataset",
                filename=repo_file,
            )
        )
        shutil.copy2(cache_path, target_path)
        downloaded_files.append(str(target_path))
    return downloaded_files


def extract_split_only(
    seven_zip: Path,
    archive_path: Path,
    extract_root: Path,
    split_name: str,
) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(seven_zip),
        "x",
        str(archive_path),
        f"-o{extract_root}",
        "-aoa",
        "-y",
        f"*\\{split_name}\\ai\\*",
        f"*\\{split_name}\\nature\\*",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"7-Zip extraction failed with exit code {result.returncode}.")


def locate_extracted_split_dirs(extract_root: Path, split_name: str) -> tuple[Path, Path]:
    ai_dirs = [p for p in extract_root.rglob("ai") if p.parent.name == split_name]
    nature_dirs = [p for p in extract_root.rglob("nature") if p.parent.name == split_name]
    if not ai_dirs or not nature_dirs:
        raise FileNotFoundError(
            f"Could not find extracted '{split_name}/ai' and '{split_name}/nature' directories in {extract_root}."
        )
    return ai_dirs[0], nature_dirs[0]


def copy_flattened_images(
    source_dir: Path,
    destination_dir: Path,
    category: str,
    generator_slug: str,
    split_name: str,
) -> list[dict[str, str]]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for source_path in sorted(source_dir.iterdir()):
        if not source_path.is_file():
            continue
        if source_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
            continue
        prefixed_name = f"genimage__{generator_slug}__{category}__{source_path.name}"
        target_path = destination_dir / prefixed_name
        if not target_path.exists():
            shutil.move(str(source_path), str(target_path))
        records.append(
            {
                "image_name": prefixed_name,
                "category": category,
                "dataset_id": "genimage",
                "domain": generator_slug,
                "source_split": split_name,
                "source_path": str(target_path),
            }
        )
    return records


def write_dataset_csv(dataset_root: Path, records: list[dict[str, str]]) -> Path:
    dataset_csv = dataset_root / "dataset.csv"
    dataframe = pd.DataFrame.from_records(
        records,
        columns=["image_name", "category", "dataset_id", "domain", "source_split", "source_path"],
    )
    dataframe.to_csv(dataset_csv, index=False)
    return dataset_csv


def write_split_manifest(project_root: Path, dataset_root: Path) -> Path:
    label_mapping = load_label_mapping(None)
    mapping_json = json.dumps(label_mapping, sort_keys=True)
    dataframe = pd.read_csv(dataset_root / "dataset.csv")
    manifest_records: list[dict[str, object]] = []

    for _, row in dataframe.iterrows():
        sample_path = (dataset_root / str(row["category"]) / str(row["image_name"])).resolve()
        relative_to_project = sample_path.relative_to(project_root).as_posix()
        manifest_records.append(
            {
                "path": f"/app/{relative_to_project}",
                "category": str(row["category"]),
                "label": int(label_mapping[str(row["category"])]),
                "split": "test",
                "group_id": "",
                "dataset_id": str(row["dataset_id"]),
                "domain": str(row["domain"]),
                "label_mapping": mapping_json,
            }
        )

    manifest_path = dataset_root / "split_manifest.csv"
    manifest = pd.DataFrame.from_records(
        manifest_records,
        columns=["path", "category", "label", "split", "group_id", "dataset_id", "domain", "label_mapping"],
    )
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def cleanup_temp_path(temp_path: Path) -> None:
    if temp_path.exists():
        shutil.rmtree(temp_path)


def cleanup_archives(download_root: Path) -> None:
    for file_path in download_root.iterdir():
        if file_path.is_file():
            file_path.unlink()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(args)
    seven_zip = find_seven_zip()
    config = GENERATOR_CONFIGS[args.generator]
    generator_slug = config["dataset_slug"]

    download_root = project_root / "Dataset" / "raw" / "genimage_subset" / generator_slug
    extract_root = project_root / "Dataset" / "raw" / "genimage_subset_extracted" / generator_slug
    dataset_root = project_root / "Dataset" / "prepared" / "external_eval" / f"genimage_{generator_slug}_{args.split}"

    archive_path = download_generator_archives(args.generator, download_root)
    extract_split_only(seven_zip, archive_path, extract_root, args.split)
    ai_dir, nature_dir = locate_extracted_split_dirs(extract_root, args.split)

    fake_dir = dataset_root / "fake"
    real_dir = dataset_root / "real"
    dataset_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    records.extend(copy_flattened_images(ai_dir, fake_dir, "fake", generator_slug, args.split))
    records.extend(copy_flattened_images(nature_dir, real_dir, "real", generator_slug, args.split))

    if not records:
        raise RuntimeError(f"No images were prepared for generator '{args.generator}'.")

    dataset_csv = write_dataset_csv(dataset_root, records)
    manifest_path = write_split_manifest(project_root, dataset_root)

    cleanup_temp_path(extract_root)
    if not args.keep_archives:
        cleanup_archives(download_root)

    print(f"Prepared dataset: {dataset_root}")
    print(f"dataset.csv: {dataset_csv}")
    print(f"split_manifest.csv: {manifest_path}")
    print(f"rows: {len(records)}")


if __name__ == "__main__":
    main()
