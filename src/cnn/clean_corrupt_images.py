import argparse
import csv
import os
import shutil
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
APP_PREFIX = "/app/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarantine corrupt training images and refresh dataset metadata.")
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("DATASET_PATH", "/app/Dataset/prepared/combined_train"),
        help="Dataset root containing fake/real folders plus dataset.csv and split_manifest.csv.",
    )
    parser.add_argument(
        "--quarantine-root",
        default=None,
        help="Where corrupt files should be moved. Defaults to a sibling quarantine folder under Dataset/prepared.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Do not move files; only generate suspicious/corrupt reports.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=89_478_485,
        help="Flag images above this pixel count as suspicious in the warning report.",
    )
    return parser.parse_args()


def get_project_root(dataset_path: Path) -> Path:
    return dataset_path.parents[2]


def get_default_quarantine_root(dataset_path: Path) -> Path:
    prepared_root = dataset_path.parent
    return prepared_root / "quarantine" / "corrupt_images" / dataset_path.name


def iter_image_files(dataset_path: Path) -> list[Path]:
    image_files: list[Path] = []
    for path in dataset_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(path)
    return sorted(image_files)


def validate_image(image_path: Path, max_pixels: int) -> tuple[bool, str, list[str]]:
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    warning_messages: list[str] = []

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Image.open(image_path) as image:
                width, height = image.size
                if width * height > max_pixels:
                    warning_messages.append(f"pixel_count_exceeds_threshold:{width * height}")
                image.verify()
            with Image.open(image_path) as image:
                image.load()

        for caught_warning in caught:
            warning_type = caught_warning.category.__name__
            warning_text = str(caught_warning.message).strip()
            warning_messages.append(f"{warning_type}:{warning_text}")

        return True, "", warning_messages
    except Exception as exc:  # Pillow raises multiple exception types for broken files.
        return False, str(exc).strip() or exc.__class__.__name__, warning_messages


def quarantine_image(image_path: Path, dataset_path: Path, quarantine_root: Path) -> Path:
    relative_path = image_path.relative_to(dataset_path)
    destination = quarantine_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(image_path), str(destination))
    return destination


def refresh_dataset_csv(dataset_path: Path) -> tuple[int, int]:
    csv_path = dataset_path / "dataset.csv"
    if not csv_path.exists():
        return 0, 0

    dataframe = pd.read_csv(csv_path)
    before = len(dataframe)
    keep_mask = dataframe.apply(
        lambda row: (dataset_path / str(row["category"]) / str(row["image_name"])).exists(),
        axis=1,
    )
    filtered = dataframe[keep_mask].copy()
    filtered.to_csv(csv_path, index=False)
    return before, len(filtered)


def resolve_manifest_path(sample_path: str, project_root: Path) -> Path:
    if os.path.exists(sample_path):
        return Path(sample_path)

    normalized = sample_path.replace("\\", "/")
    if normalized.startswith(APP_PREFIX):
        relative = normalized[len(APP_PREFIX):]
        return project_root / Path(relative)

    return Path(sample_path)


def refresh_split_manifest(dataset_path: Path) -> tuple[int, int]:
    manifest_path = dataset_path / "split_manifest.csv"
    if not manifest_path.exists():
        return 0, 0

    project_root = get_project_root(dataset_path)
    dataframe = pd.read_csv(manifest_path)
    before = len(dataframe)
    keep_mask = dataframe["path"].astype(str).map(lambda value: resolve_manifest_path(value, project_root).exists())
    filtered = dataframe[keep_mask].copy()
    filtered.to_csv(manifest_path, index=False)
    return before, len(filtered)


def write_report(report_path: Path, records: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["original_path", "quarantine_path", "reason"])
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_suspicious_report(report_path: Path, records: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "warning"])
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).resolve()
    quarantine_root = Path(args.quarantine_root).resolve() if args.quarantine_root else get_default_quarantine_root(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    image_files = iter_image_files(dataset_path)
    quarantined_records: list[dict[str, str]] = []
    suspicious_records: list[dict[str, str]] = []

    print(f"Scanning {len(image_files)} image files under: {dataset_path}")
    for index, image_path in enumerate(image_files, start=1):
        is_valid, reason, warning_messages = validate_image(image_path, args.max_pixels)
        for warning_message in warning_messages:
            suspicious_records.append(
                {
                    "path": str(image_path),
                    "warning": warning_message,
                }
            )

        if is_valid:
            continue

        destination = image_path
        if not args.report_only:
            destination = quarantine_image(image_path, dataset_path, quarantine_root)
        quarantined_records.append(
            {
                "original_path": str(image_path),
                "quarantine_path": str(destination),
                "reason": reason,
            }
        )

        if len(quarantined_records) <= 20 or len(quarantined_records) % 100 == 0:
            print(f"[{index}/{len(image_files)}] Quarantined: {image_path.name} -> {reason}")

    dataset_before, dataset_after = (0, 0)
    manifest_before, manifest_after = (0, 0)
    if not args.report_only:
        dataset_before, dataset_after = refresh_dataset_csv(dataset_path)
        manifest_before, manifest_after = refresh_split_manifest(dataset_path)

    report_path = quarantine_root / "quarantine_report.csv"
    suspicious_report_path = quarantine_root / "suspicious_report.csv"
    write_report(report_path, quarantined_records)
    write_suspicious_report(suspicious_report_path, suspicious_records)

    print("")
    print(f"Quarantined files: {len(quarantined_records)}")
    print(f"Suspicious image warnings: {len(suspicious_records)}")
    if not args.report_only:
        print(f"dataset.csv rows: {dataset_before} -> {dataset_after}")
        print(f"split_manifest.csv rows: {manifest_before} -> {manifest_after}")
    print(f"Report written to: {report_path}")
    print(f"Suspicious report written to: {suspicious_report_path}")


if __name__ == "__main__":
    main()
