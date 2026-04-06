import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve exact-duplicate train/val split leakage by aligning duplicate groups to one split."
    )
    parser.add_argument("--manifest-path", required=True, help="Path to split_manifest.csv")
    parser.add_argument("--exact-duplicates-path", required=True, help="Path to exact_duplicates.csv")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy of the original manifest before overwriting.",
    )
    return parser.parse_args()


def choose_target_split(splits: list[str]) -> str:
    train_count = sum(1 for split in splits if split == "train")
    val_count = sum(1 for split in splits if split == "val")
    if train_count >= val_count:
        return "train"
    return "val"


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path)
    duplicates_path = Path(args.exact_duplicates_path)

    manifest = pd.read_csv(manifest_path)
    duplicates = pd.read_csv(duplicates_path)

    if duplicates.empty:
        print("No exact duplicate rows found; manifest unchanged.")
        return

    path_to_indices: dict[str, list[int]] = {}
    for idx, path in enumerate(manifest["path"].astype(str).tolist()):
        path_to_indices.setdefault(path, []).append(idx)

    affected_indices: set[int] = set()
    rows_touched = 0
    groups_touched = 0

    for sha, group in duplicates.groupby("sha256"):
        group_paths = set(group["left_path"].astype(str)).union(group["right_path"].astype(str))
        manifest_indices: list[int] = []
        for path in group_paths:
            manifest_indices.extend(path_to_indices.get(path, []))

        if len(manifest_indices) <= 1:
            continue

        current_splits = manifest.loc[manifest_indices, "split"].astype(str).tolist()
        if len(set(current_splits)) <= 1:
            continue

        target_split = choose_target_split(current_splits)
        manifest.loc[manifest_indices, "split"] = target_split
        affected_indices.update(manifest_indices)
        groups_touched += 1
        rows_touched += len(set(manifest_indices))

    if args.backup:
        backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
        backup_path.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest.to_csv(manifest_path, index=False)

    split_counts = manifest["split"].value_counts().to_dict()
    print(f"Duplicate groups realigned: {groups_touched}")
    print(f"Manifest rows touched: {len(affected_indices)}")
    print(f"Updated split counts: {split_counts}")


if __name__ == "__main__":
    main()
