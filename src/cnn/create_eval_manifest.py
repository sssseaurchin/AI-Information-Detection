import argparse
import json
from pathlib import Path

import pandas as pd

from label_config import load_label_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a split_manifest.csv for a prepared eval dataset (holdout or external eval)."
    )
    parser.add_argument("--dataset-path", required=True, help="Dataset directory containing dataset.csv and class folders.")
    parser.add_argument(
        "--split-name",
        default="test",
        choices=["train", "val", "test", "real_world", "calibration"],
        help="Split name assigned to every manifest row.",
    )
    parser.add_argument(
        "--label-config",
        default=None,
        help="Optional label config path. Defaults to src/cnn/config/labels.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).resolve()
    csv_path = dataset_path / "dataset.csv"
    manifest_path = dataset_path / "split_manifest.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"dataset.csv not found: {csv_path}")

    label_mapping = load_label_mapping(args.label_config)
    mapping_json = json.dumps(label_mapping, sort_keys=True)
    dataframe = pd.read_csv(csv_path)
    project_root = dataset_path.parents[3]

    required_columns = {"image_name", "category"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"dataset.csv missing required columns: {', '.join(sorted(missing))}")

    records: list[dict[str, object]] = []
    for _, row in dataframe.iterrows():
        category = str(row["category"])
        image_name = str(row["image_name"])
        if category not in label_mapping:
            raise ValueError(f"Unknown category '{category}' in {csv_path}")

        sample_path = (dataset_path / category / image_name).resolve()
        if not sample_path.exists():
            raise FileNotFoundError(f"Image referenced in dataset.csv does not exist: {sample_path}")

        try:
            relative_to_project = sample_path.relative_to(project_root).as_posix()
            manifest_sample_path = f"/app/{relative_to_project}"
        except ValueError:
            manifest_sample_path = str(sample_path).replace("\\", "/")

        records.append(
            {
                "path": manifest_sample_path,
                "category": category,
                "label": int(label_mapping[category]),
                "split": args.split_name,
                "group_id": "",
                "dataset_id": str(row["dataset_id"]) if "dataset_id" in dataframe.columns and pd.notna(row["dataset_id"]) else "",
                "domain": str(row["domain"]) if "domain" in dataframe.columns and pd.notna(row["domain"]) else "",
                "label_mapping": mapping_json,
            }
        )

    manifest = pd.DataFrame.from_records(
        records,
        columns=["path", "category", "label", "split", "group_id", "dataset_id", "domain", "label_mapping"],
    )
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
