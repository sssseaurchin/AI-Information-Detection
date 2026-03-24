import argparse
import os
import re

import pandas as pd

from split_utils import load_manifest_dataframe


def _safe_slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("_") or "unknown"


def _build_holdout_manifest(manifest: pd.DataFrame, group_column: str, holdout_value: str) -> pd.DataFrame:
    holdout_manifest = manifest.copy()
    eligible_mask = holdout_manifest["split"].astype(str).isin(["train", "val"])
    if group_column not in holdout_manifest.columns:
        raise ValueError(f"Manifest does not contain '{group_column}'.")

    group_values = holdout_manifest[group_column].fillna("").astype(str)
    holdout_mask = eligible_mask & (group_values == holdout_value)
    train_mask = eligible_mask & (group_values != holdout_value)

    holdout_manifest.loc[train_mask, "split"] = "train"
    holdout_manifest.loc[holdout_mask, "split"] = "val"
    return holdout_manifest


def _summary_rows(manifest: pd.DataFrame, group_column: str, values: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in values:
        candidate = _build_holdout_manifest(manifest, group_column, value)
        train_count = int((candidate["split"].astype(str) == "train").sum())
        val_count = int((candidate["split"].astype(str) == "val").sum())
        rows.append(
            {
                group_column: value,
                "train_count": train_count,
                "val_count": val_count,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate leave-one-group-out manifests from an existing split manifest.")
    parser.add_argument("--manifest-path", required=True, help="Path to the source manifest CSV.")
    parser.add_argument(
        "--group-column",
        default="domain",
        choices=["domain", "dataset_id"],
        help="Manifest column used for holdout generation.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where generated manifests will be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest_dataframe(args.manifest_path)
    if args.group_column not in manifest.columns:
        raise ValueError(f"Manifest does not contain '{args.group_column}'.")

    values = [
        value for value in sorted(manifest[args.group_column].dropna().astype(str).unique().tolist())
        if value.strip()
    ]
    if not values:
        raise ValueError(f"Manifest column '{args.group_column}' does not contain any usable values.")

    os.makedirs(args.output_dir, exist_ok=True)

    for value in values:
        holdout_manifest = _build_holdout_manifest(manifest, args.group_column, value)
        output_name = f"holdout_{args.group_column}_{_safe_slug(value)}.csv"
        output_path = os.path.join(args.output_dir, output_name)
        holdout_manifest.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")

    summary = pd.DataFrame(_summary_rows(manifest, args.group_column, values))
    summary_path = os.path.join(args.output_dir, f"summary_{args.group_column}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
