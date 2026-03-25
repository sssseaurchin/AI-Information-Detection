import csv
import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MANIFEST_COLUMNS = [
    "path",
    "category",
    "label",
    "split",
    "group_id",
    "dataset_id",
    "domain",
    "label_mapping",
]

REQUIRED_MANIFEST_COLUMNS = [
    "path",
    "label",
    "split",
]


def get_default_manifest_path() -> str:
    """Return the repository-local split manifest path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "splits", "split_manifest.csv")


def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV headers to lower-case, trimmed names."""
    dataframe.columns = dataframe.columns.map(str).str.strip().str.lower()
    return dataframe


def _sniff_delimiter(csv_path: str) -> str:
    """Detect a likely CSV delimiter from a file sample."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        for delimiter in (";", "\t", "|", ","):
            if delimiter in first_line:
                return delimiter
        return ","


def load_manifest_dataframe(manifest_path: str, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Load a manifest CSV robustly across delimiter and header-format variations."""
    delimiter = _sniff_delimiter(manifest_path)
    manifest = pd.read_csv(manifest_path, sep=delimiter, encoding="utf-8-sig")
    manifest = _normalize_columns(manifest)

    # Fallback if the first parse still collapsed the file into a single delimited header column.
    if len(manifest.columns) == 1:
        only_column = manifest.columns[0]
        for candidate_delimiter in (";", "\t", "|", ","):
            if candidate_delimiter == delimiter or candidate_delimiter not in only_column:
                continue
            reparsed = pd.read_csv(manifest_path, sep=candidate_delimiter, encoding="utf-8-sig")
            reparsed = _normalize_columns(reparsed)
            if len(reparsed.columns) > 1:
                manifest = reparsed
                break

    missing = [column for column in (required_columns or REQUIRED_MANIFEST_COLUMNS) if column not in manifest.columns]
    if missing:
        available = ", ".join(manifest.columns.tolist())
        raise ValueError(
            f"Manifest is missing required columns {missing} after normalization. "
            f"Available columns in '{manifest_path}': [{available}]"
        )

    return manifest


def _normalize_path(dataset_path: str, category: str, image_name: str) -> str:
    return os.path.abspath(os.path.join(dataset_path, category, image_name))


def _discover_group_column(dataframe: pd.DataFrame) -> str | None:
    for candidate in ("group_id", "subject_id", "source_id", "generator_id"):
        if candidate in dataframe.columns and dataframe[candidate].notna().any():
            return candidate
    return None


def _derive_filename_groups(image_names: pd.Series) -> list[str | None]:
    stems = image_names.astype(str).str.rsplit(".", n=1).str[0]
    prefixes = stems.str.split(r"[_\- ]", n=1).str[0]
    counts = prefixes.value_counts()

    repeated_mask = prefixes.map(counts).fillna(0) > 1
    repeated_coverage = float(repeated_mask.mean()) if len(prefixes) else 0.0

    if repeated_coverage < 0.5:
        return [None] * len(prefixes)

    return [value if counts.get(value, 0) > 1 else None for value in prefixes]


def derive_group_ids(dataframe: pd.DataFrame) -> tuple[list[str | None], str]:
    """Derive group identifiers if the dataset exposes a stable grouping signal."""
    group_column = _discover_group_column(dataframe)
    if group_column:
        groups = dataframe[group_column].astype(str).tolist()
        return groups, f"column:{group_column}"

    if "image_name" in dataframe.columns:
        groups = _derive_filename_groups(dataframe["image_name"])
        if any(group is not None for group in groups):
            return groups, "filename_prefix"

    return [None] * len(dataframe), "none"


def _assign_labels(
    dataframe: pd.DataFrame,
    dataset_path: str,
    label_mapping: dict[str, int],
    allow_unknown: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    mapping_json = json.dumps(label_mapping, sort_keys=True)
    unknown_categories: list[str] = []
    rows: list[dict[str, Any]] = []
    group_ids, _ = derive_group_ids(dataframe)

    for position, (_, row) in enumerate(dataframe.iterrows()):
        category = str(row["category"])
        image_name = str(row["image_name"])
        sample_path = _normalize_path(dataset_path, category, image_name)
        group_id = group_ids[position]

        if category not in label_mapping:
            if category not in unknown_categories:
                unknown_categories.append(category)

            if allow_unknown:
                rows.append(
                    {
                        "path": sample_path,
                        "category": category,
                        "label": -1,
                        "split": "skipped_unknown",
                        "group_id": group_id or "",
                        "dataset_id": str(row["dataset_id"]) if "dataset_id" in dataframe.columns and pd.notna(row.get("dataset_id")) else "",
                        "domain": str(row["domain"]) if "domain" in dataframe.columns and pd.notna(row.get("domain")) else "",
                        "label_mapping": mapping_json,
                    }
                )
                continue

        rows.append(
            {
                "path": sample_path,
                "category": category,
                "label": label_mapping[category],
                "split": "",
                "group_id": group_id or "",
                "dataset_id": str(row["dataset_id"]) if "dataset_id" in dataframe.columns and pd.notna(row.get("dataset_id")) else "",
                "domain": str(row["domain"]) if "domain" in dataframe.columns and pd.notna(row.get("domain")) else "",
                "label_mapping": mapping_json,
            }
        )

    return rows, unknown_categories


def _deterministic_group_split(records: list[dict[str, Any]], validation_split: float, seed: int) -> None:
    eligible = [record for record in records if record["split"] != "skipped_unknown"]
    groups = sorted({record["group_id"] for record in eligible if record["group_id"]})

    if not groups:
        _deterministic_image_split(records, validation_split, seed)
        return

    rng = np.random.default_rng(seed)
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)

    target_val_size = max(1, int(len(eligible) * validation_split))
    val_groups: set[str] = set()
    running_size = 0

    group_sizes = {
        group: sum(1 for record in eligible if record["group_id"] == group)
        for group in shuffled_groups
    }

    for group in shuffled_groups:
        if running_size >= target_val_size and val_groups:
            break
        val_groups.add(group)
        running_size += group_sizes[group]

    for record in records:
        if record["split"] == "skipped_unknown":
            continue
        record["split"] = "val" if record["group_id"] in val_groups else "train"


def _deterministic_image_split(records: list[dict[str, Any]], validation_split: float, seed: int) -> None:
    eligible_indices = [idx for idx, record in enumerate(records) if record["split"] != "skipped_unknown"]
    rng = np.random.default_rng(seed)
    shuffled_indices = np.array(eligible_indices)
    rng.shuffle(shuffled_indices)

    val_size = max(1, int(len(shuffled_indices) * validation_split)) if len(shuffled_indices) > 1 else len(shuffled_indices)
    val_indices = set(shuffled_indices[:val_size].tolist())

    for idx, record in enumerate(records):
        if record["split"] == "skipped_unknown":
            continue
        record["split"] = "val" if idx in val_indices else "train"


def _write_manifest(manifest_path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def load_or_create_split_manifest(
    dataset_path: str,
    label_mapping: dict[str, int],
    validation_split: float,
    seed: int,
    manifest_path: str | None = None,
    regen_split: bool = False,
    allow_unknown: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load a frozen split manifest or create one deterministically."""
    resolved_manifest_path = manifest_path or get_default_manifest_path()
    if os.path.exists(resolved_manifest_path) and not regen_split:
        manifest = load_manifest_dataframe(resolved_manifest_path)
        return manifest, resolved_manifest_path

    csv_path = os.path.join(dataset_path, "dataset.csv")
    dataframe = pd.read_csv(csv_path)
    records, unknown_categories = _assign_labels(dataframe, dataset_path, label_mapping, allow_unknown)

    if unknown_categories and not allow_unknown:
        unknown_list = ", ".join(sorted(unknown_categories))
        raise ValueError(
            f"Unknown categories in dataset.csv: {unknown_list}. "
            "Update config/labels.json or rerun with --allow-unknown."
        )

    group_ids, group_strategy = derive_group_ids(dataframe)
    if any(group is not None for group in group_ids):
        _deterministic_group_split(records, validation_split, seed)
    else:
        _deterministic_image_split(records, validation_split, seed)

    _write_manifest(resolved_manifest_path, records)
    manifest = pd.DataFrame(records)
    manifest.attrs["group_strategy"] = group_strategy
    return manifest, resolved_manifest_path
