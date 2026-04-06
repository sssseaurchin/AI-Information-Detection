import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from PIL import Image, ImageOps

from split_utils import load_manifest_dataframe


def _get_default_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_sample_path(path: str, project_root: str) -> str:
    normalized = path.strip()
    if os.path.exists(normalized):
        return normalized
    if normalized.startswith("/app/"):
        candidate = os.path.join(project_root, normalized.removeprefix("/app/").replace("/", os.sep))
        return candidate
    return normalized


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _average_hash(path: str, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        grayscale = ImageOps.grayscale(image).resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(grayscale, dtype=np.float32)
    mean_value = float(pixels.mean())
    bits = (pixels >= mean_value).astype(np.uint8).flatten()
    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def _hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact and near-duplicate images from a manifest.")
    parser.add_argument("--manifest-path", required=True, help="Path to the manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory where duplicate reports will be written.")
    parser.add_argument("--near-threshold", type=int, default=5, help="Maximum Hamming distance for near-duplicate hashes.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for faster exploratory audits.")
    parser.add_argument(
        "--project-root",
        default=_get_default_project_root(),
        help="Repository root used to resolve container-style /app paths on the host.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest_dataframe(args.manifest_path)
    frame = manifest[manifest["label"] >= 0].copy()
    if args.limit is not None:
        frame = frame.head(args.limit).copy()

    os.makedirs(args.output_dir, exist_ok=True)

    sha_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    ahash_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for _, row in frame.iterrows():
        manifest_path = str(row["path"])
        path = _resolve_sample_path(manifest_path, project_root=os.path.abspath(args.project_root))
        try:
            sha = _sha256_file(path)
            ahash = _average_hash(path)
        except Exception as exc:
            errors.append({"path": manifest_path, "resolved_path": path, "error": str(exc)})
            continue

        record = {
            "path": manifest_path,
            "resolved_path": path,
            "split": str(row.get("split", "")),
            "label": str(row.get("label", "")),
            "dataset_id": str(row.get("dataset_id", "")),
            "domain": str(row.get("domain", "")),
        }
        sha_to_rows[sha].append(record)
        ahash_rows.append(
            {
                **record,
                "ahash": ahash,
                "ahash_prefix": ahash >> 48,
            }
        )

    exact_duplicates: list[dict[str, object]] = []
    for sha, rows in sha_to_rows.items():
        if len(rows) < 2:
            continue
        for left, right in combinations(rows, 2):
            exact_duplicates.append(
                {
                    "sha256": sha,
                    "left_path": left["path"],
                    "right_path": right["path"],
                    "left_split": left["split"],
                    "right_split": right["split"],
                    "left_label": left["label"],
                    "right_label": right["label"],
                    "left_dataset_id": left["dataset_id"],
                    "right_dataset_id": right["dataset_id"],
                    "left_domain": left["domain"],
                    "right_domain": right["domain"],
                    "cross_split": left["split"] != right["split"],
                }
            )

    buckets: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in ahash_rows:
        buckets[int(row["ahash_prefix"])].append(row)

    near_duplicates: list[dict[str, object]] = []
    for bucket_rows in buckets.values():
        if len(bucket_rows) < 2:
            continue
        for left, right in combinations(bucket_rows, 2):
            if left["path"] == right["path"]:
                continue
            distance = _hamming_distance(int(left["ahash"]), int(right["ahash"]))
            if distance > args.near_threshold:
                continue
            near_duplicates.append(
                {
                    "left_path": left["path"],
                    "right_path": right["path"],
                    "left_split": left["split"],
                    "right_split": right["split"],
                    "left_label": left["label"],
                    "right_label": right["label"],
                    "left_dataset_id": left["dataset_id"],
                    "right_dataset_id": right["dataset_id"],
                    "left_domain": left["domain"],
                    "right_domain": right["domain"],
                    "hamming_distance": distance,
                    "cross_split": left["split"] != right["split"],
                }
            )

    exact_path = os.path.join(args.output_dir, "exact_duplicates.csv")
    near_path = os.path.join(args.output_dir, "near_duplicates.csv")
    errors_path = os.path.join(args.output_dir, "duplicate_audit_errors.csv")
    summary_path = os.path.join(args.output_dir, "duplicate_audit_summary.json")

    with open(exact_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = list(exact_duplicates[0].keys()) if exact_duplicates else [
            "sha256",
            "left_path",
            "right_path",
            "left_split",
            "right_split",
            "left_label",
            "right_label",
            "left_dataset_id",
            "right_dataset_id",
            "left_domain",
            "right_domain",
            "cross_split",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in exact_duplicates:
            writer.writerow(row)

    with open(near_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = list(near_duplicates[0].keys()) if near_duplicates else [
            "left_path",
            "right_path",
            "left_split",
            "right_split",
            "left_label",
            "right_label",
            "left_dataset_id",
            "right_dataset_id",
            "left_domain",
            "right_domain",
            "hamming_distance",
            "cross_split",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in near_duplicates:
            writer.writerow(row)

    with open(errors_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "resolved_path", "error"])
        writer.writeheader()
        for row in errors:
            writer.writerow(row)

    summary = {
        "manifest_path": os.path.abspath(args.manifest_path),
        "output_dir": os.path.abspath(args.output_dir),
        "rows_scanned": int(len(frame)),
        "exact_duplicate_pairs": int(len(exact_duplicates)),
        "exact_cross_split_pairs": int(sum(1 for row in exact_duplicates if row["cross_split"])),
        "near_duplicate_pairs": int(len(near_duplicates)),
        "near_cross_split_pairs": int(sum(1 for row in near_duplicates if row["cross_split"])),
        "errors": int(len(errors)),
        "near_threshold": int(args.near_threshold),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Exact duplicates written to {exact_path}")
    print(f"Near duplicates written to {near_path}")
    print(f"Audit summary written to {summary_path}")


if __name__ == "__main__":
    main()
