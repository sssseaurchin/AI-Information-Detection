import argparse
import csv
import json
import os
from typing import Any


def _safe_float(value: str | None, default: float = float("-inf")) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def _load_slice_score(report_dir: str, metric: str) -> float:
    if not os.path.isdir(report_dir):
        return float("-inf")

    candidate_files = [
        os.path.join(report_dir, name)
        for name in os.listdir(report_dir)
        if name.endswith("_slices.json")
    ]
    if not candidate_files:
        return float("-inf")

    latest_path = max(candidate_files, key=os.path.getmtime)
    with open(latest_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    scores: list[float] = []
    for column_report in report.get("slice_reports", []):
        for row in column_report.get("rows", []):
            thresholded = row.get("thresholded_metrics", {})
            threshold_free = row.get("threshold_free_metrics", {})
            if metric in thresholded:
                scores.append(float(thresholded[metric]))
            elif metric in threshold_free:
                scores.append(float(threshold_free[metric]))
    if not scores:
        return float("-inf")
    return sum(scores) / len(scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank benchmark runs and pick the best candidate.")
    parser.add_argument("--summary-csv", required=True, help="Path to benchmark_summary.csv.")
    parser.add_argument(
        "--primary-metric",
        default="final_val_accuracy",
        choices=["final_val_accuracy", "final_accuracy", "slice_balanced_accuracy", "slice_accuracy", "slice_roc_auc"],
        help="Primary ranking metric.",
    )
    parser.add_argument("--output-path", required=True, help="Path for the ranked CSV output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.summary_csv, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if args.primary_metric.startswith("slice_"):
        slice_metric = args.primary_metric.replace("slice_", "", 1)
        for row in rows:
            row[args.primary_metric] = _load_slice_score(row.get("report_dir", ""), slice_metric)
    else:
        for row in rows:
            row[args.primary_metric] = _safe_float(row.get(args.primary_metric))

    scored_rows = [row for row in rows if row.get("status") in {"ok", "dry_run"}]
    scored_rows.sort(key=lambda row: _safe_float(str(row.get(args.primary_metric))), reverse=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fieldnames = list(scored_rows[0].keys()) if scored_rows else list(rows[0].keys()) if rows else [args.primary_metric]
    with open(args.output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored_rows:
            writer.writerow(row)

    if scored_rows:
        best = scored_rows[0]
        print(
            "Best run: {run_name} | {metric}={value}".format(
                run_name=best.get("run_name"),
                metric=args.primary_metric,
                value=best.get(args.primary_metric),
            )
        )
    print(f"Ranked runs written to {args.output_path}")


if __name__ == "__main__":
    main()
