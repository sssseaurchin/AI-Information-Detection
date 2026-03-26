import argparse
import csv
import json
import os


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


def _mean_from_columns(row: dict[str, str], suffix: str) -> float:
    values: list[float] = []
    for key, value in row.items():
        if key.startswith("holdout__") and key.endswith(f"__{suffix}"):
            parsed = _safe_float(value)
            if parsed != float("-inf"):
                values.append(parsed)
    return sum(values) / len(values) if values else float("-inf")


def _worst_from_columns(row: dict[str, str], suffix: str) -> float:
    values: list[float] = []
    for key, value in row.items():
        if key.startswith("holdout__") and key.endswith(f"__{suffix}"):
            parsed = _safe_float(value)
            if parsed != float("-inf"):
                values.append(parsed)
    return min(values) if values else float("-inf")


def _compute_metric(row: dict[str, str], metric_name: str) -> float:
    if metric_name.startswith("slice_"):
        slice_metric = metric_name.replace("slice_", "", 1)
        return _load_slice_score(row.get("report_dir", ""), slice_metric)

    if metric_name.startswith("holdout_mean_"):
        metric_suffix = metric_name.replace("holdout_mean_", "", 1)
        return _mean_from_columns(row, metric_suffix)

    if metric_name.startswith("holdout_worst_"):
        metric_suffix = metric_name.replace("holdout_worst_", "", 1)
        return _worst_from_columns(row, metric_suffix)

    if metric_name == "generalization_score":
        mean_bal = _compute_metric(row, "holdout_mean_balanced_accuracy")
        worst_bal = _compute_metric(row, "holdout_worst_balanced_accuracy")
        val_acc = _safe_float(row.get("final_val_accuracy"))
        if mean_bal == float("-inf") and worst_bal == float("-inf"):
            return val_acc

        mean_component = 0.0 if mean_bal == float("-inf") else mean_bal
        worst_component = 0.0 if worst_bal == float("-inf") else worst_bal
        val_component = 0.0 if val_acc == float("-inf") else val_acc
        return (0.5 * mean_component) + (0.3 * worst_component) + (0.2 * val_component)

    return _safe_float(row.get(metric_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank benchmark runs and pick the best candidate.")
    parser.add_argument("--summary-csv", required=True, help="Path to benchmark_summary.csv.")
    parser.add_argument(
        "--primary-metric",
        default="holdout_mean_balanced_accuracy",
        help="Ranking metric or derived score name.",
    )
    parser.add_argument(
        "--secondary-metric",
        default="final_val_accuracy",
        help="Tie-break metric or derived score name.",
    )
    parser.add_argument("--output-path", required=True, help="Path for the ranked CSV output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.summary_csv, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row[args.primary_metric] = _compute_metric(row, args.primary_metric)
        row[args.secondary_metric] = _compute_metric(row, args.secondary_metric)

    scored_rows = [row for row in rows if row.get("status") in {"ok", "dry_run"}]
    scored_rows.sort(
        key=lambda row: (
            _safe_float(str(row.get(args.primary_metric))),
            _safe_float(str(row.get(args.secondary_metric))),
        ),
        reverse=True,
    )

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
            "Best run: {run_name} | {metric}={value} | {tie_metric}={tie_value}".format(
                run_name=best.get("run_name"),
                metric=args.primary_metric,
                value=best.get(args.primary_metric),
                tie_metric=args.secondary_metric,
                tie_value=best.get(args.secondary_metric),
            )
        )
    print(f"Ranked runs written to {args.output_path}")


if __name__ == "__main__":
    main()
