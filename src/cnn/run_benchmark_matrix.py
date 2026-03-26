import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("_") or "run"


def _manifest_slug(manifest_path: str) -> str:
    return _safe_slug(os.path.splitext(os.path.basename(manifest_path))[0])


def _default_model_extension(arch: str) -> str:
    return ".keras" if arch.strip().lower() in {"clip_vit_b32"} else ".h5"


def _fieldnames_from_rows(rows: list[dict[str, object]]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in ordered:
                ordered.append(key)
    return ordered


def _read_training_metrics(metrics_csv_path: str) -> dict[str, str]:
    if not os.path.exists(metrics_csv_path):
        return {}
    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _load_latest_eval_metrics(report_dir: str) -> dict[str, object]:
    if not os.path.isdir(report_dir):
        return {}

    candidate_files = [
        os.path.join(report_dir, name)
        for name in os.listdir(report_dir)
        if name.endswith("_metrics.json")
    ]
    if not candidate_files:
        return {}

    latest_path = max(candidate_files, key=os.path.getmtime)
    with open(latest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_holdout_metrics(report: dict[str, object]) -> dict[str, object]:
    thresholded = dict(report.get("thresholded_metrics", {}) or {})
    threshold_free = dict(report.get("threshold_free_metrics", {}) or {})
    threshold_selection = dict(report.get("threshold_selection", {}) or {})
    metadata = dict(report.get("metadata", {}) or {})

    return {
        "accuracy": thresholded.get("accuracy"),
        "balanced_accuracy": thresholded.get("balanced_accuracy"),
        "precision": thresholded.get("precision"),
        "recall": thresholded.get("recall"),
        "f1": thresholded.get("f1"),
        "roc_auc": threshold_free.get("roc_auc"),
        "pr_auc": threshold_free.get("pr_auc"),
        "threshold": threshold_selection.get("threshold"),
        "num_samples": metadata.get("num_samples"),
    }


def _aggregate_holdout_metrics(row: dict[str, object], holdout_slugs: list[str]) -> None:
    metric_names = ("accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc")
    for metric_name in metric_names:
        values: list[float] = []
        for holdout_slug in holdout_slugs:
            value = row.get(f"holdout__{holdout_slug}__{metric_name}")
            try:
                if value not in (None, ""):
                    values.append(float(value))
            except (TypeError, ValueError):
                continue
        if values:
            row[f"holdout_mean_{metric_name}"] = sum(values) / len(values)
            row[f"holdout_worst_{metric_name}"] = min(values)
        else:
            row[f"holdout_mean_{metric_name}"] = ""
            row[f"holdout_worst_{metric_name}"] = ""

    completed = 0
    for holdout_slug in holdout_slugs:
        if row.get(f"holdout__{holdout_slug}__status") == "ok":
            completed += 1
    row["holdout_total"] = len(holdout_slugs)
    row["holdout_completed"] = completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix over multiple backbones and manifests.")
    parser.add_argument("--project-root", required=True, help="Absolute path to the repository root.")
    parser.add_argument(
        "--arches",
        nargs="+",
        default=["efficientnet_b0", "efficientnet_v2b0", "resnet50", "convnext_tiny", "dual_artifact_cnn", "clip_vit_b32"],
        help="Architectures to benchmark.",
    )
    parser.add_argument(
        "--manifest-paths",
        nargs="+",
        required=True,
        help="One or more training manifest CSV files to benchmark against.",
    )
    parser.add_argument("--preprocess-mode", default="rgb", help="Shared preprocessing mode.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--sampling-strategy", default="domain_balanced", help="Train resampling strategy.")
    parser.add_argument("--finetune-unfreeze", action="store_true", help="Enable staged backbone fine-tuning.")
    parser.add_argument("--finetune-freeze-epochs", type=int, default=2, help="Warmup epochs before unfreezing.")
    parser.add_argument("--early-stopping-patience", type=int, default=2, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument(
        "--holdout-manifests",
        nargs="*",
        default=[],
        help="Optional labeled holdout manifests evaluated after each training run.",
    )
    parser.add_argument(
        "--holdout-eval-split",
        default="test",
        help="Split name used inside each holdout manifest.",
    )
    parser.add_argument(
        "--holdout-threshold-policy",
        default="youden",
        choices=["youden", "f1", "fixed"],
        help="Threshold policy used for post-training holdout evaluations.",
    )
    parser.add_argument(
        "--holdout-calibrate",
        default="none",
        choices=["none", "temp_scaling"],
        help="Calibration mode used for post-training holdout evaluations.",
    )
    parser.add_argument(
        "--holdout-tuning-split",
        default="val",
        help="Split inside the training manifest used to tune holdout threshold/calibration.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Threshold value when --holdout-threshold-policy=fixed.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints, reports, and summary CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = os.path.abspath(args.project_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_root = os.path.join(output_dir, "checkpoints")
    reports_root = os.path.join(output_dir, "reports")
    os.makedirs(checkpoints_root, exist_ok=True)
    os.makedirs(reports_root, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    python_exe = sys.executable
    main_script = os.path.join(project_root, "src", "cnn", "main.py")
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = os.path.join(project_root, "src", "cnn")

    holdout_manifests = [os.path.abspath(path) for path in args.holdout_manifests]
    holdout_slugs = [_manifest_slug(path) for path in holdout_manifests]

    for manifest_path in args.manifest_paths:
        manifest_abspath = os.path.abspath(manifest_path)
        manifest_slug = _manifest_slug(manifest_abspath)

        for arch in args.arches:
            run_name = _safe_slug(f"{arch}__{args.preprocess_mode}__{args.sampling_strategy}__{manifest_slug}")
            run_checkpoint_dir = os.path.join(checkpoints_root, run_name)
            run_report_dir = os.path.join(reports_root, run_name)
            os.makedirs(run_checkpoint_dir, exist_ok=True)
            os.makedirs(run_report_dir, exist_ok=True)
            model_path = os.path.join(run_checkpoint_dir, f"model{_default_model_extension(arch)}")
            metrics_csv_path = os.path.join(run_checkpoint_dir, "training_metrics.csv")

            train_command = [
                python_exe,
                main_script,
                "--arch",
                arch,
                "--preprocess-mode",
                args.preprocess_mode,
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--sampling-strategy",
                args.sampling_strategy,
                "--split-manifest",
                manifest_abspath,
                "--report-dir",
                run_report_dir,
                "--model-path",
                model_path,
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--seed",
                str(args.seed),
            ]
            if args.finetune_unfreeze:
                train_command.extend(
                    [
                        "--finetune-unfreeze",
                        "--finetune-freeze-epochs",
                        str(args.finetune_freeze_epochs),
                    ]
                )

            started_at = datetime.now(timezone.utc).isoformat()
            status = "dry_run"
            return_code = None
            if args.dry_run:
                print("DRY RUN:", subprocess.list2cmdline(train_command))
            else:
                print("RUN:", subprocess.list2cmdline(train_command))
                completed = subprocess.run(train_command, cwd=project_root, env=base_env, check=False)
                return_code = int(completed.returncode)
                status = "ok" if completed.returncode == 0 else "failed"

            final_metrics = _read_training_metrics(metrics_csv_path)
            row: dict[str, object] = {
                "run_name": run_name,
                "arch": arch,
                "manifest_path": manifest_abspath,
                "preprocess_mode": args.preprocess_mode,
                "sampling_strategy": args.sampling_strategy,
                "status": status,
                "return_code": return_code,
                "started_at_utc": started_at,
                "model_path": model_path,
                "metrics_csv_path": metrics_csv_path,
                "report_dir": run_report_dir,
                "final_epoch": final_metrics.get("epoch"),
                "final_accuracy": final_metrics.get("accuracy"),
                "final_loss": final_metrics.get("loss"),
                "final_val_accuracy": final_metrics.get("val_accuracy"),
                "final_val_loss": final_metrics.get("val_loss"),
            }

            if not args.dry_run and status == "ok" and holdout_manifests:
                for holdout_manifest, holdout_slug in zip(holdout_manifests, holdout_slugs, strict=True):
                    holdout_report_dir = os.path.join(run_report_dir, f"holdout__{holdout_slug}")
                    os.makedirs(holdout_report_dir, exist_ok=True)
                    eval_command = [
                        python_exe,
                        main_script,
                        "--eval-only",
                        "--arch",
                        arch,
                        "--preprocess-mode",
                        args.preprocess_mode,
                        "--batch-size",
                        str(args.batch_size),
                        "--split-manifest",
                        holdout_manifest,
                        "--eval-split",
                        args.holdout_eval_split,
                        "--tuning-manifest",
                        manifest_abspath,
                        "--tuning-split",
                        args.holdout_tuning_split,
                        "--threshold-policy",
                        args.holdout_threshold_policy,
                        "--calibrate",
                        args.holdout_calibrate,
                        "--report-dir",
                        holdout_report_dir,
                        "--model-path",
                        model_path,
                        "--seed",
                        str(args.seed),
                        "--override-governance",
                    ]
                    if args.holdout_threshold_policy == "fixed" and args.fixed_threshold is not None:
                        eval_command.extend(["--fixed-threshold", str(args.fixed_threshold)])

                    print("EVAL:", subprocess.list2cmdline(eval_command))
                    holdout_completed = subprocess.run(eval_command, cwd=project_root, env=base_env, check=False)
                    row[f"holdout__{holdout_slug}__status"] = "ok" if holdout_completed.returncode == 0 else "failed"
                    row[f"holdout__{holdout_slug}__return_code"] = int(holdout_completed.returncode)
                    row[f"holdout__{holdout_slug}__report_dir"] = holdout_report_dir

                    if holdout_completed.returncode == 0:
                        holdout_report = _load_latest_eval_metrics(holdout_report_dir)
                        for metric_key, metric_value in _extract_holdout_metrics(holdout_report).items():
                            row[f"holdout__{holdout_slug}__{metric_key}"] = metric_value

            _aggregate_holdout_metrics(row, holdout_slugs)
            summary_rows.append(row)

    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    fieldnames = _fieldnames_from_rows(summary_rows) if summary_rows else [
        "run_name",
        "arch",
        "manifest_path",
        "preprocess_mode",
        "sampling_strategy",
        "status",
        "return_code",
        "started_at_utc",
        "model_path",
        "metrics_csv_path",
        "report_dir",
        "final_epoch",
        "final_accuracy",
        "final_loss",
        "final_val_accuracy",
        "final_val_loss",
    ]
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"Benchmark summary written to {summary_path}")


if __name__ == "__main__":
    main()
