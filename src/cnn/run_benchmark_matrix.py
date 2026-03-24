import argparse
import csv
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


def _read_training_metrics(metrics_csv_path: str) -> dict[str, str]:
    if not os.path.exists(metrics_csv_path):
        return {}
    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


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
        help="One or more manifest CSV files to benchmark against.",
    )
    parser.add_argument("--preprocess-mode", default="rgb", help="Shared preprocessing mode.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--sampling-strategy", default="domain_balanced", help="Train resampling strategy.")
    parser.add_argument("--finetune-unfreeze", action="store_true", help="Enable staged backbone fine-tuning.")
    parser.add_argument("--finetune-freeze-epochs", type=int, default=2, help="Warmup epochs before unfreezing.")
    parser.add_argument("--early-stopping-patience", type=int, default=2, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
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

            command = [
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
                command.extend(
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
                print("DRY RUN:", subprocess.list2cmdline(command))
            else:
                print("RUN:", subprocess.list2cmdline(command))
                completed = subprocess.run(command, cwd=project_root, env=base_env, check=False)
                return_code = int(completed.returncode)
                status = "ok" if completed.returncode == 0 else "failed"

            final_metrics = _read_training_metrics(metrics_csv_path)
            summary_rows.append(
                {
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
            )

    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
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
        ])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"Benchmark summary written to {summary_path}")


if __name__ == "__main__":
    main()
