from cnnModel import train_model
from CSVCreator import create_csv
import os
import argparse
from label_config import get_default_label_config_path, load_label_mapping
from preprocessing import get_default_preprocess_mode
from split_utils import get_default_manifest_path, load_manifest_dataframe
from eval_runner import run_evaluation, run_slice_evaluations
from governance import (
    append_access_log,
    finalize_access_record,
    evaluate_governance_request,
    get_default_governance_config_path,
    load_governance_config,
    requires_tuning_split,
    select_tuning_split_from_policy,
)
import hashlib
import json
import pandas as pd
import sys

SEED = 42


def parse_args():
    """Keep the existing entrypoint but allow explicit split regeneration and label handling."""
    parser = argparse.ArgumentParser(description="Train the CNN-based AI image detector.")
    parser.add_argument("--regen-split", action="store_true", help="Regenerate the split manifest instead of reusing it.")
    parser.add_argument("--allow-unknown", action="store_true", help="Skip dataset categories missing from the label config.")
    parser.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        default=True,
        help="Enable stronger training-time real-world augmentations.",
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable stronger training-time real-world augmentations.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Deterministic seed used for split generation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Early stopping patience for training.")
    parser.add_argument("--finetune-unfreeze", action="store_true", help="Enable second-stage backbone unfreezing for transfer-learning architectures.")
    parser.add_argument("--finetune-freeze-epochs", type=int, default=3, help="Warmup epochs with frozen backbone before fine-tuning.")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate used during backbone fine-tuning.")
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-5, help="Weight decay used during backbone fine-tuning.")
    parser.add_argument(
        "--sampling-strategy",
        default="domain_balanced",
        choices=["none", "domain_balanced", "domain_label_balanced"],
        help="Manifest-level train resampling strategy used to reduce domain imbalance.",
    )
    parser.add_argument(
        "--preprocess-mode",
        default=get_default_preprocess_mode(),
        choices=["rgb", "sobel", "rgb+sobel", "wavelet", "rgb+wavelet"],
        help="Shared preprocessing mode used by training and inference.",
    )
    parser.add_argument(
        "--arch",
        default="simple",
        choices=["simple", "dual_artifact_cnn", "efficientnet_b0", "efficientnet_v2b0", "resnet50", "convnext_tiny", "clip_vit_b32"],
        help="Single-backbone architecture used for training.",
    )
    parser.add_argument(
        "--label-config",
        default=get_default_label_config_path(),
        help="Path to the explicit category-to-label mapping file.",
    )
    parser.add_argument(
        "--split-manifest",
        default=get_default_manifest_path(),
        help="Path to the frozen split manifest CSV.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation on a saved model.")
    parser.add_argument(
        "--eval-split",
        default="val",
        choices=["train", "val", "test", "real_world"],
        help="Manifest split to evaluate.",
    )
    parser.add_argument("--dataset-id", default=None, help="Optional dataset_id subset filter applied within the chosen manifest split.")
    parser.add_argument("--domain", default=None, help="Optional domain subset filter applied within the chosen manifest split.")
    parser.add_argument(
        "--tuning-manifest",
        default=None,
        help="Optional manifest used only for threshold/calibration tuning during evaluation.",
    )
    parser.add_argument(
        "--tuning-split",
        default=None,
        help="Optional split name used inside --tuning-manifest during evaluation.",
    )
    parser.add_argument(
        "--threshold-policy",
        default="youden",
        choices=["youden", "f1", "fixed"],
        help="Validation-only threshold selection policy.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Threshold used when --threshold-policy=fixed.")
    parser.add_argument(
        "--calibrate",
        default="none",
        choices=["none", "temp_scaling"],
        help="Calibration method fitted on validation or calibration split only.",
    )
    parser.add_argument("--report-dir", default="reports", help="Directory used for JSON and markdown evaluation reports.")
    parser.add_argument(
        "--forbid-test-tuning",
        default=True,
        type=lambda value: str(value).lower() not in {"0", "false", "no"},
        help="Raise if evaluation would tune threshold or temperature on test or real_world.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional model path used for training outputs and --eval-only inputs. Defaults to src/cnn/model/ai_detection_model.h5.",
    )
    parser.add_argument(
        "--governance-config",
        default=get_default_governance_config_path(),
        help="Path to the machine-readable model selection governance config.",
    )
    parser.add_argument("--override-governance", action="store_true", help="Allow evaluation past governance limits and mark the run invalid for research reporting.")
    parser.add_argument("--user-id", default="unknown", help="Identifier recorded in the held-out access log.")
    parser.add_argument(
        "--slice-logging",
        dest="slice_logging",
        action="store_true",
        default=False,
        help="Log per-domain and per-dataset validation metrics at the end of each epoch.",
    )
    parser.add_argument(
        "--no-slice-logging",
        dest="slice_logging",
        action="store_false",
        help="Disable per-domain and per-dataset validation metrics during training.",
    )
    parser.add_argument(
        "--slice-reports",
        dest="slice_reports",
        action="store_true",
        default=True,
        help="Generate post-training dataset/domain slice reports for the chosen evaluation split.",
    )
    parser.add_argument(
        "--no-slice-reports",
        dest="slice_reports",
        action="store_false",
        help="Skip post-training dataset/domain slice reports.",
    )
    return parser.parse_args()


def _hash_label_mapping(label_mapping: dict[str, int]) -> str:
    payload = json.dumps(label_mapping, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_default_dataset_path() -> str:
    """Resolve the repository-local prepared training dataset path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return os.path.join(project_root, "Dataset", "prepared", "combined_train")


def resolve_default_split_manifest_path(dataset_path: str, split_manifest_path: str) -> str:
    """Keep split manifests dataset-local when the caller did not override the default path."""
    default_repo_manifest = os.path.abspath(get_default_manifest_path())
    requested_manifest = os.path.abspath(split_manifest_path)
    if requested_manifest == default_repo_manifest:
        return os.path.join(dataset_path, "split_manifest.csv")
    return split_manifest_path


def resolve_default_model_path(requested_model_path: str | None, arch: str) -> str:
    """Prefer the native Keras format for architectures with less H5-friendly serialization."""
    default_model_dir = os.path.join(os.path.dirname(__file__), 'model')
    normalized_arch = arch.strip().lower()
    default_extension = ".keras" if normalized_arch in {"clip_vit_b32"} else ".h5"
    default_model_path = os.path.join(default_model_dir, f"ai_detection_model{default_extension}")
    return os.path.abspath(requested_model_path) if requested_model_path else default_model_path


def validate_arch_preprocess_compatibility(arch: str, preprocess_mode: str) -> None:
    """Reject combinations that are architecturally misleading or redundant."""
    normalized_arch = arch.strip().lower()
    normalized_mode = preprocess_mode.strip().lower()

    if normalized_arch == "clip_vit_b32" and normalized_mode != "rgb":
        raise ValueError("clip_vit_b32 must be trained and evaluated with --preprocess-mode rgb.")
    if normalized_arch == "dual_artifact_cnn" and normalized_mode != "rgb":
        raise ValueError("dual_artifact_cnn already learns artifact branches internally; use --preprocess-mode rgb.")

def main():
    args = parse_args()
    validate_arch_preprocess_compatibility(args.arch, args.preprocess_mode)
    # Path to dataset CSV
    csv_name = 'dataset.csv'
    
    # Get dataset path from environment variable (for Docker) or use project Dataset folder
    dataset_path = os.environ.get('DATASET_PATH')
    
    if not dataset_path:
        dataset_path = get_default_dataset_path()

    dataset_path = os.path.abspath(dataset_path)
    args.split_manifest = resolve_default_split_manifest_path(dataset_path, args.split_manifest)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Split manifest path: {args.split_manifest}")
    print(f"Label config path: {args.label_config}")
    
    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}\n"
                              f"Please ensure Dataset/train folder exists in project root.")
    
    model_path = resolve_default_model_path(args.model_path, args.arch)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create CSV
    if not os.path.exists(os.path.join(dataset_path, csv_name)):
        print("CSV file not found.")
        print("Creating CSV from dataset...")
        create_csv(data_path=dataset_path, output_csv_name=csv_name) 

    label_mapping = load_label_mapping(args.label_config)

    if args.eval_only:
        governance_config = load_governance_config(args.governance_config)
        manifest = load_manifest_dataframe(args.split_manifest)
        tuning_manifest = load_manifest_dataframe(args.tuning_manifest) if args.tuning_manifest else manifest
        if requires_tuning_split(args.threshold_policy, args.calibrate):
            if args.tuning_split:
                tuning_split = args.tuning_split
            else:
                tuning_split = select_tuning_split_from_policy(
                    tuning_manifest,
                    governance_config["tuning_split_preference"],
                    args.eval_split,
                )
        else:
            tuning_split = "none"
        governance_result = evaluate_governance_request(
            governance_config=governance_config,
            governance_config_path=args.governance_config,
            manifest_path=args.split_manifest,
            label_mapping_hash=_hash_label_mapping(label_mapping),
            eval_split=args.eval_split,
            tuning_split=tuning_split,
            model_path=args.model_path or model_path,
            user_id=args.user_id,
            subset={"dataset_id": args.dataset_id, "domain": args.domain},
            threshold_policy=args.threshold_policy,
            calibrate_mode=args.calibrate,
            forbid_test_tuning=args.forbid_test_tuning,
            override_governance=args.override_governance,
        )

        if not governance_result["allowed"]:
            append_access_log(governance_result["record"], governance_result["access_log_path"])
            print(f"Governance blocked evaluation: {governance_result['reason']}")
            sys.exit(2)

        report = run_evaluation(
            model_path=args.model_path or model_path,
            manifest_path=args.split_manifest,
            split_name=args.eval_split,
            preprocess_mode=args.preprocess_mode,
            label_config_path=args.label_config,
            threshold_policy=args.threshold_policy,
            fixed_threshold=args.fixed_threshold,
            calibrate=args.calibrate,
            report_dir=args.report_dir,
            seed=args.seed,
            dataset_id=args.dataset_id,
            domain=args.domain,
            tuning_split_preference=governance_config["tuning_split_preference"],
            tuning_manifest_path=args.tuning_manifest,
            tuning_split_override=args.tuning_split,
            governance_result=governance_result,
            forbid_test_tuning=args.forbid_test_tuning,
        )
        allowed_record = finalize_access_record(
            governance_result,
            user_id=args.user_id,
            run_id=report["run_id"],
            artifacts=report.get("artifacts"),
        )
        append_access_log(allowed_record, governance_result["access_log_path"])
        print(f"Evaluation report saved to: {report['artifacts']['metrics_json']}")
        print(f"Evaluation summary saved to: {report['artifacts']['summary_md']}")
        return
    
    print("Starting CNN training for AI-generated image detection...")
    
    # Train the model with optimized settings
    model, history = train_model(
        dataset_path=dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        use_cache=True,  # Enable caching for faster subsequent epochs
        cache_in_memory=False,  # Set to True if dataset fits in memory
        use_mixed_precision=True,  # Enable mixed precision for faster GPU training
        enable_augmentation=args.augment,
        model_save_path=model_path,  # Save best model during training
        image_size=(224, 224),  # Set image size for preprocessing
        enable_early_stopping=True,
        seed=args.seed,
        label_mapping=label_mapping,
        preprocess_mode=args.preprocess_mode,
        split_manifest_path=args.split_manifest,
        regen_split=args.regen_split,
        allow_unknown=args.allow_unknown,
        arch=args.arch,
        finetune_unfreeze=args.finetune_unfreeze,
        finetune_freeze_epochs=args.finetune_freeze_epochs,
        finetune_lr=args.finetune_lr,
        finetune_weight_decay=args.finetune_weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        sampling_strategy=args.sampling_strategy,
        enable_slice_logging=args.slice_logging,
    )
    
    # Save the trained model
    model.save(model_path)
    print(f"Model saved as {model_path}")
    
    # Print final accuracy
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final Training Accuracy: {final_accuracy:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

    if args.slice_reports:
        try:
            slice_report = run_slice_evaluations(
                model_path=model_path,
                manifest_path=args.split_manifest,
                split_name="val",
                preprocess_mode=args.preprocess_mode,
                label_config_path=args.label_config,
                threshold_policy=args.threshold_policy,
                fixed_threshold=args.fixed_threshold,
                calibrate=args.calibrate,
                report_dir=args.report_dir,
                seed=args.seed,
                tuning_split_preference=["val"],
            )
            print(f"Slice metrics saved to: {slice_report['artifacts']['metrics_json']}")
            print(f"Slice summary saved to: {slice_report['artifacts']['summary_md']}")
        except Exception as exc:
            print(f"Warning: post-training slice reports failed: {exc}")

if __name__ == "__main__":
    print("Starting CNN main...")
    main()
