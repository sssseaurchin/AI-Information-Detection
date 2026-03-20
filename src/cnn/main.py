from cnnModel import train_model, predict_image
from CSVCreator import create_csv
import os
import argparse
from cnnModel import preprocess_sobel_edge
from cnnModel import preprocess_regular
from label_config import get_default_label_config_path, load_label_mapping
from preprocessing import get_default_preprocess_mode
from split_utils import get_default_manifest_path, load_manifest_dataframe
from eval_runner import run_evaluation
from governance import (
    append_access_log,
    finalize_access_record,
    evaluate_governance_request,
    get_default_governance_config_path,
    load_governance_config,
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
    parser.add_argument("--augment", action="store_true", help="Enable stronger training-time real-world augmentations.")
    parser.add_argument("--seed", type=int, default=SEED, help="Deterministic seed used for split generation.")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Early stopping patience for training.")
    parser.add_argument("--finetune-unfreeze", action="store_true", help="Enable second-stage EfficientNet backbone unfreezing.")
    parser.add_argument("--finetune-freeze-epochs", type=int, default=3, help="Warmup epochs with frozen backbone before fine-tuning.")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate used during EfficientNet fine-tuning.")
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-5, help="Weight decay used during EfficientNet fine-tuning.")
    parser.add_argument(
        "--preprocess-mode",
        default=get_default_preprocess_mode(),
        choices=["rgb", "sobel", "rgb+sobel"],
        help="Shared preprocessing mode used by training and inference.",
    )
    parser.add_argument(
        "--arch",
        default="simple",
        choices=["simple", "efficientnet_b0"],
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
        help="Path to a trained model file for --eval-only. Defaults to src/cnn/model/ai_detection_model.h5.",
    )
    parser.add_argument(
        "--governance-config",
        default=get_default_governance_config_path(),
        help="Path to the machine-readable model selection governance config.",
    )
    parser.add_argument("--override-governance", action="store_true", help="Allow evaluation past governance limits and mark the run invalid for research reporting.")
    parser.add_argument("--user-id", default="unknown", help="Identifier recorded in the held-out access log.")
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

def main():
    args = parse_args()
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
    
    model_name = 'ai_detection_model.h5'
    # Create model directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    # Create CSV
    if not os.path.exists(os.path.join(dataset_path, csv_name)):
        print("CSV file not found.")
        print("Creating CSV from dataset...")
        create_csv(data_path=dataset_path, output_csv_name=csv_name) 

    label_mapping = load_label_mapping(args.label_config)

    if args.eval_only:
        governance_config = load_governance_config(args.governance_config)
        manifest = load_manifest_dataframe(args.split_manifest)
        tuning_split = select_tuning_split_from_policy(
            manifest,
            governance_config["tuning_split_preference"],
            args.eval_split,
        )
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
        epochs=10,  # Adjust as needed
        batch_size=32,
        validation_split=0.2,
        use_cache=True,  # Enable caching for faster subsequent epochs
        cache_in_memory=False,  # Set to True if dataset fits in memory
        use_mixed_precision=True,  # Enable mixed precision for faster GPU training
        enable_augmentation=args.augment,
        model_save_path=model_path,  # Save best model during training
        preprocess_func=preprocess_regular,
        image_size=(224, 224),  # Set image size for preprocessing
        enable_early_stopping=True, 
    )
    
    # Save the trained model
    model.save(model_path)
    print("Model saved as {model_file}")
    
    # Print final accuracy
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final Training Accuracy: {final_accuracy:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

if __name__ == "__main__":
    print("Starting CNN main...")
    main()
