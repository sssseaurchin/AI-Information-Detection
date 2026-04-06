import argparse
import os
from datetime import datetime

import numpy as np


def _average_probabilities(probabilities_list: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(probabilities_list, axis=0)
    return np.mean(stacked, axis=0)


def _collect_model_probabilities(
    model_paths: list[str],
    frame,
    preprocess_mode: str,
    image_size: tuple[int, int],
    batch_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    import tensorflow as tf
    from eval_runner import _extract_logits_and_probabilities

    averaged_inputs: list[np.ndarray] = []
    labels: np.ndarray | None = None

    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path, compile=False)
        outputs = _extract_logits_and_probabilities(model, frame, preprocess_mode, image_size, batch_size)
        model_labels = np.asarray(outputs["labels"], dtype=int)
        model_probs = np.asarray(outputs["probabilities"], dtype=float)

        if labels is None:
            labels = model_labels
        elif not np.array_equal(labels, model_labels):
            raise ValueError("Ensemble members produced inconsistent filtered label sets.")

        averaged_inputs.append(model_probs)

    if labels is None:
        raise ValueError("No model probabilities were collected for ensemble evaluation.")

    return labels, averaged_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a soft-voting ensemble over one labeled manifest split.")
    parser.add_argument("--model-paths", nargs="+", required=True, help="Two or more saved model paths.")
    parser.add_argument("--split-manifest", required=True, help="Manifest to evaluate.")
    parser.add_argument("--eval-split", default="test", help="Split name inside the eval manifest.")
    parser.add_argument("--tuning-manifest", default=None, help="Optional manifest used for threshold/calibration tuning.")
    parser.add_argument("--tuning-split", default=None, help="Optional split inside the tuning manifest.")
    parser.add_argument("--preprocess-mode", default="rgb", help="Shared preprocessing mode.")
    parser.add_argument("--label-config", required=True, help="Label config path.")
    parser.add_argument("--threshold-policy", default="youden", choices=["youden", "f1", "fixed"], help="Threshold selection policy.")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Threshold value when policy is fixed.")
    parser.add_argument("--calibrate", default="none", choices=["none", "temp_scaling"], help="Calibration method.")
    parser.add_argument("--report-dir", required=True, help="Directory for ensemble reports.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed recorded in the report.")
    parser.add_argument("--dataset-id", default=None, help="Optional dataset filter.")
    parser.add_argument("--domain", default=None, help="Optional domain filter.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--ensemble-name", default="soft_voting_ensemble", help="User-facing ensemble identifier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from eval_metrics import select_threshold
    from eval_runner import (
        _apply_calibration_to_eval,
        _compute_metrics_from_outputs,
        _get_dataset_version,
        _get_split_frame,
        _hash_label_mapping,
        _load_manifest,
        _normalize_optional_filter,
        _positive_probabilities,
        _prepare_calibration,
        _report_paths,
        _write_report_files,
    )
    from label_config import load_label_mapping

    manifest = _load_manifest(args.split_manifest)
    tuning_manifest = _load_manifest(args.tuning_manifest) if args.tuning_manifest else manifest
    label_mapping = load_label_mapping(args.label_config)
    label_mapping_hash = _hash_label_mapping(label_mapping)

    eval_frame = _get_split_frame(manifest, args.eval_split, dataset_id=args.dataset_id, domain=args.domain)
    if (
        args.threshold_policy != "fixed" or args.calibrate != "none"
    ) and args.eval_split in {"test", "real_world"} and not args.tuning_manifest and not args.tuning_split:
        raise ValueError(
            "Provide --tuning-manifest and optionally --tuning-split when tuning threshold/calibration for holdout evaluation."
        )
    tuning_reference_split = args.tuning_split or ("val" if args.tuning_manifest else args.eval_split)

    eval_labels, eval_probability_members = _collect_model_probabilities(
        model_paths=[os.path.abspath(path) for path in args.model_paths],
        frame=eval_frame,
        preprocess_mode=args.preprocess_mode,
        image_size=(224, 224),
        batch_size=args.batch_size,
    )
    eval_probs = _average_probabilities(eval_probability_members)

    if args.threshold_policy == "fixed" and args.calibrate == "none":
        tuning_split = "none"
        calibration_details = {"mode": "none", "temperature": 1.0, "method": "none", "fitted_on_split": None}
        if args.fixed_threshold is None:
            raise ValueError("A fixed threshold is required when --threshold-policy fixed is used without calibration.")
        threshold_selection = {
            "policy": "fixed",
            "threshold": float(args.fixed_threshold),
            "selection_metric": None,
            "selection_value": None,
        }
    else:
        tuning_frame = _get_split_frame(
            tuning_manifest,
            tuning_reference_split,
            dataset_id=args.dataset_id,
            domain=args.domain,
        )
        tuning_labels, tuning_probability_members = _collect_model_probabilities(
            model_paths=[os.path.abspath(path) for path in args.model_paths],
            frame=tuning_frame,
            preprocess_mode=args.preprocess_mode,
            image_size=(224, 224),
            batch_size=args.batch_size,
        )
        tuning_probs = _average_probabilities(tuning_probability_members)
        tuned_probs, calibration_details = _prepare_calibration(
            calibrate=args.calibrate,
            tuning_labels=tuning_labels,
            tuning_logits=None,
            tuning_probs=tuning_probs,
        )
        threshold_selection = select_threshold(
            tuning_labels,
            _positive_probabilities(np.asarray(tuned_probs, dtype=float)),
            policy=args.threshold_policy,
            fixed_threshold=args.fixed_threshold,
        )
        tuning_split = tuning_reference_split

    calibrated_eval_probs = _apply_calibration_to_eval(None, eval_probs, calibration_details)
    metrics = _compute_metrics_from_outputs(
        labels=eval_labels,
        probabilities=calibrated_eval_probs,
        threshold=float(threshold_selection["threshold"]),
    )

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{args.eval_split}"
    report = {
        "run_id": run_id,
        "split_name": args.eval_split,
        "tuning_split": tuning_split,
        "tuning_manifest_path": os.path.abspath(args.tuning_manifest) if args.tuning_manifest else os.path.abspath(args.split_manifest),
        "dataset_version": _get_dataset_version(manifest),
        "manifest_path": os.path.abspath(args.split_manifest),
        "model_path": None,
        "model_paths": [os.path.abspath(path) for path in args.model_paths],
        "ensemble": {
            "name": args.ensemble_name,
            "size": len(args.model_paths),
            "reduction": "mean_probability",
        },
        "seed": int(args.seed),
        "preprocessing_mode": args.preprocess_mode,
        "subset": {
            "dataset_id": _normalize_optional_filter(args.dataset_id),
            "domain": _normalize_optional_filter(args.domain),
        },
        "input_hashes": None,
        "baseline_hashes": None,
        "label_mapping_hash": label_mapping_hash,
        "threshold_selection": threshold_selection,
        "calibration": calibration_details,
        "threshold_free_metrics": metrics["threshold_free_metrics"],
        "thresholded_metrics": metrics["thresholded_metrics"],
        "calibration_metrics": metrics["calibration_metrics"],
        "metadata": {
            "num_samples": int(len(eval_labels)),
            "used_logit_recovery": False,
            "forbid_test_tuning": False,
        },
        "governance": {
            "allowed": True,
            "reason": "ensemble_eval_script",
            "override_used": True,
            "valid_for_thesis": False,
            "access_log_path": None,
            "baseline_path": None,
            "hashes": None,
            "baseline_hashes": None,
        },
    }

    metrics_path, summary_path = _report_paths(args.report_dir, run_id)
    report["artifacts"] = {
        "metrics_json": os.path.abspath(metrics_path),
        "summary_md": os.path.abspath(summary_path),
    }
    metrics_path, summary_path = _write_report_files(args.report_dir, run_id, report)
    print(f"Ensemble metrics saved to: {metrics_path}")
    print(f"Ensemble summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
