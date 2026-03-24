import hashlib
import json
import os
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from calibration import apply_temperature_scaling, brier_score_binary, fit_temperature, reliability_diagram_binary
from eval_metrics import select_threshold, threshold_free_metrics, threshold_metrics
from governance import select_tuning_split_from_policy
from label_config import load_label_mapping
from preprocessing import get_preprocess_fn
from split_utils import load_manifest_dataframe


def _load_manifest(manifest_path: str) -> pd.DataFrame:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    return load_manifest_dataframe(manifest_path)


def _get_dataset_version(manifest: pd.DataFrame) -> str | None:
    if "dataset_version" in manifest.columns:
        values = manifest["dataset_version"].dropna().astype(str).unique().tolist()
        if values:
            return values[0]
    return None


def _normalize_optional_filter(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _hash_label_mapping(label_mapping: dict[str, int]) -> str:
    payload = json.dumps(label_mapping, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _get_split_frame(
    manifest: pd.DataFrame,
    split_name: str,
    dataset_id: str | None = None,
    domain: str | None = None,
) -> pd.DataFrame:
    frame = manifest[manifest["split"] == split_name].copy()
    frame = frame[frame["label"] >= 0]

    normalized_dataset_id = _normalize_optional_filter(dataset_id)
    normalized_domain = _normalize_optional_filter(domain)

    if normalized_dataset_id is not None:
        if "dataset_id" not in frame.columns:
            raise ValueError("Manifest does not contain a dataset_id column for --dataset-id filtering.")
        frame = frame[frame["dataset_id"].astype(str) == normalized_dataset_id]

    if normalized_domain is not None:
        if "domain" not in frame.columns:
            raise ValueError("Manifest does not contain a domain column for --domain filtering.")
        frame = frame[frame["domain"].astype(str) == normalized_domain]

    if frame.empty:
        raise ValueError(f"Manifest does not contain any evaluable rows for split '{split_name}'.")
    return frame


def _build_dataset(frame: pd.DataFrame, preprocess_mode: str, image_size: tuple[int, int], batch_size: int) -> tf.data.Dataset:
    preprocess_fn = get_preprocess_fn(preprocess_mode)
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label"].astype(int).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda path, label: preprocess_fn(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    dataset = dataset.filter(lambda img, label: tf.shape(img)[0] == image_size[0])
    dataset = dataset.ignore_errors()
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _labels_from_dataset(dataset: tf.data.Dataset) -> np.ndarray:
    """Collect labels after dataset filtering so metrics match evaluated samples."""
    batches = [np.asarray(batch_labels) for _, batch_labels in dataset]
    if not batches:
        return np.asarray([], dtype=int)
    return np.concatenate(batches, axis=0).astype(int)


def _extract_logits_and_probabilities(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    preprocess_mode: str,
    image_size: tuple[int, int],
    batch_size: int,
) -> dict[str, np.ndarray | bool]:
    dataset = _build_dataset(frame, preprocess_mode, image_size, batch_size)
    labels = _labels_from_dataset(dataset)

    last_layer = model.layers[-1]
    supports_logits = hasattr(last_layer, "activation") and getattr(last_layer.activation, "__name__", "") == "softmax"

    probabilities = model.predict(dataset, verbose=0)
    logits = None

    if supports_logits and hasattr(last_layer, "kernel"):
        feature_model = tf.keras.Model(inputs=model.input, outputs=last_layer.input)
        features = feature_model.predict(dataset, verbose=0)
        kernel, bias = last_layer.get_weights()
        logits = np.matmul(features, kernel) + bias

    return {
        "labels": labels,
        "probabilities": np.asarray(probabilities, dtype=float),
        "logits": None if logits is None else np.asarray(logits, dtype=float),
        "used_logit_recovery": bool(logits is not None),
    }


def _positive_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    if probabilities.shape[1] < 2:
        raise ValueError("Expected binary class probabilities with two columns.")
    return probabilities[:, 1].astype(float)


def _prepare_calibration(
    calibrate: str,
    tuning_labels: np.ndarray,
    tuning_logits: np.ndarray | None,
    tuning_probs: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = calibrate.strip().lower()
    if normalized == "none":
        return tuning_probs, {"mode": "none", "temperature": 1.0, "method": "none"}

    if normalized != "temp_scaling":
        raise ValueError(f"Unsupported calibration mode: {calibrate}")

    fit = fit_temperature(
        labels=tuning_labels,
        logits=tuning_logits,
        probabilities=_positive_probabilities(tuning_probs) if tuning_logits is None else None,
    )
    applied = apply_temperature_scaling(
        logits=tuning_logits,
        probabilities=_positive_probabilities(tuning_probs) if tuning_logits is None else None,
        temperature=float(fit["temperature"]),
    )
    details = {
        "mode": normalized,
        "temperature": float(fit["temperature"]),
        "fit_nll": float(fit["nll"]),
        "fit_method": str(fit["method"]),
        "apply_method": str(applied["method"]),
    }
    return np.asarray(applied["probabilities"], dtype=float), details


def _apply_calibration_to_eval(
    eval_logits: np.ndarray | None,
    eval_probs: np.ndarray,
    calibration_details: dict[str, Any],
) -> np.ndarray:
    if calibration_details["mode"] == "none":
        return eval_probs

    applied = apply_temperature_scaling(
        logits=eval_logits,
        probabilities=_positive_probabilities(eval_probs) if eval_logits is None else None,
        temperature=float(calibration_details["temperature"]),
    )
    return np.asarray(applied["probabilities"], dtype=float)


def _report_paths(report_dir: str, run_id: str) -> tuple[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    metrics_path = os.path.join(report_dir, f"{run_id}_metrics.json")
    summary_path = os.path.join(report_dir, f"{run_id}_summary.md")
    return metrics_path, summary_path


def _slice_report_paths(report_dir: str, run_id: str) -> tuple[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    metrics_path = os.path.join(report_dir, f"{run_id}_slices.json")
    summary_path = os.path.join(report_dir, f"{run_id}_slices.md")
    return metrics_path, summary_path


def _write_report_files(report_dir: str, run_id: str, report: dict[str, Any]) -> tuple[str, str]:
    metrics_path, summary_path = _report_paths(report_dir, run_id)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    thresholded = report["thresholded_metrics"]
    threshold_free = report["threshold_free_metrics"]
    calibration = report["calibration"]
    confusion = thresholded["confusion_matrix"]

    lines = [
        f"# Evaluation Summary - {run_id}",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Dataset filter: `{report['subset']['dataset_id']}`",
        f"- Domain filter: `{report['subset']['domain']}`",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Seed: `{report['seed']}`",
        f"- Preprocessing mode: `{report['preprocessing_mode']}`",
        f"- Threshold policy: `{report['threshold_selection']['policy']}`",
        f"- Selected threshold: `{report['threshold_selection']['threshold']:.6f}`",
        f"- Calibration: `{calibration['mode']}`",
        f"- Governance allowed: `{report['governance']['allowed']}`",
        f"- Governance reason: `{report['governance']['reason']}`",
        f"- Override used: `{report['governance']['override_used']}`",
        "",
        "## Threshold-Free Metrics",
        "",
        f"- ROC-AUC: `{threshold_free['roc_auc']:.6f}`",
        f"- PR-AUC: `{threshold_free['pr_auc']:.6f}`",
        "",
        "## Thresholded Metrics",
        "",
        f"- Accuracy: `{thresholded['accuracy']:.6f}`",
        f"- Balanced accuracy: `{thresholded['balanced_accuracy']:.6f}`",
        f"- Precision: `{thresholded['precision']:.6f}`",
        f"- Recall: `{thresholded['recall']:.6f}`",
        f"- F1: `{thresholded['f1']:.6f}`",
        "",
        "## Confusion Matrix",
        "",
        f"- TN: `{confusion['tn']}`",
        f"- FP: `{confusion['fp']}`",
        f"- FN: `{confusion['fn']}`",
        f"- TP: `{confusion['tp']}`",
        "",
        "## Calibration",
        "",
        f"- ECE: `{report['calibration_metrics']['ece']:.6f}`",
        f"- Brier score: `{report['calibration_metrics']['brier_score']:.6f}`",
    ]

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return metrics_path, summary_path


def _write_slice_report_files(report_dir: str, run_id: str, report: dict[str, Any]) -> tuple[str, str]:
    metrics_path, summary_path = _slice_report_paths(report_dir, run_id)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    lines = [
        f"# Slice Evaluation Summary - {run_id}",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Tuning split: `{report['tuning_split']}`",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Model: `{report['model_path']}`",
        f"- Preprocessing mode: `{report['preprocessing_mode']}`",
        f"- Threshold policy: `{report['threshold_selection']['policy']}`",
        f"- Selected threshold: `{report['threshold_selection']['threshold']:.6f}`",
        f"- Calibration: `{report['calibration']['mode']}`",
        "",
    ]

    for column_report in report["slice_reports"]:
        lines.extend(
            [
                f"## By `{column_report['column']}`",
                "",
                "| Slice | Samples | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in column_report["rows"]:
            thresholded = row["thresholded_metrics"]
            threshold_free = row["threshold_free_metrics"]
            lines.append(
                "| {slice_value} | {num_samples} | {accuracy:.4f} | {balanced_accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {roc_auc:.4f} | {pr_auc:.4f} |".format(
                    slice_value=row["slice_value"],
                    num_samples=row["num_samples"],
                    accuracy=thresholded["accuracy"],
                    balanced_accuracy=thresholded["balanced_accuracy"],
                    precision=thresholded["precision"],
                    recall=thresholded["recall"],
                    f1=thresholded["f1"],
                    roc_auc=threshold_free["roc_auc"],
                    pr_auc=threshold_free["pr_auc"],
                )
            )
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return metrics_path, summary_path


def _compute_metrics_from_outputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    positive_probs = _positive_probabilities(probabilities)
    thresholded = threshold_metrics(labels, positive_probs, threshold)
    threshold_free = threshold_free_metrics(labels, positive_probs)
    reliability = reliability_diagram_binary(labels, positive_probs)
    brier = brier_score_binary(labels, positive_probs)
    return {
        "thresholded_metrics": thresholded,
        "threshold_free_metrics": threshold_free,
        "calibration_metrics": {
            "ece": float(reliability["ece"]),
            "brier_score": float(brier),
            "reliability_diagram": reliability["bins"],
        },
    }


def _fit_tuning_state(
    model: tf.keras.Model,
    manifest: pd.DataFrame,
    split_name: str,
    preprocess_mode: str,
    threshold_policy: str,
    fixed_threshold: float | None,
    calibrate: str,
    image_size: tuple[int, int],
    batch_size: int,
    dataset_id: str | None = None,
    domain: str | None = None,
    tuning_split_preference: list[str] | None = None,
    forbid_test_tuning: bool = True,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    tuning_split = select_tuning_split_from_policy(
        manifest,
        tuning_split_preference or ["calibration", "val"],
        split_name,
    )
    if forbid_test_tuning and split_name in {"test", "real_world"} and tuning_split == split_name:
        raise ValueError(f"Threshold or calibration tuning on '{split_name}' is forbidden.")

    tuning_frame = _get_split_frame(manifest, tuning_split, dataset_id=dataset_id, domain=domain)
    tuning_outputs = _extract_logits_and_probabilities(model, tuning_frame, preprocess_mode, image_size, batch_size)
    tuning_probs = np.asarray(tuning_outputs["probabilities"], dtype=float)

    tuned_probs, calibration_details = _prepare_calibration(
        calibrate=calibrate,
        tuning_labels=np.asarray(tuning_outputs["labels"], dtype=int),
        tuning_logits=None if tuning_outputs["logits"] is None else np.asarray(tuning_outputs["logits"], dtype=float),
        tuning_probs=tuning_probs,
    )
    threshold_selection = select_threshold(
        np.asarray(tuning_outputs["labels"], dtype=int),
        _positive_probabilities(tuned_probs),
        policy=threshold_policy,
        fixed_threshold=fixed_threshold,
    )
    return tuning_split, calibration_details, threshold_selection


def run_evaluation(
    model_path: str,
    manifest_path: str,
    split_name: str,
    preprocess_mode: str,
    label_config_path: str,
    threshold_policy: str,
    fixed_threshold: float | None,
    calibrate: str,
    report_dir: str,
    seed: int,
    dataset_id: str | None = None,
    domain: str | None = None,
    tuning_split_preference: list[str] | None = None,
    governance_result: dict[str, Any] | None = None,
    forbid_test_tuning: bool = True,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run research-grade evaluation for a specific manifest split."""
    manifest = _load_manifest(manifest_path)
    label_mapping = load_label_mapping(label_config_path)
    label_mapping_hash = _hash_label_mapping(label_mapping)

    if split_name not in set(manifest["split"].astype(str)):
        raise ValueError(f"Requested eval split '{split_name}' does not exist in manifest.")

    model = tf.keras.models.load_model(model_path, compile=False)
    eval_frame = _get_split_frame(manifest, split_name, dataset_id=dataset_id, domain=domain)
    eval_outputs = _extract_logits_and_probabilities(model, eval_frame, preprocess_mode, image_size, batch_size)
    eval_probs = np.asarray(eval_outputs["probabilities"], dtype=float)

    tuning_split, calibration_details, threshold_selection = _fit_tuning_state(
        model=model,
        manifest=manifest,
        split_name=split_name,
        preprocess_mode=preprocess_mode,
        threshold_policy=threshold_policy,
        fixed_threshold=fixed_threshold,
        calibrate=calibrate,
        image_size=image_size,
        batch_size=batch_size,
        dataset_id=dataset_id,
        domain=domain,
        tuning_split_preference=tuning_split_preference,
        forbid_test_tuning=forbid_test_tuning,
    )

    eval_probs = _apply_calibration_to_eval(
        None if eval_outputs["logits"] is None else np.asarray(eval_outputs["logits"], dtype=float),
        eval_probs,
        calibration_details,
    )
    eval_labels = np.asarray(eval_outputs["labels"], dtype=int)
    metrics = _compute_metrics_from_outputs(
        labels=eval_labels,
        probabilities=eval_probs,
        threshold=float(threshold_selection["threshold"]),
    )

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{split_name}"
    report = {
        "run_id": run_id,
        "split_name": split_name,
        "tuning_split": tuning_split,
        "dataset_version": _get_dataset_version(manifest),
        "manifest_path": os.path.abspath(manifest_path),
        "model_path": os.path.abspath(model_path),
        "seed": int(seed),
        "preprocessing_mode": preprocess_mode,
        "subset": {
            "dataset_id": _normalize_optional_filter(dataset_id),
            "domain": _normalize_optional_filter(domain),
        },
        "input_hashes": (governance_result or {}).get("hashes"),
        "baseline_hashes": (governance_result or {}).get("baseline_hashes"),
        "label_mapping_hash": label_mapping_hash,
        "threshold_selection": threshold_selection,
        "calibration": calibration_details,
        "threshold_free_metrics": metrics["threshold_free_metrics"],
        "thresholded_metrics": metrics["thresholded_metrics"],
        "calibration_metrics": metrics["calibration_metrics"],
        "metadata": {
            "num_samples": int(len(eval_labels)),
            "used_logit_recovery": bool(eval_outputs["used_logit_recovery"]),
            "forbid_test_tuning": bool(forbid_test_tuning),
        },
        "governance": governance_result or {
            "allowed": True,
            "reason": "not_provided",
            "override_used": False,
            "valid_for_thesis": True,
            "access_log_path": None,
            "baseline_path": None,
            "hashes": None,
            "baseline_hashes": None,
        },
    }

    metrics_path, summary_path = _report_paths(report_dir, run_id)
    report["artifacts"] = {
        "metrics_json": os.path.abspath(metrics_path),
        "summary_md": os.path.abspath(summary_path),
    }
    metrics_path, summary_path = _write_report_files(report_dir, run_id, report)
    return report


def run_slice_evaluations(
    model_path: str,
    manifest_path: str,
    split_name: str,
    preprocess_mode: str,
    label_config_path: str,
    threshold_policy: str,
    fixed_threshold: float | None,
    calibrate: str,
    report_dir: str,
    seed: int,
    tuning_split_preference: list[str] | None = None,
    forbid_test_tuning: bool = True,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    slice_columns: Sequence[str] = ("domain", "dataset_id"),
) -> dict[str, Any]:
    """Evaluate the saved model across slice values using one globally tuned threshold."""
    manifest = _load_manifest(manifest_path)
    label_mapping = load_label_mapping(label_config_path)
    label_mapping_hash = _hash_label_mapping(label_mapping)

    if split_name not in set(manifest["split"].astype(str)):
        raise ValueError(f"Requested eval split '{split_name}' does not exist in manifest.")

    model = tf.keras.models.load_model(model_path, compile=False)
    tuning_split, calibration_details, threshold_selection = _fit_tuning_state(
        model=model,
        manifest=manifest,
        split_name=split_name,
        preprocess_mode=preprocess_mode,
        threshold_policy=threshold_policy,
        fixed_threshold=fixed_threshold,
        calibrate=calibrate,
        image_size=image_size,
        batch_size=batch_size,
        tuning_split_preference=tuning_split_preference,
        forbid_test_tuning=forbid_test_tuning,
    )

    full_frame = _get_split_frame(manifest, split_name)
    slice_reports: list[dict[str, Any]] = []
    for column in slice_columns:
        if column not in full_frame.columns:
            continue
        values = [
            value for value in sorted(full_frame[column].dropna().astype(str).unique().tolist())
            if value.strip()
        ]
        if not values:
            continue

        rows: list[dict[str, Any]] = []
        for value in values:
            if column == "domain":
                slice_frame = _get_split_frame(manifest, split_name, domain=value)
            elif column == "dataset_id":
                slice_frame = _get_split_frame(manifest, split_name, dataset_id=value)
            else:
                slice_frame = full_frame[full_frame[column].astype(str) == value].copy()
                if slice_frame.empty:
                    continue

            outputs = _extract_logits_and_probabilities(model, slice_frame, preprocess_mode, image_size, batch_size)
            calibrated_probs = _apply_calibration_to_eval(
                None if outputs["logits"] is None else np.asarray(outputs["logits"], dtype=float),
                np.asarray(outputs["probabilities"], dtype=float),
                calibration_details,
            )
            labels = np.asarray(outputs["labels"], dtype=int)
            metrics = _compute_metrics_from_outputs(
                labels=labels,
                probabilities=calibrated_probs,
                threshold=float(threshold_selection["threshold"]),
            )
            rows.append(
                {
                    "slice_value": value,
                    "num_samples": int(len(labels)),
                    "used_logit_recovery": bool(outputs["used_logit_recovery"]),
                    "thresholded_metrics": metrics["thresholded_metrics"],
                    "threshold_free_metrics": metrics["threshold_free_metrics"],
                    "calibration_metrics": metrics["calibration_metrics"],
                }
            )

        rows.sort(key=lambda row: row["thresholded_metrics"]["balanced_accuracy"])
        slice_reports.append({"column": column, "rows": rows})

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{split_name}"
    report = {
        "run_id": run_id,
        "split_name": split_name,
        "tuning_split": tuning_split,
        "dataset_version": _get_dataset_version(manifest),
        "manifest_path": os.path.abspath(manifest_path),
        "model_path": os.path.abspath(model_path),
        "seed": int(seed),
        "preprocessing_mode": preprocess_mode,
        "label_mapping_hash": label_mapping_hash,
        "threshold_selection": threshold_selection,
        "calibration": calibration_details,
        "slice_reports": slice_reports,
    }
    metrics_path, summary_path = _write_slice_report_files(report_dir, run_id, report)
    report["artifacts"] = {
        "metrics_json": os.path.abspath(metrics_path),
        "summary_md": os.path.abspath(summary_path),
    }
    return report
