import argparse
import json
import os
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from calibration import brier_score_binary, reliability_diagram_binary
from eval_metrics import threshold_free_metrics, threshold_metrics
from preprocessing import get_default_preprocess_mode, get_preprocess_fn
from split_utils import get_default_manifest_path, load_manifest_dataframe


def _normalize_optional_filter(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


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


def _box_blur(img: tf.Tensor) -> tf.Tensor:
    kernel = tf.constant(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=tf.float32,
    ) / 9.0
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(tf.expand_dims(img, axis=0), kernel, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(blurred, axis=0)


def _corruption_fn(name: str, image_size: tuple[int, int]) -> Callable[[tf.Tensor], tf.Tensor]:
    normalized_name = name.strip().lower()

    def _jpeg_quality(img: tf.Tensor, quality: int) -> tf.Tensor:
        encoded = tf.io.encode_jpeg(tf.cast(tf.clip_by_value(img, 0.0, 1.0) * 255.0, tf.uint8), quality=quality)
        decoded = tf.io.decode_jpeg(encoded, channels=3)
        return tf.cast(decoded, tf.float32) / 255.0

    if normalized_name == "jpeg30":
        return lambda img: _jpeg_quality(img, 30)
    if normalized_name == "jpeg50":
        return lambda img: _jpeg_quality(img, 50)
    if normalized_name == "blur":
        return _box_blur
    if normalized_name == "resize_down":
        return lambda img: tf.image.resize(
            tf.image.resize(img, (112, 112), method="area"),
            image_size,
            method="bilinear",
            antialias=True,
        )
    if normalized_name == "noise":
        return lambda img: tf.clip_by_value(img + tf.random.normal(tf.shape(img), stddev=0.05), 0.0, 1.0)
    raise ValueError(f"Unsupported corruption: {name}")


def _build_dataset(
    frame: pd.DataFrame,
    preprocess_mode: str,
    image_size: tuple[int, int],
    batch_size: int,
    corruption_name: str,
) -> tf.data.Dataset:
    preprocess_fn = get_preprocess_fn(preprocess_mode)
    corrupt = _corruption_fn(corruption_name, image_size=image_size)
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label"].astype(int).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda path, label: preprocess_fn(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    dataset = dataset.filter(lambda img, label: tf.shape(img)[0] == image_size[0])
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.map(lambda img, label: (tf.clip_by_value(corrupt(img), 0.0, 1.0), label), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _collect_labels(dataset: tf.data.Dataset) -> np.ndarray:
    batches = [np.asarray(batch_labels) for _, batch_labels in dataset]
    if not batches:
        return np.asarray([], dtype=int)
    return np.concatenate(batches, axis=0).astype(int)


def _positive_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    return probabilities[:, 1].astype(float)


def _evaluate_corruption(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    preprocess_mode: str,
    corruption_name: str,
    image_size: tuple[int, int],
    batch_size: int,
) -> dict[str, object]:
    dataset = _build_dataset(frame, preprocess_mode, image_size, batch_size, corruption_name)
    labels = _collect_labels(dataset)
    probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=float)
    positive_probs = _positive_probabilities(probabilities)

    threshold_free = threshold_free_metrics(labels, positive_probs)
    thresholded = threshold_metrics(labels, positive_probs, threshold=0.5)
    reliability = reliability_diagram_binary(labels, positive_probs)
    brier = brier_score_binary(labels, positive_probs)

    return {
        "corruption": corruption_name,
        "num_samples": int(len(labels)),
        "threshold_free_metrics": threshold_free,
        "thresholded_metrics": thresholded,
        "calibration_metrics": {
            "ece": float(reliability["ece"]),
            "brier_score": float(brier),
            "reliability_diagram": reliability["bins"],
        },
    }


def _report_paths(report_dir: str, run_id: str) -> tuple[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    return (
        os.path.join(report_dir, f"{run_id}_corruptions.json"),
        os.path.join(report_dir, f"{run_id}_corruptions.md"),
    )


def _write_reports(report_dir: str, run_id: str, report: dict[str, object]) -> tuple[str, str]:
    json_path, md_path = _report_paths(report_dir, run_id)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    rows = [
        "| Corruption | ROC-AUC | PR-AUC | Accuracy | Balanced Acc. | ECE | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        threshold_free = result["threshold_free_metrics"]
        thresholded = result["thresholded_metrics"]
        calibration = result["calibration_metrics"]
        rows.append(
            "| {corruption} | {roc_auc:.4f} | {pr_auc:.4f} | {accuracy:.4f} | {balanced_accuracy:.4f} | {ece:.4f} | {brier:.4f} |".format(
                corruption=result["corruption"],
                roc_auc=threshold_free["roc_auc"],
                pr_auc=threshold_free["pr_auc"],
                accuracy=thresholded["accuracy"],
                balanced_accuracy=thresholded["balanced_accuracy"],
                ece=calibration["ece"],
                brier=calibration["brier_score"],
            )
        )

    lines = [
        f"# Corruption Evaluation - {run_id}",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Model: `{report['model_path']}`",
        f"- Preprocessing mode: `{report['preprocess_mode']}`",
        f"- Dataset filter: `{report['subset']['dataset_id']}`",
        f"- Domain filter: `{report['subset']['domain']}`",
        "",
        *rows,
        "",
    ]

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corruption robustness evaluation on a manifest split.")
    parser.add_argument("--manifest-path", default=get_default_manifest_path(), help="Path to the split manifest CSV.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model file.")
    parser.add_argument("--split", default="val", help="Manifest split to evaluate.")
    parser.add_argument("--dataset-id", default=None, help="Optional dataset_id subset filter.")
    parser.add_argument("--domain", default=None, help="Optional domain subset filter.")
    parser.add_argument("--preprocess-mode", default=get_default_preprocess_mode(), choices=["rgb", "sobel", "rgb+sobel"])
    parser.add_argument("--image-size", type=int, default=224, help="Square image size used for preprocessing and corruptions.")
    parser.add_argument("--report-dir", default="/app/reports", help="Directory for JSON and markdown outputs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for reproducible corruption evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    manifest = load_manifest_dataframe(args.manifest_path)
    frame = _get_split_frame(manifest, args.split, dataset_id=args.dataset_id, domain=args.domain)
    model = tf.keras.models.load_model(args.model_path, compile=False)

    corruption_names = ["jpeg30", "jpeg50", "blur", "resize_down", "noise"]
    results = [
        _evaluate_corruption(
            model=model,
            frame=frame,
            preprocess_mode=args.preprocess_mode,
            corruption_name=corruption_name,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
        )
        for corruption_name in corruption_names
    ]

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{args.split}_corruptions"
    report = {
        "run_id": run_id,
        "split_name": args.split,
        "manifest_path": os.path.abspath(args.manifest_path),
        "model_path": os.path.abspath(args.model_path),
        "preprocess_mode": args.preprocess_mode,
        "image_size": int(args.image_size),
        "seed": int(args.seed),
        "subset": {
            "dataset_id": _normalize_optional_filter(args.dataset_id),
            "domain": _normalize_optional_filter(args.domain),
        },
        "results": results,
    }
    json_path, md_path = _write_reports(args.report_dir, run_id, report)
    print(f"Corruption JSON report saved to: {json_path}")
    print(f"Corruption markdown summary saved to: {md_path}")


if __name__ == "__main__":
    main()
