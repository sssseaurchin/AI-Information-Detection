# CNN Pipeline

This directory contains the active real-vs-synthetic image detection workflow.

## Main Files

- `main.py`: training and evaluation CLI
- `cnnModel.py`: model definitions and training loop
- `preprocessing.py`: shared preprocessing modes
- `eval_runner.py`: evaluation reports and calibration-aware metrics
- `corruption_eval.py`: corruption robustness evaluation
- `split_utils.py`: manifest loading and split generation

## Supported Architectures

- `simple`
- `dual_artifact_cnn`
- `efficientnet_b0`
- `efficientnet_v2b0`
- `resnet50`
- `convnext_tiny`
- `clip_vit_b32`

## Supported Preprocessing Modes

- `rgb`
- `sobel`
- `rgb+sobel`
- `wavelet`
- `rgb+wavelet`

Notes:

- `clip_vit_b32` should be used with `rgb`
- `dual_artifact_cnn` should also be used with `rgb` because it builds artifact branches internally

## Current Features

- deterministic manifest-based train/val splits
- `dataset_id` and `domain` metadata
- domain-aware augmentation
- staged fine-tuning for transfer backbones
- post-run slice reports
- multi-run benchmark tooling
- duplicate and near-duplicate audit tooling
- corrupt and oversized image hygiene tooling

## Common Commands

Train:

```bash
python -m src.cnn.main
```

Eval:

```bash
python -m src.cnn.main --eval-only
```

Benchmark matrix:

```bash
python -m src.cnn.run_benchmark_matrix --project-root /app --manifest-paths /app/Dataset/prepared/combined_train/split_manifest.csv --output-dir /app/reports/benchmark_matrix
```

Duplicate audit:

```bash
python -m src.cnn.audit_duplicates --manifest-path /app/Dataset/prepared/combined_train/split_manifest.csv --output-dir /app/reports/duplicate_audit
```

Corrupt scan:

```bash
python -m src.cnn.clean_corrupt_images --dataset-path /app/Dataset/prepared/combined_train --report-only
```

Fix exact duplicate split leakage:

```bash
python -m src.cnn.fix_exact_duplicate_split_leakage --manifest-path /app/Dataset/prepared/combined_train/split_manifest.csv --exact-duplicates-path /app/reports/duplicate_audit/exact_duplicates.csv --backup
```
