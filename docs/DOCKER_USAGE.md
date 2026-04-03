# Docker Usage

Run all commands from the repository root.

## Build

CPU image:

```bash
docker compose build aid
```

GPU image:

```bash
docker compose build aid-gpu
```

Rebuild after dependency changes:

```bash
docker compose build --no-cache aid
docker compose build --no-cache aid-gpu
```

## Training

CPU:

```bash
docker compose run --rm aid python -m src.cnn.main
```

GPU:

```bash
docker compose --profile gpu run --rm aid-gpu python -m src.cnn.main
```

Example transfer run:

```bash
docker compose --profile gpu run --rm aid-gpu python -m src.cnn.main --arch efficientnet_v2b0 --preprocess-mode rgb --batch-size 8 --epochs 6 --finetune-unfreeze --finetune-freeze-epochs 2 --sampling-strategy domain_balanced
```

Example task-specific run:

```bash
docker compose --profile gpu run --rm aid-gpu python -m src.cnn.main --arch dual_artifact_cnn --preprocess-mode rgb --batch-size 8 --epochs 8 --sampling-strategy domain_label_balanced
```

## Evaluation

Default eval:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only
```

Eval with the active prepared manifest:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only --split-manifest /app/Dataset/prepared/combined_train/split_manifest.csv
```

Write reports under the repo root:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only --report-dir /app/reports
```

Corruption evaluation:

```bash
docker compose run --rm aid python -m src.cnn.corruption_eval --model-path /app/src/cnn/model/ai_detection_model.h5 --report-dir /app/reports
```

## Benchmarking

Dry-run benchmark matrix:

```bash
docker compose --profile benchmark run --rm aid-benchmark --project-root /app --manifest-paths /app/Dataset/prepared/combined_train/split_manifest.csv --output-dir /app/reports/benchmark_matrix_dryrun --dry-run
```

GPU benchmark matrix:

```bash
docker compose --profile benchmark-gpu run --rm aid-benchmark-gpu --project-root /app --manifest-paths /app/Dataset/prepared/combined_train/split_manifest.csv --output-dir /app/reports/benchmark_matrix_gpu --finetune-unfreeze
```

Rank benchmark results:

```bash
docker compose run --rm aid python -m src.cnn.select_best_run --summary-csv /app/reports/benchmark_matrix_gpu/benchmark_summary.csv --primary-metric final_val_accuracy --output-path /app/reports/benchmark_matrix_gpu/ranked_runs.csv
```

## Dataset Utilities

Generate leave-one-domain-out manifests:

```bash
docker compose run --rm aid python -m src.cnn.generate_holdout_manifests --manifest-path /app/Dataset/prepared/combined_train/split_manifest.csv --group-column domain --output-dir /app/Dataset/prepared/combined_train/splits/leave_one_domain_out
```

Duplicate audit:

```bash
docker compose run --rm aid python -m src.cnn.audit_duplicates --manifest-path /app/Dataset/prepared/combined_train/split_manifest.csv --output-dir /app/reports/duplicate_audit
```

Corrupt and suspicious image scan:

```bash
docker compose run --rm aid python -m src.cnn.clean_corrupt_images --dataset-path /app/Dataset/prepared/combined_train --report-only
```

Fix exact duplicate train/val leakage using an existing audit:

```bash
docker compose run --rm aid python -m src.cnn.fix_exact_duplicate_split_leakage --manifest-path /app/Dataset/prepared/combined_train/split_manifest.csv --exact-duplicates-path /app/reports/duplicate_audit/exact_duplicates.csv --backup
```

## Notes

- The repository is bind-mounted to `/app`, so generated outputs stay on the host.
- `clip_vit_b32` requires rebuilding the image because it depends on `transformers`.
- `clip_vit_b32` and `dual_artifact_cnn` are intended to run with `--preprocess-mode rgb`.
