# AID

AI Information Detection is a graduation project focused on real vs synthetic image detection. The active research and training workflow is centered on the CNN pipeline under `src/cnn/`.

## Current Scope

- image-level real vs synthetic classification
- Docker-first training and evaluation
- manifest-based dataset splits
- dataset hygiene utilities for corrupt, oversized, and duplicate samples
- calibration, corruption evaluation, and slice reports
- multi-backbone benchmarking

The prepared training dataset is expected under `Dataset/prepared/combined_train`.

## Main Paths

| Area | Path |
| --- | --- |
| CNN pipeline | [src/cnn/CNN.md](./src/cnn/CNN.md) |
| Docker usage | [docs/DOCKER_USAGE.md](./docs/DOCKER_USAGE.md) |
| Compose services | [docker-compose.yml](./docker-compose.yml) |
| Docker image | [Dockerfile](./Dockerfile) |
| Helper commands | [Makefile](./Makefile), [run.ps1](./run.ps1) |

## Supported CNN Architectures

- `simple`
- `dual_artifact_cnn`
- `efficientnet_b0`
- `efficientnet_v2b0`
- `resnet50`
- `convnext_tiny`
- `clip_vit_b32`

## Quick Start

Build:

```bash
docker compose build aid
```

Train:

```bash
docker compose run --rm aid python -m src.cnn.main
```

GPU train:

```bash
docker compose --profile gpu run --rm aid-gpu python -m src.cnn.main
```

Eval:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only
```

Eval with the active prepared manifest:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only --split-manifest /app/Dataset/prepared/combined_train/split_manifest.csv
```

## Dataset Preflight

Corrupt and suspicious image scan:

```bash
docker compose run --rm aid python -m src.cnn.clean_corrupt_images --dataset-path /app/Dataset/prepared/combined_train --report-only
```

Duplicate audit:

```bash
docker compose run --rm aid python -m src.cnn.audit_duplicates --manifest-path /app/Dataset/prepared/combined_train/split_manifest.csv --output-dir /app/reports/duplicate_audit
```

Fix exact duplicate train/val leakage using an existing audit:

```bash
docker compose run --rm aid python -m src.cnn.fix_exact_duplicate_split_leakage --manifest-path /app/Dataset/prepared/combined_train/split_manifest.csv --exact-duplicates-path /app/reports/duplicate_audit/exact_duplicates.csv --backup
```

## Notes

- The project is intended to be run through Docker.
- Newer baselines such as `clip_vit_b32` require rebuilding the image so fresh dependencies are installed.
- Reports and generated benchmark artifacts are written into the repository because the project root is bind-mounted to `/app`.
