# AID

AI Information Detection is a graduation project with two main detection tracks:

- `src/cnn/`: real vs synthetic image detection
- `src/lstm/`: AI vs human text detection

The current Docker-based training and benchmarking workflow is centered on the image/CNN side, while the text/LSTM side remains part of the project and is used by the Flask API.

## Current Scope

- image-level real vs synthetic classification
- text-level AI vs human classification
- Docker-first training and evaluation
- manifest-based dataset splits
- dataset hygiene utilities for corrupt, oversized, and duplicate samples
- calibration, corruption evaluation, and slice reports
- multi-backbone benchmarking

The prepared image training dataset is expected under `Dataset/prepared/combined_train`.

## Main Paths

| Area | Path |
| --- | --- |
| CNN pipeline | [src/cnn/CNN.md](./src/cnn/CNN.md) |
| LSTM pipeline | `src/lstm/` |
| Flask API | `src/flask_server/` |
| Frontend | `src/frontend/` |
| Docker usage | [docs/DOCKER_USAGE.md](./docs/DOCKER_USAGE.md) |
| Compose services | [docker-compose.yml](./docker-compose.yml) |
| Docker image | [Dockerfile](./Dockerfile) |
| Helper commands | [Makefile](./Makefile), [run.ps1](./run.ps1) |

## Project Structure

### CNN Track

The image pipeline supports multiple backbones, manifest-based evaluation, corruption testing, duplicate audits, and benchmark sweeps.

Supported architectures:

- `simple`
- `dual_artifact_cnn`
- `efficientnet_b0`
- `efficientnet_v2b0`
- `resnet50`
- `convnext_tiny`
- `clip_vit_b32`

### LSTM Track

The text pipeline lives under `src/lstm/` and includes:

- `lstm_new.py`: main text training script
- `services.py`: text inference service used by the API
- `demo.py`: local text demo loop

### Application Layer

- `src/flask_server/run.py` exposes `/analyze_image` and `/analyze_text`
- `src/frontend/` contains the web UI assets

## Quick Start For Images

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

## Image Dataset Preflight

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

- The Docker workflow in this repo currently documents the CNN/image pipeline.
- The LSTM/text pipeline still exists and is used by the Flask layer, but it is not the part currently wired into the Docker training flow.
- Newer baselines such as `clip_vit_b32` require rebuilding the image so fresh dependencies are installed.
- Reports and generated benchmark artifacts are written into the repository because the project root is bind-mounted to `/app`.
