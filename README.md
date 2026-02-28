# AID

### AI Information Detection Tool

## Project Overview

This is a graduation project for SE4910 & COMP4910. It consists of a basic frontend, gateway, and two neural networks.
The goal of the project is to be a tool to detect AI generated media.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow?logo=javascript&logoColor=black)
![HTML](https://img.shields.io/badge/HTML5-standard-orange?logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-standard-blue?logo=css3&logoColor=white)

---

## Repository Structure

| Component            | Documentation                          |
| -------------------- | -------------------------------------- |
| **Frontend**         | [Main Directory](./src/frontend) |
| **Server**           | [Main Directory](./src/flask_server) |
| **LSTM Model** | [Main Directory](./src/lstm) |
| **CNN Model** | [README](./src/cnn/CNN.md) |

---

## Governance Baseline

`--eval-only` runs use an immutable governance baseline stored at `governance/baseline.json`.

- The first thesis-valid evaluation creates the baseline from the current governance config, split manifest, model file, and label mapping hashes.
- Later thesis-valid evaluations must match that baseline exactly.
- Hash drift blocks the run with exit code `2`.
- `--override-governance` allows exploratory evaluation to proceed, but the run is marked `valid_for_thesis=false` and the baseline is not updated.
- Access logs are persisted at `governance/access_log.jsonl`.

Minimal verification flow:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only
docker compose run --rm aid python -m src.cnn.main --eval-only
docker compose run --rm aid python -m src.cnn.main --eval-only --override-governance
```

## How To Run (Docker-Only)

Host `python` is not supported for this repo. Always run through Docker.

```bash
docker compose build aid
docker compose run --rm aid python -m src.cnn.main
docker compose run --rm aid python -m src.cnn.main --eval-only
docker compose run --rm aid python -m src.cnn.main --eval-only --split-manifest /app/src/cnn/splits/split_manifest.csv
docker compose run --rm aid python -m src.cnn.main --eval-only --report-dir /app/reports
```
