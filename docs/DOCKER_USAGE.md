# Docker Usage

This project is intended to run through Docker from the repository root.

## Standard Commands

```bash
docker compose build aid
docker compose run --rm aid python -m src.cnn.main
docker compose run --rm aid python -m src.cnn.main --eval-only
docker compose run --rm aid python -m src.cnn.main --eval-only --split-manifest /app/src/cnn/splits/split_manifest.csv
docker compose run --rm aid python -m src.cnn.main --eval-only --report-dir /app/reports
```

## Windows PowerShell

Build:

```powershell
docker compose build aid
```

Train:

```powershell
docker compose run --rm aid python -m src.cnn.main
```

Eval:

```powershell
docker compose run --rm aid python -m src.cnn.main --eval-only
```

Eval with a fixed manifest:

```powershell
docker compose run --rm aid python -m src.cnn.main --eval-only --split-manifest /app/src/cnn/splits/split_manifest.csv
```

Eval with report output under the repo root:

```powershell
docker compose run --rm aid python -m src.cnn.main --eval-only --report-dir /app/reports
```

## Linux / macOS

Build:

```bash
docker compose build aid
```

Train:

```bash
docker compose run --rm aid python -m src.cnn.main
```

Eval:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only
```

Eval with a fixed manifest:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only --split-manifest /app/src/cnn/splits/split_manifest.csv
```

Eval with report output under the repo root:

```bash
docker compose run --rm aid python -m src.cnn.main --eval-only --report-dir /app/reports
```

## Notes

- Host `python` is not supported for this workflow.
- The bind mount `.:/app` keeps `reports/` and `governance/` on the host.
- If GPU support is available, use the `aid-gpu` service/profile from `docker-compose.yml`.
