param(
    [string]$Action = "test"
)

switch ($Action) {
    "build" {
        docker compose build aid
    }
    "train" {
        docker compose run --rm aid python -m src.cnn.main
    }
    "eval" {
        docker compose run --rm aid python -m src.cnn.main --eval-only
    }
    "eval-manifest" {
        docker compose run --rm aid python -m src.cnn.main --eval-only --split-manifest /app/Dataset/prepared/combined_train/split_manifest.csv
    }
    "eval-report" {
        docker compose run --rm aid python -m src.cnn.main --eval-only --report-dir /app/reports
    }
    "shell" {
        docker compose run --rm aid sh
    }
    "clean" {
        docker compose down -v
    }
    "test" {
        docker compose run --rm aid python -m src.cnn.test
    }
    default {
        Write-Host "Unknown action. Use: build | train | eval | eval-manifest | eval-report | shell | clean"
    }
}
