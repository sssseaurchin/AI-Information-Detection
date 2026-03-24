.PHONY: build train eval eval-manifest eval-report shell clean

build:
	docker compose build aid

train:
	docker compose run --rm aid python -m src.cnn.main

eval:
	docker compose run --rm aid python -m src.cnn.main --eval-only

eval-manifest:
	docker compose run --rm aid python -m src.cnn.main --eval-only --split-manifest /app/src/cnn/splits/split_manifest.csv

eval-report:
	docker compose run --rm aid python -m src.cnn.main --eval-only --report-dir /app/reports

shell:
	docker compose run --rm aid sh

clean:
	docker compose down -v
