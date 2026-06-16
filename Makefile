.PHONY: help sync data train train-tuned evaluate run test lint format \
        docker-build docker-up docker-down docker-logs clean

UV ?= uv
PORT ?= 8000

help:
	@echo "Available targets:"
	@echo "  sync           Create venv and install runtime + dev deps via uv"
	@echo "  data           Download raw data and build processed dataset"
	@echo "  train          Train the classifier and save artifacts to models/"
	@echo "  train-tuned    Train with GridSearchCV over alpha + ngram ranges (slower)"
	@echo "  evaluate       Re-evaluate the trained model on the test split"
	@echo "  run            Run the FastAPI service locally on PORT (default 8000)"
	@echo "  test           Run pytest"
	@echo "  lint           Run ruff lint"
	@echo "  format         Run ruff format"
	@echo "  docker-build   Build the Docker image"
	@echo "  docker-up      Start the classifier service via docker-compose"
	@echo "  docker-down    Stop the docker-compose stack"
	@echo "  docker-logs    Tail container logs"
	@echo "  clean          Remove caches and build artifacts"

sync:
	$(UV) sync --all-groups

data:
	$(UV) sync --group data
	$(UV) run python -m scripts.download_data
	$(UV) run python -m scripts.build_dataset

train:
	$(UV) run python -m scripts.train

train-tuned:
	$(UV) run python -m scripts.train --grid-search

evaluate:
	$(UV) run python -m scripts.evaluate

run:
	$(UV) run uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f classifier

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
