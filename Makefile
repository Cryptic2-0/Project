.PHONY: install install-dev train export-onnx serve test lint fmt type-check ci-local docker-up clean

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

train:
	dvc repro

export-onnx:
	python -m moviesentiment.models.onnx_export

docker-up:
	docker compose -f deploy/docker-compose.yml up --build

serve:
	uvicorn moviesentiment.serve.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest --cov=moviesentiment --cov-fail-under=70 -v

lint:
	ruff check src tests
	black --check src tests

fmt:
	ruff check --fix src tests
	black src tests

type-check:
	mypy src

ci-local: lint type-check test

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
