.PHONY: install install-dev train export-onnx serve test lint fmt type-check ci-local docker-up smoke-test clean drift loadtest perf-estimate validate-data lock pre-commit docs hf-push

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
	pytest -m "not slow" --cov=moviesentiment --cov-fail-under=85 -v

test-slow:
	pytest -m slow -v

drift:
	python -m moviesentiment.cli drift

validate-data:
	python -m moviesentiment.data.validate data/interim/clean.parquet metrics/data_quality.json

perf-estimate:
	python -m moviesentiment.cli perf-estimate

loadtest:
	locust -f scripts/load_test.py --host http://localhost:8000 --headless -u 50 -r 5 -t 120s --html docs/load_test_report.html

lock:
	bash scripts/compile_requirements.sh

pre-commit:
	pre-commit run --all-files

docs:
	bash scripts/build_docs.sh

# Push v1 + v2 ONNX models to HuggingFace Hub. Requires HF_TOKEN env var
# (write scope). Idempotent — re-running uploads only changed files.
hf-push:
	python scripts/push_to_hf.py

lint:
	ruff check src tests
	black --check src tests

fmt:
	ruff check --fix src tests
	black src tests

type-check:
	mypy src tests

ci-local: lint type-check test

# Hit the live ECS Fargate endpoint. Resolves the current task's public IP, then curls /predict.
smoke-test:
	@TASK_ARN=$$(aws ecs list-tasks --cluster moviesentiment --service-name moviesentiment --region ap-southeast-2 --query 'taskArns[0]' --output text); \
	 ENI=$$(aws ecs describe-tasks --cluster moviesentiment --tasks "$$TASK_ARN" --region ap-southeast-2 --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text); \
	 IP=$$(aws ec2 describe-network-interfaces --network-interface-ids "$$ENI" --region ap-southeast-2 --query 'NetworkInterfaces[0].Association.PublicIp' --output text); \
	 echo "Live URL: http://$$IP:8000"; \
	 curl -s -X POST "http://$$IP:8000/predict" -H "Content-Type: application/json" -d '{"texts":["A masterpiece.","Worst film I have seen."]}'

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
