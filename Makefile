.PHONY: help install install-dev test test-unit test-integration test-privacy test-performance lint format typecheck clean run-edge run-central run-dashboard docker-build docker-up docker-down benchmark

help:
	@echo "Kizuna Multimodal Privacy Engine - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install all dependencies (dev + dashboard + mlops)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests with coverage"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-privacy     Run privacy verification tests"
	@echo "  make test-performance Run performance benchmarks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Format code with black and isort"
	@echo "  make typecheck        Run mypy type checker"
	@echo ""
	@echo "Running:"
	@echo "  make run-edge         Run simulated edge node"
	@echo "  make run-central      Run central aggregation node"
	@echo "  make run-dashboard    Run Streamlit dashboard"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker images"
	@echo "  make docker-up        Start all containers"
	@echo "  make docker-down      Stop all containers"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make benchmark        Run all benchmarks"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Remove cache and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-privacy:
	pytest tests/privacy/ -v -m privacy

test-performance:
	pytest tests/performance/ -v -m performance --edge

lint:
	ruff check src/ tests/ scripts/ app/

format:
	black src/ tests/ scripts/ app/
	isort src/ tests/ scripts/ app/

typecheck:
	mypy src/ --strict

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete

run-edge:
	python -m src.edge_node --config config/edge.yaml

run-central:
	python -m src.central_node --config config/default.yaml

run-dashboard:
	streamlit run app/dashboard.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

benchmark:
	python scripts/benchmark_latency.py
	python scripts/benchmark_memory.py
	python scripts/benchmark_throughput.py
	python scripts/benchmark_accuracy.py
