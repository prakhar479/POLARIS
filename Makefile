.PHONY: help install install-dev test test-verbose test-all format format-check lint type-check dependency-check dependency-policy-check pre-commit-check ci clean install-hooks run-hooks build docs docker-build-core docker-build-full docker-build-ci docker-smoke

LINE_LENGTH ?= 100
PYTHON ?= python
PIP_CONSTRAINTS ?= requirements/constraints.txt
PIP_CONSTRAINT_ARGS := $(if $(wildcard $(PIP_CONSTRAINTS)),-c $(PIP_CONSTRAINTS),)
BLACK_ARGS := --line-length=$(LINE_LENGTH)
ISORT_ARGS := --profile=black --line-length=$(LINE_LENGTH)
MYPY_ARGS := --ignore-missing-imports
FLAKE8_ARGS := polaris/ examples/ --max-line-length=$(LINE_LENGTH) --extend-ignore=E203,W503,E221,E225,E231,E272,E501,E201,E202

# Default target
help:
	@echo "Available commands:"
	@echo "  install          Install package dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run test suite"
	@echo "  test-verbose     Run test suite with verbose output"
	@echo "  test-all         Run complete test suite with coverage"
	@echo "  format           Format code with black and isort"
	@echo "  format-check     Check if code is formatted"
	@echo "  lint             Run linting checks"
	@echo "  type-check       Run mypy type checking"
	@echo "  dependency-policy-check  Validate dependency metadata consistency"
	@echo "  clean            Clean build artifacts"
	@echo "  install-hooks    Install pre-commit hooks"
	@echo "  run-hooks        Run pre-commit hooks manually"
	@echo "  build            Build package for distribution"
	@echo "  docs             Build documentation"
	@echo "  docker-build-core Build core runtime image"
	@echo "  docker-build-full Build full-feature runtime image"
	@echo "  docker-build-ci  Build CI image"
	@echo "  docker-smoke     Run Docker smoke checks"

# Installation
install:
	$(PYTHON) -m pip install $(PIP_CONSTRAINT_ARGS) -e .

install-dev:
	$(PYTHON) -m pip install $(PIP_CONSTRAINT_ARGS) -e .[dev]

# Testing
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-verbose:
	$(PYTHON) -m pytest tests/ -vv --tb=long

test-all:
	$(PYTHON) -m pytest tests/ -v --cov=polaris --cov-report=xml --cov-report=term-missing --cov-fail-under=60

# Code formatting
format:
	$(PYTHON) -m black $(BLACK_ARGS) polaris/ tests/ examples/
	$(PYTHON) -m isort $(ISORT_ARGS) polaris/ tests/ examples/

format-check:
	$(PYTHON) -m black --check --diff $(BLACK_ARGS) polaris/ tests/ examples/
	$(PYTHON) -m isort --check-only --diff $(ISORT_ARGS) polaris/ tests/ examples/

# Linting and quality
lint:
	$(PYTHON) -m flake8 $(FLAKE8_ARGS)

type-check:
	$(PYTHON) -m mypy $(MYPY_ARGS) polaris/

dependency-check:
	$(PYTHON) -m pip check

dependency-policy-check:
	$(PYTHON) scripts/check_dependency_standardization.py

# Development workflow
install-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

run-hooks:
	pre-commit run --all-files

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf .benchmark/
	rm -rf htmlcov/
	rm -rf coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	find . -type f -name "*.pyc" -delete

build: clean
	$(PYTHON) -m build --wheel --sdist .

# Docker
docker-build-core:
	docker build --target core -t polaris:core .

docker-build-full:
	docker build --target full-feature -t polaris:full-feature .

docker-build-ci:
	docker build --target ci -t polaris:ci .

docker-smoke: docker-build-core
	docker run --rm polaris:core --version
	docker run --rm polaris:core doctor --config /app/config/default.yaml

# Pre-commit-aligned local check
pre-commit-check: dependency-policy-check format-check lint type-check test dependency-check

# Full CI pipeline locally
ci: dependency-policy-check format-check lint type-check test-all dependency-check
	@echo "All CI checks passed!"

# Quick check before commit
pre-commit: pre-commit-check
	@echo "Pre-commit checks completed!"
