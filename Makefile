.PHONY: help install install-dev test test-verbose test-all format format-check lint type-check dependency-check pre-commit-check ci clean install-hooks run-hooks build docs

LINE_LENGTH ?= 100
PYTHON ?= python
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
	@echo "  clean            Clean build artifacts"
	@echo "  install-hooks    Install pre-commit hooks"
	@echo "  run-hooks        Run pre-commit hooks manually"
	@echo "  build            Build package for distribution"
	@echo "  docs             Build documentation"

# Installation
install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

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

# Pre-commit-aligned local check
pre-commit-check: format-check lint type-check test dependency-check

# Full CI pipeline locally
ci: format-check lint type-check test-all dependency-check
	@echo "All CI checks passed!"

# Quick check before commit
pre-commit: pre-commit-check
	@echo "Pre-commit checks completed!"
