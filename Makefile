.PHONY: help install install-dev test test-verbose test-all format format-check lint type-check clean install-hooks run-hooks build docs

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
	pip install -e .

install-dev:
	pip install -e .[dev]

# Testing
test:
	pytest tests/ -v --tb=short

test-verbose:
	pytest tests/ -vv --tb=long

test-all:
	pytest tests/ -v --cov=polaris --cov-report=term-missing --cov-fail-under=40

# Code formatting
format:
	black --line-length=100 polaris/ tests/ examples/
	isort --line-length=100 polaris/ tests/ examples/

format-check:
	black --check --diff --line-length=100 polaris/ tests/ examples/
	isort --check-only --diff --line-length=100 polaris/ tests/ examples/

# Linting and quality
lint:
	flake8 polaris/ examples/ --max-line-length=100 --extend-ignore=E203,W503,E221,E225,E231,E272,E501,E201,E202

type-check:
	mypy polaris/ --ignore-missing-imports

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
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build --wheel --sdist .

# Full CI pipeline locally
ci: format lint type-check test-all
	@echo "All CI checks passed!"

# Quick check before commit
pre-commit: format-check lint type-check test
	@echo "Pre-commit checks completed!"
