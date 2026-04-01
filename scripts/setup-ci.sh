#!/bin/bash
# Setup script for CI/CD pipeline
# This script helps developers set up the complete CI/CD environment locally

set -e

echo "=== POLARIS CI/CD Setup ==="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "polaris" ]; then
    echo "ERROR: Please run this script from the POLARIS project root"
    exit 1
fi

PYTHON_BIN="python"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
fi

echo "Using Python interpreter: $PYTHON_BIN"

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')"
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION#*.}"

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10 or newer is required, but found $PYTHON_VERSION"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
"$PYTHON_BIN" -m pip install -e .[dev]

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p metrics
mkdir -p benchmarks
mkdir -p docs/_build

# Check if GitHub token is configured (for auto-commit)
if [ -z "$POLARIS_PAT" ]; then
    echo "WARNING: POLARIS_PAT environment variable not set"
    echo "Auto-commit functionality in GitHub Actions may not work properly"
    echo "Set POLARIS_PAT in your repository secrets"
else
    echo "✓ POLARIS_PAT is configured"
fi

echo "Running initial quality checks via Makefile targets..."
echo ""

if make PYTHON="$PYTHON_BIN" pre-commit-check; then
    echo "✓ Pre-commit-aligned checks passed"
else
    echo "⚠ Pre-commit-aligned checks reported issues"
fi

echo ""

if make PYTHON="$PYTHON_BIN" ci; then
    echo "✓ CI-aligned checks passed"
else
    echo "⚠ CI-aligned checks reported issues"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Available commands:"
echo "  make help           - Show all available commands"
echo "  make test           - Run tests"
echo "  make format         - Format code"
echo "  make ci             - Run full CI pipeline locally"
echo "  make pre-commit     - Run pre-commit checks"
echo ""
echo "The CI/CD pipeline will automatically:"
echo "  - Run tests and coverage on push/PR"
echo "  - Format code and auto-commit changes"
echo "  - Validate code quality and security"
echo "  - Check merge readiness"
echo ""
echo "For more details, see:"
echo "  - .github/workflows/ directory"
echo "  - Makefile"
