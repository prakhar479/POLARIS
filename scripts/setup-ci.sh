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

# Install dependencies
echo "Installing dependencies..."
pip install -e .[dev]

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

# Run initial checks
echo "Running initial quality checks..."
echo ""

echo "=== Black Formatting Check ==="
if black --check polaris/ tests/ examples/; then
    echo "✓ Code is properly formatted with Black"
else
    echo "⚠ Code needs formatting. Run 'make format' to fix"
fi

echo ""
echo "=== isort Import Sorting Check ==="
if isort --check-only polaris/ tests/ examples/; then
    echo "✓ Imports are properly sorted"
else
    echo "⚠ Imports need sorting. Run 'make format' to fix"
fi

echo ""
echo "=== mypy Type Checking ==="
if mypy polaris/ --ignore-missing-imports; then
    echo "✓ Type checking passed"
else
    echo "⚠ Type checking issues found"
fi

echo ""
echo "=== Test Suite ==="
if pytest tests/ -v --tb=short; then
    echo "✓ All tests passed"
else
    echo "⚠ Some tests failed"
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
