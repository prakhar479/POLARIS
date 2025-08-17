#!/usr/bin/env python3
"""
Test runner script for POLARIS Digital Twin.

This script runs the test suite with proper async support and configuration.
"""

import sys
import subprocess
from pathlib import Path


def run_all_tests():
    """Run all tests with proper configuration."""
    print("Running all tests...")
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("✓ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Tests failed with exit code {e.returncode}")
        return False


def run_specific_test_file(test_file):
    """Run tests from a specific file."""
    print(f"Running tests from {test_file}...")
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/{test_file}",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print(f"✓ Tests in {test_file} passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Tests in {test_file} failed with exit code {e.returncode}")
        return False


def run_non_async_tests_only():
    """Run only non-async tests for basic verification."""
    print("Running non-async tests only...")
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_digital_twin_events.py",
        "tests/test_world_model.py::TestWorldModelFactory",
        "tests/test_world_model.py::TestWorldModelInterface",
        "tests/test_world_model.py::TestWorldModelExceptions",
        "tests/test_proto_wrappers.py::TestProtobufConverter",
        "-v",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("✓ Non-async tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Non-async tests failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner."""
    print("POLARIS Digital Twin - Test Runner")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--non-async":
            success = run_non_async_tests_only()
        elif sys.argv[1].startswith("test_"):
            success = run_specific_test_file(sys.argv[1])
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python run_tests.py [--non-async | test_filename.py]")
            sys.exit(1)
    else:
        success = run_all_tests()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()