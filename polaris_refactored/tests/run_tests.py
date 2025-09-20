#!/usr/bin/env python3
"""
POLARIS Test Runner

Comprehensive test runner for the POLARIS testing framework that provides
various testing modes and detailed reporting.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence
import time
import shutil

from tests.utils.coverage_utils import CoverageAnalyzer, TestMetrics


class PolarisTestRunner:
    """Main test runner for POLARIS framework."""

    def __init__(self, source_dir: str = "src", test_dir: str = "tests"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.coverage_analyzer = CoverageAnalyzer(source_dir, test_dir)
        self.test_metrics = TestMetrics()

    # ---------------------
    # Helpers
    # ---------------------
    @staticmethod
    def _looks_like_glob(pattern: str) -> bool:
        return any(ch in pattern for ch in ("*", "?", "["))

    @staticmethod
    def _looks_like_nodeid(pattern: str) -> bool:
        # pytest node ids use :: to separate file and test
        return "::" in pattern

    @staticmethod
    def _looks_like_k_expression(pattern: str) -> bool:
        # crude but practical heuristic: boolean ops or whitespace likely indicate -k expression
        lowered = pattern.lower()
        return any(op in lowered for op in (" and ", " or ", " not ", "(", ")")) or " " in pattern

    def _expand_glob_in_dir(self, directory: Path, pattern: str) -> List[str]:
        if not directory.exists():
            return []
        matches = sorted(directory.glob(pattern))
        # Convert to string paths
        return [str(p) for p in matches if p.is_file()]

    # ---------------------
    # Test runners
    # ---------------------
    def run_unit_tests(
        self,
        coverage: bool = True,
        verbose: bool = True,
        fail_fast: bool = False,
        pattern: Optional[str] = None,
    ) -> bool:
        """Run unit tests with optional coverage analysis.

        pattern behavior:
          - None: run entire tests/unit directory
          - glob (contains *, ?, [): expand relative to tests/unit and run matched files
          - nodeid (contains ::): pass directly to pytest (file::testname)
          - boolean expression or plain substring: pass to pytest -k
        """
        print("🧪 Running POLARIS Unit Tests...")

        unit_dir = self.test_dir / "unit"
        if not unit_dir.exists():
            print(f"❌ Unit tests directory not found: {unit_dir}")
            return False

        # Base command
        cmd: List[str] = [sys.executable, "-m", "pytest"]

        # Decide how to invoke pytest based on pattern
        used_files: List[str] = []
        used_k_expr: Optional[str] = None

        if pattern:
            # Node id: run directly (e.g. tests/unit/test_foo.py::test_bar)
            if self._looks_like_nodeid(pattern) or Path(pattern).is_file():
                used_files = [pattern]
            # Glob pattern relative to unit_dir (e.g. test_*.py)
            elif self._looks_like_glob(pattern):
                matched = self._expand_glob_in_dir(unit_dir, pattern)
                if matched:
                    used_files = matched
                else:
                    print(f"⚠️  Pattern {pattern!r} matched no files under {unit_dir}. Running entire unit directory instead.")
            # -k expression / substring
            elif self._looks_like_k_expression(pattern):
                used_k_expr = pattern
            else:
                # treat as simple substring for -k
                used_k_expr = pattern

        # Attach either file list or the entire unit dir
        if used_files:
            cmd.extend(used_files)
        else:
            cmd.append(str(unit_dir))

        # Add coverage if requested
        if coverage:
            cmd.extend(
                [
                    f"--cov={self.source_dir}",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov",
                    "--cov-report=json:coverage.json",
                    f"--cov-fail-under={self.coverage_analyzer.coverage_threshold}",
                ]
            )

        # Add verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        # Add fail-fast
        if fail_fast:
            cmd.append("-x")

        # Add -k expression if we decided to use one
        if used_k_expr:
            cmd.extend(["-k", used_k_expr])

        # Add markers to exclude integration and performance tests
        cmd.extend(["-m", "not integration and not performance"])

        # Print the final command for visibility (useful in CI logs)
        print("Running pytest command:", " ".join(cmd))

        start_time = time.time()
        result = subprocess.run(cmd, cwd=".")
        execution_time = time.time() - start_time

        success = result.returncode == 0

        print(
            f"✅ Unit tests completed in {execution_time:.2f}s"
            if success
            else f"❌ Unit tests failed after {execution_time:.2f}s (exit code {result.returncode})"
        )

        return success

    def run_integration_tests(self, verbose: bool = True) -> bool:
        """Run integration tests."""
        print("🔗 Running POLARIS Integration Tests...")

        integration_dir = self.test_dir / "integration"
        if not integration_dir.exists():
            print(f"❌ Integration tests directory not found: {integration_dir}")
            return False

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(integration_dir),
            "--integration",
        ]

        if verbose:
            cmd.append("-v")

        print("Running pytest command:", " ".join(cmd))

        start_time = time.time()
        result = subprocess.run(cmd, cwd=".")
        execution_time = time.time() - start_time

        success = result.returncode == 0

        print(
            f"✅ Integration tests completed in {execution_time:.2f}s"
            if success
            else f"❌ Integration tests failed after {execution_time:.2f}s (exit code {result.returncode})"
        )

        return success

    def run_performance_tests(self, verbose: bool = True) -> bool:
        """Run performance tests."""
        print("⚡ Running POLARIS Performance Tests...")

        performance_dir = self.test_dir / "performance"
        if not performance_dir.exists():
            # fallback: run tests/performance may not exist; allow running full tests dir but warn
            print(f"⚠️  Performance test directory {performance_dir} not found; running entire tests/ directory with -m performance instead.")
            target = str(self.test_dir)
        else:
            target = str(performance_dir)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            target,
            "--performance",
        ]

        if verbose:
            cmd.append("-v")

        print("Running pytest command:", " ".join(cmd))

        start_time = time.time()
        result = subprocess.run(cmd, cwd=".")
        execution_time = time.time() - start_time

        success = result.returncode == 0

        print(
            f"✅ Performance tests completed in {execution_time:.2f}s"
            if success
            else f"❌ Performance tests failed after {execution_time:.2f}s (exit code {result.returncode})"
        )

        return success

    def run_all_tests(self, coverage: bool = True, verbose: bool = True) -> bool:
        """Run all test suites."""
        print("🚀 Running All POLARIS Tests...")

        results = []

        # Run unit tests
        results.append(self.run_unit_tests(coverage=coverage, verbose=verbose))

        # Run integration tests
        results.append(self.run_integration_tests(verbose=verbose))

        # Run performance tests
        results.append(self.run_performance_tests(verbose=verbose))

        all_passed = all(results)

        print(f"\n{'='*50}")
        print(f"📊 Test Summary:")
        print(f"Unit Tests: {'✅ PASS' if results[0] else '❌ FAIL'}")
        print(f"Integration Tests: {'✅ PASS' if results[1] else '❌ FAIL'}")
        print(f"Performance Tests: {'✅ PASS' if results[2] else '❌ FAIL'}")
        print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        print(f"{'='*50}")

        return all_passed

    def run_coverage_analysis(self) -> None:
        """Run detailed coverage analysis and reporting."""
        print("📈 Running Coverage Analysis...")

        coverage_data = self.coverage_analyzer.run_coverage_analysis()

        if coverage_data:
            report = self.coverage_analyzer.generate_coverage_report(coverage_data)
            print(report)

            # Save detailed report
            report_file = Path("coverage_report.txt")
            with open(report_file, "w") as f:
                f.write(report)
            print(f"\n📄 Detailed coverage report saved to: {report_file}")

            # Generate suggestions
            suggestions = self.coverage_analyzer.suggest_test_improvements(coverage_data)
            if suggestions:
                print("\n💡 Suggestions for improvement:")
                for suggestion in suggestions:
                    print(suggestion)
        else:
            print("❌ Failed to generate coverage analysis")

    def create_test_templates(self) -> None:
        """Create test templates for untested modules."""
        print("📝 Creating Test Templates...")

        from tests.utils.coverage_utils import create_missing_test_files

        created_files = create_missing_test_files(str(self.source_dir), str(self.test_dir))

        if created_files:
            print(f"✅ Created {len(created_files)} test template files:")
            for file_path in created_files:
                print(f"  📄 {file_path}")
            print("\n💡 Please implement the TODO items in the generated test files.")
        else:
            print("✅ All modules already have test files.")

    def run_specific_test(self, test_path: str, verbose: bool = True) -> bool:
        """Run a specific test file or test function (supports nodeid like file::test_name)."""
        print(f"🎯 Running Specific Test: {test_path}")

        cmd = [sys.executable, "-m", "pytest", test_path]

        if verbose:
            cmd.append("-v")

        print("Running pytest command:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=".")
        success = result.returncode == 0

        print("✅ Test completed successfully" if success else f"❌ Test failed (exit code {result.returncode})")

        return success

    def run_tests_with_watch(self, pattern: str = "test_*.py") -> None:
        """Run tests in watch mode (requires pytest-watch)."""
        print("👀 Running Tests in Watch Mode...")
        print("Press Ctrl+C to stop watching")

        ptw = shutil.which("ptw")
        if not ptw:
            print("❌ pytest-watch not installed. Install with: pip install pytest-watch")
            return

        try:
            # Build watch command; prefer to watch the tests directory and apply -k if pattern is -k expression
            cmd: List[str] = [sys.executable, "-m", "ptw", str(self.test_dir)]

            # If pattern looks like a glob, pass it as a path argument to ptw (ptw will watch that path)
            if self._looks_like_glob(pattern):
                cmd = [sys.executable, "-m", "ptw", str(self.test_dir)]
                # ptw doesn't accept a file-glob arg in the same way — simpler to use -k for filtered runs:
                # Use -k if it's not literally a glob with star (fallback)
                cmd.extend(["--", "-v", f"--cov={self.source_dir}", "--cov-report=term-missing"])
            else:
                # treat as -k
                cmd.extend(["--", "-v", f"--cov={self.source_dir}", "--cov-report=term-missing", "-k", pattern])

            print("Running watch command:", " ".join(cmd))
            subprocess.run(cmd, cwd=".")
        except KeyboardInterrupt:
            print("\n👋 Watch mode stopped")
        except FileNotFoundError:
            print("❌ pytest-watch not installed. Install with: pip install pytest-watch")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="POLARIS Test Runner - Comprehensive testing framework for POLARIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tests.run_tests --unit                    # Run unit tests only
  python -m tests.run_tests --integration             # Run integration tests only
  python -m tests.run_tests --performance             # Run performance tests only
  python -m tests.run_tests --all                     # Run all test suites
  python -m tests.run_tests --coverage                # Run coverage analysis
  python -m tests.run_tests --create-templates        # Create test templates
  python -m tests.run_tests --test path/to/test.py    # Run specific test
  python -m tests.run_tests --watch                   # Run in watch mode
        """
    )

    # Test execution options
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--all", action="store_true", help="Run all test suites")

    # Analysis options
    parser.add_argument("--coverage", action="store_true", help="Run coverage analysis")
    parser.add_argument("--create-templates", action="store_true", help="Create test templates for untested modules")

    # Specific test options
    parser.add_argument("--test", type=str, help="Run a specific test file or function")
    parser.add_argument("--watch", action="store_true", help="Run tests in watch mode")

    # Configuration options
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--pattern", type=str, default=None, help="Test file pattern (glob), nodeid (file::test), or -k expression")
    parser.add_argument("--threshold", type=float, default=70.0, help="Coverage threshold percentage")

    args = parser.parse_args()

    # Initialize test runner
    runner = PolarisTestRunner()
    runner.coverage_analyzer.coverage_threshold = args.threshold

    verbose = not args.quiet
    coverage = not args.no_coverage

    success = True

    try:
        if args.unit:
            success = runner.run_unit_tests(
                coverage=coverage, verbose=verbose, fail_fast=args.fail_fast, pattern=args.pattern
            )
        elif args.integration:
            success = runner.run_integration_tests(verbose=verbose)
        elif args.performance:
            success = runner.run_performance_tests(verbose=verbose)
        elif args.all:
            success = runner.run_all_tests(coverage=coverage, verbose=verbose)
        elif args.coverage:
            runner.run_coverage_analysis()
        elif args.create_templates:
            runner.create_test_templates()
        elif args.test:
            success = runner.run_specific_test(args.test, verbose=verbose)
        elif args.watch:
            runner.run_tests_with_watch(args.pattern or "test_*.py")
        else:
            # Default: run unit tests
            success = runner.run_unit_tests(
                coverage=coverage, verbose=verbose, fail_fast=args.fail_fast, pattern=args.pattern
            )

    except KeyboardInterrupt:
        print("\n👋 Test execution interrupted")
        success = False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        success = False

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
