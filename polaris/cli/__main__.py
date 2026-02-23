"""
CLI package main entry point.

Allows running the CLI with: python -m polaris.cli
"""

import sys

from polaris.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
