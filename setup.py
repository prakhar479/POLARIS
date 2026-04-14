"""Compatibility shim for legacy setuptools invocations.

Project metadata, dependencies, and entry points are defined in ``pyproject.toml``.
This file intentionally avoids duplicating metadata to prevent drift.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
