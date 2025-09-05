"""
POLARIS - Proactive Optimization & Learning Architecture for Resilient Intelligent Systems

This is the refactored version of POLARIS implementing a clean layered architecture
with proper separation of concerns and dependency injection.
"""

__version__ = "2.0.0"
__author__ = "POLARIS Development Team"

# Core framework exports
from .framework import PolarisFramework
from .infrastructure.di import DIContainer

__all__ = [
    "PolarisFramework",
    "DIContainer",
]