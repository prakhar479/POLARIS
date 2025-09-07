
"""
This module initializes the data storage sub-package by importing the necessary
classes and backends. It provides a unified interface for managing data storage
and provides several concrete implementations for different storage backends.
"""
from .data_store import PolarisDataStore, PolarisUnitOfWork
from .repository import (
    Repository, AdaptationActionRepository,
    SystemDependencyRepository, LearnedPatternRepository,
    SystemStateRepository
)
from .storage_backend import StorageBackend, GraphStorageBackend, InMemoryGraphStorageBackend

__all__ = [
    # Data store
    "PolarisDataStore",
    "PolarisUnitOfWork",

    # Repositories
    "Repository",
    "AdaptationActionRepository",
    "SystemDependencyRepository",
    "LearnedPatternRepository",
    "SystemStateRepository",

    # Storage backends
    "StorageBackend",
    "GraphStorageBackend",
    "InMemoryGraphStorageBackend"
]
