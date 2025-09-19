
"""
This module initializes the data storage sub-package by importing the necessary
classes and backends. It provides a unified interface for managing data storage
and provides several concrete implementations for different storage backends.
"""
from .data_store import PolarisDataStore, PolarisUnitOfWork
from .repository import (
    Repository, AdaptationActionRepository,
    SystemDependencyRepository, LearnedPatternRepository,
    SystemStateRepository, ExecutionResultRepository
)
from .storage_backend import (
    StorageBackend, GraphStorageBackend, InMemoryGraphStorageBackend,
    TimeSeriesStorageBackend, DocumentStorageBackend,
    InMemoryTimeSeriesBackend, InMemoryDocumentBackend
)
from .factory import DataStoreFactory

__all__ = [
    # Data store
    "PolarisDataStore",
    "PolarisUnitOfWork",
    "DataStoreFactory",

    # Repositories
    "Repository",
    "AdaptationActionRepository",
    "SystemDependencyRepository",
    "LearnedPatternRepository",
    "SystemStateRepository",
    "ExecutionResultRepository",

    # Storage backends
    "StorageBackend",
    "GraphStorageBackend",
    "InMemoryGraphStorageBackend",
    "TimeSeriesStorageBackend",
    "DocumentStorageBackend",
    "InMemoryTimeSeriesBackend",
    "InMemoryDocumentBackend"
]
