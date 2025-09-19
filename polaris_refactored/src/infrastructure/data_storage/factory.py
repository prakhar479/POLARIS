"""
Data Storage Factory

Factory for creating PolarisDataStore instances with common configurations.
"""

from typing import Dict, Optional
from .data_store import PolarisDataStore
from .storage_backend import (
    StorageBackend, InMemoryGraphStorageBackend,
    InMemoryTimeSeriesBackend, InMemoryDocumentBackend
)


class DataStoreFactory:
    """Factory for creating PolarisDataStore instances."""
    
    @staticmethod
    def create_in_memory_store() -> PolarisDataStore:
        """Create a data store with all in-memory backends for testing."""
        backends = {
            "time_series": InMemoryTimeSeriesBackend(),
            "document": InMemoryDocumentBackend(),
            "graph": InMemoryGraphStorageBackend()
        }
        return PolarisDataStore(backends)
    
    @staticmethod
    def create_minimal_store() -> PolarisDataStore:
        """Create a minimal data store with just document backend."""
        backends = {
            "document": InMemoryDocumentBackend()
        }
        return PolarisDataStore(backends)
    
    @staticmethod
    def create_custom_store(backends: Dict[str, StorageBackend]) -> PolarisDataStore:
        """Create a data store with custom backends."""
        return PolarisDataStore(backends)
    
    @staticmethod
    def create_production_store(
        time_series_backend: Optional[StorageBackend] = None,
        document_backend: Optional[StorageBackend] = None,
        graph_backend: Optional[StorageBackend] = None
    ) -> PolarisDataStore:
        """Create a production data store with specified backends."""
        backends = {}
        
        if time_series_backend:
            backends["time_series"] = time_series_backend
        
        if document_backend:
            backends["document"] = document_backend
        
        if graph_backend:
            backends["graph"] = graph_backend
        
        # Fallback to in-memory if no backends provided
        if not backends:
            return DataStoreFactory.create_in_memory_store()
        
        return PolarisDataStore(backends)