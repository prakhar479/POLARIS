"""
Data Storage Infrastructure Module

The main data storage abstraction for Polaris. It provides a class `PolarisDataStore`
which manages the repositories and the storage backend and implements the Unit of Work
pattern for transactional operations. This module is the entry point for managing data
storage in Polaris.
"""



from typing import Dict, List, Any, TypeVar, AsyncContextManager
from contextlib import asynccontextmanager

from ..exceptions import DataStoreError
from ..di import Injectable

from .repository import (
    Repository, AdaptationActionRepository,
    SystemDependencyRepository, LearnedPatternRepository,
    SystemStateRepository
)
from .storage_backend import StorageBackend, GraphStorageBackend

T = TypeVar('T')

class PolarisUnitOfWork(AsyncContextManager):
    """
    Unit of Work pattern implementation for transactional operations.
    """
    
    def __init__(self, repositories: Dict[str, Repository]):
        self.repositories = repositories
        self._transaction_active = False
        self._changes: List[tuple] = []  # (operation, repository, entity)
    
    async def __aenter__(self):
        """Start a transaction."""
        self._transaction_active = True
        self._changes.clear()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End transaction - commit if no exception, rollback otherwise."""
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()
        
        self._transaction_active = False
        self._changes.clear()
    
    async def commit(self) -> None:
        """Commit all changes in the transaction."""
        if not self._transaction_active:
            raise DataStoreError("No active transaction to commit")
        
        try:
            # Execute all changes
            for operation, repository, entity in self._changes:
                if operation == "save":
                    await repository.save(entity)
                elif operation == "delete":
                    await repository.delete(entity)
        except Exception as e:
            await self.rollback()
            raise DataStoreError("Failed to commit transaction", cause=e)
    
    async def rollback(self) -> None:
        """Rollback all changes in the transaction."""
        # In a real implementation, this would undo changes
        # For now, we just clear the changes list
        self._changes.clear()
    
    def add_save(self, repository_name: str, entity: Any) -> None:
        """Add a save operation to the transaction."""
        if not self._transaction_active:
            raise DataStoreError("No active transaction")
        
        repository = self.repositories.get(repository_name)
        if not repository:
            raise DataStoreError(f"Repository {repository_name} not found")
        
        self._changes.append(("save", repository, entity))
    
    def add_delete(self, repository_name: str, entity_id: str) -> None:
        """Add a delete operation to the transaction."""
        if not self._transaction_active:
            raise DataStoreError("No active transaction")
        
        repository = self.repositories.get(repository_name)
        if not repository:
            raise DataStoreError(f"Repository {repository_name} not found")
        
        self._changes.append(("delete", repository, entity_id))


class PolarisDataStore(Injectable):
    """
    Main data store class that provides access to all repositories
    and manages storage backends.
    """
    
    def __init__(self, storage_backends: Dict[str, StorageBackend]):
        self.storage_backends = storage_backends
        self._repositories: Dict[str, Repository] = {}
        self._connected = False
    
    async def start(self) -> None:
        """Start the data store and connect to all backends."""
        try:
            for backend in self.storage_backends.values():
                await backend.connect()
            
            # Initialize repositories
            self._initialize_repositories()
            self._connected = True
            
        except Exception as e:
            raise DataStoreError("Failed to start data store", cause=e)
    
    async def stop(self) -> None:
        """Stop the data store and disconnect from all backends."""
        try:
            for backend in self.storage_backends.values():
                await backend.disconnect()
            
            self._repositories.clear()
            self._connected = False
            
        except Exception as e:
            raise DataStoreError("Failed to stop data store", cause=e)
    
    def _initialize_repositories(self) -> None:
        """Initialize all repositories with appropriate backends."""
        # Use time-series backend for system states if available
        time_series_backend = self.storage_backends.get("time_series")
        if time_series_backend:
            self._repositories["system_states"] = SystemStateRepository(time_series_backend)
        
        # Use document backend for actions if available
        document_backend = self.storage_backends.get("document")
        if document_backend:
            self._repositories["adaptation_actions"] = AdaptationActionRepository(document_backend)
            # Learned patterns are document-oriented
            self._repositories["learned_patterns"] = LearnedPatternRepository(document_backend)
        
        # Use graph backend for system dependencies if available
        graph_backend = self.storage_backends.get("graph")
        if graph_backend and isinstance(graph_backend, GraphStorageBackend):
            self._repositories["system_dependencies"] = SystemDependencyRepository(graph_backend)
    
    def get_repository(self, name: str) -> Repository:
        """Get a repository by name."""
        if not self._connected:
            raise DataStoreError("Data store is not connected")
        
        repository = self._repositories.get(name)
        if not repository:
            raise DataStoreError(f"Repository {name} not found")
        
        return repository
    
    @asynccontextmanager
    async def unit_of_work(self) -> PolarisUnitOfWork:
        """Create a unit of work for transactional operations."""
        uow = PolarisUnitOfWork(self._repositories)
        async with uow:
            yield uow