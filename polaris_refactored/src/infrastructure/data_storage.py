"""
Data Storage Infrastructure

Provides data storage abstraction using Repository and Unit of Work patterns
with support for multiple storage backends (time-series, document, graph databases).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic, AsyncContextManager
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from ..domain.models import SystemState, AdaptationAction, ExecutionResult, SystemDependency, LearnedPattern
from .exceptions import DataStoreError
from .di import Injectable

T = TypeVar('T')


class StorageBackend(ABC):
    """Abstract interface for storage backends."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the storage backend."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        pass
    
    @abstractmethod
    async def store(self, collection: str, key: str, data: Dict[str, Any]) -> None:
        """Store data in the backend."""
        pass
    
    @abstractmethod
    async def retrieve(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from the backend."""
        pass
    
    @abstractmethod
    async def query(self, collection: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query data from the backend."""
        pass
    
    @abstractmethod
    async def delete(self, collection: str, key: str) -> bool:
        """Delete data from the backend."""
        pass


class Repository(Generic[T], ABC):
    """Abstract base class for repositories."""
    
    def __init__(self, storage_backend: StorageBackend, collection_name: str):
        self.storage_backend = storage_backend
        self.collection_name = collection_name
    
    @abstractmethod
    async def save(self, entity: T) -> None:
        """Save an entity to the repository."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get an entity by its ID."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity by its ID."""
        pass
    
    @abstractmethod
    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dictionary for storage."""
        pass
    
    @abstractmethod
    def _dict_to_entity(self, data: Dict[str, Any]) -> T:
        """Convert dictionary from storage to entity."""
        pass


class SystemStateRepository(Repository[SystemState]):
    """Repository for managing system state data."""
    
    def __init__(self, storage_backend: StorageBackend):
        super().__init__(storage_backend, "system_states")
    
    async def save(self, state: SystemState) -> None:
        """Save a system state."""
        try:
            key = f"{state.system_id}_{state.timestamp.isoformat()}"
            data = self._entity_to_dict(state)
            await self.storage_backend.store(self.collection_name, key, data)
        except Exception as e:
            raise DataStoreError(
                f"Failed to save system state for {state.system_id}",
                operation="save",
                entity_type="SystemState",
                cause=e
            )
    
    async def get_by_id(self, entity_id: str) -> Optional[SystemState]:
        """Get system state by ID."""
        try:
            data = await self.storage_backend.retrieve(self.collection_name, entity_id)
            return self._dict_to_entity(data) if data else None
        except Exception as e:
            raise DataStoreError(
                f"Failed to retrieve system state {entity_id}",
                operation="get_by_id",
                entity_type="SystemState",
                cause=e
            )
    
    async def get_current_state(self, system_id: str) -> Optional[SystemState]:
        """Get the most recent state for a system."""
        try:
            # Query for the most recent state
            filters = {"system_id": system_id}
            results = await self.storage_backend.query(self.collection_name, filters)
            
            if not results:
                return None
            
            # Sort by timestamp and get the most recent
            sorted_results = sorted(results, key=lambda x: x["timestamp"], reverse=True)
            return self._dict_to_entity(sorted_results[0])
            
        except Exception as e:
            raise DataStoreError(
                f"Failed to get current state for system {system_id}",
                operation="get_current_state",
                entity_type="SystemState",
                cause=e
            )
    
    async def get_states_in_range(
        self, 
        system_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[SystemState]:
        """Get system states within a time range."""
        try:
            filters = {
                "system_id": system_id,
                "timestamp": {"$gte": start_time.isoformat(), "$lte": end_time.isoformat()}
            }
            results = await self.storage_backend.query(self.collection_name, filters)
            return [self._dict_to_entity(data) for data in results]
        except Exception as e:
            raise DataStoreError(
                f"Failed to get states in range for system {system_id}",
                operation="get_states_in_range",
                entity_type="SystemState",
                cause=e
            )
    
    async def delete(self, entity_id: str) -> bool:
        """Delete a system state."""
        try:
            return await self.storage_backend.delete(self.collection_name, entity_id)
        except Exception as e:
            raise DataStoreError(
                f"Failed to delete system state {entity_id}",
                operation="delete",
                entity_type="SystemState",
                cause=e
            )
    
    def _entity_to_dict(self, state: SystemState) -> Dict[str, Any]:
        """Convert SystemState to dictionary."""
        return {
            "system_id": state.system_id,
            "timestamp": state.timestamp.isoformat(),
            "metrics": {k: {
                "name": v.name,
                "value": v.value,
                "unit": v.unit,
                "timestamp": v.timestamp.isoformat() if v.timestamp else None,
                "tags": v.tags
            } for k, v in state.metrics.items()},
            "health_status": state.health_status.value,
            "metadata": state.metadata
        }
    
    def _dict_to_entity(self, data: Dict[str, Any]) -> SystemState:
        """Convert dictionary to SystemState."""
        from ..domain.models import MetricValue, HealthStatus
        
        metrics = {}
        for k, v in data["metrics"].items():
            metrics[k] = MetricValue(
                name=v["name"],
                value=v["value"],
                unit=v["unit"],
                timestamp=datetime.fromisoformat(v["timestamp"]) if v["timestamp"] else None,
                tags=v["tags"]
            )
        
        return SystemState(
            system_id=data["system_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=metrics,
            health_status=HealthStatus(data["health_status"]),
            metadata=data["metadata"]
        )


class AdaptationActionRepository(Repository[AdaptationAction]):
    """Repository for managing adaptation actions."""
    
    def __init__(self, storage_backend: StorageBackend):
        super().__init__(storage_backend, "adaptation_actions")
    
    async def save(self, action: AdaptationAction) -> None:
        """Save an adaptation action."""
        try:
            data = self._entity_to_dict(action)
            await self.storage_backend.store(self.collection_name, action.action_id, data)
        except Exception as e:
            raise DataStoreError(
                f"Failed to save adaptation action {action.action_id}",
                operation="save",
                entity_type="AdaptationAction",
                cause=e
            )
    
    async def get_by_id(self, action_id: str) -> Optional[AdaptationAction]:
        """Get adaptation action by ID."""
        try:
            data = await self.storage_backend.retrieve(self.collection_name, action_id)
            return self._dict_to_entity(data) if data else None
        except Exception as e:
            raise DataStoreError(
                f"Failed to retrieve adaptation action {action_id}",
                operation="get_by_id",
                entity_type="AdaptationAction",
                cause=e
            )
    
    async def delete(self, action_id: str) -> bool:
        """Delete an adaptation action."""
        try:
            return await self.storage_backend.delete(self.collection_name, action_id)
        except Exception as e:
            raise DataStoreError(
                f"Failed to delete adaptation action {action_id}",
                operation="delete",
                entity_type="AdaptationAction",
                cause=e
            )
    
    def _entity_to_dict(self, action: AdaptationAction) -> Dict[str, Any]:
        """Convert AdaptationAction to dictionary."""
        return {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "target_system": action.target_system,
            "parameters": action.parameters,
            "priority": action.priority,
            "timeout_seconds": action.timeout_seconds,
            "created_at": action.created_at.isoformat() if action.created_at else None
        }
    
    def _dict_to_entity(self, data: Dict[str, Any]) -> AdaptationAction:
        """Convert dictionary to AdaptationAction."""
        return AdaptationAction(
            action_id=data["action_id"],
            action_type=data["action_type"],
            target_system=data["target_system"],
            parameters=data["parameters"],
            priority=data["priority"],
            timeout_seconds=data["timeout_seconds"],
            created_at=datetime.fromisoformat(data["created_at"]) if data["created_at"] else None
        )


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