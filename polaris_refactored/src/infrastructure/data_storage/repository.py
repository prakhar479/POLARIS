"""
Repository Module

This module provides the abstract base classes for repositories.
Repositories are objects that encapsulate the data storage abstraction
for a specific domain model. They provide simple CRUD operations and
possibly additional query methods.
"""


from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic
from datetime import datetime

from ...domain.models import SystemState, AdaptationAction, SystemDependency, LearnedPattern
from ..exceptions import DataStoreError

from .storage_backend import StorageBackend, GraphStorageBackend

T = TypeVar('T')

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
        from ...domain.models import MetricValue, HealthStatus
        
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


class LearnedPatternRepository(Repository[LearnedPattern]):
    """Repository for managing learned patterns (document-oriented)."""
    
    def __init__(self, storage_backend: StorageBackend):
        super().__init__(storage_backend, "learned_patterns")
    
    async def save(self, pattern: LearnedPattern) -> None:
        try:
            data = self._entity_to_dict(pattern)
            await self.storage_backend.store(self.collection_name, pattern.pattern_id, data)
        except Exception as e:
            raise DataStoreError(
                f"Failed to save learned pattern {pattern.pattern_id}",
                operation="save",
                entity_type="LearnedPattern",
                cause=e,
            )
    
    async def get_by_id(self, pattern_id: str) -> Optional[LearnedPattern]:
        try:
            data = await self.storage_backend.retrieve(self.collection_name, pattern_id)
            return self._dict_to_entity(data) if data else None
        except Exception as e:
            raise DataStoreError(
                f"Failed to retrieve learned pattern {pattern_id}",
                operation="get_by_id",
                entity_type="LearnedPattern",
                cause=e,
            )
    
    async def delete(self, pattern_id: str) -> bool:
        try:
            return await self.storage_backend.delete(self.collection_name, pattern_id)
        except Exception as e:
            raise DataStoreError(
                f"Failed to delete learned pattern {pattern_id}",
                operation="delete",
                entity_type="LearnedPattern",
                cause=e,
            )
    
    def _entity_to_dict(self, pattern: LearnedPattern) -> Dict[str, Any]:
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "conditions": pattern.conditions,
            "outcomes": pattern.outcomes,
            "confidence": pattern.confidence,
            "learned_at": pattern.learned_at.isoformat(),
            "usage_count": pattern.usage_count,
        }
    
    def _dict_to_entity(self, data: Dict[str, Any]) -> LearnedPattern:
        return LearnedPattern(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            conditions=data["conditions"],
            outcomes=data["outcomes"],
            confidence=data["confidence"],
            learned_at=datetime.fromisoformat(data["learned_at"]),
            usage_count=data.get("usage_count", 0),
        )

    async def list_all(self) -> List[LearnedPattern]:
        """Return all learned patterns (for simple in-memory filtering)."""
        try:
            raw = await self.storage_backend.query(self.collection_name, {})
            return [self._dict_to_entity(d) for d in raw]
        except Exception as e:
            raise DataStoreError(
                "Failed to list learned patterns",
                operation="query",
                entity_type="LearnedPattern",
                cause=e,
            )

    async def query(self, filters: Dict[str, Any]) -> List[LearnedPattern]:
        """A thin query helper; for nested conditions, callers should post-filter."""
        try:
            raw = await self.storage_backend.query(self.collection_name, filters)
            return [self._dict_to_entity(d) for d in raw]
        except Exception as e:
            raise DataStoreError(
                "Failed to query learned patterns",
                operation="query",
                entity_type="LearnedPattern",
                cause=e,
            )


class SystemDependencyRepository(Repository[SystemDependency]):
    """Repository for system dependencies using a graph backend.

    Stores a document record for simple CRUD and mirrors edges in the graph backend
    for neighbor and dependency chain queries.
    """
    
    def __init__(self, storage_backend: GraphStorageBackend):
        super().__init__(storage_backend, "system_dependencies")
        self._graph_backend = storage_backend
    
    def _key(self, dep: SystemDependency) -> str:
        return f"{dep.source_system}:{dep.dependency_type}:{dep.target_system}"
    
    async def save(self, dep: SystemDependency) -> None:
        try:
            key = self._key(dep)
            await self.storage_backend.store(self.collection_name, key, self._entity_to_dict(dep))
            await self._graph_backend.add_edge(
                dep.source_system,
                dep.target_system,
                dep.dependency_type,
                strength=dep.strength,
                metadata=dep.metadata,
            )
        except Exception as e:
            raise DataStoreError(
                f"Failed to save system dependency {dep}",
                operation="save",
                entity_type="SystemDependency",
                cause=e,
            )
    
    async def get_by_id(self, entity_id: str) -> Optional[SystemDependency]:
        try:
            data = await self.storage_backend.retrieve(self.collection_name, entity_id)
            return self._dict_to_entity(data) if data else None
        except Exception as e:
            raise DataStoreError(
                f"Failed to retrieve system dependency {entity_id}",
                operation="get_by_id",
                entity_type="SystemDependency",
                cause=e,
            )
    
    async def delete(self, entity_id: str) -> bool:
        try:
            # best-effort remove edge as well by parsing the key
            parts = entity_id.split(":", 2)
            if len(parts) == 3:
                source, dep_type, target = parts
                await self._graph_backend.remove_edge(source, target, relationship_type=dep_type)
            return await self.storage_backend.delete(self.collection_name, entity_id)
        except Exception as e:
            raise DataStoreError(
                f"Failed to delete system dependency {entity_id}",
                operation="delete",
                entity_type="SystemDependency",
                cause=e,
            )
    
    def _entity_to_dict(self, dep: SystemDependency) -> Dict[str, Any]:
        return {
            "source_system": dep.source_system,
            "target_system": dep.target_system,
            "dependency_type": dep.dependency_type,
            "strength": dep.strength,
            "metadata": dep.metadata or {},
        }
    
    def _dict_to_entity(self, data: Dict[str, Any]) -> SystemDependency:
        return SystemDependency(
            source_system=data["source_system"],
            target_system=data["target_system"],
            dependency_type=data["dependency_type"],
            strength=float(data["strength"]),
            metadata=data.get("metadata", {}),
        )
    
    # Graph-specific helpers
    async def get_neighbors(self, system_id: str, direction: str = "out", relationship_type: Optional[str] = None) -> List[SystemDependency]:
        edges = await self._graph_backend.get_neighbors(system_id, direction=direction, relationship_type=relationship_type)
        deps: List[SystemDependency] = []
        for e in edges:
            deps.append(SystemDependency(
                source_system=e["source"],
                target_system=e["target"],
                dependency_type=e.get("relationship_type", "depends_on"),
                strength=float(e.get("strength", 1.0)),
                metadata=e.get("metadata", {}),
            ))
        return deps
    
    async def get_dependency_chain(self, system_id: str, max_depth: int = 3, direction: str = "out") -> Dict[str, Any]:
        return await self._graph_backend.get_dependency_chain(system_id, max_depth=max_depth, direction=direction)


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

