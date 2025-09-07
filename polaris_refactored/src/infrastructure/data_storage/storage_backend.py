"""
Storage Backend Infrastructure

This module provides the abstract base classes for storage backends.
Storage backends are objects that encapsulate the data storage abstraction
for the POLARIS system. They provide simple CRUD and query operations
and possibly additional functionality for specific storage technologies.

Supported storage backends:

- InMemoryGraphStorageBackend: simple in-memory graph database
- GraphStorageBackend: abstract interface for graph databases

All storage backends implement the StorageBackend abstract base class.
"""


from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar
from datetime import datetime

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


class GraphStorageBackend(StorageBackend):
    """Minimal graph-specific storage backend interface.

    Extends StorageBackend with edge-oriented operations for representing
    system relationships (dependencies). Implementations should provide
    efficient neighbor queries and simple traversal for dependency chains.
    """

    @abstractmethod
    async def add_edge(
        self,
        source_system: str,
        target_system: str,
        relationship_type: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a directed relationship edge from source -> target."""
        pass

    @abstractmethod
    async def remove_edge(
        self,
        source_system: str,
        target_system: str,
        relationship_type: Optional[str] = None,
    ) -> bool:
        """Remove an edge; if relationship_type is None, remove all between the pair."""
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        system_id: str,
        direction: str = "out",
        relationship_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get neighboring systems with optional direction and type filter.

        Returns a list of edge dicts with keys: source, target, relationship_type, strength, metadata, created_at.
        """
        pass

    @abstractmethod
    async def get_dependency_chain(
        self,
        system_id: str,
        max_depth: int = 3,
        direction: str = "out",
    ) -> Dict[str, Any]:
        """Compute a simple BFS-based dependency chain up to max_depth."""
        pass


class InMemoryGraphStorageBackend(GraphStorageBackend):
    """In-memory graph backend suitable for tests and local runs.

    Also fulfills the base StorageBackend API for compatibility, using
    simple dictionaries per collection for generic CRUD operations.
    """

    def __init__(self) -> None:
        # Generic collections store (e.g., for optional node/edge documents)
        self._collections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Edge stores
        # outgoing[source] -> List[edge]
        self._outgoing: Dict[str, List[Dict[str, Any]]] = {}
        # incoming[target] -> List[edge]
        self._incoming: Dict[str, List[Dict[str, Any]]] = {}
        self._connected: bool = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        # do not clear data to allow reuse across start/stop in tests

    async def store(self, collection: str, key: str, data: Dict[str, Any]) -> None:
        col = self._collections.setdefault(collection, {})
        col[key] = dict(data)

    async def retrieve(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        col = self._collections.get(collection, {})
        val = col.get(key)
        return dict(val) if val is not None else None

    async def query(self, collection: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        col = self._collections.get(collection, {})
        results: List[Dict[str, Any]] = []
        for item in col.values():
            if _match_filters(item, filters):
                results.append(dict(item))
        return results

    async def delete(self, collection: str, key: str) -> bool:
        col = self._collections.get(collection, {})
        return col.pop(key, None) is not None

    async def add_edge(
        self,
        source_system: str,
        target_system: str,
        relationship_type: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        edge = {
            "source": source_system,
            "target": target_system,
            "relationship_type": relationship_type,
            "strength": strength,
            "metadata": dict(metadata) if metadata else {},
            "created_at": datetime.utcnow().isoformat(),
        }
        self._outgoing.setdefault(source_system, []).append(edge)
        self._incoming.setdefault(target_system, []).append(edge)

    async def remove_edge(
        self,
        source_system: str,
        target_system: str,
        relationship_type: Optional[str] = None,
    ) -> bool:
        removed = False
        def _keep(edge: Dict[str, Any]) -> bool:
            if edge["source"] != source_system or edge["target"] != target_system:
                return True
            if relationship_type is not None and edge["relationship_type"] != relationship_type:
                return True
            nonlocal removed
            removed = True
            return False

        self._outgoing[source_system] = [e for e in self._outgoing.get(source_system, []) if _keep(e)]
        self._incoming[target_system] = [e for e in self._incoming.get(target_system, []) if _keep(e)]
        return removed

    async def get_neighbors(
        self,
        system_id: str,
        direction: str = "out",
        relationship_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if direction not in ("out", "in"):
            direction = "out"
        edges = self._outgoing.get(system_id, []) if direction == "out" else self._incoming.get(system_id, [])
        if relationship_type is None:
            return [dict(e) for e in edges]
        return [dict(e) for e in edges if e.get("relationship_type") == relationship_type]

    async def get_dependency_chain(
        self,
        system_id: str,
        max_depth: int = 3,
        direction: str = "out",
    ) -> Dict[str, Any]:
        # BFS traversal
        if direction not in ("out", "in"):
            direction = "out"
        visited = {system_id}
        frontier = [system_id]
        depth = 0
        graph: Dict[str, List[str]] = {}
        while frontier and depth < max_depth:
            next_frontier: List[str] = []
            for node in frontier:
                neighbors = await self.get_neighbors(node, direction=direction)
                ids = []
                for e in neighbors:
                    nid = e["target"] if direction == "out" else e["source"]
                    ids.append(nid)
                    if nid not in visited:
                        visited.add(nid)
                        next_frontier.append(nid)
                graph[node] = ids
            frontier = next_frontier
            depth += 1
        return {"root": system_id, "depth": depth, "graph": graph}


def _match_filters(item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Very small helper to match dicts to filters with optional $gte/$lte.

    This is intentionally minimal for tests and should not be used for
    production-grade querying.
    """
    for k, v in filters.items():
        if isinstance(v, dict):
            # support simple range filters
            gte = v.get("$gte")
            lte = v.get("$lte")
            val = item.get(k)
            if gte is not None and (val is None or val < gte):
                return False
            if lte is not None and (val is None or val > lte):
                return False
        else:
            if item.get(k) != v:
                return False
    return True
