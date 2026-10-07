"""Multi-system topology graph for Polaris.

Represents dependency graphs between managed systems (upstream callers and downstream dependencies).
Used for cascade failure prevention, cross-system co-adaptation, and blast radius calculation.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


@dataclass
class SystemTopology:
    """Directed dependency topology graph across managed systems.

    Edges represent directed dependencies:
    source_system -> target_system (meaning source_system depends on target_system).
    For example: api_gateway -> order_service -> database.
    """

    dependencies: Dict[str, List[str]] = field(default_factory=dict)

    def add_node(self, system_id: str, dependencies: Optional[Sequence[str]] = None) -> None:
        """Add a system node and optional dependencies to the topology graph.

        Args:
            system_id: System ID to register in topology.
            dependencies: Optional list of downstream system IDs it depends on.
        """
        sid = system_id.strip()
        if not sid:
            return
        if sid not in self.dependencies:
            self.dependencies[sid] = []
        if dependencies:
            for dep in dependencies:
                self.add_dependency(sid, dep)

    def remove_node(self, system_id: str) -> None:
        """Remove a system node and all its incoming and outgoing dependencies.

        Args:
            system_id: System ID to remove.
        """
        sid = system_id.strip()
        if not sid:
            return
        self.dependencies.pop(sid, None)
        for src, targets in list(self.dependencies.items()):
            self.dependencies[src] = [tgt for tgt in targets if tgt != sid]

    def add_dependency(self, source_system: str, target_system: str) -> None:
        """Add a directed dependency: source_system depends on target_system.

        Args:
            source_system: Upstream system (caller/consumer)
            target_system: Downstream system (callee/dependency)
        """
        src = source_system.strip()
        tgt = target_system.strip()
        if not src or not tgt or src == tgt:
            return

        if src not in self.dependencies:
            self.dependencies[src] = []
        if tgt not in self.dependencies[src]:
            self.dependencies[src].append(tgt)
        if tgt not in self.dependencies:
            self.dependencies[tgt] = []

    def get_dependencies(self, system_id: str) -> List[str]:
        """Return systems that system_id depends on (downstream dependencies).

        Args:
            system_id: System ID

        Returns:
            List of downstream system IDs
        """
        return list(self.dependencies.get(system_id.strip(), []))

    def get_dependents(self, system_id: str) -> List[str]:
        """Return systems that depend on system_id (upstream callers).

        Args:
            system_id: System ID

        Returns:
            List of upstream system IDs
        """
        target = system_id.strip()
        dependents = []
        for src, targets in self.dependencies.items():
            if target in targets:
                dependents.append(src)
        return dependents

    def get_impact_radius(self, system_id: str) -> Set[str]:
        """Return all transitive upstream and downstream systems connected to system_id.

        Args:
            system_id: System ID

        Returns:
            Set of all transitively connected system IDs including system_id
        """
        sid = system_id.strip()
        visited: Set[str] = set()
        if sid not in self.dependencies and not any(
            sid in targets for targets in self.dependencies.values()
        ):
            return {sid}

        queue: deque[str] = deque([sid])
        visited.add(sid)

        while queue:
            curr = queue.popleft()
            neighbors = set(self.get_dependencies(curr)) | set(self.get_dependents(curr))
            for nbr in neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return visited

    def topological_sort(self) -> List[str]:
        """Return topological order from downstream dependencies to upstream callers.

        Systems with no downstream dependencies appear first, followed by callers.
        If cycles exist, falls back to remaining nodes in arbitrary order.

        Returns:
            Topologically sorted list of system IDs
        """
        all_nodes = set(self.dependencies.keys())
        for targets in self.dependencies.values():
            all_nodes.update(targets)

        # In-degree here is count of downstream dependencies
        in_degree: Dict[str, int] = dict.fromkeys(all_nodes, 0)
        for node, targets in self.dependencies.items():
            in_degree[node] = len(targets)

        queue: deque[str] = deque([node for node, deg in in_degree.items() if deg == 0])
        result: List[str] = []

        # Reverse adjacency: target -> callers
        reverse_adj: Dict[str, List[str]] = {node: [] for node in all_nodes}
        for src, targets in self.dependencies.items():
            for tgt in targets:
                reverse_adj[tgt].append(src)

        while queue:
            curr = queue.popleft()
            result.append(curr)

            for caller in reverse_adj.get(curr, []):
                in_degree[caller] -= 1
                if in_degree[caller] == 0:
                    queue.append(caller)

        # If cyclic nodes remain, append them
        for node in all_nodes:
            if node not in result:
                result.append(node)

        return result

    def to_dict(self) -> Dict[str, List[str]]:
        """Serialize topology to dict."""
        return {k: list(v) for k, v in self.dependencies.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "SystemTopology":
        """Deserialize topology from dict."""
        topo = cls()
        for src, targets in data.items():
            if isinstance(targets, list):
                for tgt in targets:
                    topo.add_dependency(src, tgt)
            if src not in topo.dependencies:
                topo.dependencies[src] = []
        return topo
