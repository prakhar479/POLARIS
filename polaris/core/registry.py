"""Component registry for Polaris."""

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from polaris.abstractions.connector import Connector
    from polaris.abstractions.observability import MetricsCollector


class ConnectorRegistry:
    """Simple registry for system connectors."""

    def __init__(self, metrics: Optional["MetricsCollector"] = None):
        """Initialize connector registry with optional metrics collection."""
        self._connectors: Dict[str, "Connector"] = {}
        self._metrics = metrics

    async def register(self, connector: "Connector") -> None:
        """
        Register a connector.

        Args:
            connector: Connector to register
        """
        system_id = await connector.get_system_id()
        self._connectors[system_id] = connector

        if self._metrics:
            self._metrics.increment(
                "polaris.registry.connector_registered", tags={"system_id": system_id}
            )
            self._metrics.gauge("polaris.registry.total_connectors", len(self._connectors))

    def get(self, system_id: str) -> Optional["Connector"]:
        """
        Get a connector by system ID.

        Args:
            system_id: System ID

        Returns:
            Connector if found, None otherwise
        """
        connector = self._connectors.get(system_id)

        if self._metrics:
            self._metrics.increment(
                "polaris.registry.connector_accessed",
                tags={"system_id": system_id, "found": str(connector is not None).lower()},
            )

        return connector

    def all(self) -> List["Connector"]:
        """
        Get all registered connectors.

        Returns:
            List of all connectors
        """
        if self._metrics:
            self._metrics.increment("polaris.registry.all_connectors_accessed")

        return list(self._connectors.values())

    def system_ids(self) -> List[str]:
        """Get all registered system IDs."""
        if self._metrics:
            self._metrics.increment("polaris.registry.system_ids_accessed")

        return list(self._connectors.keys())
