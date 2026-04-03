"""Component registry for Polaris."""

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from polaris.abstractions.connector import Connector
    from polaris.abstractions.observability import MetricsCollector
    from polaris.abstractions.system_contract import SystemContract


class ConnectorRegistry:
    """Simple registry for system connectors."""

    def __init__(self, metrics: Optional["MetricsCollector"] = None):
        """Initialize connector registry with optional metrics collection."""
        self._connectors: Dict[str, "Connector"] = {}
        self._contracts: Dict[str, "SystemContract"] = {}
        self._metrics = metrics

    async def register(
        self,
        connector: "Connector",
        contract: "SystemContract",
    ) -> None:
        """Register a connector.

        Args:
            connector: Connector to register
            contract: System contract for this connector
        """
        system_id = await connector.get_system_id()
        self._connectors[system_id] = connector
        self._contracts[system_id] = contract

        if self._metrics:
            self._metrics.increment(
                "polaris.registry.connector_registered", tags={"system_id": system_id}
            )
            self._metrics.gauge("polaris.registry.total_connectors", len(self._connectors))

    def get(self, system_id: str) -> Optional["Connector"]:
        """Get a connector by system ID.

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

    def register_contract(self, contract: "SystemContract") -> None:
        """Register or replace a system contract."""
        self._contracts[contract.system_id] = contract

    def get_contract(self, system_id: str) -> Optional["SystemContract"]:
        """Get a system contract by system ID."""
        contract = self._contracts.get(system_id)

        if self._metrics:
            self._metrics.increment(
                "polaris.registry.contract_accessed",
                tags={"system_id": system_id, "found": str(contract is not None).lower()},
            )

        return contract

    def contracts(self) -> List["SystemContract"]:
        """Get all registered system contracts."""
        return list(self._contracts.values())

    def all(self) -> List["Connector"]:
        """Get all registered connectors.

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
