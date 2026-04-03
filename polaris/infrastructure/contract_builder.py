"""Helpers for building runtime system contracts from connectors."""

from typing import TYPE_CHECKING, Dict, Iterable

from polaris.abstractions.connector_capabilities import ConnectorCapabilities
from polaris.abstractions.system_contract import SystemContract

if TYPE_CHECKING:
    from polaris.abstractions.connector import Connector
    from polaris.abstractions.observability import Logger


async def build_system_contract(
    connector: "Connector",
    logger: "Logger | None" = None,
) -> SystemContract:
    """Build a contract for a single connector."""
    system_id = await connector.get_system_id()
    connector_type = connector.__class__.__name__

    capabilities = await connector.get_capabilities()
    if not isinstance(capabilities, ConnectorCapabilities):
        raise TypeError("connector.get_capabilities() must return ConnectorCapabilities")
    if not capabilities.supported_action_types:
        raise ValueError(
            f"Connector '{connector_type}' for system '{system_id}' returned empty supported actions"
        )

    return SystemContract.from_capabilities(
        system_id=system_id,
        connector_type=connector_type,
        capabilities=capabilities,
    )


async def build_system_contracts(
    connectors: Iterable["Connector"],
    logger: "Logger | None" = None,
) -> Dict[str, SystemContract]:
    """Build contracts for multiple connectors keyed by system_id."""
    contracts: Dict[str, SystemContract] = {}
    for connector in connectors:
        contract = await build_system_contract(connector, logger=logger)
        contracts[contract.system_id] = contract
    return contracts
