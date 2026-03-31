"""Tests for runtime system contract assembly helpers."""

import pytest

from polaris.abstractions.connector_capabilities import ConnectorCapabilities
from polaris.infrastructure.contract_builder import build_system_contract
from tests.conftest import MockConnector, MockLogger


class CapabilityConnector(MockConnector):
    async def get_capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities.from_supported_action_types(
            ["scale_up", "scale_down"],
            action_aliases={"add_server": "scale_up"},
            metadata={"source": "test"},
        )


class BrokenCapabilityConnector(MockConnector):
    async def get_capabilities(self) -> ConnectorCapabilities:
        raise RuntimeError("capabilities unavailable")


@pytest.mark.asyncio
async def test_build_system_contract_uses_connector_capabilities():
    connector = CapabilityConnector("sys-1")

    contract = await build_system_contract(connector)

    assert contract.system_id == "sys-1"
    assert contract.connector_type == "CapabilityConnector"
    assert contract.supported_action_types == ("scale_up", "scale_down")
    assert contract.action_aliases["add_server"] == "scale_up"
    assert contract.metadata["source"] == "test"


@pytest.mark.asyncio
async def test_build_system_contract_raises_on_capability_error():
    """System contract building should fail if capabilities cannot be fetched."""
    connector = BrokenCapabilityConnector("sys-2")
    logger = MockLogger()

    with pytest.raises(RuntimeError, match="capabilities unavailable"):
        await build_system_contract(connector, logger=logger)
