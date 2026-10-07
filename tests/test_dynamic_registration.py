"""Tests for dynamic system and connector registration in Polaris."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.connector import Connector
from polaris.abstractions.system_contract import SystemContract
from polaris.core.polaris import Polaris, PolarisConfig
from polaris.core.registry import ConnectorRegistry
from polaris.core.topology import SystemTopology


def test_topology_add_and_remove_node():
    """Test SystemTopology add_node and remove_node methods."""
    topo = SystemTopology()
    topo.add_node("gateway", ["auth_svc", "order_svc"])

    assert topo.get_dependencies("gateway") == ["auth_svc", "order_svc"]
    assert topo.get_dependents("auth_svc") == ["gateway"]
    assert topo.get_dependents("order_svc") == ["gateway"]

    # Add downstream dependency
    topo.add_node("order_svc", ["order_db"])
    assert "order_db" in topo.get_dependencies("order_svc")
    assert "order_db" in topo.get_impact_radius("gateway")

    # Remove node
    topo.remove_node("auth_svc")
    assert "auth_svc" not in topo.get_dependencies("gateway")
    assert topo.get_dependents("auth_svc") == []

    # Remove middle node
    topo.remove_node("order_svc")
    assert "order_svc" not in topo.get_dependencies("gateway")
    assert topo.get_dependents("order_db") == []


def test_registry_unregister():
    """Test ConnectorRegistry unregister method."""
    mock_metrics = MagicMock()
    registry = ConnectorRegistry(metrics=mock_metrics)

    mock_conn = MagicMock(spec=Connector)
    contract = SystemContract(system_id="test_sys")

    # Register
    registry._connectors["test_sys"] = mock_conn
    registry._contracts["test_sys"] = contract

    assert registry.get("test_sys") is mock_conn
    assert registry.get_contract("test_sys") is contract

    # Unregister
    removed = registry.unregister("test_sys")
    assert removed is mock_conn
    assert registry.get("test_sys") is None
    assert registry.get_contract("test_sys") is None

    unregistered_calls = [
        call
        for call in mock_metrics.increment.call_args_list
        if call[0][0] == "polaris.registry.connector_unregistered"
    ]
    assert len(unregistered_calls) == 1
    assert unregistered_calls[0][1]["tags"] == {"system_id": "test_sys"}

    # Unregister unknown
    assert registry.unregister("unknown") is None


@pytest.mark.asyncio
async def test_polaris_register_and_unregister_system():
    """Test dynamic runtime registration and unregistration of systems on Polaris coordinator."""
    from polaris.abstractions.connector_capabilities import ConnectorCapabilities

    mock_ks = MagicMock()
    mock_ks.store_topology = AsyncMock()

    mock_logger = MagicMock()
    mock_metrics = MagicMock()

    polaris = Polaris(
        config=PolarisConfig(),
        knowledge_store=mock_ks,
        logger=mock_logger,
        metrics=mock_metrics,
    )

    caps = ConnectorCapabilities(
        supported_action_types=("refund", "charge"),
    )

    mock_conn = MagicMock(spec=Connector)
    mock_conn.__class__.__name__ = "PaymentConnector"
    mock_conn.get_system_id = AsyncMock(return_value="payment_api")
    mock_conn.connect = AsyncMock(return_value=True)
    mock_conn.disconnect = AsyncMock(return_value=True)
    mock_conn.get_supported_actions = MagicMock(return_value=["refund", "charge"])
    mock_conn.get_capabilities = AsyncMock(return_value=caps)

    # Register dynamically
    success = await polaris.register_system(
        mock_conn,
        dependencies=["stripe_gateway"],
    )

    assert success is True
    assert mock_conn.connect.called
    assert polaris.registry.get("payment_api") is mock_conn
    contract = polaris.registry.get_contract("payment_api")
    assert contract is not None
    assert contract.system_id == "payment_api"
    assert "payment_api" in polaris.topology.dependencies
    assert polaris.topology.get_dependencies("payment_api") == ["stripe_gateway"]
    assert mock_ks.store_topology.called

    # Attach mock monitoring loop
    mock_loop = MagicMock()
    polaris._monitoring_loop = mock_loop

    # Unregister dynamically
    unreg_success = await polaris.unregister_system("payment_api", disconnect=True)
    assert unreg_success is True
    assert mock_conn.disconnect.called
    assert polaris.registry.get("payment_api") is None
    assert polaris.registry.get_contract("payment_api") is None
    assert "payment_api" not in polaris.topology.dependencies
    mock_loop.unregister_system.assert_called_with("payment_api")


@pytest.mark.asyncio
async def test_polaris_register_system_connect_failure():
    """Test dynamic registration fails gracefully if connector.connect fails."""
    polaris = Polaris(config=PolarisConfig())

    mock_conn = MagicMock(spec=Connector)
    mock_conn.get_system_id = AsyncMock(return_value="failing_sys")
    mock_conn.connect = AsyncMock(return_value=False)

    success = await polaris.register_system(mock_conn)
    assert success is False
    assert polaris.registry.get("failing_sys") is None


@pytest.mark.asyncio
async def test_polaris_unregister_unknown_system():
    """Test unregistering a non-existent system returns False."""
    polaris = Polaris(config=PolarisConfig())
    success = await polaris.unregister_system("non_existent")
    assert success is False
