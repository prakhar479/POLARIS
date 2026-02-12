"""Tests for connector registry."""

import pytest

from polaris.core.registry import ConnectorRegistry


class TestConnectorRegistry:
    """Test ConnectorRegistry functionality."""

    @pytest.fixture
    def registry(self, mock_metrics):
        """Create registry with mock metrics."""
        return ConnectorRegistry(metrics=mock_metrics)

    @pytest.mark.asyncio
    async def test_register_connector(self, registry, mock_connector):
        """Test registering a connector."""
        await registry.register(mock_connector)

        # Verify connector is registered
        retrieved = registry.get("test-system")
        assert retrieved == mock_connector

    @pytest.mark.asyncio
    async def test_register_multiple_connectors(self, registry):
        """Test registering multiple connectors."""
        from tests.conftest import MockConnector

        connector1 = MockConnector("system-1")
        connector2 = MockConnector("system-2")

        await registry.register(connector1)
        await registry.register(connector2)

        # Verify both connectors are registered
        assert registry.get("system-1") == connector1
        assert registry.get("system-2") == connector2

        # Verify all() returns both
        all_connectors = registry.all()
        assert len(all_connectors) == 2
        assert connector1 in all_connectors
        assert connector2 in all_connectors

    def test_get_nonexistent_connector(self, registry):
        """Test getting a connector that doesn't exist."""
        result = registry.get("nonexistent-system")
        assert result is None

    @pytest.mark.asyncio
    async def test_system_ids(self, registry):
        """Test getting all system IDs."""
        from tests.conftest import MockConnector

        connector1 = MockConnector("system-1")
        connector2 = MockConnector("system-2")

        await registry.register(connector1)
        await registry.register(connector2)

        system_ids = registry.system_ids()
        assert len(system_ids) == 2
        assert "system-1" in system_ids
        assert "system-2" in system_ids

    def test_empty_registry(self, registry):
        """Test operations on empty registry."""
        assert registry.all() == []
        assert registry.system_ids() == []
        assert registry.get("any-system") is None

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, mock_metrics):
        """Test that metrics are tracked correctly."""
        from tests.conftest import MockConnector

        registry = ConnectorRegistry(metrics=mock_metrics)
        connector = MockConnector("test-system")

        await registry.register(connector)
        registry.get("test-system")
        registry.get("nonexistent")
        registry.all()
        registry.system_ids()

        # Verify metrics were recorded
        metric_calls = mock_metrics.metrics

        # Check for registration metrics
        registration_calls = [call for call in metric_calls if "connector_registered" in call[1]]
        assert len(registration_calls) == 1

        # Check for access metrics
        access_calls = [call for call in metric_calls if "connector_accessed" in call[1]]
        assert len(access_calls) == 2  # One found, one not found

        # Check for gauge metrics
        gauge_calls = [call for call in metric_calls if call[0] == "gauge"]
        assert len(gauge_calls) >= 1  # At least one total_connectors gauge
