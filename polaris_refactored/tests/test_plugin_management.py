"""
Test Plugin Management System

Tests for the PolarisPluginRegistry and ManagedSystemConnectorFactory.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from ..src.framework.plugin_management import (
    PolarisPluginRegistry,
    ManagedSystemConnectorFactory,
    PluginDescriptor,
    PluginValidator
)
from ..src.domain.interfaces import ManagedSystemConnector
from ..src.domain.models import SystemState, AdaptationAction, ExecutionResult, MetricValue
from ..src.infrastructure.exceptions import ConnectorError


class MockManagedSystemConnector(ManagedSystemConnector):
    """Mock connector for testing."""

    def __init__(self, system_config: Dict[str, Any] = None):
        self.system_config = system_config or {}
        self.connected = False

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> bool:
        self.connected = False
        return True

    async def get_system_id(self) -> str:
        return "mock_system"

    async def collect_metrics(self) -> Dict[str, MetricValue]:
        return {
            "cpu_usage": MetricValue("cpu_usage", 50.0, "percent"),
            "memory_usage": MetricValue("memory_usage", 60.0, "percent")
        }

    async def get_system_state(self) -> SystemState:
        metrics = await self.collect_metrics()
        return SystemState(
            system_id="mock_system",
            metrics=metrics,
            health_status="healthy"
        )

    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        return ExecutionResult(
            action_id=action.action_id,
            status="success",
            result_data={"executed": True}
        )

    async def validate_action(self, action: AdaptationAction) -> bool:
        return True

    async def get_supported_actions(self) -> List[str]:
        return ["scale_up", "scale_down", "restart"]


class InvalidConnector:
    """Invalid connector that doesn't inherit from ManagedSystemConnector."""
    pass


class IncompleteConnector(ManagedSystemConnector):
    """Incomplete connector missing required methods."""

    async def connect(self) -> bool:
        return True

    # Missing other required methods


class TestPluginValidator:
    """Test the plugin validator."""

    def test_validate_valid_connector_class(self):
        """Test validation of a valid connector class."""
        errors = PluginValidator.validate_connector_class(
            MockManagedSystemConnector)
        assert len(errors) == 0

    def test_validate_invalid_connector_class(self):
        """Test validation of invalid connector class."""
        errors = PluginValidator.validate_connector_class(InvalidConnector)
        assert len(errors) > 0
        assert any(
            "must inherit from ManagedSystemConnector" in error for error in errors)

    def test_validate_incomplete_connector_class(self):
        """Test validation of incomplete connector class."""
        errors = PluginValidator.validate_connector_class(IncompleteConnector)
        assert len(errors) > 0
        # Should have errors for missing methods
        missing_methods = ["disconnect", "get_system_id", "collect_metrics",
                           "get_system_state", "execute_action", "validate_action",
                           "get_supported_actions"]
        for method in missing_methods:
            assert any(method in error for error in errors)

    def test_validate_plugin_metadata_valid(self):
        """Test validation of valid plugin metadata."""
        metadata = {
            "name": "test_plugin",
            "version": "1.0.0",
            "connector_class": "TestConnector"
        }
        errors = PluginValidator.validate_plugin_metadata(metadata)
        assert len(errors) == 0

    def test_validate_plugin_metadata_missing_fields(self):
        """Test validation of metadata with missing fields."""
        metadata = {
            "name": "test_plugin"
            # Missing version and connector_class
        }
        errors = PluginValidator.validate_plugin_metadata(metadata)
        assert len(errors) == 2
        assert any("version" in error for error in errors)
        assert any("connector_class" in error for error in errors)


class TestPolarisPluginRegistry:
    """Test the PolarisPluginRegistry."""

    @pytest.fixture
    def temp_plugin_dir(self):
        """Create a temporary directory for plugin testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def registry(self):
        """Create a plugin registry for testing."""
        return PolarisPluginRegistry()

    def create_mock_plugin(self, plugin_dir: Path, plugin_name: str, with_metadata: bool = True):
        """Create a mock plugin in the given directory."""
        plugin_path = plugin_dir / plugin_name
        plugin_path.mkdir(exist_ok=True)

        # Create connector.py with absolute imports
        connector_code = f"""
from polaris_refactored.src.domain.interfaces import ManagedSystemConnector
from polaris_refactored.src.domain.models import SystemState, AdaptationAction, ExecutionResult, MetricValue
from typing import Dict, List

class {plugin_name.title()}Connector(ManagedSystemConnector):
    async def connect(self) -> bool:
        return True
    
    async def disconnect(self) -> bool:
        return True
    
    async def get_system_id(self) -> str:
        return "{plugin_name}"
    
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        return {{"test_metric": MetricValue("test_metric", 42.0, "units")}}
    
    async def get_system_state(self) -> SystemState:
        return SystemState(system_id="{plugin_name}", metrics={{}}, health_status="healthy")
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        return ExecutionResult(action_id=action.action_id, status="success", result_data={{}})
    
    async def validate_action(self, action: AdaptationAction) -> bool:
        return True
    
    async def get_supported_actions(self) -> List[str]:
        return ["test_action"]
"""

        with open(plugin_path / "connector.py", "w") as f:
            f.write(connector_code)

        if with_metadata:
            # Create plugin.yaml
            metadata = f'''
name: {plugin_name}
version: 1.0.0
connector_class: {plugin_name.title()}Connector
module: connector
description: Test plugin for {plugin_name}
'''
            with open(plugin_path / "plugin.yaml", "w") as f:
                f.write(metadata)

        return plugin_path

    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert registry is not None
        assert len(registry._connectors) == 0
        assert len(registry._plugin_descriptors) == 0

    @pytest.mark.asyncio
    async def test_initialize_with_search_paths(self, registry, temp_plugin_dir):
        """Test initializing registry with search paths."""
        # Create a mock plugin
        self.create_mock_plugin(temp_plugin_dir, "test_plugin")

        await registry.initialize(search_paths=[temp_plugin_dir])

        descriptors = registry.get_plugin_descriptors()
        assert len(descriptors) > 0
        assert "test_plugin" in descriptors

        plugin = descriptors["test_plugin"]
        assert plugin.plugin_id == "test_plugin"
        assert plugin.version == "1.0.0"
        assert plugin.is_valid

    def test_discover_plugins_with_metadata(self, registry, temp_plugin_dir):
        """Test discovering plugins with metadata files."""
        # Create plugins
        self.create_mock_plugin(temp_plugin_dir, "plugin1", with_metadata=True)
        self.create_mock_plugin(temp_plugin_dir, "plugin2", with_metadata=True)

        discovered = registry.discover_managed_system_plugins(
            [temp_plugin_dir])

        assert len(discovered) == 2
        plugin_ids = [p.plugin_id for p in discovered]
        assert "plugin1" in plugin_ids
        assert "plugin2" in plugin_ids

    def test_discover_plugins_without_metadata(self, registry, temp_plugin_dir):
        """Test discovering plugins without metadata (inferred)."""
        # Create plugin without metadata
        self.create_mock_plugin(
            temp_plugin_dir, "inferred_plugin", with_metadata=False)

        discovered = registry.discover_managed_system_plugins(
            [temp_plugin_dir])

        assert len(discovered) == 1
        plugin = discovered[0]
        assert plugin.plugin_id == "inferred_plugin"
        assert plugin.metadata.get("inferred") is True

    @pytest.mark.asyncio
    async def test_load_connector_success(self, registry, temp_plugin_dir):
        """Test successfully loading a connector."""
        # Create and initialize registry
        self.create_mock_plugin(temp_plugin_dir, "loadable_plugin")
        await registry.initialize(search_paths=[temp_plugin_dir])

        # Load connector
        with patch('sys.path'):  # Mock sys.path to avoid import issues
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                with patch('importlib.util.module_from_spec') as mock_module:
                    # Mock the import process
                    mock_spec.return_value = Mock()
                    mock_spec.return_value.loader = Mock()

                    mock_mod = Mock()
                    # Set the connector class to the actual MockManagedSystemConnector class
                    # Note: plugin_name.title() converts "loadable_plugin" to "Loadable_Plugin"
                    setattr(mock_mod, "Loadable_PluginConnector",
                            MockManagedSystemConnector)
                    mock_module.return_value = mock_mod

                    connector = registry.load_managed_system_connector(
                        "loadable_plugin")

                    # Should successfully load the connector
                    assert connector is not None
                    assert isinstance(connector, MockManagedSystemConnector)

    @pytest.mark.asyncio
    async def test_load_nonexistent_connector(self, registry):
        """Test loading a non-existent connector."""
        await registry.initialize()

        connector = registry.load_managed_system_connector("nonexistent")
        assert connector is None

    @pytest.mark.asyncio
    async def test_reload_plugin(self, registry, temp_plugin_dir):
        """Test reloading a plugin."""
        # Create and initialize registry
        self.create_mock_plugin(temp_plugin_dir, "reloadable_plugin")
        await registry.initialize(search_paths=[temp_plugin_dir])

        # Mock a loaded connector
        mock_connector = Mock()
        mock_connector.disconnect = Mock(return_value=asyncio.Future())
        mock_connector.disconnect.return_value.set_result(None)

        registry._connectors["reloadable_plugin"] = mock_connector

        # Test reload
        await registry.reload_plugin("reloadable_plugin")

        # Connector should be removed from registry
        assert "reloadable_plugin" not in registry._connectors

    @pytest.mark.asyncio
    async def test_unload_all_connectors(self, registry):
        """Test unloading all connectors."""
        # Mock some loaded connectors
        mock_connector1 = Mock()
        mock_connector1.disconnect = Mock(return_value=asyncio.Future())
        mock_connector1.disconnect.return_value.set_result(None)

        mock_connector2 = Mock()
        mock_connector2.disconnect = Mock(return_value=asyncio.Future())
        mock_connector2.disconnect.return_value.set_result(None)

        registry._connectors["connector1"] = mock_connector1
        registry._connectors["connector2"] = mock_connector2

        await registry.unload_all_connectors()

        assert len(registry._connectors) == 0
        mock_connector1.disconnect.assert_called_once()
        mock_connector2.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown(self, registry):
        """Test registry shutdown."""
        await registry.initialize()

        # Mock some state
        registry._connectors["test"] = Mock()
        registry._plugin_descriptors["test"] = Mock()

        await registry.shutdown()

        assert len(registry._connectors) == 0
        assert len(registry._plugin_descriptors) == 0


class TestManagedSystemConnectorFactory:
    """Test the ManagedSystemConnectorFactory."""

    @pytest.fixture
    def registry(self):
        """Create a mock plugin registry."""
        return Mock(spec=PolarisPluginRegistry)

    @pytest.fixture
    def factory(self, registry):
        """Create a connector factory."""
        return ManagedSystemConnectorFactory(registry)

    def test_factory_initialization(self, factory, registry):
        """Test factory initialization."""
        assert factory.plugin_registry is registry

    def test_create_connector_success(self, factory, registry):
        """Test successfully creating a connector."""
        mock_connector = MockManagedSystemConnector()
        registry.load_managed_system_connector.return_value = mock_connector

        connector = factory.create_connector("test_system")

        assert connector is mock_connector
        registry.load_managed_system_connector.assert_called_once_with(
            "test_system")

    def test_create_connector_with_config(self, factory, registry):
        """Test creating a connector with configuration."""
        mock_connector = Mock()
        mock_connector.configure = Mock()
        registry.load_managed_system_connector.return_value = mock_connector

        config = {"host": "localhost", "port": 8080}
        connector = factory.create_connector("test_system", config)

        assert connector is mock_connector
        mock_connector.configure.assert_called_once_with(config)

    def test_create_connector_failure(self, factory, registry):
        """Test connector creation failure."""
        registry.load_managed_system_connector.side_effect = Exception(
            "Load failed")

        with pytest.raises(ConnectorError):
            factory.create_connector("test_system")

    def test_get_available_connectors(self, factory, registry):
        """Test getting available connectors."""
        mock_descriptors = {
            "system1": Mock(is_valid=True),
            "system2": Mock(is_valid=False),
            "system3": Mock(is_valid=True)
        }
        registry.get_plugin_descriptors.return_value = mock_descriptors

        available = factory.get_available_connectors()

        assert len(available) == 2
        assert "system1" in available
        assert "system3" in available
        assert "system2" not in available

    def test_get_connector_info(self, factory, registry):
        """Test getting connector information."""
        mock_plugin = Mock()
        mock_plugin.plugin_id = "test_system"
        mock_plugin.version = "1.0.0"
        mock_plugin.is_valid = True
        mock_plugin.metadata = {"description": "Test connector"}
        mock_plugin.validation_errors = []
        mock_plugin.path = Path("/test/path")

        registry.get_plugin_descriptors.return_value = {
            "test_system": mock_plugin}
        registry.is_plugin_loaded.return_value = True

        info = factory.get_connector_info("test_system")

        assert info is not None
        assert info["plugin_id"] == "test_system"
        assert info["version"] == "1.0.0"
        assert info["is_valid"] is True
        assert info["is_loaded"] is True
        assert info["metadata"]["description"] == "Test connector"

    def test_get_connector_info_not_found(self, factory, registry):
        """Test getting info for non-existent connector."""
        registry.get_plugin_descriptors.return_value = {}

        info = factory.get_connector_info("nonexistent")

        assert info is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
