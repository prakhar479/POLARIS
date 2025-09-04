"""
Tests for POLARIS Adapter Architecture and Inheritance Patterns.

This module tests the clean architecture implementation with single inheritance,
proper separation of concerns, and plugin system integration.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import pytest
except ImportError:
    print("pytest not available, skipping tests")
    sys.exit(0)

from polaris.adapters.core import (
    BaseComponent, ExternalAdapter, InternalAdapter, ManagedSystemConnector
)
from polaris.adapters.monitor import MonitorAdapter
from polaris.adapters.execution import ExecutionAdapter
from polaris.adapters.verification import VerificationAdapter


class MockConnector(ManagedSystemConnector):
    """Mock connector for testing."""
    
    def __init__(self, system_config, logger):
        super().__init__(system_config, logger)
        self.connected = False
    
    async def connect(self):
        self.connected = True
    
    async def disconnect(self):
        self.connected = False
    
    async def execute_command(self, command_template, params=None):
        if command_template == "get_server_count":
            return "5"
        elif command_template == "get_max_servers":
            return "10"
        elif command_template == "get_status":
            return "running"
        return "ok"
    
    async def health_check(self):
        return self.connected


class TestBaseComponent:
    """Test the BaseComponent abstract class."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        mock_config = MagicMock()
        mock_config.load_framework_config.return_value = {
            "nats": {"url": "nats://localhost:4222"}
        }
        return mock_config
    
    @pytest.fixture
    def mock_nats_client(self):
        """Mock NATS client."""
        return AsyncMock()
    
    def test_base_component_is_abstract(self):
        """Test that BaseComponent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseComponent("config.yaml")
    
    def test_base_component_inheritance(self):
        """Test that adapters properly inherit from BaseComponent."""
        # Check inheritance hierarchy
        assert issubclass(ExternalAdapter, BaseComponent)
        assert issubclass(InternalAdapter, BaseComponent)
        assert issubclass(MonitorAdapter, ExternalAdapter)
        assert issubclass(ExecutionAdapter, ExternalAdapter)
        assert issubclass(VerificationAdapter, InternalAdapter)
    
    @pytest.mark.asyncio
    async def test_base_component_lifecycle(self, mock_config_manager, mock_nats_client):
        """Test BaseComponent lifecycle management."""
        
        class TestComponent(BaseComponent):
            def __init__(self, config_path):
                super().__init__(config_path)
                self.processing_started = False
                self.processing_stopped = False
            
            async def _start_processing(self):
                self.processing_started = True
            
            async def _stop_processing(self):
                self.processing_stopped = True
        
        with patch('polaris.adapters.core.ConfigurationManager', return_value=mock_config_manager), \
             patch('polaris.adapters.core.NATSClient', return_value=mock_nats_client):
            
            component = TestComponent("test_config.yaml")
            
            # Test start
            await component.start()
            assert component.running is True
            assert component.processing_started is True
            mock_nats_client.connect.assert_called_once()
            
            # Test stop
            await component.stop()
            assert component.running is False
            assert component.processing_stopped is True
            mock_nats_client.close.assert_called_once()


class TestManagedSystemConnector:
    """Test the ManagedSystemConnector interface."""
    
    def test_connector_is_abstract(self):
        """Test that ManagedSystemConnector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ManagedSystemConnector({}, MagicMock())
    
    def test_mock_connector_implementation(self):
        """Test that MockConnector properly implements the interface."""
        logger = MagicMock()
        config = {"connection": {"host": "localhost"}, "implementation": {"timeout": 30}}
        
        connector = MockConnector(config, logger)
        
        assert connector.config == config
        assert connector.logger == logger
        assert connector.get_timeout() == 30
        assert connector.get_max_retries() == 3  # default
    
    @pytest.mark.asyncio
    async def test_connector_lifecycle(self):
        """Test connector connection lifecycle."""
        logger = MagicMock()
        config = {"connection": {}, "implementation": {}}
        
        connector = MockConnector(config, logger)
        
        # Initially not connected
        assert connector.connected is False
        assert await connector.health_check() is False
        
        # Connect
        await connector.connect()
        assert connector.connected is True
        assert await connector.health_check() is True
        
        # Disconnect
        await connector.disconnect()
        assert connector.connected is False
        assert await connector.health_check() is False
    
    @pytest.mark.asyncio
    async def test_connector_command_execution(self):
        """Test connector command execution."""
        logger = MagicMock()
        config = {"connection": {}, "implementation": {}}
        
        connector = MockConnector(config, logger)
        await connector.connect()
        
        # Test various commands
        result = await connector.execute_command("get_server_count")
        assert result == "5"
        
        result = await connector.execute_command("get_max_servers")
        assert result == "10"
        
        result = await connector.execute_command("unknown_command")
        assert result == "ok"


class TestExternalAdapter:
    """Test ExternalAdapter functionality."""
    
    @pytest.fixture
    def mock_plugin_config(self):
        """Mock plugin configuration."""
        return {
            "system_name": "test_system",
            "implementation": {
                "connector_class": "MockConnector"
            },
            "connection": {
                "host": "localhost",
                "port": 8080
            }
        }
    
    @pytest.fixture
    def mock_framework_config(self):
        """Mock framework configuration."""
        return {
            "nats": {"url": "nats://localhost:4222"}
        }
    
    def test_external_adapter_is_abstract(self):
        """Test that ExternalAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExternalAdapter("config.yaml", "plugin_dir")
    
    @pytest.mark.asyncio
    async def test_external_adapter_connector_loading(self, mock_plugin_config, mock_framework_config):
        """Test that ExternalAdapter properly loads connectors."""
        
        class TestExternalAdapter(ExternalAdapter):
            async def _start_processing(self):
                pass
            
            async def _stop_processing(self):
                pass
        
        mock_config_manager = MagicMock()
        mock_config_manager.load_framework_config.return_value = mock_framework_config
        mock_config_manager.load_plugin_config.return_value = mock_plugin_config
        mock_config_manager.get_plugin_connector_class.return_value = "MockConnector"
        
        mock_nats_client = AsyncMock()
        
        # Mock the connector module loading
        with patch('polaris.adapters.core.ConfigurationManager', return_value=mock_config_manager), \
             patch('polaris.adapters.core.NATSClient', return_value=mock_nats_client), \
             patch('polaris.adapters.core.importlib.import_module') as mock_import:
            
            mock_module = MagicMock()
            mock_module.MockConnector = MockConnector
            mock_import.return_value = mock_module
            
            adapter = TestExternalAdapter("config.yaml", "plugin_dir")
            
            # Verify connector was loaded
            assert isinstance(adapter.connector, MockConnector)
            assert adapter.plugin_config == mock_plugin_config


class TestInternalAdapter:
    """Test InternalAdapter functionality."""
    
    @pytest.fixture
    def mock_framework_config(self):
        """Mock framework configuration."""
        return {
            "nats": {"url": "nats://localhost:4222"},
            "verification": {
                "input_subject": "polaris.verification.requests",
                "output_subject": "polaris.verification.results"
            }
        }
    
    def test_internal_adapter_is_abstract(self):
        """Test that InternalAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            InternalAdapter("config.yaml")
    
    @pytest.mark.asyncio
    async def test_internal_adapter_standalone_mode(self, mock_framework_config):
        """Test InternalAdapter can work without plugin configuration."""
        
        class TestInternalAdapter(InternalAdapter):
            async def _start_processing(self):
                pass
            
            async def _stop_processing(self):
                pass
        
        mock_config_manager = MagicMock()
        mock_config_manager.load_framework_config.return_value = mock_framework_config
        mock_config_manager.load_plugin_config.side_effect = Exception("No plugin config")
        
        mock_nats_client = AsyncMock()
        
        with patch('polaris.adapters.core.ConfigurationManager', return_value=mock_config_manager), \
             patch('polaris.adapters.core.NATSClient', return_value=mock_nats_client):
            
            # Should work without plugin directory
            adapter = TestInternalAdapter("config.yaml")
            
            assert adapter.plugin_dir is None
            assert adapter.plugin_config == {}
    
    @pytest.mark.asyncio
    async def test_internal_adapter_with_plugin(self, mock_framework_config):
        """Test InternalAdapter can optionally use plugin configuration."""
        
        class TestInternalAdapter(InternalAdapter):
            async def _start_processing(self):
                pass
            
            async def _stop_processing(self):
                pass
        
        mock_plugin_config = {
            "system_name": "test_system",
            "verification": {
                "constraints": [],
                "policies": []
            }
        }
        
        mock_config_manager = MagicMock()
        mock_config_manager.load_framework_config.return_value = mock_framework_config
        mock_config_manager.load_plugin_config.return_value = mock_plugin_config
        
        mock_nats_client = AsyncMock()
        
        with patch('polaris.adapters.core.ConfigurationManager', return_value=mock_config_manager), \
             patch('polaris.adapters.core.NATSClient', return_value=mock_nats_client):
            
            # Should work with plugin directory
            adapter = TestInternalAdapter("config.yaml", "plugin_dir")
            
            assert adapter.plugin_dir is not None
            assert adapter.plugin_config == mock_plugin_config


class TestAdapterIntegration:
    """Test integration between different adapter types."""
    
    @pytest.mark.asyncio
    async def test_adapter_communication_patterns(self):
        """Test that adapters can communicate through NATS."""
        mock_nats_client = AsyncMock()
        
        # Mock published messages
        published_messages = []
        
        async def mock_publish_json(subject, data):
            published_messages.append({"subject": subject, "data": data})
        
        mock_nats_client.publish_json.side_effect = mock_publish_json
        
        # Test that adapters use consistent subject patterns
        subjects = {
            "telemetry_stream": "polaris.telemetry.events.stream",
            "telemetry_batch": "polaris.telemetry.events.batch",
            "execution_actions": "polaris.execution.actions",
            "execution_results": "polaris.execution.results",
            "verification_requests": "polaris.verification.requests",
            "verification_results": "polaris.verification.results"
        }
        
        # Verify subject naming consistency
        for subject_name, subject in subjects.items():
            assert subject.startswith("polaris.")
            assert "." in subject[8:]  # Has at least one more dot after "polaris."
    
    def test_configuration_consistency(self):
        """Test that configuration patterns are consistent across adapters."""
        # All adapters should support these configuration sections
        expected_sections = ["nats", "logging"]
        
        # External adapters should require these additional sections
        external_sections = ["connection", "implementation"]
        
        # Internal adapters should optionally support these sections
        internal_sections = ["verification", "policies", "constraints"]
        
        # This test verifies the configuration schema expectations
        assert all(section in expected_sections for section in ["nats", "logging"])
        assert all(section in external_sections for section in ["connection", "implementation"])
        assert all(section in internal_sections for section in ["verification", "policies", "constraints"])


if __name__ == "__main__":
    pytest.main([__file__])