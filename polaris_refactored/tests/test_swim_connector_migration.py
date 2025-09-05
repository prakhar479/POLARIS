"""
Test SWIM Connector Migration

Tests for the migrated SWIM connector to ensure it works with the refactored POLARIS architecture.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ..plugins.swim.connector import SwimTCPConnector
from ..src.domain.models import AdaptationAction, HealthStatus, ExecutionStatus


class TestSwimConnectorMigration:
    """Test the migrated SWIM connector."""
    
    @pytest.fixture
    def swim_config(self):
        """Create SWIM configuration for testing."""
        return {
            "system_name": "swim_test",
            "connection": {
                "host": "localhost",
                "port": 4242
            },
            "implementation": {
                "timeout": 30.0,
                "max_retries": 2
            }
        }
    
    @pytest.fixture
    def connector(self, swim_config):
        """Create SWIM connector for testing."""
        return SwimTCPConnector(swim_config)
    
    def test_connector_initialization(self, connector, swim_config):
        """Test connector initialization with configuration."""
        assert connector.host == "localhost"
        assert connector.port == 4242
        assert connector.timeout == 30.0
        assert connector.max_retries == 2
        assert connector._system_id == "swim_test"
    
    def test_connector_initialization_defaults(self):
        """Test connector initialization with default values."""
        connector = SwimTCPConnector()
        assert connector.host == "localhost"
        assert connector.port == 4242
        assert connector.timeout == 30.0
        assert connector.max_retries == 3
        assert connector._system_id == "swim"
    
    @pytest.mark.asyncio
    async def test_get_system_id(self, connector):
        """Test getting system ID."""
        system_id = await connector.get_system_id()
        assert system_id == "swim_test"
    
    @pytest.mark.asyncio
    async def test_get_supported_actions(self, connector):
        """Test getting supported actions."""
        actions = await connector.get_supported_actions()
        expected_actions = [
            "ADD_SERVER", "REMOVE_SERVER", "SCALE_UP", 
            "SCALE_DOWN", "SET_DIMMER", "ADJUST_QOS"
        ]
        assert all(action in actions for action in expected_actions)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, connector):
        """Test successful connection."""
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = "5"  # Mock server count response
            
            result = await connector.connect()
            
            assert result is True
            assert connector._connected is True
            mock_cmd.assert_called_once_with("get_servers")
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, connector):
        """Test connection failure."""
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = ConnectionError("Connection failed")
            
            result = await connector.connect()
            
            assert result is False
            assert connector._connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnection."""
        connector._connected = True
        
        result = await connector.disconnect()
        
        assert result is True
        assert connector._connected is False
    
    @pytest.mark.asyncio
    async def test_collect_metrics_success(self, connector):
        """Test successful metrics collection."""
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            # Mock SWIM command responses
            mock_cmd.side_effect = [
                "5",    # get_servers
                "4",    # get_active_servers  
                "10",   # get_max_servers
                "0.8",  # dimmer
                "150.5", # get_basic_rt (optional)
                "200.3"  # get_opt_rt (optional)
            ]
            
            metrics = await connector.collect_metrics()
            
            assert len(metrics) >= 4  # At least the basic metrics
            assert "server_count" in metrics
            assert "active_servers" in metrics
            assert "max_servers" in metrics
            assert "dimmer" in metrics
            assert "server_utilization" in metrics
            
            # Check metric values
            assert metrics["server_count"].value == 5
            assert metrics["active_servers"].value == 4
            assert metrics["max_servers"].value == 10
            assert metrics["dimmer"].value == 0.8
            assert metrics["server_utilization"].value == 0.4  # 4/10
    
    @pytest.mark.asyncio
    async def test_collect_metrics_failure(self, connector):
        """Test metrics collection failure."""
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = Exception("Command failed")
            
            metrics = await connector.collect_metrics()
            
            assert metrics == {}
    
    @pytest.mark.asyncio
    async def test_get_system_state_healthy(self, connector):
        """Test getting system state when healthy."""
        connector._connected = True
        
        with patch.object(connector, 'collect_metrics', new_callable=AsyncMock) as mock_metrics:
            mock_metrics.return_value = {
                "server_count": Mock(value=5),
                "active_servers": Mock(value=4)
            }
            
            state = await connector.get_system_state()
            
            assert state.system_id == "swim_test"
            assert state.health_status == HealthStatus.HEALTHY
            assert len(state.metrics) == 2
            assert state.metadata["connected"] is True
    
    @pytest.mark.asyncio
    async def test_get_system_state_unhealthy(self, connector):
        """Test getting system state when unhealthy."""
        connector._connected = False
        
        state = await connector.get_system_state()
        
        assert state.system_id == "swim_test"
        assert state.health_status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_execute_action_add_server(self, connector):
        """Test executing ADD_SERVER action."""
        action = AdaptationAction(
            action_id="test_add",
            action_type="ADD_SERVER",
            target_system="swim_test",
            parameters={}
        )
        
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = "Server added"
            
            result = await connector.execute_action(action)
            
            assert result.action_id == "test_add"
            assert result.status == ExecutionStatus.SUCCESS
            assert result.result_data["swim_response"] == "Server added"
            mock_cmd.assert_called_once_with("add_server")
    
    @pytest.mark.asyncio
    async def test_execute_action_remove_server(self, connector):
        """Test executing REMOVE_SERVER action."""
        action = AdaptationAction(
            action_id="test_remove",
            action_type="REMOVE_SERVER", 
            target_system="swim_test",
            parameters={}
        )
        
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = "Server removed"
            
            result = await connector.execute_action(action)
            
            assert result.action_id == "test_remove"
            assert result.status == ExecutionStatus.SUCCESS
            mock_cmd.assert_called_once_with("remove_server")
    
    @pytest.mark.asyncio
    async def test_execute_action_set_dimmer(self, connector):
        """Test executing SET_DIMMER action."""
        action = AdaptationAction(
            action_id="test_dimmer",
            action_type="SET_DIMMER",
            target_system="swim_test", 
            parameters={"value": 0.7}
        )
        
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = "Dimmer set"
            
            result = await connector.execute_action(action)
            
            assert result.action_id == "test_dimmer"
            assert result.status == ExecutionStatus.SUCCESS
            mock_cmd.assert_called_once_with("set_dimmer 0.7")
    
    @pytest.mark.asyncio
    async def test_execute_action_invalid_dimmer(self, connector):
        """Test executing SET_DIMMER action with invalid value."""
        action = AdaptationAction(
            action_id="test_invalid_dimmer",
            action_type="SET_DIMMER",
            target_system="swim_test",
            parameters={"value": 1.5}  # Invalid value > 1.0
        )
        
        result = await connector.execute_action(action)
        
        assert result.action_id == "test_invalid_dimmer"
        assert result.status == ExecutionStatus.FAILED
        assert "must be between 0.0 and 1.0" in result.result_data["error"]
    
    @pytest.mark.asyncio
    async def test_execute_action_unsupported(self, connector):
        """Test executing unsupported action."""
        action = AdaptationAction(
            action_id="test_unsupported",
            action_type="UNSUPPORTED_ACTION",
            target_system="swim_test",
            parameters={}
        )
        
        result = await connector.execute_action(action)
        
        assert result.action_id == "test_unsupported"
        assert result.status == ExecutionStatus.FAILED
        assert "Unsupported action type" in result.result_data["error"]
    
    @pytest.mark.asyncio
    async def test_validate_action_add_server_valid(self, connector):
        """Test validating ADD_SERVER action when valid."""
        action = AdaptationAction(
            action_id="test",
            action_type="ADD_SERVER",
            target_system="swim_test",
            parameters={}
        )
        
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = ["5", "10"]  # current=5, max=10
            
            result = await connector.validate_action(action)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_action_add_server_invalid(self, connector):
        """Test validating ADD_SERVER action when invalid."""
        action = AdaptationAction(
            action_id="test",
            action_type="ADD_SERVER",
            target_system="swim_test",
            parameters={}
        )
        
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = ["10", "10"]  # current=max=10
            
            result = await connector.validate_action(action)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_action_remove_server_valid(self, connector):
        """Test validating REMOVE_SERVER action when valid."""
        action = AdaptationAction(
            action_id="test",
            action_type="REMOVE_SERVER",
            target_system="swim_test",
            parameters={}
        )
        
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = "5"  # current=5 > 1
            
            result = await connector.validate_action(action)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_action_remove_server_invalid(self, connector):
        """Test validating REMOVE_SERVER action when invalid."""
        action = AdaptationAction(
            action_id="test",
            action_type="REMOVE_SERVER",
            target_system="swim_test",
            parameters={}
        )
        
        with patch.object(connector, '_execute_swim_command', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = "1"  # current=1, cannot remove
            
            result = await connector.validate_action(action)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_action_set_dimmer_valid(self, connector):
        """Test validating SET_DIMMER action with valid value."""
        action = AdaptationAction(
            action_id="test",
            action_type="SET_DIMMER",
            target_system="swim_test",
            parameters={"value": 0.5}
        )
        
        result = await connector.validate_action(action)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_action_set_dimmer_invalid(self, connector):
        """Test validating SET_DIMMER action with invalid value."""
        action = AdaptationAction(
            action_id="test",
            action_type="SET_DIMMER",
            target_system="swim_test",
            parameters={"value": 1.5}
        )
        
        result = await connector.validate_action(action)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_action_unsupported(self, connector):
        """Test validating unsupported action."""
        action = AdaptationAction(
            action_id="test",
            action_type="UNSUPPORTED",
            target_system="swim_test",
            parameters={}
        )
        
        result = await connector.validate_action(action)
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])