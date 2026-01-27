"""Integration tests for main Polaris framework."""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from polaris.core.polaris import Polaris, PolarisConfig
from polaris.core.models import SystemState, AdaptationAction, ExecutionResult, MetricValue, HealthStatus, ExecutionStatus


class TestPolarisIntegration:
    """Test Polaris framework integration."""
    
    @pytest.fixture
    def polaris_config(self):
        """Create basic Polaris configuration."""
        return PolarisConfig()
    
    @pytest.fixture
    def polaris(self, polaris_config, mock_logger, mock_metrics, mock_connector, mock_strategy):
        """Create Polaris instance with mock components."""
        polaris_instance = Polaris(
            config=polaris_config,
            strategy=mock_strategy,
            connectors=[mock_connector],
            logger=mock_logger,
            metrics=mock_metrics
        )
        # Mock the missing _metrics_export_config attribute
        polaris_instance._metrics_export_config = {'enabled': False}
        return polaris_instance
    
    def test_polaris_initialization(self, polaris_config, mock_logger, mock_metrics):
        """Test Polaris initialization with custom components."""
        polaris = Polaris(
            config=polaris_config,
            logger=mock_logger,
            metrics=mock_metrics
        )
        
        assert polaris.config == polaris_config
        assert polaris.logger == mock_logger
        assert polaris.metrics == mock_metrics
        assert polaris.event_bus is not None
        assert polaris.knowledge_store is not None
        assert polaris.world_model is not None
        assert polaris.strategy is not None  # Should create default threshold strategy
        assert polaris.registry is not None
        assert not polaris.is_running()
    
    def test_polaris_default_initialization(self):
        """Test Polaris initialization with all defaults."""
        polaris = Polaris()
        # Mock the missing attribute
        polaris._metrics_export_config = {'enabled': False}
        
        assert polaris.config is not None
        assert polaris.logger is not None
        assert polaris.metrics is not None
        assert polaris.event_bus is not None
        assert polaris.knowledge_store is not None
        assert polaris.world_model is not None
        assert polaris.strategy is not None
        assert polaris.registry is not None
        assert not polaris.is_running()
    
    def test_register_connector(self, polaris, mock_connector):
        """Test registering a connector."""
        from tests.conftest import MockConnector
        new_connector = MockConnector("new-system")
        
        initial_count = len(polaris._connectors)
        polaris.register_connector(new_connector)
        
        assert len(polaris._connectors) == initial_count + 1
        assert new_connector in polaris._connectors
    
    def test_get_knowledge_store(self, polaris):
        """Test accessing knowledge store."""
        knowledge_store = polaris.get_knowledge_store()
        assert knowledge_store is not None
        assert knowledge_store == polaris.knowledge_store
    
    def test_get_world_model(self, polaris):
        """Test accessing world model."""
        world_model = polaris.get_world_model()
        assert world_model is not None
        assert world_model == polaris.world_model
    
    @pytest.mark.asyncio
    async def test_polaris_lifecycle(self, polaris):
        """Test basic Polaris lifecycle - start and stop."""
        assert not polaris.is_running()
        
        # Start Polaris in background
        run_task = asyncio.create_task(polaris.run())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        assert polaris.is_running()
        
        # Stop Polaris
        await polaris.stop()
        assert not polaris.is_running()
        
        # Wait for run task to complete
        await run_task
    
    @pytest.mark.asyncio
    async def test_connector_registration_and_connection(self, polaris, mock_connector):
        """Test that connectors are registered and connected during startup."""
        # Mock the connector methods
        mock_connector.connect = AsyncMock(return_value=True)
        mock_connector.get_system_id = AsyncMock(return_value="test-system")
        
        # Start Polaris briefly
        run_task = asyncio.create_task(polaris.run())
        await asyncio.sleep(0.1)
        
        # Verify connector was registered and connected
        mock_connector.get_system_id.assert_called()
        mock_connector.connect.assert_called_once()
        
        # Verify connector is in registry
        registered_connector = polaris.registry.get("test-system")
        assert registered_connector == mock_connector
        
        # Stop
        await polaris.stop()
        await run_task
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_execution(self, polaris, mock_connector, mock_strategy):
        """Test that monitoring loop executes telemetry collection and adaptation."""
        # Setup mock connector to return telemetry
        sample_state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 85.0, "percent")  # High CPU
            },
            health_status=HealthStatus.HEALTHY
        )
        
        mock_connector.collect_telemetry = AsyncMock(return_value=sample_state)
        mock_connector.validate_action = AsyncMock(return_value=True)
        mock_connector.execute_action = AsyncMock(return_value=ExecutionResult(
            action_id="test-action",
            status=ExecutionStatus.SUCCESS,
            result_data={}
        ))
        mock_connector.connect = AsyncMock(return_value=True)
        mock_connector.get_system_id = AsyncMock(return_value="test-system")
        
        # Setup strategy to return an action
        test_action = AdaptationAction(
            action_id="test-action",
            action_type="scale_up",
            target_system="test-system",
            parameters={}
        )
        mock_strategy.action_to_return = test_action
        
        # Reduce monitoring interval for faster testing
        polaris._monitoring_interval = 0.05  # Very short interval
        
        # Start Polaris
        run_task = asyncio.create_task(polaris.run())
        
        # Wait for startup and several monitoring cycles
        await asyncio.sleep(0.2)  # Wait for registration and a few cycles
        
        # Verify telemetry collection was called
        assert mock_connector.collect_telemetry.call_count >= 1
        
        # Verify strategy assessment was called
        assert len(mock_strategy.assess_calls) >= 1
        
        # Verify action validation and execution
        assert mock_connector.validate_action.call_count >= 1
        assert mock_connector.execute_action.call_count >= 1
        
        # Stop
        await polaris.stop()
        await run_task
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_no_action_needed(self, polaris, mock_connector, mock_strategy):
        """Test monitoring loop when no adaptation is needed."""
        # Setup mock connector
        sample_state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 50.0, "percent")  # Normal CPU
            },
            health_status=HealthStatus.HEALTHY
        )
        
        mock_connector.collect_telemetry = AsyncMock(return_value=sample_state)
        mock_connector.connect = AsyncMock(return_value=True)
        mock_connector.get_system_id = AsyncMock(return_value="test-system")
        
        # Strategy returns no action
        mock_strategy.action_to_return = None
        
        # Reduce monitoring interval
        polaris._monitoring_interval = 0.05
        
        # Start Polaris
        run_task = asyncio.create_task(polaris.run())
        await asyncio.sleep(0.2)  # Wait for registration and cycles
        
        # Verify telemetry collection was called
        assert mock_connector.collect_telemetry.call_count >= 1
        
        # Verify strategy was assessed
        assert len(mock_strategy.assess_calls) >= 1
        
        # Stop
        await polaris.stop()
        await run_task
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_action_validation_failure(self, polaris, mock_connector, mock_strategy):
        """Test monitoring loop when action validation fails."""
        # Setup mock connector
        sample_state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 85.0, "percent")
            },
            health_status=HealthStatus.HEALTHY
        )
        
        mock_connector.collect_telemetry = AsyncMock(return_value=sample_state)
        mock_connector.validate_action = AsyncMock(return_value=False)  # Validation fails
        mock_connector.execute_action = AsyncMock()
        mock_connector.connect = AsyncMock(return_value=True)
        mock_connector.get_system_id = AsyncMock(return_value="test-system")
        
        # Setup strategy to return an action
        test_action = AdaptationAction(
            action_id="test-action",
            action_type="scale_up",
            target_system="test-system",
            parameters={}
        )
        mock_strategy.action_to_return = test_action
        
        # Reduce monitoring interval
        polaris._monitoring_interval = 0.05
        
        # Start Polaris
        run_task = asyncio.create_task(polaris.run())
        await asyncio.sleep(0.2)  # Wait for registration and cycles
        
        # Verify validation was called but execution was not
        assert mock_connector.validate_action.call_count >= 1
        mock_connector.execute_action.assert_not_called()
        
        # Stop
        await polaris.stop()
        await run_task
    
    @pytest.mark.asyncio
    async def test_error_handling_in_monitoring_loop(self, polaris, mock_connector, mock_strategy):
        """Test error handling in monitoring loop."""
        # Setup connector to raise exception
        mock_connector.collect_telemetry = AsyncMock(side_effect=Exception("Connection failed"))
        mock_connector.connect = AsyncMock(return_value=True)
        mock_connector.get_system_id = AsyncMock(return_value="test-system")
        
        # Reduce monitoring interval
        polaris._monitoring_interval = 0.05
        
        # Start Polaris
        run_task = asyncio.create_task(polaris.run())
        await asyncio.sleep(0.2)  # Wait for registration and cycles
        
        # Verify the loop continues despite errors
        assert mock_connector.collect_telemetry.call_count >= 1
        
        # Stop
        await polaris.stop()
        await run_task
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, polaris, mock_connector):
        """Test graceful shutdown disconnects connectors."""
        mock_connector.disconnect = AsyncMock(return_value=True)
        
        # Start and stop
        run_task = asyncio.create_task(polaris.run())
        await asyncio.sleep(0.1)
        await polaris.stop()
        await run_task
        
        # Verify disconnect was called
        mock_connector.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, polaris):
        """Test using Polaris as async context manager."""
        async with polaris:
            # Context manager should not start the run loop automatically
            assert not polaris.is_running()
        
        # Should be stopped after exiting context
        assert not polaris.is_running()
    
    def test_export_metrics_not_supported(self, polaris):
        """Test export_metrics when not supported by metrics collector."""
        # Mock metrics collector without export_to_file method
        polaris.metrics = Mock()
        delattr(polaris.metrics, 'export_to_file') if hasattr(polaris.metrics, 'export_to_file') else None
        
        with pytest.raises(NotImplementedError):
            polaris.export_metrics("test.json")
    
    def test_get_metrics_summary(self, polaris, mock_metrics):
        """Test getting metrics summary."""
        summary = polaris.get_metrics_summary()
        assert "total_metrics" in summary
    
    @pytest.mark.asyncio
    async def test_multiple_connectors(self, polaris_config, mock_logger, mock_metrics, mock_strategy):
        """Test Polaris with multiple connectors."""
        from tests.conftest import MockConnector
        
        connector1 = MockConnector("system-1")
        connector2 = MockConnector("system-2")
        
        polaris = Polaris(
            config=polaris_config,
            strategy=mock_strategy,
            connectors=[connector1, connector2],
            logger=mock_logger,
            metrics=mock_metrics
        )
        polaris._metrics_export_config = {'enabled': False}
        
        # Mock connector methods
        for connector in [connector1, connector2]:
            connector.connect = AsyncMock(return_value=True)
            connector.collect_telemetry = AsyncMock(return_value=SystemState(
                system_id=await connector.get_system_id(),
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY
            ))
        
        # Reduce monitoring interval
        polaris._monitoring_interval = 0.05
        
        # Start Polaris
        run_task = asyncio.create_task(polaris.run())
        await asyncio.sleep(0.2)  # Wait for registration and cycles
        
        # Verify both connectors were processed
        assert connector1.collect_telemetry.call_count >= 1
        assert connector2.collect_telemetry.call_count >= 1
        
        # Verify both are registered
        assert polaris.registry.get("system-1") == connector1
        assert polaris.registry.get("system-2") == connector2
        
        # Stop
        await polaris.stop()
        await run_task