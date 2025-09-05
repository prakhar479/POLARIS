"""
Tests for the POLARIS Monitor Adapter

Comprehensive tests for the Strategy pattern implementation, metric collection,
telemetry publishing, and integration with the plugin system.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from ..src.adapters.monitor_adapter import (
    MonitorAdapter, MetricCollectionStrategy, DirectConnectorStrategy,
    PollingStrategy, BatchCollectionStrategy, MonitoringTarget,
    CollectionResult, MetricCollectionMode, RetryingStrategyDecorator
)
from ..src.adapters.base_adapter import (
    AdapterConfiguration, AdapterHealthStatus, AdapterState
)
from ..src.framework.events import PolarisEventBus, TelemetryEvent
from ..src.framework.plugin_management import (
    PolarisPluginRegistry, ManagedSystemConnectorFactory
)
from ..src.domain.models import MetricValue, SystemState, HealthStatus
from ..src.domain.interfaces import ManagedSystemConnector


class MockManagedSystemConnector(ManagedSystemConnector):
    """Mock connector for testing."""
    
    def __init__(self, system_id: str = "test_system"):
        self.system_id = system_id
        self.should_fail = False
        self.metrics_to_return = {
            "cpu_usage": MetricValue("cpu_usage", 50.0, "percent"),
            "memory_usage": MetricValue("memory_usage", 60.0, "percent")
        }
    
    async def connect(self) -> bool:
        return True
    
    async def disconnect(self) -> bool:
        return True
    
    def get_system_id(self) -> str:
        return self.system_id
    
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        if self.should_fail:
            raise Exception("Mock collection failure")
        return self.metrics_to_return
    
    async def get_system_state(self) -> SystemState:
        metrics = await self.collect_metrics()
        return SystemState(
            system_id=self.system_id,
            health_status=HealthStatus.HEALTHY,
            metrics=metrics,
            timestamp=datetime.utcnow()
        )
    
    async def execute_action(self, action) -> Any:
        from ..src.domain.models import ExecutionResult, ExecutionStatus
        return ExecutionResult(
            action_id="test_action",
            status=ExecutionStatus.SUCCESS,
            result_data={}
        )
    
    async def validate_action(self, action) -> bool:
        return True
    
    async def get_supported_actions(self) -> list:
        return []


class MockCollectionStrategy(MetricCollectionStrategy):
    """Mock collection strategy for testing."""
    
    def __init__(self, name: str = "mock_strategy"):
        self.name = name
        self.should_fail = False
        self.should_support = True
        self.collection_calls = []
    
    async def collect_metrics(self, target, connector_factory) -> CollectionResult:
        self.collection_calls.append(target.system_id)
        
        if self.should_fail:
            return CollectionResult(
                system_id=target.system_id,
                metrics={},
                timestamp=datetime.utcnow(),
                success=False,
                error="Mock strategy failure"
            )
        
        return CollectionResult(
            system_id=target.system_id,
            metrics={"test_metric": MetricValue("test_metric", 42.0, "units")},
            timestamp=datetime.utcnow(),
            success=True,
            collection_duration=timedelta(milliseconds=100)
        )
    
    def get_strategy_name(self) -> str:
        return self.name
    
    def supports_target(self, target) -> bool:
        return self.should_support


class TestMonitoringTarget:
    """Test the MonitoringTarget class."""
    
    def test_valid_target_creation(self):
        """Test creating a valid monitoring target."""
        target = MonitoringTarget(
            system_id="test_system",
            connector_type="test_connector",
            collection_interval=30.0,
            enabled=True,
            config={"key": "value"}
        )
        
        assert target.system_id == "test_system"
        assert target.connector_type == "test_connector"
        assert target.collection_interval == 30.0
        assert target.enabled is True
        assert target.config == {"key": "value"}
    
    def test_target_validation_success(self):
        """Test successful target validation."""
        target = MonitoringTarget(
            system_id="test_system",
            connector_type="test_connector"
        )
        
        errors = target.validate()
        assert len(errors) == 0
    
    def test_target_validation_failures(self):
        """Test target validation failures."""
        target = MonitoringTarget(
            system_id="",  # Invalid empty ID
            connector_type="",  # Invalid empty type
            collection_interval=-1,  # Invalid negative interval
            enabled="not_bool",  # Invalid type
            config="not_dict"  # Invalid type
        )
        
        errors = target.validate()
        assert len(errors) > 0
        assert any("system_id" in error for error in errors)
        assert any("connector_type" in error for error in errors)
        assert any("collection_interval" in error for error in errors)
        assert any("enabled" in error for error in errors)
        assert any("config" in error for error in errors)


class TestCollectionStrategies:
    """Test the collection strategy implementations."""
    
    @pytest.fixture
    def mock_connector_factory(self):
        """Create a mock connector factory."""
        factory = Mock(spec=ManagedSystemConnectorFactory)
        mock_connector = MockManagedSystemConnector()
        factory.create_connector.return_value = mock_connector
        return factory, mock_connector
    
    @pytest.fixture
    def sample_target(self):
        """Create a sample monitoring target."""
        return MonitoringTarget(
            system_id="test_system",
            connector_type="test_connector",
            collection_interval=30.0
        )
    
    def test_direct_connector_strategy_name(self):
        """Test DirectConnectorStrategy name."""
        strategy = DirectConnectorStrategy()
        assert strategy.get_strategy_name() == "direct_connector"
    
    def test_direct_connector_strategy_supports_target(self, sample_target):
        """Test DirectConnectorStrategy target support."""
        strategy = DirectConnectorStrategy()
        assert strategy.supports_target(sample_target)
        
        # Test with empty connector type
        empty_target = MonitoringTarget(system_id="test", connector_type="")
        assert not strategy.supports_target(empty_target)
    
    @pytest.mark.asyncio
    async def test_direct_connector_strategy_success(self, mock_connector_factory, sample_target):
        """Test successful metric collection with DirectConnectorStrategy."""
        factory, mock_connector = mock_connector_factory
        strategy = DirectConnectorStrategy()
        
        result = await strategy.collect_metrics(sample_target, factory)
        
        assert result.success is True
        assert result.system_id == "test_system"
        assert len(result.metrics) > 0
        assert result.error is None
        assert result.collection_duration is not None
        
        # Verify factory was called correctly
        factory.create_connector.assert_called_once_with("test_connector", sample_target.config)
    
    @pytest.mark.asyncio
    async def test_direct_connector_strategy_connector_creation_failure(self, sample_target):
        """Test DirectConnectorStrategy when connector creation fails."""
        factory = Mock(spec=ManagedSystemConnectorFactory)
        factory.create_connector.return_value = None
        
        strategy = DirectConnectorStrategy()
        result = await strategy.collect_metrics(sample_target, factory)
        
        assert result.success is False
        assert result.system_id == "test_system"
        assert len(result.metrics) == 0
        assert "Failed to create connector" in result.error
    
    @pytest.mark.asyncio
    async def test_direct_connector_strategy_collection_failure(self, mock_connector_factory, sample_target):
        """Test DirectConnectorStrategy when metric collection fails."""
        factory, mock_connector = mock_connector_factory
        mock_connector.should_fail = True
        
        strategy = DirectConnectorStrategy()
        result = await strategy.collect_metrics(sample_target, factory)
        
        assert result.success is False
        assert result.system_id == "test_system"
        assert len(result.metrics) == 0
        assert result.error is not None
    
    def test_polling_strategy_name(self):
        """Test PollingStrategy name."""
        base_strategy = DirectConnectorStrategy()
        strategy = PollingStrategy(base_strategy)
        assert strategy.get_strategy_name() == "polling_direct_connector"
    
    def test_batch_strategy_name(self):
        """Test BatchCollectionStrategy name."""
        base_strategy = DirectConnectorStrategy()
        strategy = BatchCollectionStrategy(base_strategy)
        assert strategy.get_strategy_name() == "batch_direct_connector"


class TestMonitorAdapter:
    """Test the MonitorAdapter class."""
    
    @pytest.fixture
    def valid_configuration(self):
        """Create a valid monitor adapter configuration."""
        return AdapterConfiguration(
            adapter_id="test_monitor",
            adapter_type="monitor",
            enabled=True,
            config={
                "collection_mode": "pull",
                "monitoring_targets": [
                    {
                        "system_id": "system1",
                        "connector_type": "test_connector",
                        "collection_interval": 1.0,  # Short interval for testing
                        "enabled": True,
                        "config": {}
                    },
                    {
                        "system_id": "system2",
                        "connector_type": "test_connector",
                        "collection_interval": 1.0,
                        "enabled": True,
                        "config": {}
                    }
                ]
            }
        )
    
    @pytest.fixture
    async def event_bus(self):
        """Create an event bus for testing."""
        bus = PolarisEventBus(worker_count=1)
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    def mock_plugin_registry(self):
        """Create a mock plugin registry."""
        registry = Mock(spec=PolarisPluginRegistry)
        return registry
    
    @pytest.fixture
    def monitor_adapter(self, valid_configuration, event_bus, mock_plugin_registry):
        """Create a monitor adapter for testing."""
        return MonitorAdapter(
            configuration=valid_configuration,
            event_bus=event_bus,
            plugin_registry=mock_plugin_registry
        )
    
    @pytest.mark.asyncio
    async def test_monitor_adapter_initialization(self, monitor_adapter):
        """Test monitor adapter initialization."""
        assert monitor_adapter.adapter_id == "test_monitor"
        assert monitor_adapter.adapter_type == "monitor"
        assert monitor_adapter.state == AdapterState.STOPPED
        
        # Targets are loaded during validation
        await monitor_adapter._validate_configuration()
        assert len(monitor_adapter._monitoring_targets) == 2
        assert "system1" in monitor_adapter._monitoring_targets
        assert "system2" in monitor_adapter._monitoring_targets
    
    @pytest.mark.asyncio
    async def test_configuration_validation_success(self, monitor_adapter):
        """Test successful configuration validation."""
        # Should not raise any exceptions
        await monitor_adapter._validate_configuration()
        
        # Check that targets were loaded
        assert len(monitor_adapter._monitoring_targets) == 2
    
    @pytest.mark.asyncio
    async def test_configuration_validation_invalid_mode(self, valid_configuration, mock_plugin_registry):
        """Test configuration validation with invalid collection mode."""
        valid_configuration.config["collection_mode"] = "invalid_mode"
        
        adapter = MonitorAdapter(
            configuration=valid_configuration,
            plugin_registry=mock_plugin_registry
        )
        
        with pytest.raises(Exception):  # Should raise validation error
            await adapter._validate_configuration()
    
    @pytest.mark.asyncio
    async def test_configuration_validation_no_plugin_registry(self, valid_configuration):
        """Test configuration validation without plugin registry."""
        adapter = MonitorAdapter(
            configuration=valid_configuration,
            plugin_registry=None
        )
        
        with pytest.raises(Exception):  # Should raise validation error
            await adapter._validate_configuration()
    
    @pytest.mark.asyncio
    async def test_resource_initialization(self, monitor_adapter):
        """Test resource initialization."""
        await monitor_adapter._validate_configuration()
        await monitor_adapter._initialize_resources()
        
        assert monitor_adapter.connector_factory is not None
        assert len(monitor_adapter._collection_strategies) > 0
        assert monitor_adapter._default_strategy is not None
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, monitor_adapter):
        """Test monitor adapter start and stop lifecycle."""
        # Mock the connector factory to avoid actual connector creation
        with patch.object(monitor_adapter, 'connector_factory') as mock_factory:
            mock_connector = MockManagedSystemConnector()
            mock_factory.create_connector.return_value = mock_connector
            
            # Start adapter
            await monitor_adapter.start()
            
            assert monitor_adapter.is_running()
            assert len(monitor_adapter._collection_tasks) == 2  # Two enabled targets
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            # Stop adapter
            await monitor_adapter.stop()
            
            assert not monitor_adapter.is_running()
            assert len(monitor_adapter._collection_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_metric_collection_with_strategy(self, monitor_adapter):
        """Test metric collection using strategies."""
        await monitor_adapter._validate_configuration()
        await monitor_adapter._initialize_resources()
        
        # Get a target before we modify the strategies
        target = list(monitor_adapter._monitoring_targets.values())[0]
        
        # Clear existing strategies to ensure only our mock is used
        monitor_adapter._collection_strategies = {}
        monitor_adapter._default_strategy = None
        
        # Add our mock strategy as the only strategy
        mock_strategy = MockCollectionStrategy("test_strategy")
        monitor_adapter.add_collection_strategy(mock_strategy)
        
        # Set it as the default strategy
        monitor_adapter.set_default_strategy("test_strategy")
        
        # Mock the connector factory to return a proper mock connector
        mock_connector = MockManagedSystemConnector()
        monitor_adapter.connector_factory = Mock()
        monitor_adapter.connector_factory.create_connector = Mock(return_value=mock_connector)
        
        # Debug: Print available strategies and selection
        print(f"Available strategies: {monitor_adapter._collection_strategies}")
        print(f"Default strategy: {monitor_adapter._default_strategy}")
        
        # Force the target to use our strategy
        target.config["collection_strategy"] = "test_strategy"
        
        result = await monitor_adapter._collect_target_metrics(target)
        
        # Debug: Print the result and collection calls
        print(f"Collection result: {result}")
        print(f"Collection calls: {mock_strategy.collection_calls}")
        
        # Verify the strategy was called
        assert len(mock_strategy.collection_calls) == 1, "Mock strategy was not called"
        assert mock_strategy.collection_calls[0] == target.system_id, "Unexpected system ID in collection calls"
        
        # Verify the result
        assert result.success is True
        assert result.system_id == target.system_id
    
    @pytest.mark.asyncio
    async def test_telemetry_event_publishing(self, monitor_adapter):
        """Test telemetry event publishing."""
        await monitor_adapter._validate_configuration()
        await monitor_adapter._initialize_resources()
        
        # Mock event bus
        published_events = []
        
        async def mock_publish_telemetry(event):
            published_events.append(event)
        
        monitor_adapter.event_bus.publish_telemetry = mock_publish_telemetry
        
        # Create a successful collection result
        result = CollectionResult(
            system_id="test_system",
            metrics={"test_metric": MetricValue("test_metric", 42.0, "units")},
            timestamp=datetime.utcnow(),
            success=True
        )
        
        # Publish telemetry event
        await monitor_adapter._publish_telemetry_event(result)
        
        # Verify event was published
        assert len(published_events) == 1
        assert isinstance(published_events[0], TelemetryEvent)
        assert published_events[0].system_id == "test_system"
    
    def test_monitoring_target_management(self, monitor_adapter):
        """Test adding and removing monitoring targets."""
        initial_count = len(monitor_adapter.get_monitoring_targets())
        
        # Add new target
        new_target = MonitoringTarget(
            system_id="new_system",
            connector_type="new_connector",
            collection_interval=60.0
        )
        
        monitor_adapter.add_monitoring_target(new_target)
        
        targets = monitor_adapter.get_monitoring_targets()
        assert len(targets) == initial_count + 1
        assert "new_system" in targets
        
        # Remove target
        success = monitor_adapter.remove_monitoring_target("new_system")
        assert success is True
        
        targets = monitor_adapter.get_monitoring_targets()
        assert len(targets) == initial_count
        assert "new_system" not in targets
        
        # Try to remove non-existent target
        success = monitor_adapter.remove_monitoring_target("non_existent")
        assert success is False
    
    def test_collection_statistics(self, monitor_adapter):
        """Test collection statistics tracking."""
        # Initial stats
        stats = monitor_adapter.get_collection_statistics()
        assert stats["total_collections"] == 0
        assert stats["successful_collections"] == 0
        assert stats["failed_collections"] == 0
        
        # Update stats with successful collection
        successful_result = CollectionResult(
            system_id="test_system",
            metrics={},
            timestamp=datetime.utcnow(),
            success=True,
            collection_duration=timedelta(milliseconds=100)
        )
        
        monitor_adapter._update_collection_stats(successful_result)
        
        stats = monitor_adapter.get_collection_statistics()
        assert stats["total_collections"] == 1
        assert stats["successful_collections"] == 1
        assert stats["failed_collections"] == 0
        assert stats["average_collection_time"] > 0
        
        # Update stats with failed collection
        failed_result = CollectionResult(
            system_id="test_system",
            metrics={},
            timestamp=datetime.utcnow(),
            success=False,
            error="Test error"
        )
        
        monitor_adapter._update_collection_stats(failed_result)
        
        stats = monitor_adapter.get_collection_statistics()
        assert stats["total_collections"] == 2
        assert stats["successful_collections"] == 1
        assert stats["failed_collections"] == 1
    
    def test_strategy_management(self, monitor_adapter):
        """Test collection strategy management."""
        # Add custom strategy
        custom_strategy = MockCollectionStrategy("custom_strategy")
        monitor_adapter.add_collection_strategy(custom_strategy)
        
        stats = monitor_adapter.get_collection_statistics()
        assert "custom_strategy" in stats["available_strategies"]
        
        # Set as default strategy
        success = monitor_adapter.set_default_strategy("custom_strategy")
        assert success is True
        assert monitor_adapter._default_strategy == custom_strategy
        
        # Try to set non-existent strategy as default
        success = monitor_adapter.set_default_strategy("non_existent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_health_check(self, monitor_adapter):
        """Test adapter health check functionality."""
        await monitor_adapter._validate_configuration()
        await monitor_adapter._initialize_resources()
        
        # Initial health check (no collections yet)
        health = await monitor_adapter._check_health()
        assert health == AdapterHealthStatus.DEGRADED  # No active tasks
        
        # Simulate some successful collections with recent timestamp
        for _ in range(10):
            result = CollectionResult(
                system_id="test_system",
                metrics={},
                timestamp=datetime.utcnow(),
                success=True
            )
            monitor_adapter._update_collection_stats(result)
        
        # Mock having active collection tasks to simulate running state
        monitor_adapter._collection_tasks["test_system"] = Mock()
        
        # Should be healthy now
        health = await monitor_adapter._check_health()
        assert health == AdapterHealthStatus.HEALTHY
        
        # Simulate many failed collections
        for _ in range(20):
            result = CollectionResult(
                system_id="test_system",
                metrics={},
                timestamp=datetime.utcnow(),
                success=False
            )
            monitor_adapter._update_collection_stats(result)
        
        # Should be unhealthy now due to low success rate
        health = await monitor_adapter._check_health()
        assert health == AdapterHealthStatus.UNHEALTHY
    
    def test_strategy_selection(self, monitor_adapter):
        """Test collection strategy selection logic."""
        # Add strategies
        strategy1 = MockCollectionStrategy("strategy1")
        strategy2 = MockCollectionStrategy("strategy2")
        
        monitor_adapter.add_collection_strategy(strategy1)
        monitor_adapter.add_collection_strategy(strategy2)
        
        # Test target without preferred strategy
        target = MonitoringTarget(
            system_id="test_system",
            connector_type="test_connector"
        )
        
        selected = monitor_adapter._select_strategy(target)
        assert selected is not None
        
        # Test target with preferred strategy
        target_with_preference = MonitoringTarget(
            system_id="test_system",
            connector_type="test_connector",
            config={"collection_strategy": "strategy2"}
        )
        
        selected = monitor_adapter._select_strategy(target_with_preference)
        assert selected == strategy2
        
        # Test target with non-existent preferred strategy
        target_with_invalid_preference = MonitoringTarget(
            system_id="test_system",
            connector_type="test_connector",
            config={"collection_strategy": "non_existent"}
        )
        
        selected = monitor_adapter._select_strategy(target_with_invalid_preference)
        assert selected is not None  # Should fall back to a supported strategy


class TestRetryingStrategyAndAdaptiveInterval:
    """Tests for retrying decorator and adaptive interval behavior."""
    @pytest.fixture
    async def event_bus(self):
        """Create an event bus for testing."""
        bus = PolarisEventBus(worker_count=1)
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    def mock_plugin_registry(self):
        """Create a mock plugin registry."""
        registry = Mock(spec=PolarisPluginRegistry)
        return registry

    @pytest.mark.asyncio
    async def test_retrying_strategy_succeeds_after_failures(self):
        """Verify RetryingStrategyDecorator retries and eventually succeeds."""
        class FlakyStrategy(MetricCollectionStrategy):
            def __init__(self, name: str = "flaky", fail_times: int = 2):
                self.name = name
                self.remaining = fail_times
            def get_strategy_name(self) -> str:
                return self.name
            def supports_target(self, target) -> bool:
                return True
            async def collect_metrics(self, target, connector_factory) -> CollectionResult:
                if self.remaining > 0:
                    self.remaining -= 1
                    return CollectionResult(
                        system_id=target.system_id,
                        metrics={},
                        timestamp=datetime.utcnow(),
                        success=False,
                        error="transient"
                    )
                return CollectionResult(
                    system_id=target.system_id,
                    metrics={"ok": MetricValue("ok", 1, "u")},
                    timestamp=datetime.utcnow(),
                    success=True
                )

        base = FlakyStrategy(fail_times=2)
        retrying = RetryingStrategyDecorator(base, max_retries=3, backoff_base=0.0, backoff_factor=1.0, max_backoff=0.0, jitter=0.0)
        target = MonitoringTarget(system_id="s1", connector_type="x")
        factory = Mock(spec=ManagedSystemConnectorFactory)
        result = await retrying.collect_metrics(target, factory)
        assert result.success is True
        assert result.strategy_name.startswith("retrying_")

    @pytest.mark.asyncio
    async def test_retrying_strategy_fails_after_max_retries(self):
        """Verify RetryingStrategyDecorator returns failure when retries exhausted."""
        class AlwaysFailStrategy(MetricCollectionStrategy):
            def get_strategy_name(self) -> str:
                return "always_fail"
            def supports_target(self, target) -> bool:
                return True
            async def collect_metrics(self, target, connector_factory) -> CollectionResult:
                return CollectionResult(
                    system_id=target.system_id,
                    metrics={},
                    timestamp=datetime.utcnow(),
                    success=False,
                    error="always"
                )

        retrying = RetryingStrategyDecorator(AlwaysFailStrategy(), max_retries=2, backoff_base=0.0, backoff_factor=1.0, max_backoff=0.0, jitter=0.0)
        target = MonitoringTarget(system_id="s2", connector_type="x")
        factory = Mock(spec=ManagedSystemConnectorFactory)
        result = await retrying.collect_metrics(target, factory)
        assert result.success is False
        assert result.strategy_name == "retrying_always_fail"
        assert result.collection_duration is not None

    @pytest.mark.asyncio
    async def test_adaptive_interval_success_path(event_bus):
        """Ensure adaptive interval computes expected next_interval on success."""
        cfg = AdapterConfiguration(
            adapter_id="m1",
            adapter_type="monitor",
            enabled=True,
            config={
                "collection_mode": "pull",
                "monitoring_targets": [
                    {
                        "system_id": "sysA",
                        "connector_type": "c",
                        "collection_interval": 10.0,
                        "enabled": True,
                        "config": {"success_adjustment": 0.9, "min_interval": 3.0},
                    }
                ]
            },
        )
        registry = Mock(spec=PolarisPluginRegistry)
        adapter = MonitorAdapter(cfg, event_bus=event_bus, plugin_registry=registry)
        await adapter._validate_configuration()
        await adapter._initialize_resources()

        # Patch collection to return success once
        async def mock_collect(target):
            return CollectionResult(
                system_id=target.system_id,
                metrics={},
                timestamp=datetime.utcnow(),
                success=True,
            )

        adapter._collect_target_metrics = mock_collect  # type: ignore

        # Monkeypatch asyncio.sleep to capture interval and cancel loop
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            raise asyncio.CancelledError()

        target = adapter.get_monitoring_targets()["sysA"]
        adapter._state = AdapterState.RUNNING
        with patch("asyncio.sleep", side_effect=fake_sleep):
            # Run one iteration
            task = asyncio.create_task(adapter._collection_loop(target))
            await task
        # Expected next_interval = max(min_interval, base * success_adjustment) = max(3, 10*0.9) = 9
        assert len(sleep_calls) == 1
        assert abs(sleep_calls[0] - 9.0) < 1e-6



    @pytest.mark.asyncio
    async def test_adaptive_interval_failure_path(event_bus, mock_plugin_registry):
        """Ensure adaptive interval computes expected next_interval on failure."""
        cfg = AdapterConfiguration(
            adapter_id="m2",
            adapter_type="monitor",
            enabled=True,
            config={
                "collection_mode": "pull",
                "monitoring_targets": [
                    {
                        "system_id": "sysB",
                        "connector_type": "c",
                        "collection_interval": 5.0,
                        "enabled": True,
                        "config": {"failure_backoff": 2.5, "max_interval": 11.0},
                    }
                ]
            },
        )

        adapter = MonitorAdapter(cfg, event_bus=event_bus, plugin_registry=mock_plugin_registry)
        await adapter._validate_configuration()
        await adapter._initialize_resources()

        # Patch collection to return failure once
        async def mock_collect(target):
            return CollectionResult(
                system_id=target.system_id,
                metrics={},
                timestamp=datetime.utcnow(),
                success=False,
            )

        adapter._collect_target_metrics = mock_collect  # type: ignore

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            raise asyncio.CancelledError()

        target = adapter.get_monitoring_targets()["sysB"]
        adapter._state = AdapterState.RUNNING
        with patch("asyncio.sleep", side_effect=fake_sleep):
            task = asyncio.create_task(adapter._collection_loop(target))
            await task

        # Expected next_interval = min(max_interval, base * failure_backoff) = min(11, 5*2.5) = 11
        assert len(sleep_calls) == 1
        assert abs(sleep_calls[0] - 11.0) < 1e-6




if __name__ == "__main__":
    pytest.main([__file__, "-v"])