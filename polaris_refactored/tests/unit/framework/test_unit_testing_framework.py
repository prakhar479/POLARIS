"""
Unit tests for the POLARIS unit testing framework itself.

This module demonstrates the testing framework capabilities and ensures
the testing infrastructure works correctly.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from tests.fixtures.test_fixtures import *
from tests.fixtures.mock_objects import *
from tests.utils.test_helpers import TestAssertions, AsyncTestHelper, MockHelper, PerformanceTestHelper

from src.domain.models import SystemState, AdaptationAction, ExecutionResult, HealthStatus, ExecutionStatus
from src.framework.events import TelemetryEvent

class TestMockObjects:
    """Test suite for mock objects and test doubles."""
    
    def test_mock_connector_initialization(self):
        """Test MockManagedSystemConnector initialization."""
        connector = MockManagedSystemConnector("test_system")
        
        assert connector.system_id == "test_system"
        assert not connector.connected
        assert len(connector.collected_metrics) == 0
        assert len(connector.executed_actions) == 0
        assert connector.connection_attempts == 0
    
    @pytest.mark.asyncio
    async def test_mock_connector_connection_lifecycle(self):
        """Test connection lifecycle of mock connector."""
        connector = MockManagedSystemConnector()
        
        # Test successful connection
        await connector.connect()
        assert connector.connected
        assert connector.connection_attempts == 1
        
        # Test disconnection
        await connector.disconnect()
        assert not connector.connected
    
    @pytest.mark.asyncio
    async def test_mock_connector_failure_scenarios(self):
        """Test failure scenarios in mock connector."""
        connector = MockManagedSystemConnector()
        
        # Test connection failure
        connector.should_fail_connection = True
        with pytest.raises(ConnectionError):
            await connector.connect()
        
        # Test metrics collection failure
        connector.should_fail_connection = False
        await connector.connect()
        connector.should_fail_metrics = True
        
        with pytest.raises(RuntimeError):
            await connector.collect_metrics()
    
    @pytest.mark.asyncio
    async def test_mock_message_broker(self):
        """Test MockMessageBroker functionality."""
        broker = MockMessageBroker()
        
        # Test connection
        await broker.connect()
        assert broker.connected
        
        # Test subscription
        received_messages = []
        
        def handler(message: bytes):
            received_messages.append(message)
        
        sub_id = await broker.subscribe("test_topic", handler)
        assert sub_id is not None
        
        # Test publishing
        test_message = b"test message"
        await broker.publish("test_topic", test_message)
        
        # Allow async delivery
        await asyncio.sleep(0.1)
        
        assert len(received_messages) == 1
        assert received_messages[0] == test_message
        assert len(broker.published_messages) == 1
    
    def test_mock_data_store(self):
        """Test MockDataStore functionality."""
        store = MockDataStore()
        
        # Test basic operations
        asyncio.run(self._test_data_store_operations(store))
    
    async def _test_data_store_operations(self, store):
        """Helper for testing data store operations."""
        # Test store and retrieve
        await store.store("key1", {"data": "value1"})
        result = await store.retrieve("key1")
        assert result == {"data": "value1"}
        
        # Test query
        await store.store("key2", {"data": "value2"})
        results = await store.query("key")
        assert len(results) == 2
        
        # Test delete
        await store.delete("key1")
        result = await store.retrieve("key1")
        assert result is None


class TestDataBuilder:
    """Test suite for DataBuilder utility."""
    
    def test_system_state_builder(self, test_data_builder):
        """Test SystemState creation with builder."""
        state = test_data_builder.system_state(
            system_id="test_system",
            health_status=HealthStatus.HEALTHY
        )
        
        TestAssertions.assert_system_state_valid(state)
        assert state.system_id == "test_system"
        assert state.health_status == HealthStatus.HEALTHY
        assert isinstance(state.metrics, dict)
    
    def test_adaptation_action_builder(self, test_data_builder):
        """Test AdaptationAction creation with builder."""
        action = test_data_builder.adaptation_action(
            action_id="test_action",
            action_type="scale_up",
            target_system="test_system"
        )
        
        assert action.action_id == "test_action"
        assert action.action_type == "scale_up"
        assert action.target_system == "test_system"
        assert isinstance(action.parameters, dict)
    
    def test_telemetry_event_builder(self, test_data_builder):
        """Test TelemetryEvent creation with builder."""
        event = test_data_builder.telemetry_event(system_id="test_system")
        
        TestAssertions.assert_telemetry_event_valid(event)
        assert event.system_id == "test_system"


class TestTestAssertions:
    """Test suite for custom test assertions."""
    
    def test_system_state_assertions(self, sample_system_state):
        """Test system state validation assertions."""
        # Should not raise any exceptions
        TestAssertions.assert_system_state_valid(sample_system_state)
        
        # Test with invalid state
        invalid_state = SystemState(
            system_id=None,  # Invalid
            timestamp=datetime.now(),
            metrics={},
            health_status=HealthStatus.HEALTHY
        )
        
        with pytest.raises(AssertionError):
            TestAssertions.assert_system_state_valid(invalid_state)
    
    def test_metrics_assertions(self, mock_metrics_collector):
        """Test metrics collection assertions."""
        # Record some metrics
        mock_metrics_collector.increment_counter("test_metric")
        mock_metrics_collector.increment_counter("test_metric")
        
        # Test assertions
        TestAssertions.assert_metrics_collected(mock_metrics_collector, "test_metric", 2)
        TestAssertions.assert_metrics_collected(mock_metrics_collector, "test_metric")
        
        # Test failure case
        with pytest.raises(AssertionError):
            TestAssertions.assert_metrics_collected(mock_metrics_collector, "nonexistent_metric")
    
    def test_log_assertions(self, mock_logger):
        """Test log assertion functionality."""
        # Generate some logs
        mock_logger.info("Test info message")
        mock_logger.error("Test error message")
        
        # Test assertions
        TestAssertions.assert_logs_contain(mock_logger, "INFO", "info message")
        TestAssertions.assert_logs_contain(mock_logger, "ERROR", "error message")
        
        # Test failure case
        with pytest.raises(AssertionError):
            TestAssertions.assert_logs_contain(mock_logger, "DEBUG", "nonexistent message")


class TestAsyncTestHelper:
    """Test suite for async testing utilities."""
    
    @pytest.mark.asyncio
    async def test_wait_for_condition(self):
        """Test waiting for conditions to become true."""
        # Test condition that becomes true
        counter = {"value": 0}
        
        def increment_counter():
            counter["value"] += 1
        
        def condition():
            return counter["value"] >= 3
        
        # Start incrementing in background
        async def background_task():
            for _ in range(5):
                await asyncio.sleep(0.05)
                increment_counter()
        
        task = asyncio.create_task(background_task())
        
        # Wait for condition
        result = await AsyncTestHelper.wait_for_condition(condition, timeout=1.0, interval=0.01)
        
        assert result is True
        await task
    
    @pytest.mark.asyncio
    async def test_timeout_context(self):
        """Test timeout context manager."""
        # Test operation that completes within timeout
        async with AsyncTestHelper.timeout_context(1.0):
            await asyncio.sleep(0.1)
        
        # Test would timeout (but we can't easily test the failure case without it actually failing)
    
    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        """Test running coroutines with timeout."""
        # Test successful operation
        async def quick_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await AsyncTestHelper.run_with_timeout(quick_operation(), timeout=1.0)
        assert result == "success"


class TestPerformanceTestHelper:
    """Test suite for performance testing utilities."""
    
    def test_measure_time(self):
        """Test time measurement utility."""
        import time
        
        with PerformanceTestHelper.measure_time() as result:
            time.sleep(0.1)  # Simulate work
        
        assert result["duration"] is not None
        assert result["duration"] >= 0.1
        assert result["duration"] < 0.2  # Should be close to 0.1
    
    @pytest.mark.asyncio
    async def test_measure_async_time(self):
        """Test async time measurement."""
        async def test_operation():
            await asyncio.sleep(0.1)
            return "completed"
        
        result, duration = await PerformanceTestHelper.measure_async_time(test_operation())
        
        assert result == "completed"
        assert duration >= 0.1
        assert duration < 0.2
    
    def test_performance_assertions(self):
        """Test performance assertion utilities."""
        # Test passing assertion
        PerformanceTestHelper.assert_performance_within_bounds(0.1, 0.2)
        
        # Test failing assertion
        with pytest.raises(AssertionError):
            PerformanceTestHelper.assert_performance_within_bounds(0.3, 0.2)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_load_test_utility(self):
        """Test load testing utility."""
        call_count = {"value": 0}
        
        async def test_operation():
            call_count["value"] += 1
            await asyncio.sleep(0.01)  # Simulate work
            return f"result_{call_count['value']}"
        
        # Run load test
        results = await PerformanceTestHelper.run_load_test(
            operation=test_operation,
            concurrent_requests=5,
            total_requests=20
        )
        
        assert results["total_requests"] == 20
        assert results["successful_requests"] == 20
        assert results["failed_requests"] == 0
        assert results["success_rate"] == 1.0
        assert results["requests_per_second"] > 0
        assert results["avg_response_time"] > 0


class TestScenarioBuilder:
    """Test suite for test scenario builder."""
    
    def test_multi_system_scenario(self, test_scenario_builder):
        """Test building multi-system scenarios."""
        scenario = (test_scenario_builder
                   .add_system("web_server")
                   .add_system("database")
                   .add_system("cache")
                   .build())
        
        assert len(scenario["connectors"]) == 3
        assert "web_server" in scenario["connectors"]
        assert "database" in scenario["connectors"]
        assert "cache" in scenario["connectors"]
        assert scenario["message_broker"] is not None
        assert scenario["data_store"] is not None
    
    def test_failure_scenario(self, test_scenario_builder):
        """Test building failure scenarios."""
        scenario = (test_scenario_builder
                   .add_system("healthy_system")
                   .with_failing_system("failing_system", "connection")
                   .with_failing_infrastructure("data_store")
                   .build())
        
        # Test that failing system is configured correctly
        failing_connector = scenario["connectors"]["failing_system"]
        assert failing_connector.should_fail_connection is True
        
        # Test that infrastructure failure is configured
        assert scenario["data_store"].should_fail_operations is True


class TestDependencyInjection:
    """Test suite for dependency injection in tests."""
    
    def test_di_container_fixture(self, di_container):
        """Test dependency injection container fixture."""
        # Test that container has required services
        message_broker = di_container.get("message_broker")
        assert message_broker is not None
        
        data_store = di_container.get("data_store")
        assert data_store is not None
        
        metrics_collector = di_container.get("metrics_collector")
        assert metrics_collector is not None
    
    def test_mock_di_container(self, mock_di_container):
        """Test mocked DI container."""
        # Configure mock
        mock_di_container.get.return_value = "mocked_service"
        
        # Test usage
        service = mock_di_container.get("any_service")
        assert service == "mocked_service"
        
        mock_di_container.get.assert_called_once_with("any_service")


# Parametrized tests to demonstrate the framework
class TestParametrizedScenarios:
    """Test suite demonstrating parametrized testing."""
    
    @pytest.mark.parametrize("system_id,expected_prefix", [
        ("web_server", "web"),
        ("database_server", "database"),
        ("cache_server", "cache"),
    ])
    def test_system_id_processing(self, system_id, expected_prefix):
        """Test system ID processing with different inputs."""
        # Simple example of parametrized testing
        actual_prefix = system_id.split("_")[0]
        assert actual_prefix == expected_prefix
    
    @pytest.mark.parametrize("health_status", [
        HealthStatus.HEALTHY,
        HealthStatus.WARNING,
        HealthStatus.UNHEALTHY
    ])
    def test_system_state_with_different_health(self, health_status, test_data_builder):
        """Test system state creation with different health statuses."""
        state = test_data_builder.system_state(health_status=health_status)
        assert state.health_status == health_status
        TestAssertions.assert_system_state_valid(state)
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50])
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, batch_size):
        """Test performance with different batch sizes."""
        items = list(range(batch_size))
        
        async def process_batch(batch):
            await asyncio.sleep(0.001 * len(batch))  # Simulate processing time
            return len(batch)
        
        start_time = asyncio.get_event_loop().time()
        result = await process_batch(items)
        end_time = asyncio.get_event_loop().time()
        
        assert result == batch_size
        
        # Performance should scale reasonably with batch size
        execution_time = end_time - start_time
        expected_max_time = 0.001 * batch_size + 0.01  # Allow some overhead
        assert execution_time <= expected_max_time


# Integration test examples
class TestFrameworkIntegration:
    """Integration tests for the testing framework components."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_mock_scenario(self, multi_system_scenario):
        """Test end-to-end scenario with multiple mock components."""
        connectors = multi_system_scenario["connectors"]
        message_broker = multi_system_scenario["message_broker"]
        
        # Connect all systems
        for connector in connectors.values():
            await connector.connect()
            assert connector.connected
        
        # Connect message broker
        await message_broker.connect()
        assert message_broker.connected
        
        # Simulate telemetry collection
        for system_id, connector in connectors.items():
            metrics = await connector.collect_metrics()
            assert len(metrics) > 0
            
            # Publish telemetry event
            event_data = json.dumps({
                "system_id": system_id,
                "metrics": {k: str(v.value) for k, v in metrics.items()},
                "timestamp": datetime.now().isoformat()
            }).encode()
            
            await message_broker.publish(f"telemetry.{system_id}", event_data)
        
        # Verify events were published
        assert len(message_broker.published_messages) == len(connectors)
        
        # Cleanup
        for connector in connectors.values():
            await connector.disconnect()
        await message_broker.disconnect()
    
    @pytest.mark.integration
    def test_failure_handling_integration(self, failure_scenario):
        """Test integration of failure handling across components."""
        # Test that failure scenarios are properly configured
        failing_connector = failure_scenario["connectors"]["failing_connector"]
        data_store = failure_scenario["data_store"]
        
        # Verify failure configurations
        assert failing_connector.should_fail_connection is True
        assert data_store.should_fail_operations is True
        
        # Test that failures propagate correctly
        asyncio.run(self._test_failure_propagation(failing_connector, data_store))
    
    async def _test_failure_propagation(self, connector, data_store):
        """Helper to test failure propagation."""
        # Test connector failure
        with pytest.raises(ConnectionError):
            await connector.connect()
        
        # Test data store failure
        with pytest.raises(RuntimeError):
            await data_store.store("key", "value")