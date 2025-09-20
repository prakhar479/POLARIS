"""
Pytest fixtures for POLARIS unit testing framework.

This module provides comprehensive fixtures for dependency injection and test isolation.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from .mock_objects import (
    MockManagedSystemConnector, MockMessageBroker, MockDataStore,
    MockCacheStrategy, MockMetricsCollector, MockLogger, MockTracer,
    DataBuilder, MockComponentFactory, TestScenarioBuilder
)

from src.domain.models import SystemState, AdaptationAction
from src.infrastructure.di import DIContainer


@pytest.fixture
def mock_connector():
    """Provides a mock managed system connector."""
    return MockComponentFactory.create_mock_connector()


@pytest.fixture
def mock_message_broker():
    """Provides a mock message broker."""
    return MockComponentFactory.create_mock_message_broker()


@pytest.fixture
def mock_data_store():
    """Provides a mock data store."""
    return MockComponentFactory.create_mock_data_store()


@pytest.fixture
def mock_cache():
    """Provides a mock cache strategy."""
    return MockComponentFactory.create_mock_cache()


@pytest.fixture
def mock_metrics_collector():
    """Provides a mock metrics collector."""
    return MockComponentFactory.create_mock_metrics_collector()


@pytest.fixture
def mock_logger():
    """Provides a mock logger."""
    return MockComponentFactory.create_mock_logger()


@pytest.fixture
def mock_tracer():
    """Provides a mock tracer."""
    return MockComponentFactory.create_mock_tracer()


@pytest.fixture
def test_data_builder():
    """Provides a test data builder for creating test objects."""
    return DataBuilder()


@pytest.fixture
def test_scenario_builder():
    """Provides a test scenario builder for complex test setups."""
    return TestScenarioBuilder()


@pytest.fixture
def sample_system_state(test_data_builder):
    """Provides a sample system state for testing."""
    return test_data_builder.system_state()


@pytest.fixture
def sample_adaptation_action(test_data_builder):
    """Provides a sample adaptation action for testing."""
    return test_data_builder.adaptation_action()


@pytest.fixture
def sample_telemetry_event(test_data_builder):
    """Provides a sample telemetry event for testing."""
    return test_data_builder.telemetry_event()


@pytest.fixture
def sample_adaptation_event(test_data_builder):
    """Provides a sample adaptation event for testing."""
    return test_data_builder.adaptation_event()


@pytest.fixture
def multi_system_scenario(test_scenario_builder):
    """Provides a multi-system test scenario."""
    return (test_scenario_builder
            .add_system("web_server")
            .add_system("database")
            .add_system("cache_server")
            .build())


@pytest.fixture
def failure_scenario(test_scenario_builder):
    """Provides a test scenario with various failure conditions."""
    return (test_scenario_builder
            .add_system("healthy_system")
            .with_failing_system("failing_connector", "connection")
            .with_failing_system("failing_metrics", "metrics")
            .with_failing_system("failing_execution", "execution")
            .with_failing_infrastructure("data_store")
            .build())


@pytest.fixture
def di_container():
    """Provides a dependency injection container for testing."""
    container = DIContainer()
    
    # Register mock implementations
    container.register("message_broker", MockComponentFactory.create_mock_message_broker())
    container.register("data_store", MockComponentFactory.create_mock_data_store())
    container.register("cache_strategy", MockComponentFactory.create_mock_cache())
    container.register("metrics_collector", MockComponentFactory.create_mock_metrics_collector())
    container.register("logger", MockComponentFactory.create_mock_logger())
    container.register("tracer", MockComponentFactory.create_mock_tracer())
    
    return container


@pytest.fixture
def mock_di_container():
    """Provides a fully mocked DI container."""
    container = MagicMock()
    container.get = MagicMock()
    container.register = MagicMock()
    return container


@pytest.fixture(autouse=True)
def reset_singletons():
    """Automatically reset singleton instances between tests."""
    # This fixture runs before each test to ensure clean state
    yield
    # Cleanup code can go here if needed


@pytest.fixture
def event_loop():
    """Provides a fresh event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def connected_message_broker(mock_message_broker):
    """Provides a connected message broker."""
    await mock_message_broker.connect()
    yield mock_message_broker
    await mock_message_broker.disconnect()


@pytest.fixture
async def connected_data_store(mock_data_store):
    """Provides a connected data store."""
    await mock_data_store.connect()
    yield mock_data_store
    await mock_data_store.disconnect()


@pytest.fixture
def time_frozen():
    """Freezes time for deterministic testing."""
    frozen_time = datetime(2024, 1, 1, 12, 0, 0)
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = frozen_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield frozen_time


@pytest.fixture
def performance_metrics():
    """Provides performance metrics tracking for tests."""
    metrics = {
        "start_time": None,
        "end_time": None,
        "duration": None,
        "memory_usage": None
    }
    
    def start_measurement():
        metrics["start_time"] = datetime.now()
    
    def end_measurement():
        metrics["end_time"] = datetime.now()
        if metrics["start_time"]:
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
    
    metrics["start"] = start_measurement
    metrics["end"] = end_measurement
    
    return metrics


@pytest.fixture
def async_test_timeout():
    """Provides a timeout for async tests to prevent hanging."""
    return 5.0  # 5 seconds timeout


@pytest.fixture
def test_config():
    """Provides test configuration settings."""
    return {
        "test_timeout": 5.0,
        "max_retries": 3,
        "batch_size": 10,
        "cache_ttl": timedelta(minutes=5),
        "metrics_interval": timedelta(seconds=1)
    }


class AsyncContextManager:
    """Helper for testing async context managers."""
    
    def __init__(self, mock_obj):
        self.mock_obj = mock_obj
        self.entered = False
        self.exited = False
    
    async def __aenter__(self):
        self.entered = True
        return self.mock_obj
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False


@pytest.fixture
def async_context_manager():
    """Provides an async context manager for testing."""
    return AsyncContextManager


@pytest.fixture
def mock_sleep():
    """Mocks asyncio.sleep to speed up tests."""
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def capture_logs(mock_logger):
    """Captures logs for assertion in tests."""
    def get_logs_by_level(level: str) -> List[Dict[str, Any]]:
        return [log for log in mock_logger.logs if log["level"] == level]
    
    def get_logs_containing(message: str) -> List[Dict[str, Any]]:
        return [log for log in mock_logger.logs if message in log["message"]]
    
    mock_logger.get_logs_by_level = get_logs_by_level
    mock_logger.get_logs_containing = get_logs_containing
    
    return mock_logger


@pytest.fixture
def capture_metrics(mock_metrics_collector):
    """Captures metrics for assertion in tests."""
    def get_counter_value(name: str) -> int:
        return mock_metrics_collector.counters.get(name, 0)
    
    def get_gauge_value(name: str) -> float:
        return mock_metrics_collector.gauges.get(name, 0.0)
    
    def get_histogram_values(name: str) -> List[float]:
        return mock_metrics_collector.histograms.get(name, [])
    
    mock_metrics_collector.get_counter_value = get_counter_value
    mock_metrics_collector.get_gauge_value = get_gauge_value
    mock_metrics_collector.get_histogram_values = get_histogram_values
    
    return mock_metrics_collector


@pytest.fixture
def capture_traces(mock_tracer):
    """Captures traces for assertion in tests."""
    def get_spans_by_operation(operation_name: str) -> List[Dict[str, Any]]:
        return [span for span in mock_tracer.spans if span["operation_name"] == operation_name]
    
    def get_active_spans() -> List[str]:
        return mock_tracer.active_spans.copy()
    
    mock_tracer.get_spans_by_operation = get_spans_by_operation
    mock_tracer.get_active_spans = get_active_spans
    
    return mock_tracer


# Parametrized fixtures for testing different scenarios
@pytest.fixture(params=["healthy", "degraded", "unhealthy"])
def system_health_status(request):
    """Parametrized fixture for different system health statuses."""
    from src.domain.models import HealthStatus
    status_map = {
        "healthy": HealthStatus.HEALTHY,
        "degraded": HealthStatus.DEGRADED,
        "unhealthy": HealthStatus.UNHEALTHY
    }
    return status_map[request.param]


@pytest.fixture(params=["success", "failed", "timeout"])
def execution_status(request):
    """Parametrized fixture for different execution statuses."""
    from src.domain.models import ExecutionStatus
    status_map = {
        "success": ExecutionStatus.SUCCESS,
        "failed": ExecutionStatus.FAILED,
        "timeout": ExecutionStatus.TIMEOUT
    }
    return status_map[request.param]


@pytest.fixture(params=[1, 5, 10, 50])
def batch_sizes(request):
    """Parametrized fixture for different batch sizes."""
    return request.param


@pytest.fixture(params=[0.1, 0.5, 1.0, 2.0])
def timeout_values(request):
    """Parametrized fixture for different timeout values."""
    return request.param