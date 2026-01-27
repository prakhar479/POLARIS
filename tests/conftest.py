"""Shared test fixtures and configuration."""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock, Mock

from polaris.core.models import (
    SystemState, AdaptationAction, ExecutionResult, MetricValue, 
    HealthStatus, ExecutionStatus
)
from polaris.abstractions.connector import Connector
from polaris.abstractions.strategy import AdaptationStrategy, AdaptationContext
from polaris.abstractions.observability import Logger, MetricsCollector


class MockConnector(Connector):
    """Mock connector for testing."""
    
    def __init__(self, system_id: str = "test-system"):
        self.system_id = system_id
        self.connected = False
        self.telemetry_data = None
        self.supported_actions = []
        
    async def connect(self) -> bool:
        self.connected = True
        return True
        
    async def disconnect(self) -> bool:
        self.connected = False
        return True
        
    async def get_system_id(self) -> str:
        return self.system_id
        
    async def collect_telemetry(self) -> SystemState:
        if self.telemetry_data:
            return self.telemetry_data
        return SystemState(
            system_id=self.system_id,
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 50.0, "percent"),
                "memory_usage": MetricValue("memory_usage", 60.0, "percent")
            },
            health_status=HealthStatus.HEALTHY
        )
        
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        return ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.SUCCESS,
            result_data={"message": "Action executed successfully"}
        )
        
    async def validate_action(self, action: AdaptationAction) -> bool:
        return action.action_type in [a.action_type for a in self.supported_actions]


class MockStrategy(AdaptationStrategy):
    """Mock strategy for testing."""
    
    def __init__(self):
        self.assess_calls = []
        self.action_to_return = None
        self.parameters = {"threshold": 80.0}
        
    async def assess(self, state: SystemState, context: AdaptationContext) -> Optional[AdaptationAction]:
        self.assess_calls.append((state, context))
        return self.action_to_return
        
    def get_tunable_parameters(self) -> Dict[str, Any]:
        return {"threshold": {"current_value": self.parameters["threshold"], "type": float}}
        
    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        if parameter_path in self.parameters:
            self.parameters[parameter_path] = new_value
            return True
        return False


class MockLogger(Logger):
    """Mock logger for testing."""
    
    def __init__(self):
        self.logs = []
        
    def info(self, message: str, **context) -> None:
        self.logs.append(("info", message, context))
        
    def error(self, message: str, **context) -> None:
        self.logs.append(("error", message, context))
        
    def warning(self, message: str, **context) -> None:
        self.logs.append(("warning", message, context))
        
    def debug(self, message: str, **context) -> None:
        self.logs.append(("debug", message, context))


class MockMetricsCollector(MetricsCollector):
    """Mock metrics collector for testing."""
    
    def __init__(self):
        self.metrics = []
        
    def increment(self, metric: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        self.metrics.append(("increment", metric, value, tags))
        
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        self.metrics.append(("gauge", metric, value, tags))
        
    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        self.metrics.append(("histogram", metric, value, tags))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {"total_metrics": len(self.metrics)}


@pytest.fixture
def mock_connector():
    """Provide a mock connector."""
    return MockConnector()


@pytest.fixture
def mock_strategy():
    """Provide a mock strategy."""
    return MockStrategy()


@pytest.fixture
def mock_logger():
    """Provide a mock logger."""
    return MockLogger()


@pytest.fixture
def mock_metrics():
    """Provide a mock metrics collector."""
    return MockMetricsCollector()


@pytest.fixture
def sample_system_state():
    """Provide a sample system state."""
    return SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={
            "cpu_usage": MetricValue("cpu_usage", 75.0, "percent"),
            "memory_usage": MetricValue("memory_usage", 65.0, "percent")
        },
        health_status=HealthStatus.HEALTHY
    )


@pytest.fixture
def sample_adaptation_action():
    """Provide a sample adaptation action."""
    return AdaptationAction(
        action_id="test-action-1",
        action_type="scale_up",
        target_system="test-system",
        parameters={"instances": 2}
    )