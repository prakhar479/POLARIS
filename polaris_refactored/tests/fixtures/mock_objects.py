"""
Mock objects and test doubles for POLARIS unit testing.

This module provides comprehensive mock implementations for all major POLARIS components
to enable isolated unit testing with dependency injection.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field
from datetime import timedelta
import json

from src.domain.models import (
    SystemState, AdaptationAction, ExecutionResult,
    MetricValue, HealthStatus, ExecutionStatus
)
from src.domain.interfaces import ManagedSystemConnector
from src.infrastructure.message_bus import MessageBroker


class MockManagedSystemConnector:
    """Mock implementation of ManagedSystemConnector for testing."""
    
    def __init__(self, system_id: str = "test_system"):
        self.system_id = system_id
        self.connected = False
        self.collected_metrics: List[Dict[str, MetricValue]] = []
        self.executed_actions: List[AdaptationAction] = []
        self.connection_attempts = 0
        self.should_fail_connection = False
        self.should_fail_metrics = False
        self.should_fail_execution = False
        
    async def connect(self) -> None:
        self.connection_attempts += 1
        if self.should_fail_connection:
            raise ConnectionError(f"Failed to connect to {self.system_id}")
        self.connected = True
        
    async def disconnect(self) -> None:
        self.connected = False
        
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        if self.should_fail_metrics:
            raise RuntimeError("Failed to collect metrics")
        
        # If metrics have been overridden for testing, use those
        if self.collected_metrics:
            # Return the most recently set metrics
            return self.collected_metrics[-1]
        
        # Default metrics
        metrics = {
            "cpu_usage": MetricValue(name="cpu_usage", value=50.0, unit="percent", timestamp=datetime.now()),
            "memory_usage": MetricValue(name="memory_usage", value=1024.0, unit="MB", timestamp=datetime.now()),
            "response_time": MetricValue(name="response_time", value=100.0, unit="ms", timestamp=datetime.now())
        }
        self.collected_metrics.append(metrics)
        return metrics
        
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        if self.should_fail_execution:
            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={"error": "Execution failed"},
                execution_time_ms=50
            )
        else:
            self.executed_actions.append(action)
            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.SUCCESS,
                result_data={"message": "Action executed successfully"},
                execution_time_ms=100
            )
        
        # Add compatibility properties for contract tests
        if not hasattr(result, 'timestamp'):
            object.__setattr__(result, 'timestamp', result.completed_at)
        if not hasattr(result, 'execution_time'):
            execution_time_seconds = (result.execution_time_ms or 0) / 1000.0
            object.__setattr__(result, 'execution_time', timedelta(seconds=execution_time_seconds))
        
        return result
        
    def get_system_id(self) -> str:
        return self.system_id


class MockMessageBroker:
    """Mock message broker for testing event publishing and subscription."""
    
    def __init__(self):
        self.connected = False
        self.published_messages: List[Dict[str, Any]] = []
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.subscription_counter = 0
        
    async def connect(self) -> None:
        self.connected = True
        
    async def disconnect(self) -> None:
        self.connected = False
        self.subscriptions.clear()
        
    async def publish(self, topic: str, message: bytes) -> None:
        if not self.connected:
            raise RuntimeError("Broker not connected")
            
        self.published_messages.append({
            "topic": topic,
            "message": message,
            "timestamp": datetime.now()
        })
        
        # Deliver to subscribers (support wildcard patterns)
        for pattern, handlers in self.subscriptions.items():
            if self._topic_matches_pattern(topic, pattern):
                for handler in handlers:
                    await asyncio.create_task(self._invoke_handler(handler, message))
                
    def _topic_matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if a topic matches a subscription pattern (supports * wildcard)."""
        if pattern == topic:
            return True
        if '*' in pattern:
            # Simple wildcard matching - replace * with .* for regex
            import re
            regex_pattern = pattern.replace('*', '.*')
            return bool(re.match(f"^{regex_pattern}$", topic))
        return False
    
    async def _invoke_handler(self, handler: Callable, message: bytes) -> None:
        """Safely invoke message handler."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
        except Exception:
            pass  # Ignore handler exceptions in tests
            
    async def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> str:
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(handler)
        
        self.subscription_counter += 1
        return f"sub_{self.subscription_counter}"
        
    async def unsubscribe(self, subscription_id: str) -> None:
        # Simple implementation - remove all handlers for testing
        pass


class MockDataStore:
    """Mock data store for testing data persistence."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.transaction_active = False
        self.should_fail_operations = False
    
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def connect(self) -> None:
        pass
        
    async def disconnect(self) -> None:
        self.data.clear()
        
    async def store(self, key: str, value: Any) -> None:
        if self.should_fail_operations:
            raise RuntimeError("Store operation failed")
        self.data[key] = value
        
    async def retrieve(self, key: str) -> Optional[Any]:
        if self.should_fail_operations:
            raise RuntimeError("Retrieve operation failed")
        return self.data.get(key)
        
    async def delete(self, key: str) -> None:
        if self.should_fail_operations:
            raise RuntimeError("Delete operation failed")
        self.data.pop(key, None)
        
    async def query(self, pattern: str) -> List[Any]:
        if self.should_fail_operations:
            raise RuntimeError("Query operation failed")
        # Simple pattern matching for tests
        return [v for k, v in self.data.items() if pattern in k]
        
    async def begin_transaction(self) -> None:
        self.transaction_active = True
        
    async def commit_transaction(self) -> None:
        self.transaction_active = False
        
    async def rollback_transaction(self) -> None:
        self.transaction_active = False


class MockCacheStrategy:
    """Mock cache strategy for testing caching behavior."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self.should_fail = False
        
    async def get(self, key: str) -> Optional[Any]:
        if self.should_fail:
            raise RuntimeError("Cache get failed")
        self.access_count[key] = self.access_count.get(key, 0) + 1
        return self.cache.get(key)
        
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        if self.should_fail:
            raise RuntimeError("Cache set failed")
        self.cache[key] = value
        
    async def delete(self, key: str) -> None:
        if self.should_fail:
            raise RuntimeError("Cache delete failed")
        self.cache.pop(key, None)
        self.access_count.pop(key, None)
        
    async def clear(self) -> None:
        self.cache.clear()
        self.access_count.clear()


class MockMetricsCollector:
    """Mock metrics collector for testing observability."""
    
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        key = f"{name}_{labels}" if labels else name
        self.counters[key] = self.counters.get(key, 0) + 1
        
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = f"{name}_{labels}" if labels else name
        self.gauges[key] = value
        
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = f"{name}_{labels}" if labels else name
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)


class MockLogger:
    """Mock logger for testing logging behavior."""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log("DEBUG", message, extra)
        
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log("INFO", message, extra)
        
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log("WARNING", message, extra)
        
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log("ERROR", message, extra)
        
    def _log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.now(),
            "extra": extra or {}
        }
        self.logs.append(log_entry)


class MockTracer:
    """Mock tracer for testing distributed tracing."""
    
    def __init__(self):
        self.spans: List[Dict[str, Any]] = []
        self.active_spans: List[str] = []
        
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None) -> str:
        span_id = f"span_{len(self.spans) + 1}"
        span = {
            "span_id": span_id,
            "operation_name": operation_name,
            "parent_span_id": parent_span_id,
            "start_time": datetime.now(),
            "tags": {},
            "logs": []
        }
        self.spans.append(span)
        self.active_spans.append(span_id)
        return span_id
        
    def finish_span(self, span_id: str) -> None:
        if span_id in self.active_spans:
            self.active_spans.remove(span_id)
        for span in self.spans:
            if span["span_id"] == span_id:
                span["end_time"] = datetime.now()
                break
                
    def add_tag(self, span_id: str, key: str, value: Any) -> None:
        for span in self.spans:
            if span["span_id"] == span_id:
                span["tags"][key] = value
                break
                
    def log(self, span_id: str, message: str) -> None:
        for span in self.spans:
            if span["span_id"] == span_id:
                span["logs"].append({
                    "timestamp": datetime.now(),
                    "message": message
                })
                break


class DataBuilder:
    """Builder for creating test data objects with sensible defaults."""
    
    @staticmethod
    def system_state(
        system_id: str = "test_system",
        metrics: Optional[Dict[str, MetricValue]] = None,
        health_status: HealthStatus = HealthStatus.HEALTHY
    ) -> SystemState:
        if metrics is None:
            metrics = {
                "cpu_usage": MetricValue(name="cpu_usage", value=50.0, unit="percent", timestamp=datetime.now()),
                "memory_usage": MetricValue(name="memory_usage", value=1024.0, unit="MB", timestamp=datetime.now())
            }
        
        return SystemState(
            system_id=system_id,
            timestamp=datetime.now(),
            metrics=metrics,
            health_status=health_status
        )
    
    @staticmethod
    def adaptation_action(
        action_id: str = "test_action",
        action_type: str = "scale_up",
        target_system: str = "test_system",
        parameters: Optional[Dict[str, Any]] = None
    ) -> AdaptationAction:
        if parameters is None:
            parameters = {"replicas": 3, "cpu_limit": "500m"}
            
        return AdaptationAction(
            action_id=action_id,
            action_type=action_type,
            target_system=target_system,
            parameters=parameters
        )
    
    @staticmethod
    def execution_result(
        action_id: str = "test_action",
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        result_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        if result_data is None:
            result_data = {"message": "Action executed successfully"}
            
        return ExecutionResult(
            action_id=action_id,
            status=status,
            result_data=result_data
        )
    
    @staticmethod
    def telemetry_event(
        system_id: str = "test_system",
        **kwargs
    ):
        """Create a telemetry event for testing."""
        from src.framework.events import TelemetryEvent
        
        # Create a system state first (required by TelemetryEvent)
        system_state = DataBuilder.system_state(system_id=system_id)
        
        return TelemetryEvent(
            system_state=system_state,
            **kwargs
        )
    
    @staticmethod
    def adaptation_event(
        system_id: str = "test_system",
        reason: str = "test_adaptation_needed",
        severity: str = "normal",
        **kwargs
    ):
        """Create an adaptation event for testing."""
        from src.framework.events import AdaptationEvent
        
        return AdaptationEvent(
            system_id=system_id,
            reason=reason,
            severity=severity,
            **kwargs
        )


class MockComponentFactory:
    """Factory for creating mock components with dependency injection."""
    
    @staticmethod
    def create_mock_connector(system_id: str = "test_system") -> MockManagedSystemConnector:
        return MockManagedSystemConnector(system_id)
    
    @staticmethod
    def create_mock_message_broker() -> MockMessageBroker:
        return MockMessageBroker()
    
    @staticmethod
    def create_mock_data_store() -> MockDataStore:
        return MockDataStore()
    
    @staticmethod
    def create_mock_cache() -> MockCacheStrategy:
        return MockCacheStrategy()
    
    @staticmethod
    def create_mock_metrics_collector() -> MockMetricsCollector:
        return MockMetricsCollector()
    
    @staticmethod
    def create_mock_logger() -> MockLogger:
        return MockLogger()
    
    @staticmethod
    def create_mock_tracer() -> MockTracer:
        return MockTracer()


class TestScenarioBuilder:
    """Builder for creating complex test scenarios with multiple components."""
    
    def __init__(self):
        self.connectors: Dict[str, MockManagedSystemConnector] = {}
        self.message_broker = MockComponentFactory.create_mock_message_broker()
        self.data_store = MockComponentFactory.create_mock_data_store()
        self.cache = MockComponentFactory.create_mock_cache()
        self.metrics_collector = MockComponentFactory.create_mock_metrics_collector()
        self.logger = MockComponentFactory.create_mock_logger()
        self.tracer = MockComponentFactory.create_mock_tracer()
        
    def add_system(self, system_id: str) -> 'TestScenarioBuilder':
        """Add a managed system to the test scenario."""
        self.connectors[system_id] = MockComponentFactory.create_mock_connector(system_id)
        return self
        
    def with_failing_system(self, system_id: str, fail_type: str = "connection") -> 'TestScenarioBuilder':
        """Add a system that will fail in specific ways."""
        connector = MockComponentFactory.create_mock_connector(system_id)
        if fail_type == "connection":
            connector.should_fail_connection = True
        elif fail_type == "metrics":
            connector.should_fail_metrics = True
        elif fail_type == "execution":
            connector.should_fail_execution = True
        self.connectors[system_id] = connector
        return self
        
    def with_failing_infrastructure(self, component: str) -> 'TestScenarioBuilder':
        """Configure infrastructure components to fail."""
        if component == "data_store":
            self.data_store.should_fail_operations = True
        elif component == "cache":
            self.cache.should_fail = True
        return self
        
    def build(self) -> Dict[str, Any]:
        """Build the complete test scenario."""
        return {
            "connectors": self.connectors,
            "message_broker": self.message_broker,
            "data_store": self.data_store,
            "cache": self.cache,
            "metrics_collector": self.metrics_collector,
            "logger": self.logger,
            "tracer": self.tracer
        }