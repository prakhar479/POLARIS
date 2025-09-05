"""
Tests for the POLARIS Event System

Comprehensive tests for event publishing, subscription, correlation, and processing.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from typing import List

from ..src.framework.events import (
    PolarisEvent, TelemetryEvent, AdaptationEvent, ExecutionResultEvent,
    PolarisEventBus, EventSubscription, EventMetadata, EventProcessingError
)
from ..src.domain.models import (
    SystemState, AdaptationAction, ExecutionResult, MetricValue, HealthStatus, ExecutionStatus
)
from ..src.domain.interfaces import EventHandler


class MockEventHandler(EventHandler):
    """Mock event handler for testing."""
    
    def __init__(self, handler_id: str = "mock_handler"):
        self.handler_id = handler_id
        self.handled_events: List[PolarisEvent] = []
        self.should_handle = True
        self.should_fail = False
    
    async def handle(self, event: PolarisEvent) -> None:
        """Handle an event."""
        if self.should_fail:
            raise Exception(f"Handler {self.handler_id} failed")
        
        self.handled_events.append(event)
    
    def can_handle(self, event: PolarisEvent) -> bool:
        """Check if this handler can handle the event."""
        return self.should_handle


class TestPolarisEvent:
    """Test the base PolarisEvent class."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.system_id == "test_system"
        assert event.metadata is not None
    
    def test_event_correlation(self):
        """Test event correlation functionality."""
        event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        correlation_id = "test-correlation-123"
        event.add_correlation(correlation_id)
        
        assert event.correlation_id == correlation_id
    
    def test_event_processing_tracking(self):
        """Test event processing tracking."""
        event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        handler_id = "test_handler"
        assert not event.was_processed_by(handler_id)
        
        event.mark_processed_by(handler_id)
        assert event.was_processed_by(handler_id)
    
    def test_event_serialization(self):
        """Test event serialization to dictionary."""
        system_state = SystemState(
            system_id="test_system",
            health_status=HealthStatus.HEALTHY,
            metrics=[],
            timestamp=datetime.utcnow()
        )
        
        event = TelemetryEvent(system_state=system_state)
        event_dict = event.to_dict()
        
        assert "event_id" in event_dict
        assert "event_type" in event_dict
        assert "timestamp" in event_dict
        assert "system_id" in event_dict
        assert event_dict["event_type"] == "TelemetryEvent"
        assert event_dict["system_id"] == "test_system"


class TestAdaptationEvent:
    """Test the AdaptationEvent class."""
    
    def test_adaptation_event_creation(self):
        """Test adaptation event creation."""
        event = AdaptationEvent(
            system_id="test_system",
            reason="High CPU usage detected",
            severity="high"
        )
        
        assert event.system_id == "test_system"
        assert event.reason == "High CPU usage detected"
        assert event.severity == "high"
        assert event.metadata.priority == 3  # high severity
    
    def test_adaptation_event_with_actions(self):
        """Test adaptation event with suggested actions."""
        action = AdaptationAction(
            action_id="scale_up_001",
            action_type="SCALE_UP",
            target_system="test_system",
            parameters={"instances": 2}
        )
        
        event = AdaptationEvent(
            system_id="test_system",
            reason="High load detected",
            suggested_actions=[action]
        )
        
        assert len(event.suggested_actions) == 1
        assert event.suggested_actions[0].action_id == "scale_up_001"


class TestExecutionResultEvent:
    """Test the ExecutionResultEvent class."""
    
    def test_execution_result_event_creation(self):
        """Test execution result event creation."""
        execution_result = ExecutionResult(
            action_id="scale_up_001",
            status=ExecutionStatus.SUCCESS,
            result_data={"instances_added": 2}
        )
        
        event = ExecutionResultEvent(execution_result=execution_result)
        
        assert event.action_id == "scale_up_001"
        assert event.execution_result.status == ExecutionStatus.SUCCESS


class TestEventSubscription:
    """Test the EventSubscription class."""
    
    def test_subscription_matching(self):
        """Test subscription event matching."""
        handler = MockEventHandler()
        subscription = EventSubscription(
            subscription_id="test_sub",
            event_type=TelemetryEvent,
            handler=handler
        )
        
        # Create matching event
        telemetry_event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        # Create non-matching event
        adaptation_event = AdaptationEvent(
            system_id="test_system",
            reason="Test reason"
        )
        
        assert subscription.matches(telemetry_event)
        assert not subscription.matches(adaptation_event)
    
    def test_subscription_with_filter(self):
        """Test subscription with filter function."""
        handler = MockEventHandler()
        filter_func = lambda event: event.system_id == "target_system"
        
        subscription = EventSubscription(
            subscription_id="test_sub",
            event_type=TelemetryEvent,
            handler=handler,
            filter_func=filter_func
        )
        
        # Create matching event
        matching_event = TelemetryEvent(
            system_state=SystemState(
                system_id="target_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        # Create non-matching event
        non_matching_event = TelemetryEvent(
            system_state=SystemState(
                system_id="other_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        assert subscription.matches(matching_event)
        assert not subscription.matches(non_matching_event)


class TestPolarisEventBus:
    """Test the PolarisEventBus class."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create an event bus for testing."""
        bus = PolarisEventBus(worker_count=2)
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_event_bus_lifecycle(self):
        """Test event bus start and stop."""
        bus = PolarisEventBus()
        
        assert not bus._running
        
        await bus.start()
        assert bus._running
        assert len(bus._workers) > 0
        
        await bus.stop()
        assert not bus._running
        assert len(bus._workers) == 0
    
    @pytest.mark.asyncio
    async def test_event_publishing_and_subscription(self, event_bus):
        """Test basic event publishing and subscription."""
        handler = MockEventHandler()
        
        # Subscribe to telemetry events
        subscription_id = event_bus.subscribe(TelemetryEvent, handler)
        
        # Create and publish event
        event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        await event_bus.publish_telemetry(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify handler was called
        assert len(handler.handled_events) == 1
        assert handler.handled_events[0].event_id == event.event_id
        
        # Cleanup
        await event_bus.unsubscribe(subscription_id)
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers for the same event type."""
        handler1 = MockEventHandler("handler1")
        handler2 = MockEventHandler("handler2")
        
        # Subscribe both handlers
        sub1 = event_bus.subscribe(TelemetryEvent, handler1)
        sub2 = event_bus.subscribe(TelemetryEvent, handler2)
        
        # Publish event
        event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        await event_bus.publish_telemetry(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Both handlers should have received the event
        assert len(handler1.handled_events) == 1
        assert len(handler2.handled_events) == 1
        
        # Cleanup
        await event_bus.unsubscribe(sub1)
        await event_bus.unsubscribe(sub2)
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, event_bus):
        """Test event filtering in subscriptions."""
        handler = MockEventHandler()
        
        # Subscribe with filter for specific system
        subscription_id = await event_bus.subscribe_to_telemetry(
            handler.handle,
            system_id_filter="target_system"
        )
        
        # Publish matching event
        matching_event = TelemetryEvent(
            system_state=SystemState(
                system_id="target_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        # Publish non-matching event
        non_matching_event = TelemetryEvent(
            system_state=SystemState(
                system_id="other_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        await event_bus.publish_telemetry(matching_event)
        await event_bus.publish_telemetry(non_matching_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Only matching event should be handled
        assert len(handler.handled_events) == 1
        assert handler.handled_events[0].system_id == "target_system"
        
        # Cleanup
        await event_bus.unsubscribe(subscription_id)
    
    @pytest.mark.asyncio
    async def test_event_correlation_tracking(self, event_bus):
        """Test event correlation tracking."""
        correlation_id = "test-correlation-123"
        
        # Create events with same correlation ID
        event1 = TelemetryEvent(
            system_state=SystemState(
                system_id="system1",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            ),
            correlation_id=correlation_id
        )
        
        event2 = AdaptationEvent(
            system_id="system1",
            reason="Test reason",
            correlation_id=correlation_id
        )
        
        await event_bus.publish_telemetry(event1)
        await event_bus.publish_adaptation_needed(event2)
        
        # Wait for processing (longer wait to ensure async processing completes)
        await asyncio.sleep(0.2)
        
        # Check correlation tracking
        correlated_events = event_bus.get_correlated_events(correlation_id)
        assert len(correlated_events) == 2
        assert event1.event_id in correlated_events
        assert event2.event_id in correlated_events
    
    @pytest.mark.asyncio
    async def test_event_history(self, event_bus):
        """Test event history tracking."""
        # Publish some events
        for i in range(5):
            event = TelemetryEvent(
                system_state=SystemState(
                    system_id=f"system{i}",
                    health_status=HealthStatus.HEALTHY,
                    metrics=[],
                    timestamp=datetime.utcnow()
                )
            )
            await event_bus.publish_telemetry(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check history
        history = event_bus.get_event_history()
        assert len(history) == 5
        
        # Check limited history
        limited_history = event_bus.get_event_history(limit=3)
        assert len(limited_history) == 3
    
    @pytest.mark.asyncio
    async def test_processing_stats(self, event_bus):
        """Test processing statistics."""
        handler = MockEventHandler()
        subscription_id = event_bus.subscribe(TelemetryEvent, handler)
        
        # Publish some events
        for i in range(3):
            event = TelemetryEvent(
                system_state=SystemState(
                    system_id=f"system{i}",
                    health_status=HealthStatus.HEALTHY,
                    metrics=[],
                    timestamp=datetime.utcnow()
                )
            )
            await event_bus.publish_telemetry(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check stats
        stats = event_bus.get_processing_stats()
        assert stats["events_published"] == 3
        assert stats["events_processed"] >= 3
        assert stats["active_subscriptions"] == 1
        assert stats["is_running"] is True
        
        # Cleanup
        await event_bus.unsubscribe(subscription_id)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retry(self, event_bus):
        """Test error handling and retry mechanism."""
        handler = MockEventHandler()
        handler.should_fail = True  # Make handler fail
        
        subscription_id = event_bus.subscribe(TelemetryEvent, handler)
        
        # Create event with low max retries for testing
        event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        event.metadata.max_retries = 1
        
        await event_bus.publish_telemetry(event)
        
        # Wait for processing and retries
        await asyncio.sleep(0.2)
        
        # Handler should have been called multiple times due to retries
        assert len(handler.handled_events) == 0  # No successful handling
        
        # Cleanup
        await event_bus.unsubscribe(subscription_id)
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        handler = MockEventHandler()
        
        # Subscribe and then unsubscribe
        subscription_id = event_bus.subscribe(TelemetryEvent, handler)
        assert await event_bus.unsubscribe(subscription_id)
        
        # Try to unsubscribe again (should return False)
        assert not await event_bus.unsubscribe(subscription_id)
        
        # Publish event - handler should not receive it
        event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        await event_bus.publish_telemetry(event)
        await asyncio.sleep(0.1)
        
        assert len(handler.handled_events) == 0
    
    @pytest.mark.asyncio
    async def test_callable_handler(self, event_bus):
        """Test using callable functions as handlers."""
        handled_events = []
        
        async def async_handler(event):
            handled_events.append(event)
        
        def sync_handler(event):
            handled_events.append(event)
        
        # Test async handler
        sub1 = event_bus.subscribe(TelemetryEvent, async_handler)
        
        # Test sync handler
        sub2 = event_bus.subscribe(AdaptationEvent, sync_handler)
        
        # Publish events
        telemetry_event = TelemetryEvent(
            system_state=SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics=[],
                timestamp=datetime.utcnow()
            )
        )
        
        adaptation_event = AdaptationEvent(
            system_id="test_system",
            reason="Test reason"
        )
        
        await event_bus.publish_telemetry(telemetry_event)
        await event_bus.publish_adaptation_needed(adaptation_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Both handlers should have been called
        assert len(handled_events) == 2
        
        # Cleanup
        await event_bus.unsubscribe(sub1)
        await event_bus.unsubscribe(sub2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])