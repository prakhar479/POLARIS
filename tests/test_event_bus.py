"""Tests for event bus system."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from polaris.core.events import AdaptationEvent, EventBus, TelemetryEvent
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)


class TestEventBus:
    """Test EventBus functionality."""

    @pytest.fixture
    def event_bus(self, mock_metrics):
        """Create event bus with mock metrics."""
        return EventBus(metrics=mock_metrics)

    @pytest.mark.asyncio
    async def test_event_bus_lifecycle(self, event_bus):
        """Test event bus start/stop lifecycle."""
        assert not event_bus._running

        await event_bus.start()
        assert event_bus._running

        await event_bus.stop()
        assert not event_bus._running

    @pytest.mark.asyncio
    async def test_publish_without_handlers(self, event_bus):
        """Test publishing event with no handlers."""
        await event_bus.start()

        event = TelemetryEvent(
            system_id="test-system",
            state=SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY,
            ),
            timestamp=datetime.now(timezone.utc),
        )

        # Should not raise exception
        await event_bus.publish(event)

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test subscribing to events and publishing."""
        await event_bus.start()

        # Create mock handler
        handler = AsyncMock()

        # Subscribe to TelemetryEvent
        subscription_id = event_bus.subscribe(TelemetryEvent, handler)
        assert subscription_id.startswith("TelemetryEvent:")

        # Create and publish event
        event = TelemetryEvent(
            system_id="test-system",
            state=SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY,
            ),
            timestamp=datetime.now(timezone.utc),
        )

        await event_bus.publish(event)

        # Verify handler was called
        handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event type."""
        await event_bus.start()

        handler1 = AsyncMock()
        handler2 = AsyncMock()

        event_bus.subscribe(TelemetryEvent, handler1)
        event_bus.subscribe(TelemetryEvent, handler2)

        event = TelemetryEvent(
            system_id="test-system",
            state=SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY,
            ),
            timestamp=datetime.now(timezone.utc),
        )

        await event_bus.publish(event)

        # Both handlers should be called
        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_sync_handler(self, event_bus):
        """Test synchronous handler execution."""
        await event_bus.start()

        # Create sync handler
        handler_calls = []

        def sync_handler(event):
            handler_calls.append(event)

        event_bus.subscribe(TelemetryEvent, sync_handler)

        event = TelemetryEvent(
            system_id="test-system",
            state=SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY,
            ),
            timestamp=datetime.now(timezone.utc),
        )

        await event_bus.publish(event)

        # Verify sync handler was called
        assert len(handler_calls) == 1
        assert handler_calls[0] == event

    @pytest.mark.asyncio
    async def test_handler_exception_handling(self, event_bus):
        """Test that handler exceptions don't break event bus."""
        await event_bus.start()

        # Create handlers - one that fails, one that succeeds
        failing_handler = AsyncMock(side_effect=Exception("Handler failed"))
        success_handler = AsyncMock()

        event_bus.subscribe(TelemetryEvent, failing_handler)
        event_bus.subscribe(TelemetryEvent, success_handler)

        event = TelemetryEvent(
            system_id="test-system",
            state=SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY,
            ),
            timestamp=datetime.now(timezone.utc),
        )

        # Should not raise exception despite failing handler
        await event_bus.publish(event)

        # Both handlers should have been called
        failing_handler.assert_called_once_with(event)
        success_handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_handler_exception_logging(self, mock_metrics, mock_logger):
        """Test that handler exceptions are logged when a logger is provided."""
        bus = EventBus(metrics=mock_metrics, logger=mock_logger)
        await bus.start()

        async def failing_handler(event):
            raise RuntimeError("boom")

        bus.subscribe(TelemetryEvent, failing_handler)

        event = TelemetryEvent(
            system_id="test-system",
            state=SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY,
            ),
            timestamp=datetime.now(timezone.utc),
        )

        # Should not raise despite handler failure
        await bus.publish(event)

        # One error log should have been recorded with expected context keys
        error_logs = [
            log
            for log in mock_logger.logs
            if log[0] == "error" and log[1] == "EventBus handler error"
        ]
        assert len(error_logs) == 1
        _level, _msg, ctx = error_logs[0]
        assert ctx.get("event_type") == "TelemetryEvent"
        assert ctx.get("handler_index") == 0
        assert "boom" in ctx.get("error", "")

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        await event_bus.start()

        handler = AsyncMock()
        event_bus.subscribe(TelemetryEvent, handler)

        # Unsubscribe
        event_bus.unsubscribe(TelemetryEvent, handler)

        # Publish event
        event = TelemetryEvent(
            system_id="test-system",
            state=SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY,
            ),
            timestamp=datetime.now(timezone.utc),
        )

        await event_bus.publish(event)

        # Handler should not be called
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_when_stopped(self, event_bus):
        """Test publishing when event bus is stopped."""
        handler = AsyncMock()
        event_bus.subscribe(TelemetryEvent, handler)

        event = TelemetryEvent(
            system_id="test-system",
            state=SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.HEALTHY,
            ),
            timestamp=datetime.now(timezone.utc),
        )

        # Publish when stopped - should not call handlers
        await event_bus.publish(event)
        handler.assert_not_called()

    def test_metrics_tracking(self, mock_metrics):
        """Test that metrics are tracked correctly."""
        event_bus = EventBus(metrics=mock_metrics)

        # Check start/stop metrics
        asyncio.run(event_bus.start())
        asyncio.run(event_bus.stop())

        # Verify metrics were recorded
        metric_calls = [call for call in mock_metrics.metrics if call[0] == "increment"]
        metric_names = [call[1] for call in metric_calls]

        assert "polaris.event_bus.started" in metric_names
        assert "polaris.event_bus.stopped" in metric_names


class TestEventTypes:
    """Test event type classes."""

    def test_telemetry_event(self):
        """Test TelemetryEvent creation."""
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu": MetricValue("cpu", 50.0)},
            health_status=HealthStatus.HEALTHY,
        )

        event = TelemetryEvent(
            system_id="test-system", state=state, timestamp=datetime.now(timezone.utc)
        )

        assert event.system_id == "test-system"
        assert event.state == state
        assert isinstance(event.timestamp, datetime)

    def test_adaptation_event(self):
        """Test AdaptationEvent creation."""
        action = AdaptationAction(
            action_id="test-action",
            action_type="scale_up",
            target_system="test-system",
            parameters={},
        )

        result = ExecutionResult(
            action_id="test-action", status=ExecutionStatus.SUCCESS, result_data={}
        )

        event = AdaptationEvent(action=action, result=result, timestamp=datetime.now(timezone.utc))

        assert event.action == action
        assert event.result == result
        assert isinstance(event.timestamp, datetime)
