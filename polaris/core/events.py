"""Event system for Polaris."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.core.models import AdaptationAction, ExecutionResult, SystemState


@dataclass
class TelemetryEvent:
    """Telemetry data from a managed system."""

    system_id: str
    state: SystemState
    timestamp: datetime


@dataclass
class AdaptationEvent:
    """Adaptation action execution event."""

    action: AdaptationAction
    result: ExecutionResult
    timestamp: datetime


class EventBus:
    """Simple async event bus for component communication."""

    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        logger: Optional[Logger] = None,
    ):
        """Initialize event bus with optional metrics and logging."""
        self._handlers: Dict[Type, List[Callable]] = defaultdict(list)
        self._running = False
        self._metrics = metrics
        self._logger = logger

    async def start(self) -> None:
        """Start the event bus."""
        self._running = True
        if self._metrics:
            self._metrics.increment("polaris.event_bus.started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False
        if self._metrics:
            self._metrics.increment("polaris.event_bus.stopped")

    async def publish(self, event: Any) -> None:
        """
        Publish an event to all subscribed handlers.

        Args:
            event: Event to publish
        """
        if not self._running:
            return

        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        if self._metrics:
            self._metrics.increment(
                "polaris.event_bus.events_published", tags={"event_type": event_type.__name__}
            )
            self._metrics.gauge(
                "polaris.event_bus.handler_count",
                len(handlers),
                tags={"event_type": event_type.__name__},
            )

        # Call all handlers concurrently
        tasks: List[asyncio.Future] = []
        handler_names: List[str] = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                # Wrap sync handlers
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(None, handler, event))

            # Best-effort handler identifier for logging
            name = getattr(handler, "__name__", None) or repr(handler)
            handler_names.append(str(name))

        if tasks:
            start_time = datetime.now(timezone.utc)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            if self._metrics:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                self._metrics.histogram(
                    "polaris.event_bus.handler_duration_seconds",
                    duration,
                    tags={"event_type": event_type.__name__},
                )

            # Log any exceptions from handlers
            error_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_count += 1
                    if self._logger:
                        handler_name = handler_names[i] if i < len(handler_names) else str(i)
                        self._logger.error(
                            "EventBus handler error",
                            event_type=event_type.__name__,
                            handler_index=i,
                            handler=handler_name,
                            error=str(result),
                        )

            if self._metrics and error_count > 0:
                self._metrics.increment(
                    "polaris.event_bus.handler_errors",
                    value=error_count,
                    tags={"event_type": event_type.__name__},
                )

    def subscribe(self, event_type: Type, handler: Callable) -> str:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async or sync callable to handle events

        Returns:
            Subscription ID
        """
        self._handlers[event_type].append(handler)

        if self._metrics:
            self._metrics.increment(
                "polaris.event_bus.subscriptions_added", tags={"event_type": event_type.__name__}
            )
            self._metrics.gauge(
                "polaris.event_bus.total_handlers",
                sum(len(handlers) for handlers in self._handlers.values()),
            )

        return f"{event_type.__name__}: {id(handler)}"

    def unsubscribe(self, event_type: Type, handler: Callable) -> None:
        """Unsubscribe a handler from event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                if self._metrics:
                    self._metrics.increment(
                        "polaris.event_bus.subscriptions_removed",
                        tags={"event_type": event_type.__name__},
                    )
                    self._metrics.gauge(
                        "polaris.event_bus.total_handlers",
                        sum(len(handlers) for handlers in self._handlers.values()),
                    )
            except ValueError:
                pass
