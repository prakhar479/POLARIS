"""
Telemetry Subscriber

Provides a simple EventHandler wired to PolarisEventBus to persist TelemetryEvent
into the PolarisKnowledgeBase automatically with comprehensive observability.
"""

from typing import Optional

from ..framework.events import TelemetryEvent, PolarisEventBus
from ..domain.interfaces import EventHandler
from ..infrastructure.observability import (
    observe_polaris_component, trace_telemetry_processing, get_logger,
    get_metrics_collector
)
from .knowledge_base import PolarisKnowledgeBase


@observe_polaris_component("telemetry_handler", auto_trace=True, auto_metrics=True)
class TelemetryToKnowledgeBaseHandler(EventHandler):
    """Event handler that forwards telemetry into the knowledge base with observability."""

    def __init__(self, knowledge_base: PolarisKnowledgeBase):
        self._kb = knowledge_base
        self.logger = get_logger("polaris.telemetry.handler")
        self.metrics = get_metrics_collector()

    def can_handle(self, event) -> bool:
        return isinstance(event, TelemetryEvent)

    @trace_telemetry_processing("store_to_knowledge_base")
    async def handle(self, event) -> None:
        if isinstance(event, TelemetryEvent):
            try:
                self.logger.debug("Processing telemetry event", extra={
                    "system_id": event.system_state.system_id,
                    "event_timestamp": event.system_state.timestamp.isoformat(),
                    "metrics_count": len(event.system_state.metrics),
                    "health_status": event.system_state.health_status.value
                })
                
                # Store telemetry with timing
                with self.metrics.time_telemetry_processing(event.system_state.system_id):
                    await self._kb.store_telemetry(event)
                
                # Update metrics
                self.metrics.increment_telemetry_events_received(
                    event.system_state.system_id, 
                    "telemetry_event"
                )
                
                self.logger.debug("Telemetry event processed successfully", extra={
                    "system_id": event.system_state.system_id
                })
                
            except Exception as e:
                self.logger.error("Failed to process telemetry event", extra={
                    "system_id": event.system_state.system_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }, exc_info=e)
                raise


async def subscribe_telemetry_persistence(
    event_bus: PolarisEventBus,
    knowledge_base: PolarisKnowledgeBase,
    system_id_filter: Optional[str] = None,
) -> str:
    """Subscribe a persistence handler to the event bus for telemetry events.

    Returns the subscription ID so callers can unsubscribe later if needed.
    """
    handler = TelemetryToKnowledgeBaseHandler(knowledge_base)
    return event_bus.subscribe(TelemetryEvent, handler, (
        (lambda e: isinstance(e, TelemetryEvent) and e.system_id == system_id_filter)
        if system_id_filter else None
    ))
