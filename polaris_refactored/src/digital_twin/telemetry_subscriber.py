"""
Telemetry Subscriber

Provides a simple EventHandler wired to PolarisEventBus to persist TelemetryEvent
into the PolarisKnowledgeBase automatically.
"""

from typing import Optional

from ..framework.events import TelemetryEvent, PolarisEventBus
from ..domain.interfaces import EventHandler
from .knowledge_base import PolarisKnowledgeBase


class TelemetryToKnowledgeBaseHandler(EventHandler):
    """Event handler that forwards telemetry into the knowledge base."""

    def __init__(self, knowledge_base: PolarisKnowledgeBase):
        self._kb = knowledge_base

    def can_handle(self, event) -> bool:
        return isinstance(event, TelemetryEvent)

    async def handle(self, event) -> None:
        if isinstance(event, TelemetryEvent):
            await self._kb.store_telemetry(event)


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
