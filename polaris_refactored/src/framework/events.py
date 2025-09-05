"""
Event System for POLARIS

Comprehensive event-driven architecture with event publishing, subscription management,
correlation tracking, and asynchronous event processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Set, Union
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import uuid
import weakref

from ..domain.models import SystemState, AdaptationAction, ExecutionResult
from ..domain.interfaces import EventHandler
from ..infrastructure.di import Injectable

logger = logging.getLogger(__name__)


@dataclass
class EventMetadata:
    """Metadata associated with events for tracking and correlation."""
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    priority: int = 0  # 0 = normal, higher = more important
    retry_count: int = 0
    max_retries: int = 3


class PolarisEvent(ABC):
    """Base class for all POLARIS events with correlation and metadata support."""
    
    def __init__(
        self, 
        event_id: Optional[str] = None, 
        timestamp: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[EventMetadata] = None
    ):
        self.event_id = event_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.utcnow()
        self.correlation_id = correlation_id
        self.metadata = metadata or EventMetadata()
        self._processed_by: Set[str] = set()
    
    def add_correlation(self, correlation_id: str) -> None:
        """Add correlation ID to track related events."""
        self.correlation_id = correlation_id
    
    def mark_processed_by(self, handler_id: str) -> None:
        """Mark this event as processed by a specific handler."""
        self._processed_by.add(handler_id)
    
    def was_processed_by(self, handler_id: str) -> bool:
        """Check if this event was processed by a specific handler."""
        return handler_id in self._processed_by
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "metadata": {
                "source": self.metadata.source,
                "tags": self.metadata.tags,
                "priority": self.metadata.priority,
                "retry_count": self.metadata.retry_count
            }
        }


class TelemetryEvent(PolarisEvent):
    """Event containing telemetry data from managed systems."""
    
    def __init__(self, system_state: SystemState, **kwargs):
        super().__init__(**kwargs)
        self.system_state = system_state
        self.system_id = system_state.system_id
        if not self.metadata.source:
            self.metadata.source = f"system:{self.system_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "system_id": self.system_id,
            "system_state": {
                "system_id": self.system_state.system_id,
                "health_status": str(self.system_state.health_status),
                "metrics_count": len(self.system_state.metrics) if self.system_state.metrics else 0,
                "timestamp": self.system_state.timestamp.isoformat() if self.system_state.timestamp else None
            }
        })
        return base_dict


class AdaptationEvent(PolarisEvent):
    """Event indicating that adaptation is needed."""
    
    def __init__(
        self, 
        system_id: str, 
        reason: str, 
        suggested_actions: Optional[List[AdaptationAction]] = None,
        severity: str = "normal",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.system_id = system_id
        self.reason = reason
        self.suggested_actions = suggested_actions or []
        self.severity = severity  # low, normal, high, critical
        if not self.metadata.source:
            self.metadata.source = f"adaptation_engine:{system_id}"
        
        # Set priority based on severity
        severity_priority = {"low": 1, "normal": 2, "high": 3, "critical": 4}
        self.metadata.priority = severity_priority.get(severity, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert adaptation event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "system_id": self.system_id,
            "reason": self.reason,
            "severity": self.severity,
            "suggested_actions_count": len(self.suggested_actions)
        })
        return base_dict


class ExecutionResultEvent(PolarisEvent):
    """Event containing the result of an adaptation action execution."""
    
    def __init__(self, execution_result: ExecutionResult, **kwargs):
        super().__init__(**kwargs)
        self.execution_result = execution_result
        self.action_id = execution_result.action_id
        if not self.metadata.source:
            self.metadata.source = f"execution_engine:{self.action_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result event to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "action_id": self.action_id,
            "execution_status": str(self.execution_result.status),
            "has_result_data": bool(self.execution_result.result_data)
        })
        return base_dict


@dataclass
class EventSubscription:
    """Represents a subscription to events."""
    subscription_id: str
    event_type: type
    handler: Union[EventHandler, Callable]
    filter_func: Optional[Callable[[PolarisEvent], bool]] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches(self, event: PolarisEvent) -> bool:
        """Check if this subscription matches the given event."""
        if not self.is_active:
            return False
        
        # Check event type
        if not isinstance(event, self.event_type):
            return False
        
        # Apply filter if present
        if self.filter_func and not self.filter_func(event):
            return False
        
        return True


class EventProcessingError(Exception):
    """Exception raised when event processing fails."""
    
    def __init__(self, message: str, event: PolarisEvent, handler_id: str, cause: Exception = None):
        super().__init__(message)
        self.event = event
        self.handler_id = handler_id
        self.cause = cause


class PolarisEventBus(Injectable):
    """
    Comprehensive event bus for POLARIS framework with subscription management,
    correlation tracking, and asynchronous event processing.
    
    Features:
    - Type-safe event publishing and subscription
    - Event correlation and tracking
    - Asynchronous event processing with error handling
    - Event filtering and priority handling
    - Subscription lifecycle management
    - Event replay and debugging capabilities
    """
    
    def __init__(self, max_queue_size: int = 10000, worker_count: int = 4):
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._event_type_subscriptions: Dict[type, List[str]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._worker_count = worker_count
        self._event_history: List[PolarisEvent] = []
        self._max_history_size = 1000
        self._correlation_tracker: Dict[str, List[str]] = defaultdict(list)
        self._processing_stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "active_subscriptions": 0
        }
        
        logger.info(f"PolarisEventBus initialized with {worker_count} workers")
    
    async def start(self) -> None:
        """Start the event bus and processing workers."""
        if self._running:
            logger.warning("Event bus is already running")
            return
        
        self._running = True
        
        # Start worker tasks
        for i in range(self._worker_count):
            worker = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Event bus started with {len(self._workers)} workers")
    
    async def stop(self) -> None:
        """Stop the event bus and cleanup resources."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        
        # Clear remaining events in queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("Event bus stopped")
    
    async def _event_worker(self, worker_id: str) -> None:
        """Worker task that processes events from the queue."""
        logger.info(f"Event worker {worker_id} started")
        
        while self._running:
            try:
                # Wait for event with timeout to allow graceful shutdown
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event, worker_id)
                self._processing_stats["events_processed"] += 1
                
            except asyncio.TimeoutError:
                continue  # Normal timeout, check if still running
            except Exception as e:
                logger.error(f"Event worker {worker_id} error: {e}")
                self._processing_stats["events_failed"] += 1
        
        logger.info(f"Event worker {worker_id} stopped")
    
    async def _process_event(self, event: PolarisEvent, worker_id: str) -> None:
        """Process a single event by notifying all matching subscribers."""
        # Track correlation first (regardless of subscribers)
        if event.correlation_id:
            self._correlation_tracker[event.correlation_id].append(event.event_id)
        
        event_type = type(event)
        subscription_ids = self._event_type_subscriptions.get(event_type, [])
        
        if not subscription_ids:
            logger.debug(f"No subscribers for event type {event_type.__name__}")
            return
        
        # Process each matching subscription
        for subscription_id in subscription_ids:
            subscription = self._subscriptions.get(subscription_id)
            if not subscription or not subscription.matches(event):
                continue
            
            try:
                await self._invoke_handler(subscription, event, worker_id)
                event.mark_processed_by(subscription_id)
                
            except Exception as e:
                logger.error(f"Handler {subscription_id} failed to process event {event.event_id}: {e}")
                
                # Retry logic
                if event.metadata.retry_count < event.metadata.max_retries:
                    event.metadata.retry_count += 1
                    await self._event_queue.put(event)
                    logger.info(f"Retrying event {event.event_id} (attempt {event.metadata.retry_count})")
                else:
                    logger.error(f"Event {event.event_id} failed after {event.metadata.max_retries} retries")
                    raise EventProcessingError(
                        f"Event processing failed after retries",
                        event, subscription_id, e
                    )
    
    async def _invoke_handler(self, subscription: EventSubscription, event: PolarisEvent, worker_id: str) -> None:
        """Invoke a handler for an event."""
        handler = subscription.handler
        
        if isinstance(handler, EventHandler):
            if handler.can_handle(event):
                await handler.handle(event)
        elif callable(handler):
            # Direct callable handler
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        else:
            raise ValueError(f"Invalid handler type: {type(handler)}")
    
    async def publish(self, event: PolarisEvent) -> None:
        """Publish an event to the event bus."""
        if not self._running:
            raise RuntimeError("Event bus is not running")
        
        # Add to history
        self._add_to_history(event)
        
        # Queue for processing
        try:
            await self._event_queue.put(event)
            self._processing_stats["events_published"] += 1
            logger.debug(f"Published event {event.event_id} of type {type(event).__name__}")
        except asyncio.QueueFull:
            logger.error(f"Event queue is full, dropping event {event.event_id}")
            raise RuntimeError("Event queue is full")
    
    async def publish_telemetry(self, telemetry: TelemetryEvent) -> None:
        """Publish a telemetry event."""
        await self.publish(telemetry)
    
    async def publish_adaptation_needed(self, adaptation: AdaptationEvent) -> None:
        """Publish an adaptation needed event."""
        await self.publish(adaptation)
    
    async def publish_execution_result(self, result: ExecutionResultEvent) -> None:
        """Publish an execution result event."""
        await self.publish(result)
    
    def subscribe(
        self, 
        event_type: type, 
        handler: Union[EventHandler, Callable],
        filter_func: Optional[Callable[[PolarisEvent], bool]] = None
    ) -> str:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Handler function or EventHandler instance
            filter_func: Optional filter function to apply to events
            
        Returns:
            str: Subscription ID for managing the subscription
        """
        subscription_id = str(uuid.uuid4())
        
        subscription = EventSubscription(
            subscription_id=subscription_id,
            event_type=event_type,
            handler=handler,
            filter_func=filter_func
        )
        
        self._subscriptions[subscription_id] = subscription
        self._event_type_subscriptions[event_type].append(subscription_id)
        self._processing_stats["active_subscriptions"] += 1
        
        logger.info(f"Created subscription {subscription_id} for event type {event_type.__name__}")
        return subscription_id
    
    async def subscribe_to_telemetry(
        self, 
        handler: Callable[[TelemetryEvent], None],
        system_id_filter: Optional[str] = None
    ) -> str:
        """Subscribe to telemetry events with optional system ID filtering."""
        filter_func = None
        if system_id_filter:
            filter_func = lambda event: event.system_id == system_id_filter
        
        return self.subscribe(TelemetryEvent, handler, filter_func)
    
    async def subscribe_to_adaptations(
        self, 
        handler: Callable[[AdaptationEvent], None],
        severity_filter: Optional[str] = None
    ) -> str:
        """Subscribe to adaptation events with optional severity filtering."""
        filter_func = None
        if severity_filter:
            filter_func = lambda event: event.severity == severity_filter
        
        return self.subscribe(AdaptationEvent, handler, filter_func)
    
    async def subscribe_to_execution_results(
        self, 
        handler: Callable[[ExecutionResultEvent], None],
        action_id_filter: Optional[str] = None
    ) -> str:
        """Subscribe to execution result events with optional action ID filtering."""
        filter_func = None
        if action_id_filter:
            filter_func = lambda event: event.action_id == action_id_filter
        
        return self.subscribe(ExecutionResultEvent, handler, filter_func)
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID of the subscription to remove
            
        Returns:
            bool: True if subscription was found and removed
        """
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            logger.warning(f"Subscription {subscription_id} not found")
            return False
        
        # Remove from subscriptions
        del self._subscriptions[subscription_id]
        
        # Remove from event type mapping
        event_type_subs = self._event_type_subscriptions[subscription.event_type]
        if subscription_id in event_type_subs:
            event_type_subs.remove(subscription_id)
        
        self._processing_stats["active_subscriptions"] -= 1
        
        logger.info(f"Removed subscription {subscription_id}")
        return True
    
    def get_correlated_events(self, correlation_id: str) -> List[str]:
        """Get all event IDs associated with a correlation ID."""
        return self._correlation_tracker.get(correlation_id, [])
    
    def get_event_history(self, limit: int = 100) -> List[PolarisEvent]:
        """Get recent event history."""
        return self._event_history[-limit:]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get event processing statistics."""
        stats = self._processing_stats.copy()
        stats.update({
            "queue_size": self._event_queue.qsize(),
            "history_size": len(self._event_history),
            "correlation_count": len(self._correlation_tracker),
            "is_running": self._running,
            "worker_count": len(self._workers)
        })
        return stats
    
    def _add_to_history(self, event: PolarisEvent) -> None:
        """Add event to history with size limit."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
    
    async def replay_events(
        self, 
        correlation_id: str, 
        handler: Callable[[PolarisEvent], None]
    ) -> int:
        """
        Replay all events associated with a correlation ID.
        
        Args:
            correlation_id: Correlation ID to replay events for
            handler: Handler to process replayed events
            
        Returns:
            int: Number of events replayed
        """
        event_ids = self.get_correlated_events(correlation_id)
        replayed_count = 0
        
        for event in self._event_history:
            if event.event_id in event_ids:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                    replayed_count += 1
                except Exception as e:
                    logger.error(f"Error replaying event {event.event_id}: {e}")
        
        logger.info(f"Replayed {replayed_count} events for correlation {correlation_id}")
        return replayed_count