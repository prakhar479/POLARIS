"""
Message Bus Infrastructure

Provides reliable, scalable message passing using the Adapter pattern
over message brokers like NATS, with middleware support for cross-cutting concerns.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Dict
import asyncio
import json
from datetime import datetime

from ..domain.interfaces import EventHandler
from .exceptions import EventBusError
from .di import Injectable


class MessageBroker(ABC):
    """Abstract interface for message brokers."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the message broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the message broker."""
        pass
    
    @abstractmethod
    async def publish(self, topic: str, message: bytes) -> None:
        """Publish a message to a topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> str:
        """Subscribe to a topic and return subscription ID."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a topic."""
        pass


class Middleware(ABC):
    """Abstract base class for message bus middleware."""
    
    @abstractmethod
    async def process_outbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process outbound message before publishing."""
        pass
    
    @abstractmethod
    async def process_inbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process inbound message after receiving."""
        pass


class LoggingMiddleware(Middleware):
    """Middleware for logging message bus activity."""
    
    def __init__(self, logger):
        self.logger = logger
    
    async def process_outbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug(f"Publishing message to topic '{topic}'", extra={
            "topic": topic,
            "message_type": message.get("type"),
            "correlation_id": message.get("correlation_id")
        })
        return message
    
    async def process_inbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug(f"Received message from topic '{topic}'", extra={
            "topic": topic,
            "message_type": message.get("type"),
            "correlation_id": message.get("correlation_id")
        })
        return message


class MetricsMiddleware(Middleware):
    """Middleware for collecting message bus metrics."""
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
    
    async def process_outbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        self.metrics_collector.increment_counter("messages_published_total", {"topic": topic})
        return message
    
    async def process_inbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        self.metrics_collector.increment_counter("messages_received_total", {"topic": topic})
        return message


class MiddlewareChain:
    """Chain of middleware for processing messages."""
    
    def __init__(self, middleware_list: List[Middleware]):
        self.middleware_list = middleware_list
    
    async def process_outbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message through outbound middleware chain."""
        for middleware in self.middleware_list:
            message = await middleware.process_outbound(topic, message)
        return message
    
    async def process_inbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message through inbound middleware chain."""
        for middleware in reversed(self.middleware_list):
            message = await middleware.process_inbound(topic, message)
        return message


class PolarisMessageBus(Injectable):
    """
    POLARIS Message Bus implementation using the Adapter pattern.
    
    Provides reliable, scalable messaging with middleware support
    for cross-cutting concerns like logging and metrics.
    """
    
    def __init__(
        self,
        broker: MessageBroker,
        middleware_chain: Optional[MiddlewareChain] = None
    ):
        self.broker = broker
        self.middleware_chain = middleware_chain or MiddlewareChain([])
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._subscriptions: Dict[str, str] = {}
        self._connected = False
    
    async def start(self) -> None:
        """Start the message bus."""
        try:
            await self.broker.connect()
            self._connected = True
        except Exception as e:
            raise EventBusError(
                "Failed to start message bus",
                cause=e,
                context={"broker_type": type(self.broker).__name__}
            )
    
    async def stop(self) -> None:
        """Stop the message bus."""
        try:
            # Unsubscribe from all topics
            for topic, subscription_id in self._subscriptions.items():
                await self.broker.unsubscribe(subscription_id)
            
            self._subscriptions.clear()
            self._handlers.clear()
            
            await self.broker.disconnect()
            self._connected = False
        except Exception as e:
            raise EventBusError(
                "Failed to stop message bus",
                cause=e,
                context={"broker_type": type(self.broker).__name__}
            )
    
    async def publish(self, topic: str, event: Any) -> None:
        """Publish an event to a topic."""
        if not self._connected:
            raise EventBusError("Message bus is not connected")
        
        try:
            # Convert event to message
            message = self._event_to_message(event)
            
            # Process through middleware
            message = await self.middleware_chain.process_outbound(topic, message)
            
            # Serialize and publish
            message_bytes = json.dumps(message).encode('utf-8')
            await self.broker.publish(topic, message_bytes)
            
        except Exception as e:
            raise EventBusError(
                f"Failed to publish event to topic '{topic}'",
                event_type=type(event).__name__,
                topic=topic,
                cause=e
            )
    
    async def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribe to a topic with an event handler."""
        if not self._connected:
            raise EventBusError("Message bus is not connected")
        
        try:
            # Add handler to list
            if topic not in self._handlers:
                self._handlers[topic] = []
                # Create broker subscription
                subscription_id = await self.broker.subscribe(
                    topic, 
                    lambda msg: asyncio.create_task(self._handle_message(topic, msg))
                )
                self._subscriptions[topic] = subscription_id
            
            self._handlers[topic].append(handler)
            
        except Exception as e:
            raise EventBusError(
                f"Failed to subscribe to topic '{topic}'",
                topic=topic,
                cause=e
            )
    
    async def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from a topic."""
        if topic in self._handlers and handler in self._handlers[topic]:
            self._handlers[topic].remove(handler)
            
            # If no more handlers, unsubscribe from broker
            if not self._handlers[topic]:
                if topic in self._subscriptions:
                    await self.broker.unsubscribe(self._subscriptions[topic])
                    del self._subscriptions[topic]
                del self._handlers[topic]
    
    async def _handle_message(self, topic: str, message_bytes: bytes) -> None:
        """Handle incoming message from broker."""
        try:
            # Deserialize message
            message = json.loads(message_bytes.decode('utf-8'))
            
            # Process through middleware
            message = await self.middleware_chain.process_inbound(topic, message)
            
            # Convert to event
            event = self._message_to_event(message)
            
            # Dispatch to handlers
            if topic in self._handlers:
                for handler in self._handlers[topic]:
                    if handler.can_handle(event):
                        await handler.handle(event)
                        
        except Exception as e:
            # Log error but don't raise to avoid breaking message processing
            # This should be logged through the logging system
            pass
    
    def _event_to_message(self, event: Any) -> Dict[str, Any]:
        """Convert an event object to a message dictionary."""
        message = {
            "type": type(event).__name__,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {}
        }
        
        # Convert event attributes to message data
        if hasattr(event, '__dict__'):
            for key, value in event.__dict__.items():
                if not key.startswith('_'):
                    message["data"][key] = self._serialize_value(value)
        
        return message
    
    def _message_to_event(self, message: Dict[str, Any]) -> Any:
        """Convert a message dictionary back to an event object."""
        # This is a simplified implementation
        # In practice, you'd want proper event deserialization
        event_type = message.get("type")
        event_data = message.get("data", {})
        
        # Create a simple event object
        class GenericEvent:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        return GenericEvent(**event_data)
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON encoding."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif hasattr(value, '__dict__'):
            return {k: self._serialize_value(v) for k, v in value.__dict__.items()}
        else:
            return value