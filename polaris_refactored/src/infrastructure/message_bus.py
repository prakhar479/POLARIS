"""
Message Bus Infrastructure

Provides reliable, scalable message passing using the Adapter pattern
over message brokers like NATS, with middleware support for cross-cutting concerns.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Dict
import asyncio
import json
import gzip
import logging
from datetime import datetime, timezone

import nats
from nats.aio.client import Client as NATS

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


class CompressionMiddleware(Middleware):
    """Middleware for compressing/decompressing messages."""
    
    def __init__(self, compression_threshold: int = 1024):
        """
        Initialize compression middleware.
        
        Args:
            compression_threshold: Minimum message size in bytes to trigger compression
        """
        self.compression_threshold = compression_threshold
    
    async def process_outbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Compress message if it exceeds threshold."""
        message_str = json.dumps(message)
        message_bytes = message_str.encode('utf-8')
        
        if len(message_bytes) >= self.compression_threshold:
            compressed_data = gzip.compress(message_bytes)
            return {
                "_compressed": True,
                "_original_size": len(message_bytes),
                "_compressed_size": len(compressed_data),
                "data": compressed_data.hex()  # Convert to hex for JSON serialization
            }
        
        return message
    
    async def process_inbound(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress message if it was compressed."""
        if message.get("_compressed"):
            compressed_data = bytes.fromhex(message["data"])
            decompressed_bytes = gzip.decompress(compressed_data)
            return json.loads(decompressed_bytes.decode('utf-8'))
        
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


class NATSMessageBroker(MessageBroker):
    """NATS message broker implementation."""
    
    def __init__(
        self,
        servers: List[str] = None,
        name: str = "polaris-message-bus",
        max_reconnect_attempts: int = 10,
        reconnect_time_wait: int = 2
    ):
        """
        Initialize NATS message broker.
        
        Args:
            servers: List of NATS server URLs
            name: Client name for identification
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_time_wait: Time to wait between reconnection attempts
        """
        self.servers = servers or ["nats://localhost:4222"]
        self.name = name
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_time_wait = reconnect_time_wait
        self.nc: Optional[NATS] = None
        self._subscriptions: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> None:
        """Connect to NATS server."""
        try:
            self.nc = await nats.connect(
                servers=self.servers,
                name=self.name,
                max_reconnect_attempts=self.max_reconnect_attempts,
                reconnect_time_wait=self.reconnect_time_wait,
                error_cb=self._error_callback,
                disconnected_cb=self._disconnected_callback,
                reconnected_cb=self._reconnected_callback
            )
            self.logger.info(f"Connected to NATS servers: {self.servers}")
        except Exception as e:
            raise EventBusError(f"Failed to connect to NATS: {e}", cause=e)
    
    async def disconnect(self) -> None:
        """Disconnect from NATS server."""
        if self.nc:
            try:
                # Unsubscribe from all subscriptions
                for sub in self._subscriptions.values():
                    await sub.unsubscribe()
                self._subscriptions.clear()
                
                await self.nc.close()
                self.logger.info("Disconnected from NATS")
            except Exception as e:
                raise EventBusError(f"Failed to disconnect from NATS: {e}", cause=e)
            finally:
                self.nc = None
    
    async def publish(self, topic: str, message: bytes) -> None:
        """Publish a message to a topic."""
        if not self.nc:
            raise EventBusError("NATS client is not connected")
        
        try:
            await self.nc.publish(topic, message)
        except Exception as e:
            raise EventBusError(f"Failed to publish to topic '{topic}': {e}", cause=e)
    
    async def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> str:
        """Subscribe to a topic and return subscription ID."""
        if not self.nc:
            raise EventBusError("NATS client is not connected")
        
        try:
            async def message_handler(msg):
                try:
                    await handler(msg.data)
                except Exception as e:
                    self.logger.error(f"Error handling message on topic '{topic}': {e}")
            
            sub = await self.nc.subscribe(topic, cb=message_handler)
            subscription_id = f"{topic}_{id(sub)}"
            self._subscriptions[subscription_id] = sub
            
            self.logger.debug(f"Subscribed to topic '{topic}' with ID '{subscription_id}'")
            return subscription_id
            
        except Exception as e:
            raise EventBusError(f"Failed to subscribe to topic '{topic}': {e}", cause=e)
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a topic."""
        if subscription_id in self._subscriptions:
            try:
                sub = self._subscriptions[subscription_id]
                await sub.unsubscribe()
                del self._subscriptions[subscription_id]
                self.logger.debug(f"Unsubscribed from subscription '{subscription_id}'")
            except Exception as e:
                raise EventBusError(f"Failed to unsubscribe '{subscription_id}': {e}", cause=e)
    
    async def _error_callback(self, e):
        """Handle NATS connection errors."""
        self.logger.error(f"NATS connection error: {e}")
    
    async def _disconnected_callback(self):
        """Handle NATS disconnection."""
        self.logger.warning("Disconnected from NATS server")
    
    async def _reconnected_callback(self):
        """Handle NATS reconnection."""
        self.logger.info("Reconnected to NATS server")


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
    
    async def publish_telemetry(self, telemetry_event: Any) -> None:
        """Publish a telemetry event to the telemetry topic."""
        await self.publish("polaris.telemetry", telemetry_event)
    
    async def publish_adaptation_needed(self, adaptation_event: Any) -> None:
        """Publish an adaptation needed event to the adaptation topic."""
        await self.publish("polaris.adaptation", adaptation_event)
    
    async def publish_execution_result(self, result_event: Any) -> None:
        """Publish an execution result event to the execution topic."""
        await self.publish("polaris.execution", result_event)
    
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
        # Check if it's a POLARIS event with to_dict method
        if hasattr(event, 'to_dict') and callable(getattr(event, 'to_dict')):
            return event.to_dict()
        
        # Fallback to generic serialization
        message = {
            "type": type(event).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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