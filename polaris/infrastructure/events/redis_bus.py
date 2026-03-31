"""Redis-based distributed event bus implementation."""

import asyncio
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type

import redis.asyncio as redis

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.core.events import EventBus


class RedisEventBus(EventBus):
    """Redis-backed distributed async event bus for component communication."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        metrics: Optional[MetricsCollector] = None,
        logger: Optional[Logger] = None,
    ):
        """Initialize Redis event bus.

        Args:
            redis_url: The connection URL for the Redis server.
            metrics: Optional metrics collector.
            logger: Optional logger.
        """
        self.redis_url = redis_url
        self._metrics = metrics
        self._logger = logger
        self._handlers: Dict[Type, List[Callable]] = defaultdict(list)
        self._running = False

        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._listen_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Connect to Redis and start listening for events."""
        if self._running:
            return

        self._running = True
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.pubsub = self.redis_client.pubsub()

            # Start the background task to listen to messages
            self._listen_task = asyncio.create_task(self._listen_loop())

            if self._logger:
                self._logger.info("RedisEventBus started", url=self.redis_url)
            if self._metrics:
                self._metrics.increment("polaris.event_bus.redis.started")
        except Exception as e:
            self._running = False
            if self._logger:
                self._logger.error("Failed to start RedisEventBus", error=str(e))
            raise

    async def stop(self) -> None:
        """Disconnect from Redis and stop the listener."""
        if not self._running:
            return

        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self.pubsub:
            await self.pubsub.close()

        if self.redis_client:
            await self.redis_client.aclose()

        if self._logger:
            self._logger.info("RedisEventBus stopped")
        if self._metrics:
            self._metrics.increment("polaris.event_bus.redis.stopped")

    async def publish(self, event: Any) -> None:
        """Publish an event to the Redis channel.

        Args:
            event: Event to publish (must be pickleable)
        """
        if not self._running or not self.redis_client:
            return

        event_type = type(event)
        channel_name = f"polaris:events:{event_type.__name__}"

        try:
            payload = pickle.dumps(event)
            await self.redis_client.publish(channel_name, payload)

            if self._metrics:
                self._metrics.increment(
                    "polaris.event_bus.events_published",
                    tags={"event_type": event_type.__name__, "bus": "redis"},
                )
        except Exception as e:
            if self._logger:
                self._logger.error(
                    "Failed to publish event to Redis", event_type=event_type.__name__, error=str(e)
                )
            if self._metrics:
                self._metrics.increment(
                    "polaris.event_bus.redis.publish_errors",
                    tags={"event_type": event_type.__name__},
                )

    def subscribe(self, event_type: Type, handler: Callable) -> str:
        """Subscribe to events of a specific type locally and on Redis."""
        self._handlers[event_type].append(handler)

        # If we're already running, subscribe the pubsub to this channel dynamically
        if self._running and self.pubsub:
            channel_name = f"polaris:events:{event_type.__name__}"
            asyncio.create_task(self._dynamic_subscribe(channel_name))

        if self._metrics:
            self._metrics.increment(
                "polaris.event_bus.subscriptions_added",
                tags={"event_type": event_type.__name__, "bus": "redis"},
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

                # If no more handlers for this type, unsubscribe from Redis channel
                if not self._handlers[event_type] and self._running and self.pubsub:
                    channel_name = f"polaris:events:{event_type.__name__}"
                    asyncio.create_task(self._dynamic_unsubscribe(channel_name))

                if self._metrics:
                    self._metrics.increment(
                        "polaris.event_bus.subscriptions_removed",
                        tags={"event_type": event_type.__name__, "bus": "redis"},
                    )
                    self._metrics.gauge(
                        "polaris.event_bus.total_handlers",
                        sum(len(handlers) for handlers in self._handlers.values()),
                    )
            except ValueError:
                pass

    async def _listen_loop(self) -> None:
        """Background task to listen for messages from Redis and dispatch them."""
        if not self.pubsub:
            return

        # Subscribe to all currently registered channels
        for event_type in self._handlers.keys():
            channel_name = f"polaris:events:{event_type.__name__}"
            await self.pubsub.subscribe(channel_name)

        try:
            async for message in self.pubsub.listen():
                if not self._running:
                    break

                if message["type"] == "message":
                    payload = message["data"]
                    try:
                        event = pickle.loads(payload)
                        await self._dispatch_local(event)
                    except Exception as e:
                        if self._logger:
                            self._logger.error(
                                "Failed to deserialize event from Redis",
                                channel=message["channel"],
                                error=str(e),
                            )
                        if self._metrics:
                            self._metrics.increment("polaris.event_bus.redis.deserialize_errors")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._logger:
                self._logger.error("Redis EventBus listener loop crashed", error=str(e))
            if self._metrics:
                self._metrics.increment("polaris.event_bus.redis.listener_crashes")

    async def _dispatch_local(self, event: Any) -> None:
        """Dispatch a deserialized event to local handlers concurrently."""
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return

        if self._metrics:
            self._metrics.gauge(
                "polaris.event_bus.handler_count",
                len(handlers),
                tags={"event_type": event_type.__name__, "bus": "redis"},
            )

        tasks: List[asyncio.Future] = []
        handler_names: List[str] = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(asyncio.create_task(handler(event)))
            else:
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(None, handler, event))

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
                    tags={"event_type": event_type.__name__, "bus": "redis"},
                )

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
                    tags={"event_type": event_type.__name__, "bus": "redis"},
                )

    async def _dynamic_subscribe(self, channel_name: str) -> None:
        """Coroutine wrapper for dynamic redis subscription."""
        if self.pubsub:
            await self.pubsub.subscribe(channel_name)

    async def _dynamic_unsubscribe(self, channel_name: str) -> None:
        """Coroutine wrapper for dynamic redis unsubscription."""
        if self.pubsub:
            await self.pubsub.unsubscribe(channel_name)
