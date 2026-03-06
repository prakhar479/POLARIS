"""Tests for the RedisEventBus."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polaris.infrastructure.events.redis_bus import RedisEventBus


class DummyEvent:
    def __init__(self, value: int):
        self.value = value


async def async_generator_mock():
    """Helper async generator for mocking pubsub.listen()."""
    yield


@pytest.fixture
def redis_bus():
    return RedisEventBus(redis_url="redis://localhost:6379")


@pytest.mark.asyncio
@patch("polaris.infrastructure.events.redis_bus.redis.from_url")
async def test_redis_bus_start_stop(mock_from_url, redis_bus):
    mock_redis_client = MagicMock()
    mock_pubsub = MagicMock()
    mock_redis_client.pubsub.return_value = mock_pubsub
    mock_redis_client.aclose = AsyncMock()

    # Mock listen() to return an async generator
    mock_pubsub.listen = MagicMock(side_effect=async_generator_mock)
    # Mock close() to be async
    mock_pubsub.close = AsyncMock()
    # Mock subscribe() to be async
    mock_pubsub.subscribe = AsyncMock()

    mock_from_url.return_value = mock_redis_client

    await redis_bus.start()
    assert redis_bus._running is True
    assert redis_bus.redis_client is mock_redis_client
    assert redis_bus.pubsub is mock_pubsub

    await redis_bus.stop()
    assert redis_bus._running is False
    mock_pubsub.close.assert_awaited_once()
    mock_redis_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
@patch("polaris.infrastructure.events.redis_bus.redis.from_url")
async def test_redis_bus_publish(mock_from_url, redis_bus):
    mock_redis_client = MagicMock()
    mock_pubsub = MagicMock()
    mock_redis_client.pubsub.return_value = mock_pubsub
    mock_redis_client.publish = AsyncMock()
    mock_redis_client.aclose = AsyncMock()

    # Mock listen() to return an async generator
    mock_pubsub.listen = MagicMock(side_effect=async_generator_mock)
    # Mock close() to be async
    mock_pubsub.close = AsyncMock()
    # Mock subscribe() to be async
    mock_pubsub.subscribe = AsyncMock()

    mock_from_url.return_value = mock_redis_client

    await redis_bus.start()

    event = DummyEvent(42)
    # Patch pickle.dumps to avoid dealing with bytes assertions if needed, but it's fine.
    await redis_bus.publish(event)

    mock_redis_client.publish.assert_awaited_once()
    args, _ = mock_redis_client.publish.call_args
    assert args[0] == "polaris:events:DummyEvent"

    await redis_bus.stop()


@pytest.mark.asyncio
@patch("polaris.infrastructure.events.redis_bus.redis.from_url")
async def test_redis_bus_subscribe_unsubscribe(mock_from_url, redis_bus):
    handler = MagicMock()

    # Subscribe before start
    sub_id = redis_bus.subscribe(DummyEvent, handler)
    assert DummyEvent in redis_bus._handlers
    assert handler in redis_bus._handlers[DummyEvent]

    mock_redis_client = MagicMock()
    mock_pubsub = MagicMock()
    mock_redis_client.pubsub.return_value = mock_pubsub
    mock_redis_client.aclose = AsyncMock()

    # Mock listen() to return an async generator
    mock_pubsub.listen = MagicMock(side_effect=async_generator_mock)
    # Mock close() to be async
    mock_pubsub.close = AsyncMock()
    # Mock subscribe() to be async
    mock_pubsub.subscribe = AsyncMock()
    # Mock unsubscribe() to be async
    mock_pubsub.unsubscribe = AsyncMock()

    mock_from_url.return_value = mock_redis_client

    await redis_bus.start()

    # Subscribe after start
    handler2 = MagicMock()
    redis_bus.subscribe(DummyEvent, handler2)

    # Yield to let the async task run subscription
    await asyncio.sleep(0.01)
    mock_pubsub.subscribe.assert_called_with("polaris:events:DummyEvent")

    # Unsubscribe
    redis_bus.unsubscribe(DummyEvent, handler)
    assert handler not in redis_bus._handlers[DummyEvent]

    # Unsubscribe last handler
    redis_bus.unsubscribe(DummyEvent, handler2)
    assert not redis_bus._handlers[DummyEvent]

    await asyncio.sleep(0.01)
    mock_pubsub.unsubscribe.assert_called_with("polaris:events:DummyEvent")

    await redis_bus.stop()
