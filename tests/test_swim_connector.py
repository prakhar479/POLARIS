"""Tests for SWIMConnector resilience behavior."""

from unittest.mock import AsyncMock

import pytest

from polaris.connectors.swim import SWIMConnector


@pytest.mark.asyncio
async def test_swim_connect_retries_then_succeeds():
    connector = SWIMConnector(host="localhost", port=4242)
    connector._send_command = AsyncMock(side_effect=[ConnectionError("empty"), "3"])

    result = await connector.connect()

    assert result is True
    assert connector._connected is True
    assert connector._send_command.await_count == 2


@pytest.mark.asyncio
async def test_swim_connect_fails_after_retry_budget():
    connector = SWIMConnector(host="localhost", port=4242)
    connector._send_command = AsyncMock(side_effect=ConnectionError("empty"))

    result = await connector.connect()

    assert result is False
    assert connector._connected is False
    assert connector._send_command.await_count == 3
