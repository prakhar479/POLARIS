"""Tests for SWIMConnector resilience behavior."""

import asyncio
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


@pytest.mark.asyncio
async def test_send_command_closes_writer_on_timeout(monkeypatch):
    connector = SWIMConnector(host="localhost", port=4242, timeout=0.01)

    class FakeReader:
        async def readline(self):
            raise asyncio.TimeoutError("read timeout")

    class FakeWriter:
        def __init__(self):
            self.closed = False
            self.wait_closed_called = False

        def write(self, _data):
            return None

        async def drain(self):
            return None

        def close(self):
            self.closed = True

        async def wait_closed(self):
            self.wait_closed_called = True

    fake_writer = FakeWriter()

    async def fake_open_connection(_host, _port):
        return FakeReader(), fake_writer

    monkeypatch.setattr("asyncio.open_connection", fake_open_connection)

    with pytest.raises(TimeoutError, match="timed out"):
        await connector._send_command("get_servers")

    assert fake_writer.closed is True
    assert fake_writer.wait_closed_called is True
