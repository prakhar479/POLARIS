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


@pytest.mark.asyncio
async def test_swim_execute_set_dimmer_supports_both_value_and_dimmer_keys():
    connector = SWIMConnector(host="localhost", port=4242)
    connector._connected = True
    connector._send_command = AsyncMock(return_value="OK")

    from polaris.core.models import AdaptationAction, ExecutionStatus

    # Test with {"dimmer": 0.5}
    action1 = AdaptationAction(
        action_id="act-dim-1",
        action_type="set_dimmer",
        target_system="swim",
        parameters={"dimmer": 0.5},
    )
    res1 = await connector.execute_action(action1)
    assert res1.status == ExecutionStatus.SUCCESS
    connector._send_command.assert_called_with("set_dimmer 0.5")

    # Test with {"value": 0.75}
    action2 = AdaptationAction(
        action_id="act-dim-2",
        action_type="set_dimmer",
        target_system="swim",
        parameters={"value": 0.75},
    )
    res2 = await connector.execute_action(action2)
    assert res2.status == ExecutionStatus.SUCCESS
    connector._send_command.assert_called_with("set_dimmer 0.75")
