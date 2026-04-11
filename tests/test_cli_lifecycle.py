"""Lifecycle tests for interactive CLI cancellation and orchestration."""

import asyncio
import importlib
import threading
import time

import pytest

from polaris.cli import interactive as interactive_module

main_module = importlib.import_module("polaris.cli.main")


class _NotifyingQueue(list):
    def __init__(self, on_quit: threading.Event):
        super().__init__()
        self._on_quit = on_quit

    def append(self, item):
        super().append(item)
        if item == "quit":
            self._on_quit.set()


@pytest.mark.asyncio
async def test_run_interactive_cli_cancellation_requests_quit_and_joins(monkeypatch) -> None:
    started = threading.Event()
    quit_requested = threading.Event()
    exited = threading.Event()

    class FakeInteractiveCLI:
        def __init__(self, _polaris):
            self.use_rawinput = True
            self.stdin = None
            self.cmdqueue = _NotifyingQueue(quit_requested)

        def cmdloop(self) -> None:
            started.set()
            while not quit_requested.is_set():
                time.sleep(0.01)
            exited.set()

    monkeypatch.setattr(interactive_module, "PolarisInteractiveCLI", FakeInteractiveCLI)

    task = asyncio.create_task(interactive_module.run_interactive_cli(object()))

    started_ok = await asyncio.to_thread(started.wait, 1.0)
    assert started_ok is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert quit_requested.is_set() is True
    exited_ok = await asyncio.to_thread(exited.wait, 1.0)
    assert exited_ok is True


@pytest.mark.asyncio
async def test_run_with_interactive_cli_stops_when_interactive_finishes(monkeypatch) -> None:
    class FakePolaris:
        def __init__(self):
            self.stop_calls = 0
            self.run_cancelled = False

        async def run(self) -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.run_cancelled = True
                raise

        async def stop(self) -> None:
            self.stop_calls += 1

    async def fake_run_interactive_cli(_polaris) -> None:
        return None

    monkeypatch.setattr("polaris.cli.interactive.run_interactive_cli", fake_run_interactive_cli)

    polaris = FakePolaris()
    await main_module.run_with_interactive_cli(polaris)

    assert polaris.stop_calls == 1
    assert polaris.run_cancelled is True


@pytest.mark.asyncio
async def test_run_with_interactive_cli_cancels_interactive_when_polaris_finishes(
    monkeypatch,
) -> None:
    cancelled = {"value": False}

    class FakePolaris:
        def __init__(self):
            self.stop_calls = 0

        async def run(self) -> None:
            return None

        async def stop(self) -> None:
            self.stop_calls += 1

    async def fake_run_interactive_cli(_polaris) -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled["value"] = True
            raise

    monkeypatch.setattr("polaris.cli.interactive.run_interactive_cli", fake_run_interactive_cli)

    polaris = FakePolaris()
    await main_module.run_with_interactive_cli(polaris)

    assert polaris.stop_calls == 1
    assert cancelled["value"] is True


@pytest.mark.asyncio
async def test_run_with_interactive_cli_propagates_task_error_and_stops(monkeypatch) -> None:
    class FakePolaris:
        def __init__(self):
            self.stop_calls = 0
            self.run_cancelled = False

        async def run(self) -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.run_cancelled = True
                raise

        async def stop(self) -> None:
            self.stop_calls += 1

    async def fake_run_interactive_cli(_polaris) -> None:
        raise RuntimeError("interactive failed")

    monkeypatch.setattr("polaris.cli.interactive.run_interactive_cli", fake_run_interactive_cli)

    polaris = FakePolaris()
    with pytest.raises(RuntimeError, match="interactive failed"):
        await main_module.run_with_interactive_cli(polaris)

    assert polaris.stop_calls == 1
    assert polaris.run_cancelled is True
