"""Tests for dashboard rendering, event handling, and loop control."""

from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.layout import Layout

from polaris.cli import dashboard as dashboard_module
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)


class _FakeEventBus:
    def __init__(self):
        self.subscriptions = []

    def subscribe(self, event_type, handler):
        self.subscriptions.append((event_type, handler))
        return "sub"


class _FakeRegistry:
    def __init__(self, systems=None):
        self._systems = list(systems or [])

    def system_ids(self):
        return list(self._systems)


class _FakeMetrics:
    def __init__(self, summary=None, error=None):
        self._summary = summary or {}
        self._error = error

    def get_summary(self):
        if self._error is not None:
            raise self._error
        return self._summary


class _FakeStrategy:
    def __init__(self, perf=None, fail=False):
        self._perf = perf or {"success_rate": 0.5}
        self._fail = fail
        self.calls = 0

    async def get_performance_metrics(self):
        self.calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return self._perf


class _FakeLive:
    def __init__(self, _renderable, **_kwargs):
        self.updated = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type, exc, tb
        return False

    def start(self, refresh=False):
        _ = refresh
        pass

    def stop(self):
        pass

    def update(self, renderable):
        self.updated.append(renderable)


class _FakeRunner:
    def __init__(self, _polaris, _output):
        self.commands = []

    def command_names(self):
        return ["help", "status", "quit"]

    def execute(self, command):
        self.commands.append(command)
        return command.strip() == "quit"


def _make_polaris(*, running=True, systems=None, metrics=None, strategy=None):
    polaris = SimpleNamespace()
    polaris.event_bus = _FakeEventBus()
    polaris.registry = _FakeRegistry(systems=systems)
    polaris.metrics = metrics
    polaris.strategy = strategy
    polaris.meta_learner = object()
    polaris._running = running
    polaris.is_running = lambda: polaris._running
    return polaris


@pytest.fixture
def dashboard_fixture():
    polaris = _make_polaris(
        systems=["sys-a"],
        metrics=_FakeMetrics(
            summary={
                "counters": {
                    "polaris.monitoring.cycles": 2,
                    "polaris.adaptations.success": 1,
                },
                "gauges": {"polaris.monitoring.queue_depth": 3},
                "histograms": {"polaris.monitoring.loop_duration": {"avg": 0.12}},
            }
        ),
        strategy=_FakeStrategy(perf={"success_rate": 0.75, "total_adaptations": 4}),
    )
    dash = dashboard_module.Dashboard(polaris)
    yield dash, polaris
    if dash._log_handler is not None:
        try:
            import logging

            logging.getLogger("polaris").removeHandler(dash._log_handler)
        except Exception:
            pass


def test_dashboard_init_subscribes_event_handlers(dashboard_fixture):
    dash, polaris = dashboard_fixture
    assert dash._log_handler is not None
    assert len(polaris.event_bus.subscriptions) == 2
    names = {event_type.__name__ for event_type, _ in polaris.event_bus.subscriptions}
    assert names == {"TelemetryEvent", "AdaptationEvent"}


def test_dashboard_event_buffers_prune_to_limits(dashboard_fixture):
    dash, _ = dashboard_fixture
    dash.max_history = 2
    dash.max_events = 2

    for value in [1.0, 2.0, 3.0]:
        state = SystemState(
            system_id="sys-a",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu": MetricValue(name="cpu", value=value, unit="percent")},
            health_status=HealthStatus.HEALTHY,
        )
        telemetry_event = SimpleNamespace(timestamp=datetime.now(timezone.utc), state=state)
        dash._on_telemetry(telemetry_event)

        action = AdaptationAction(
            action_id=f"act-{int(value)}",
            action_type=f"scale-{int(value)}",
            target_system="sys-a",
        )
        result = ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.SUCCESS,
            result_data={},
        )
        adaptation_event = SimpleNamespace(
            timestamp=datetime.now(timezone.utc),
            action=action,
            result=result,
        )
        dash._on_adaptation(adaptation_event)

    assert [entry[1] for entry in dash.metric_history["cpu"]] == [2.0, 3.0]
    assert len(dash.recent_events) == 2
    assert dash.recent_events[-1]["action"] == "scale-3"


def test_dashboard_render_includes_metrics_and_error_fallback(dashboard_fixture):
    dash, polaris = dashboard_fixture
    dash.metric_history["cpu"] = [
        (datetime.now(timezone.utc), 10.0),
        (datetime.now(timezone.utc), 12.0),
    ]
    dash.recent_events.append(
        {
            "time": datetime.now(timezone.utc),
            "action": "scale_up",
            "status": "success",
            "system": "sys-a",
        }
    )

    layout = dash._render()
    assert isinstance(layout, Layout)
    assert layout["header"].renderable is not None
    assert layout["system_metrics"].renderable is not None

    polaris.metrics = _FakeMetrics(error=RuntimeError("metrics failed"))
    layout_with_error = dash._render()
    console = Console(record=True, force_terminal=False, color_system=None, width=140)
    console.print(layout_with_error["system_metrics"].renderable)
    assert "Failed to load: metrics failed" in console.export_text()


def test_dashboard_value_and_trend_helpers(dashboard_fixture):
    dash, _ = dashboard_fixture

    assert dash._format_metric_value(1200) == "1,200"
    assert dash._format_metric_value(25.6789) == "25.68"
    assert dash._format_metric_value(1.23456) == "1.235"
    assert dash._format_metric_value("raw") == "raw"

    up = [(datetime.now(timezone.utc), v) for v in [1.0, 1.1, 1.3, 1.5]]
    down = [(datetime.now(timezone.utc), v) for v in [2.0, 1.8, 1.5, 1.2]]
    flat = [(datetime.now(timezone.utc), v) for v in [1.0, 1.01, 1.0, 1.02]]

    assert dash._calculate_trend(up) == "↑"
    assert dash._calculate_trend(down) == "↓"
    assert dash._calculate_trend(flat) == "→"
    assert dash._calculate_trend([(datetime.now(timezone.utc), "bad")]) == "—"


@pytest.mark.asyncio
async def test_dashboard_uptime_and_non_tty_key_read(dashboard_fixture, monkeypatch):
    dash, _ = dashboard_fixture
    dash._started_at = datetime.now() - timedelta(seconds=65)
    assert dash._format_uptime() == "1m 5s"

    monkeypatch.setattr(dashboard_module.sys.stdin, "isatty", lambda: False)
    assert await dash._read_key_nonblocking() is None


@pytest.mark.asyncio
async def test_update_metrics_cache_updates_and_handles_errors(dashboard_fixture, monkeypatch):
    dash, polaris = dashboard_fixture
    polaris.strategy = _FakeStrategy(perf={"success_rate": 0.91})
    dash.running = True

    async def fake_sleep(_seconds):
        dash.running = False

    monkeypatch.setattr(dashboard_module.asyncio, "sleep", fake_sleep)
    await dash._update_metrics_cache()
    assert dash._cached_perf_metrics["success_rate"] == 0.91

    polaris.strategy = _FakeStrategy(fail=True)
    dash.running = True
    await dash._update_metrics_cache()


@pytest.mark.asyncio
async def test_run_exits_and_detaches_handler(dashboard_fixture, monkeypatch):
    dash, polaris = dashboard_fixture
    polaris._running = False

    monkeypatch.setattr(dashboard_module, "Live", _FakeLive)
    await dash.run(refresh_rate=0.01)

    assert dash.running is False
    import logging

    assert dash._log_handler not in logging.getLogger("polaris").handlers


def test_live_display_safe_uses_manual_refresh_mode(dashboard_fixture, monkeypatch):
    dash, _ = dashboard_fixture
    observed = {}

    class _SpyLive:
        def __init__(self, _renderable, **kwargs):
            observed["kwargs"] = kwargs
            self.started = None
            self.stopped = False

        def start(self, refresh=False):
            self.started = refresh

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(dashboard_module, "Live", _SpyLive)

    with dash._live_display_safe(Layout(name="root"), refresh_per_second=7) as live:
        assert live.started is True

    assert observed["kwargs"]["auto_refresh"] is False
    assert observed["kwargs"]["refresh_per_second"] == 7


def test_safe_live_update_forces_refresh_when_supported(dashboard_fixture):
    dash, _ = dashboard_fixture

    class _RefreshAwareLive:
        def __init__(self):
            self.calls = []

        def update(self, renderable, refresh=False):
            self.calls.append((renderable, refresh))

    live = _RefreshAwareLive()
    renderable = Layout(name="root")
    dash._safe_live_update(live, renderable)

    assert len(live.calls) == 1
    _, refresh = live.calls[0]
    assert refresh is True


@pytest.mark.asyncio
async def test_run_with_interactive_cli_handles_ctrl_d_exit(dashboard_fixture, monkeypatch):
    dash, polaris = dashboard_fixture
    polaris._running = True

    monkeypatch.setattr(dashboard_module, "Live", _FakeLive)
    monkeypatch.setattr(dashboard_module, "_EmbeddedInteractiveCLI", _FakeRunner)

    keys = iter(["\x04", None, None])

    async def mock_read_key():
        return next(keys, None)

    monkeypatch.setattr(dash, "_read_key_nonblocking", mock_read_key)
    monkeypatch.setattr(dash, "_render_with_interactive", lambda **_kwargs: Layout(name="root"))

    async def fast_update_cache():
        while dash.running:
            await dashboard_module.asyncio.sleep(0)

    async def fast_sleep(_seconds):
        return None

    monkeypatch.setattr(dash, "_update_metrics_cache", fast_update_cache)
    monkeypatch.setattr(dashboard_module.asyncio, "sleep", fast_sleep)

    await dash.run_with_interactive_cli(refresh_rate=0.01)

    assert dash.running is False


def test_embedded_interactive_cli_helpers(monkeypatch):
    class FakePolarisInteractiveCLI:
        def __init__(self, _polaris):
            self.printed = []

        def onecmd(self, command):
            return command == "quit"

        def do_help(self, _arg=""):
            return None

        def do_status(self, _arg=""):
            return None

    monkeypatch.setattr(
        "polaris.cli.interactive.PolarisInteractiveCLI",
        FakePolarisInteractiveCLI,
    )

    output = deque(maxlen=20)
    runner = dashboard_module._EmbeddedInteractiveCLI(object(), output)

    runner._print("hello")
    runner._print_json({"k": 1})
    assert runner.execute("noop") is False
    assert runner.execute("quit") is True
    assert runner.command_names() == ["clear", "help", "status"]

    monkeypatch.setattr(dashboard_module, "RICH_AVAILABLE", False)
    runner._print_table(object())
    runner.do_clear("")
    assert list(output) == ["Cleared interactive output."]
