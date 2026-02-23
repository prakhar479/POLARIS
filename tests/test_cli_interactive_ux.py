"""UX-focused tests for the interactive Polaris CLI."""

from datetime import datetime, timezone
from typing import Any, Dict, List

from polaris.cli.interactive import PolarisInteractiveCLI
from polaris.core.models import HealthStatus, MetricValue, SystemState


class _DummyRegistry:
    def __init__(self) -> None:
        self._systems = {"system-a", "system-b"}

    def system_ids(self):
        return self._systems

    def all(self):
        return []


class _DummyMetrics:
    def get_summary(self) -> Dict[str, Any]:
        return {"counters": {}, "gauges": {}, "histograms": {}}


class _DummyKnowledgeStore:
    async def query_states(self, system_id, start, end):  # noqa: ANN001
        _ = start
        _ = end
        return [
            SystemState(
                system_id=system_id,
                timestamp=datetime.now(timezone.utc),
                metrics={"cpu_usage": MetricValue(name="cpu_usage", value=50.0, unit="percent")},
                health_status=HealthStatus.HEALTHY,
            )
        ]

    async def query_actions(self, system_id, start, end):  # noqa: ANN001
        _ = system_id
        _ = start
        _ = end
        return []


class _DummyWorldModel:
    async def get_insights(self):
        return {"system-a": {"summary": "ok"}}


class _DummyPolaris:
    def __init__(self) -> None:
        self.registry = _DummyRegistry()
        self.metrics = _DummyMetrics()
        self.knowledge_store = _DummyKnowledgeStore()
        self.world_model = _DummyWorldModel()
        self.meta_learner = None
        self.strategy = None
        self.exported: List[Any] = []

    def is_running(self) -> bool:
        return True

    def export_metrics(self, file_path: str, format_type: str) -> None:
        self.exported.append((file_path, format_type))


def test_alias_and_replay_history() -> None:
    cli = PolarisInteractiveCLI(_DummyPolaris())

    first = cli.precmd("wm system-a")
    second = cli.precmd("!!")

    assert first == "worldmodel system-a"
    assert second == "worldmodel system-a"


def test_history_command_prints_recent_lines() -> None:
    cli = PolarisInteractiveCLI(_DummyPolaris())
    output: List[str] = []
    cli._print = lambda content, style=None: output.append(str(content))  # type: ignore[method-assign]

    _ = cli.precmd("status")
    _ = cli.precmd("systems")
    cli.do_history("2")

    combined = "\n".join(output)
    assert "status" in combined
    assert "systems" in combined


def test_export_supports_quoted_paths() -> None:
    polaris = _DummyPolaris()
    cli = PolarisInteractiveCLI(polaris)

    cli.do_export('"metrics report.json" json')

    assert polaris.exported == [("metrics report.json", "json")]


def test_system_id_completion_for_knowledge() -> None:
    cli = PolarisInteractiveCLI(_DummyPolaris())

    completions = cli.complete_knowledge("system", "knowledge system", len("knowledge "), 16)

    assert "system-a" in completions
    assert "system-b" in completions


def test_unknown_command_shows_suggestion() -> None:
    cli = PolarisInteractiveCLI(_DummyPolaris())
    output: List[str] = []
    cli._print = lambda content, style=None: output.append(str(content))  # type: ignore[method-assign]

    cli.default("sttus")

    assert any("Did you mean" in line and "status" in line for line in output)
