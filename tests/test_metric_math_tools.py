from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from polaris.tools.base import ToolDependencies
from polaris.tools.builtin import ComputeMetricMathTool, ListMetricFieldsTool


@dataclass
class _MV:
    value: float


@dataclass
class _State:
    system_id: str
    timestamp: datetime
    metrics: dict


class _KS:
    def __init__(self, states):
        self._states = list(states)

    async def query_states(self, system_id, start, end):
        return [s for s in self._states if s.system_id == system_id and start <= s.timestamp <= end]

    async def query_actions(self, system_id, start, end):
        return []


class _NullWorldModel:
    async def get_insights(self):
        return {}

    async def predict(self, candidate, state):
        return {}


def test_list_metric_fields_identifies_numeric_fields():
    now = datetime.now(timezone.utc)
    states = [
        _State("sys", now - timedelta(seconds=10), {"a": _MV(1.0), "b": _MV(2.0)}),
        _State("sys", now - timedelta(seconds=5), {"a": _MV(3.0), "c": _MV("x")}),
    ]
    deps = ToolDependencies(
        knowledge_store=_KS(states),
        world_model=_NullWorldModel(),
        metrics=None,
        connector=None,
        logger=None,
    )
    tool = ListMetricFieldsTool()
    out = asyncio.run(
        tool.execute({"window_seconds": 60, "limit": 50}, states[-1], None, deps)  # type: ignore[arg-type]
    )

    assert "fields" in out
    assert "numeric_fields" in out
    assert set(out["numeric_fields"]) >= {"a", "b"}
    assert "c" not in set(out["numeric_fields"])


def test_compute_metric_math_avg_and_delta():
    now = datetime.now(timezone.utc)
    states = [
        _State("sys", now - timedelta(seconds=10), {"x": _MV(1.0)}),
        _State("sys", now - timedelta(seconds=5), {"x": _MV(3.0)}),
        _State("sys", now, {"x": _MV(6.0)}),
    ]
    deps = ToolDependencies(
        knowledge_store=_KS(states),
        world_model=_NullWorldModel(),
        metrics=None,
        connector=None,
        logger=None,
    )
    tool = ComputeMetricMathTool()
    out_avg = asyncio.run(
        tool.execute({"metric": "x", "op": "avg", "window_seconds": 60}, states[-1], None, deps)  # type: ignore[arg-type]
    )
    assert out_avg["count"] == 3
    assert abs(out_avg["value"] - (1.0 + 3.0 + 6.0) / 3.0) < 1e-9

    out_delta = asyncio.run(
        tool.execute({"metric": "x", "op": "delta", "window_seconds": 60}, states[-1], None, deps)  # type: ignore[arg-type]
    )
    assert out_delta["value"] == pytest.approx(5.0)


def test_compute_metric_math_rejects_unsafe_expression():
    now = datetime.now(timezone.utc)
    states = [_State("sys", now, {"x": _MV(1.0)})]
    deps = ToolDependencies(
        knowledge_store=_KS(states),
        world_model=_NullWorldModel(),
        metrics=None,
        connector=None,
        logger=None,
    )
    tool = ComputeMetricMathTool()

    # Attribute access should be rejected by safety checks.
    out = asyncio.run(
        tool.execute(
            {
                "expression": "__import__('os').system('echo pwn')",
                "op": "avg",
                "window_seconds": 60,
            },
            states[-1],
            None,
            deps,
        )  # type: ignore[arg-type]
    )

    assert out.get("error_code") in {"unsafe_expression", "execution_error"}
