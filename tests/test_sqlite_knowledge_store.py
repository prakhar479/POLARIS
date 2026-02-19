"""Tests for SQLiteKnowledgeStore and NullMetricsCollector."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector
from polaris.knowledge.sqlite_store import SQLiteKnowledgeStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(system_id: str, metric_val: float, offset_seconds: int = 0) -> SystemState:
    ts = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return SystemState(
        system_id=system_id,
        timestamp=ts,
        metrics={
            "cpu": MetricValue(name="cpu", value=metric_val, unit="%", timestamp=ts),
        },
        health_status=HealthStatus.HEALTHY,
    )


def _make_action_result(system_id: str) -> tuple:
    action = AdaptationAction(
        action_id="test-action-1",
        action_type="scale_up",
        target_system=system_id,
        parameters={"replicas": 3},
    )
    result = ExecutionResult(
        action_id="test-action-1",
        status=ExecutionStatus.SUCCESS,
        result_data={"replicas_after": 3},
    )
    return action, result


# ---------------------------------------------------------------------------
# NullMetricsCollector
# ---------------------------------------------------------------------------


class TestNullMetricsCollector:
    def test_increment_does_not_raise(self) -> None:
        m = NullMetricsCollector()
        m.increment("any.metric")
        m.increment("any.metric", value=5.0, tags={"env": "test"})

    def test_gauge_does_not_raise(self) -> None:
        m = NullMetricsCollector()
        m.gauge("cpu.pct", 42.0)
        m.gauge("cpu.pct", 42.0, tags={"host": "h1"})

    def test_histogram_does_not_raise(self) -> None:
        m = NullMetricsCollector()
        m.histogram("latency_ms", 5.3)

    def test_get_summary_returns_empty_dict(self) -> None:
        m = NullMetricsCollector()
        m.increment("foo")
        m.gauge("bar", 1.0)
        assert m.get_summary() == {}

    def test_is_metrics_collector_subclass(self) -> None:
        from polaris.abstractions.observability import MetricsCollector

        assert isinstance(NullMetricsCollector(), MetricsCollector)


# ---------------------------------------------------------------------------
# SQLiteKnowledgeStore  (all tests use ":memory:" for isolation)
# ---------------------------------------------------------------------------


class TestSQLiteKnowledgeStore:
    @pytest.fixture()
    def store(self) -> SQLiteKnowledgeStore:
        return SQLiteKnowledgeStore(db_path=":memory:")

    def test_store_and_query_states(self, store: SQLiteKnowledgeStore) -> None:
        state = _make_state("web-01", 55.0)

        async def _run() -> None:
            await store.store_state(state)
            start = state.timestamp - timedelta(minutes=1)
            end = state.timestamp + timedelta(minutes=1)
            results = await store.query_states("web-01", start, end)
            assert len(results) == 1
            assert results[0].system_id == "web-01"
            assert abs(results[0].metrics["cpu"].value - 55.0) < 1e-6

        asyncio.run(_run())

    def test_query_states_empty_range(self, store: SQLiteKnowledgeStore) -> None:
        state = _make_state("web-01", 55.0)

        async def _run() -> None:
            await store.store_state(state)
            # Query a range in the distant past — should return nothing
            past = datetime(2000, 1, 1, tzinfo=timezone.utc)
            results = await store.query_states("web-01", past, past)
            assert results == []

        asyncio.run(_run())

    def test_query_states_wrong_system(self, store: SQLiteKnowledgeStore) -> None:
        state = _make_state("web-01", 55.0)

        async def _run() -> None:
            await store.store_state(state)
            start = state.timestamp - timedelta(minutes=1)
            end = state.timestamp + timedelta(minutes=1)
            results = await store.query_states("db-01", start, end)
            assert results == []

        asyncio.run(_run())

    def test_store_and_query_actions(self, store: SQLiteKnowledgeStore) -> None:
        action, result = _make_action_result("web-01")

        async def _run() -> None:
            await store.store_action(action, result)
            pairs = await store.query_actions("web-01", None, None)
            assert len(pairs) == 1
            stored_action, stored_result = pairs[0]
            assert stored_action.action_type == "scale_up"
            assert stored_result.status == ExecutionStatus.SUCCESS
            assert stored_result.result_data.get("replicas_after") == 3

        asyncio.run(_run())

    def test_max_states_pruning(self) -> None:
        store = SQLiteKnowledgeStore(db_path=":memory:", max_states_per_system=3)

        async def _run() -> None:
            for i in range(6):
                await store.store_state(_make_state("web-01", float(i), offset_seconds=i))

            start = datetime(2000, 1, 1, tzinfo=timezone.utc)
            end = datetime(2099, 1, 1, tzinfo=timezone.utc)
            results = await store.query_states("web-01", start, end)
            # Should be capped at 3
            assert len(results) == 3

        asyncio.run(_run())

    def test_multiple_systems_isolated(self, store: SQLiteKnowledgeStore) -> None:
        s1 = _make_state("web-01", 10.0, offset_seconds=0)
        s2 = _make_state("db-01", 20.0, offset_seconds=0)

        async def _run() -> None:
            await store.store_state(s1)
            await store.store_state(s2)
            start = datetime(2000, 1, 1, tzinfo=timezone.utc)
            end = datetime(2099, 1, 1, tzinfo=timezone.utc)
            r1 = await store.query_states("web-01", start, end)
            r2 = await store.query_states("db-01", start, end)
            assert len(r1) == 1 and r1[0].metrics["cpu"].value == 10.0
            assert len(r2) == 1 and r2[0].metrics["cpu"].value == 20.0

        asyncio.run(_run())

    def test_null_metrics_wired_in(self) -> None:
        """SQLiteKnowledgeStore should work fine with NullMetricsCollector."""
        null_m = NullMetricsCollector()
        store = SQLiteKnowledgeStore(db_path=":memory:", metrics=null_m)
        state = _make_state("sys-x", 77.0)

        async def _run() -> None:
            await store.store_state(state)
            start = state.timestamp - timedelta(minutes=1)
            end = state.timestamp + timedelta(minutes=1)
            results = await store.query_states("sys-x", start, end)
            assert len(results) == 1

        asyncio.run(_run())
