"""Tests for shared strategy runtime utilities."""

from datetime import datetime, timezone

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import HealthStatus, MetricValue, SystemState
from polaris.strategies.utils import (
    bounded_tool_data,
    build_tool_result_message,
    execute_strategy_tool,
    extract_connector_from_context,
)


class _ToolRegistrySuccess:
    async def execute(self, **kwargs):
        _ = kwargs
        return {"ok": True}


class _ToolRegistryFailure:
    async def execute(self, **kwargs):
        _ = kwargs
        raise RuntimeError("tool exploded")


@pytest.fixture
def sample_state() -> SystemState:
    return SystemState(
        system_id="sys-1",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue("cpu", 0.5)},
        health_status=HealthStatus.HEALTHY,
    )


def test_extract_connector_from_context_prefers_typed_field():
    typed_connector = object()
    metadata_connector = object()
    context = AdaptationContext(
        system_id="sys-1",
        historical_states=[],
        connector=typed_connector,
        metadata={"connector": metadata_connector},
    )

    assert extract_connector_from_context(context) is typed_connector


def test_extract_connector_from_context_falls_back_to_metadata():
    metadata_connector = object()
    context = AdaptationContext(
        system_id="sys-1",
        historical_states=[],
        metadata={"connector": metadata_connector},
    )

    assert extract_connector_from_context(context) is metadata_connector


def test_build_tool_result_message_is_bounded():
    payload = {"blob": "x" * 5000}
    msg = build_tool_result_message("demo", payload, max_chars=120)

    assert "tool_result" in msg
    bounded = bounded_tool_data(payload, max_chars=120)
    assert bounded["_truncated"] is True
    assert bounded["original_chars"] > 120


@pytest.mark.asyncio
async def test_execute_strategy_tool_records_success_metric(
    sample_state, mock_logger, mock_metrics
):
    context = AdaptationContext(system_id="sys-1", historical_states=[])

    result = await execute_strategy_tool(
        tool_registry=_ToolRegistrySuccess(),
        tool_name="get_recent_states",
        args={},
        state=sample_state,
        context=context,
        knowledge_store=object(),
        world_model=object(),
        logger=mock_logger,
        metrics=mock_metrics,
        metric_prefix="polaris.strategy.agentic",
        error_log_message="tool error",
    )

    assert result == {"ok": True}
    assert (
        "increment",
        "polaris.strategy.agentic.tool_called",
        1.0,
        {"tool": "get_recent_states", "system_id": "sys-1"},
    ) in mock_metrics.metrics


@pytest.mark.asyncio
async def test_execute_strategy_tool_returns_error_payload(sample_state, mock_logger, mock_metrics):
    context = AdaptationContext(system_id="sys-1", historical_states=[])

    result = await execute_strategy_tool(
        tool_registry=_ToolRegistryFailure(),
        tool_name="get_recent_states",
        args={},
        state=sample_state,
        context=context,
        knowledge_store=object(),
        world_model=object(),
        logger=mock_logger,
        metrics=mock_metrics,
        metric_prefix="polaris.strategy.agentic",
        error_log_message="tool error",
    )

    assert "error" in result
    assert "tool_error: RuntimeError" in result["error"]
    assert (
        "increment",
        "polaris.strategy.agentic.tool_error",
        1.0,
        {"tool": "get_recent_states", "system_id": "sys-1"},
    ) in mock_metrics.metrics
