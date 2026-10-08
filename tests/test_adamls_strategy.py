"""Tests for AdaMLS baseline strategy."""

from datetime import datetime, timezone

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.strategies.adamls import AdaMLSStrategy


def _make_state(
    system_id: str = "switch",
    latency: float = 0.10,
    cpu: float = 50.0,
    confidence: float = 0.70,
    active_model: str = "yolov5m",
) -> SystemState:
    now = datetime.now(timezone.utc)
    metrics = {
        "response_time": MetricValue("response_time", latency, unit="s", timestamp=now),
        "cpu_usage": MetricValue("cpu_usage", cpu, unit="percent", timestamp=now),
        "confidence_mean": MetricValue("confidence_mean", confidence, timestamp=now),
    }
    return SystemState(
        system_id=system_id,
        timestamp=now,
        metrics=metrics,
        health_status=HealthStatus.HEALTHY,
        metadata={"active_model": active_model},
    )


@pytest.mark.asyncio
async def test_adamls_downgrade_on_latency_violation():
    strategy = AdaMLSStrategy(
        latency_sla=0.15,
        cpu_sla=70.0,
        cooldown_seconds=0.0,
    )
    context = AdaptationContext(system_id="switch", historical_states=[])

    # yolov5m with latency 0.18s (> 0.15s SLA) -> downgrade to yolov5s
    state = _make_state(latency=0.18, cpu=45.0, active_model="yolov5m")
    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    assert actions[0].action_type == "switch_model"
    assert actions[0].parameters["model_name"] == "yolov5s"
    assert "downgrade" in (actions[0].metadata or {}).get("reasoning", "").lower()


@pytest.mark.asyncio
async def test_adamls_downgrade_on_cpu_violation():
    strategy = AdaMLSStrategy(
        latency_sla=0.15,
        cpu_sla=70.0,
        cooldown_seconds=0.0,
    )
    context = AdaptationContext(system_id="switch", historical_states=[])

    # yolov5m with normal latency (0.09s) but excessive cpu (85% > 70%) -> downgrade to yolov5s
    state = _make_state(latency=0.09, cpu=85.0, active_model="yolov5m")
    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    assert actions[0].parameters["model_name"] == "yolov5s"


@pytest.mark.asyncio
async def test_adamls_upgrade_on_headroom_and_low_confidence():
    strategy = AdaMLSStrategy(
        latency_sla=0.15,
        cpu_sla=70.0,
        confidence_target=0.75,
        headroom_factor=0.70,
        cooldown_seconds=0.0,
    )
    context = AdaptationContext(system_id="switch", historical_states=[])

    # latency = 0.05 (< 0.15 * 0.70 = 0.105), cpu = 35% (< 49%), confidence = 0.60 (< 0.75)
    # yolov5s -> upgrade to yolov5m
    state = _make_state(latency=0.05, cpu=35.0, confidence=0.60, active_model="yolov5s")
    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    assert actions[0].parameters["model_name"] == "yolov5m"
    assert "upgrade" in (actions[0].metadata or {}).get("reasoning", "").lower()


@pytest.mark.asyncio
async def test_adamls_cooldown_enforcement():
    strategy = AdaMLSStrategy(
        latency_sla=0.15,
        cooldown_seconds=10.0,
    )
    context = AdaptationContext(system_id="switch", historical_states=[])

    # First assess triggers switch
    state = _make_state(latency=0.19, active_model="yolov5m")
    actions1 = await strategy.assess(state, context)
    assert len(actions1) == 1

    # Immediate second assess is blocked by cooldown
    actions2 = await strategy.assess(state, context)
    assert len(actions2) == 0


@pytest.mark.asyncio
async def test_adamls_on_action_executed():
    strategy = AdaMLSStrategy()
    action = AdaptationAction(
        action_id="act-adamls-1",
        action_type="switch_model",
        target_system="switch",
        parameters={"model_name": "yolov5l"},
    )
    result = ExecutionResult(
        action_id=action.action_id,
        status=ExecutionStatus.SUCCESS,
        result_data={},
    )

    await strategy.on_action_executed(action, result)
    assert strategy._current_model == "yolov5l"


@pytest.mark.asyncio
async def test_adamls_tunable_parameters_and_update():
    strategy = AdaMLSStrategy(latency_sla=0.15, cpu_sla=70.0)
    params = strategy.get_tunable_parameters()
    assert "latency_sla" in params

    assert "cpu_sla" in params
    assert "confidence_target" in params

    updated = await strategy.update_parameter("latency_sla", 0.12)
    assert updated is True
    assert strategy.latency_sla == 0.12

    with pytest.raises(ValueError):
        await strategy.update_parameter("latency_sla", -0.5)

    with pytest.raises(ValueError):
        await strategy.update_parameter("cpu_sla", 150.0)

    perf = await strategy.get_performance_metrics()
    assert "total_assessments" in perf
    assert "total_switches" in perf
