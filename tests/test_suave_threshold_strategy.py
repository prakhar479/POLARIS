"""Tests for SuaveThresholdStrategy."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.strategies.suave_threshold import SuaveThresholdStrategy


@pytest.fixture
def mock_logger():
    return Mock(spec=Logger)


@pytest.fixture
def mock_metrics():
    return Mock(spec=MetricsCollector)


@pytest.fixture
def suave_strategy(mock_logger, mock_metrics):
    return SuaveThresholdStrategy(
        trigger_visibility_below=1.0,
        trigger_thruster_failure_at_or_above=0.5,
        visibility_medium_at_or_above=1.0,
        visibility_high_at_or_above=2.0,
        cooldown_seconds=5,
        logger=mock_logger,
        metrics=mock_metrics,
    )


@pytest.fixture
def context():
    return AdaptationContext(
        system_id="suave-1",
        historical_states=[],
    )


@pytest.mark.asyncio
async def test_suave_no_action_needed_under_normal_conditions(suave_strategy, context):
    """When visibility is good and thrusters are fine, no action is emitted."""
    state = SystemState(
        system_id="suave-1",
        timestamp=datetime.now(timezone.utc),
        metrics={
            "water_visibility": MetricValue("water_visibility", 2.5),
            "thruster_failure_detected": MetricValue("thruster_failure_detected", 0.0),
        },
        health_status=HealthStatus.HEALTHY,
    )

    actions = await suave_strategy.assess(state, context)
    assert actions == []


@pytest.mark.asyncio
async def test_suave_low_visibility_triggers_spiral_low(suave_strategy, context):
    """Low visibility (< 1.0) triggers spiral_low and all_thrusters modes."""
    state = SystemState(
        system_id="suave-1",
        timestamp=datetime.now(timezone.utc),
        metrics={
            "water_visibility": MetricValue("water_visibility", 0.4),
            "thruster_failure_detected": MetricValue("thruster_failure_detected", 0.0),
        },
        health_status=HealthStatus.HEALTHY,
    )

    actions = await suave_strategy.assess(state, context)
    assert len(actions) == 2

    r1_action = next(
        a for a in actions if a.parameters["function_node"] == "f_generate_search_path"
    )
    r2_action = next(a for a in actions if a.parameters["function_node"] == "f_maintain_motion")

    assert r1_action.parameters["mode"] == "fd_spiral_low"
    assert r2_action.parameters["mode"] == "fd_all_thrusters"


@pytest.mark.asyncio
async def test_suave_thruster_failure_triggers_recovery(suave_strategy, context):
    """Thruster failure triggers recover_thrusters motion mode."""
    state = SystemState(
        system_id="suave-1",
        timestamp=datetime.now(timezone.utc),
        metrics={
            "water_visibility": MetricValue("water_visibility", 1.5),
            "thruster_failure_detected": MetricValue("thruster_failure_detected", 1.0),
        },
        health_status=HealthStatus.WARNING,
    )

    actions = await suave_strategy.assess(state, context)
    assert len(actions) == 2

    r1_action = next(
        a for a in actions if a.parameters["function_node"] == "f_generate_search_path"
    )
    r2_action = next(a for a in actions if a.parameters["function_node"] == "f_maintain_motion")

    # Visibility 1.5 is medium mode
    assert r1_action.parameters["mode"] == "fd_spiral_medium"
    assert r2_action.parameters["mode"] == "fd_recover_thrusters"


@pytest.mark.asyncio
async def test_suave_cooldown_blocks_frequent_assessment(suave_strategy, context):
    """Cooldown prevents consecutive actions from firing too rapidly."""
    state = SystemState(
        system_id="suave-1",
        timestamp=datetime.now(timezone.utc),
        metrics={
            "water_visibility": MetricValue("water_visibility", 0.2),
            "thruster_failure_detected": MetricValue("thruster_failure_detected", 0.0),
        },
        health_status=HealthStatus.WARNING,
    )

    actions1 = await suave_strategy.assess(state, context)
    assert len(actions1) == 2

    # Immediate second call should be blocked by cooldown
    actions2 = await suave_strategy.assess(state, context)
    assert actions2 == []


@pytest.mark.asyncio
async def test_suave_tunable_parameters_and_updates(suave_strategy):
    """Check get_tunable_parameters, update_parameter, and apply_config_update."""
    params = suave_strategy.get_tunable_parameters()
    assert "trigger_visibility_below" in params
    assert "cooldown_seconds" in params

    # Update valid parameter
    ok = await suave_strategy.update_parameter("trigger_visibility_below", 0.8)
    assert ok is True
    assert suave_strategy.trigger_visibility_below == 0.8

    ok = await suave_strategy.update_parameter("cooldown_seconds", 12)
    assert ok is True
    assert suave_strategy.cooldown_seconds == 12

    ok = await suave_strategy.update_parameter("unknown_param", 100)
    assert ok is False

    # Config update dictionary
    await suave_strategy.apply_config_update(
        {
            "trigger_thruster_failure_at_or_above": 0.75,
            "visibility_high_at_or_above": 3.0,
        }
    )
    assert suave_strategy.trigger_thruster_failure_at_or_above == 0.75
    assert suave_strategy.visibility_high_at_or_above == 3.0


@pytest.mark.asyncio
async def test_suave_action_execution_metrics(suave_strategy):
    """Test on_action_executed tracking success and total adaptations."""
    metrics_initial = await suave_strategy.get_performance_metrics()
    assert metrics_initial["success_rate"] == 0.0

    action = AdaptationAction(action_id="act-1", action_type="change_mode", target_system="suave-1")
    success_result = ExecutionResult(
        action_id="act-1", status=ExecutionStatus.SUCCESS, result_data={}
    )
    fail_result = ExecutionResult(action_id="act-1", status=ExecutionStatus.FAILED, result_data={})

    await suave_strategy.on_action_executed(action, success_result)
    await suave_strategy.on_action_executed(action, fail_result)

    perf = await suave_strategy.get_performance_metrics()
    assert perf["total_adaptations"] == 2.0
    assert perf["success_rate"] == 0.5
