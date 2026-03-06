"""Tests for HybridStrategy with multiple actions."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import AdaptationAction, HealthStatus, MetricValue, SystemState
from polaris.strategies.hybrid import HybridStrategy


class MockSubStrategy:
    def __init__(self, actions=None, confidence=0.8):
        self.actions = actions or []
        self.confidence = confidence
        self.on_action_executed = AsyncMock()

    async def assess(self, state, context):
        return self.actions


@pytest.fixture
def context():
    return AdaptationContext(system_id="test-system", historical_states=[])


@pytest.fixture
def state():
    return SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )


@pytest.mark.asyncio
async def test_hybrid_strategy_selects_all_actions_from_chosen_strategy(state, context):
    """Test that HybridStrategy returns the full list of actions from the selected strategy."""
    action1 = AdaptationAction(action_id="1", action_type="a", target_system="test-system")
    action2 = AdaptationAction(action_id="2", action_type="b", target_system="test-system")

    strat1 = MockSubStrategy(actions=[action1, action2])
    strat2 = MockSubStrategy(
        actions=[AdaptationAction(action_id="3", action_type="c", target_system="test-system")]
    )

    # Priority mode: strat2 has higher priority but strat1 has better outcomes?
    # Actually let's just use "first" mode for simplicity in one test
    hybrid = HybridStrategy(strategies=[(strat1, 10), (strat2, 5)], selection_mode="first")

    actions = await hybrid.assess(state, context)

    assert len(actions) == 2
    assert actions[0].action_id == "1"
    assert actions[1].action_id == "2"


@pytest.mark.asyncio
async def test_hybrid_strategy_priority_selection(state, context):
    """Test selection based on priority."""
    action_p1 = [AdaptationAction(action_id="p1", action_type="low", target_system="test-system")]
    action_p10 = [
        AdaptationAction(action_id="p10", action_type="high", target_system="test-system")
    ]

    strat_low = MockSubStrategy(actions=action_p1)
    strat_high = MockSubStrategy(actions=action_p10)

    hybrid = HybridStrategy(
        strategies=[(strat_low, 1), (strat_high, 10)], selection_mode="priority"
    )

    # We need to mock _estimate_confidence because it's called in priority/confidence modes
    hybrid._estimate_confidence = AsyncMock(return_value=0.9)

    actions = await hybrid.assess(state, context)

    assert len(actions) == 1
    assert actions[0].action_id == "p10"


@pytest.mark.asyncio
async def test_hybrid_strategy_confidence_selection(state, context):
    """Test selection based on confidence."""
    strat_low_conf = MockSubStrategy(
        actions=[AdaptationAction(action_id="l", action_type="l", target_system="test-system")]
    )
    strat_high_conf = MockSubStrategy(
        actions=[AdaptationAction(action_id="h", action_type="h", target_system="test-system")]
    )

    hybrid = HybridStrategy(
        strategies=[(strat_low_conf, 10), (strat_high_conf, 1)], selection_mode="confidence"
    )

    # Mock different confidences
    async def side_effect(strat, action, state):
        if strat == strat_low_conf:
            return 0.5
        return 0.9

    hybrid._estimate_confidence = AsyncMock(side_effect=side_effect)

    actions = await hybrid.assess(state, context)

    assert len(actions) == 1
    assert actions[0].action_id == "h"


@pytest.mark.asyncio
async def test_hybrid_strategy_get_performance_metrics():
    """Test getting performance metrics."""
    strat = MockSubStrategy()
    hybrid = HybridStrategy(strategies=[(strat, 1)])

    # Initially 0
    metrics = await hybrid.get_performance_metrics()
    assert metrics.get("success_rate") is None
    assert metrics.get("strategy_0_usage") == 0.0

    # Simulate some usage
    hybrid._adaptation_count = 2
    hybrid._success_count = 1
    hybrid._strategy_usage[0] = 2

    metrics = await hybrid.get_performance_metrics()
    assert metrics["success_rate"] == 0.5
    assert metrics["total_adaptations"] == 2.0
    assert metrics["strategy_0_usage"] == 2.0


@pytest.mark.asyncio
async def test_hybrid_strategy_on_action_executed():
    """Test tracking action execution propagates to sub-strategies."""
    strat1 = MockSubStrategy()
    strat2 = MockSubStrategy()
    hybrid = HybridStrategy(strategies=[(strat1, 10), (strat2, 5)])

    action = AdaptationAction(action_id="1", action_type="test", target_system="sys")

    from polaris.core.models import ExecutionResult, ExecutionStatus

    res_success = ExecutionResult(action_id="1", status=ExecutionStatus.SUCCESS, result_data={})

    await hybrid.on_action_executed(action, res_success)

    assert hybrid._adaptation_count == 1
    assert hybrid._success_count == 1

    strat1.on_action_executed.assert_called_once_with(action, res_success)
    strat2.on_action_executed.assert_called_once_with(action, res_success)


@pytest.mark.asyncio
async def test_hybrid_strategy_estimate_confidence_fallback(state):
    """Test fallback confidence when sub-strategy lacks metrics."""
    strat = MockSubStrategy()
    # Mocking get_performance_metrics to return something invalid or raise
    strat.get_performance_metrics = AsyncMock(side_effect=Exception("No metrics"))

    hybrid = HybridStrategy(strategies=[(strat, 1)])
    action = AdaptationAction(action_id="1", action_type="test", target_system="sys")

    conf = await hybrid._estimate_confidence(strat, action, state)
    assert conf == 0.7  # Default


@pytest.mark.asyncio
async def test_hybrid_strategy_estimate_confidence_with_metrics(state):
    """Test confidence estimation based on sub-strategy metrics."""
    strat = MockSubStrategy()
    strat.get_performance_metrics = AsyncMock(return_value={"success_rate": 0.8})

    hybrid = HybridStrategy(strategies=[(strat, 1)])
    action = AdaptationAction(action_id="1", action_type="test", target_system="sys")

    conf = await hybrid._estimate_confidence(strat, action, state)
    # 0.6 + 0.4 * 0.8 = 0.92
    assert conf == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_hybrid_strategy_parameter_updates():
    """Test delegating parameter updates."""
    strat = MockSubStrategy()
    strat.update_parameter = AsyncMock(return_value=True)

    hybrid = HybridStrategy(strategies=[(strat, 1)])

    # Update hybrid param
    res = await hybrid.update_parameter("selection_mode", "priority")
    assert res is True
    assert hybrid.selection_mode == "priority"

    res = await hybrid.update_parameter("min_confidence", 0.5)
    assert res is True
    assert hybrid.min_confidence == 0.5

    # Delegate to sub-strategy
    res = await hybrid.update_parameter("strategy_0.some_param", "value")
    assert res is True
    strat.update_parameter.assert_called_once_with("some_param", "value")

    # Unknown
    res = await hybrid.update_parameter("unknown", "val")
    assert res is False


@pytest.mark.asyncio
async def test_hybrid_strategy_apply_config_update():
    """Test applying a broader config update to itself and sub-strategies."""

    class MockUpdateStrategy(MockSubStrategy):
        def __init__(self, actions=None, confidence=0.8):
            super().__init__(actions, confidence)
            self.update_parameter = AsyncMock()

    strat = MockUpdateStrategy()
    hybrid = HybridStrategy(strategies=[(strat, 1)])

    config = {
        "selection_mode": "first",
        "min_confidence": 0.9,
        "strategies": [
            {
                "type": "threshold",
                "threshold": {"cooldown_seconds": 30, "thresholds": {"cpu": {"high": 80}}},
            }
        ],
    }

    await hybrid.apply_config_update(config)

    assert hybrid.selection_mode == "first"
    assert hybrid.min_confidence == 0.9
    strat.update_parameter.assert_any_call("cooldown_seconds", 30)
    strat.update_parameter.assert_any_call("thresholds.cpu.high", 80)


@pytest.mark.asyncio
async def test_hybrid_strategy_assess_no_proposals(state, context):
    """Test assess when no sub-strategies return valid actions."""
    strat = MockSubStrategy(actions=[])
    hybrid = HybridStrategy(strategies=[(strat, 1)])

    actions = await hybrid.assess(state, context)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_hybrid_strategy_assess_exception_in_sub_strategy(state, context):
    """Test assess when sub-strategy throws an exception."""
    strat = MockSubStrategy()
    strat.assess = AsyncMock(side_effect=Exception("Failed"))

    hybrid = HybridStrategy(strategies=[(strat, 1)])

    actions = await hybrid.assess(state, context)
    assert len(actions) == 0
