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
