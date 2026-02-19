"""Tests for AgenticLLMStrategy with multiple actions."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import AdaptationAction, HealthStatus, MetricValue, SystemState
from polaris.strategies.agentic_llm import AgenticLLMStrategy


class MockLLMResponse:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def strategy():
    llm = Mock()
    llm.generate = AsyncMock()
    # We need a knowledge_store and world_model for AgenticLLMStrategy
    ks = Mock()
    wm = Mock()
    return AgenticLLMStrategy(llm_client=llm, knowledge_store=ks, world_model=wm)


@pytest.mark.asyncio
async def test_agentic_llm_returns_multiple_actions(strategy):
    """Test that AgenticLLMStrategy can return multiple actions from LLM reasoning."""
    # Setup LLM to return a final response with multiple actions
    final_response = {
        "final": {
            "needs_adaptation": True,
            "reasoning": "High load and high cost",
            "actions": [
                {"type": "scale_up", "parameters": {"instances": 1}},
                {"type": "optimize_cost", "parameters": {"mode": "aggressive"}},
            ],
        }
    }
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(final_response))

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue("cpu", 90)},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 2
    assert actions[0].action_type == "scale_up"
    assert actions[1].action_type == "optimize_cost"
    assert actions[0].parameters["llm_reasoning"] == "High load and high cost"


@pytest.mark.asyncio
async def test_agentic_llm_backward_compatibility(strategy):
    """Test that AgenticLLMStrategy still supports single 'action' field."""
    final_response = {
        "final": {
            "needs_adaptation": True,
            "reasoning": "One thing only",
            "action": {"type": "scale_up", "parameters": {}},
        }
    }
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(final_response))

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"


@pytest.mark.asyncio
async def test_agentic_llm_no_adaptation(strategy):
    """Test that AgenticLLMStrategy returns empty list when no adaptation is needed."""
    final_response = {"final": {"needs_adaptation": False, "reasoning": "Stable"}}
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(final_response))

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 0
