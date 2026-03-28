"""Tests for THREAD-inspired recursive agentic strategy."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import HealthStatus, MetricValue, SystemState
from polaris.strategies.thread_agentic import ThreadAgenticStrategy


class MockLLMResponse:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def strategy():
    llm = Mock()
    llm.generate = AsyncMock()
    ks = Mock()
    wm = Mock()
    return ThreadAgenticStrategy(llm_client=llm, knowledge_store=ks, world_model=wm)


@pytest.fixture
def sample_state():
    return SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue("cpu", 91)},
        health_status=HealthStatus.HEALTHY,
    )


@pytest.fixture
def sample_context():
    return AdaptationContext(system_id="test-system", historical_states=[])


@pytest.mark.asyncio
async def test_thread_agentic_returns_action_from_root_final(
    strategy, sample_state, sample_context
):
    final = {
        "final": {
            "needs_adaptation": True,
            "reasoning": "CPU is consistently high",
            "actions": [{"type": "scale_up", "parameters": {"instances": 1}}],
        }
    }
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(final))

    actions = await strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"
    assert actions[0].parameters["llm_reasoning"] == "CPU is consistently high"
    assert actions[0].parameters["thread_count"] >= 1


@pytest.mark.asyncio
async def test_thread_agentic_spawn_join_child_feedback(strategy, sample_state, sample_context):
    strategy.max_thread_depth = 2
    strategy.llm.generate.side_effect = [
        MockLLMResponse(json.dumps({"spawn": {"objective": "investigate cpu trend"}})),
        MockLLMResponse(json.dumps({"final": {"return_payload": "cpu trend is rising"}})),
        MockLLMResponse(
            json.dumps(
                {
                    "final": {
                        "needs_adaptation": True,
                        "reasoning": "child thread found rising trend",
                        "actions": [{"type": "scale_up", "parameters": {"instances": 2}}],
                    }
                }
            )
        ),
    ]

    actions = await strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"
    assert actions[0].parameters["thread_count"] >= 2
    assert strategy.llm.generate.call_count == 3


@pytest.mark.asyncio
async def test_thread_agentic_respects_depth_limit(strategy, sample_state, sample_context):
    strategy.max_thread_depth = 0
    strategy.llm.generate.side_effect = [
        MockLLMResponse(json.dumps({"spawn": {"objective": "deep analysis"}})),
        MockLLMResponse(json.dumps({"final": {"needs_adaptation": False, "reasoning": "stable"}})),
    ]

    actions = await strategy.assess(sample_state, sample_context)

    assert actions == []
    assert strategy.llm.generate.call_count == 2


@pytest.mark.asyncio
async def test_thread_agentic_uses_tool_then_final(strategy, sample_state, sample_context):
    strategy.llm.generate.side_effect = [
        MockLLMResponse(json.dumps({"tool": "get_recent_states", "args": {"window_seconds": 300}})),
        MockLLMResponse(
            json.dumps({"final": {"needs_adaptation": False, "reasoning": "no change"}})
        ),
    ]
    strategy.knowledge_store.query_states = AsyncMock(return_value=[])

    actions = await strategy.assess(sample_state, sample_context)

    assert actions == []
    assert strategy.llm.generate.call_count == 2
    strategy.knowledge_store.query_states.assert_called_once()


@pytest.mark.asyncio
async def test_thread_agentic_step_limit_reached(strategy, sample_state, sample_context):
    strategy.steps_limit = 2
    strategy.llm.generate.return_value = MockLLMResponse(
        json.dumps({"tool": "get_recent_states", "args": {}})
    )
    strategy.knowledge_store.query_states = AsyncMock(return_value=[])

    actions = await strategy.assess(sample_state, sample_context)

    assert actions == []
    assert strategy.llm.generate.call_count == 2


@pytest.mark.asyncio
async def test_thread_agentic_apply_config_update(strategy):
    await strategy.apply_config_update(
        {
            "temperature": 0.6,
            "steps_limit": 7,
            "max_thread_depth": 4,
            "max_total_threads": 20,
            "tools": {"enabled": ["get_recent_states", "get_action_history"]},
            "listen_token": "[",
            "return_token": "]",
        }
    )

    assert strategy.temperature == 0.6
    assert strategy.steps_limit == 7
    assert strategy.max_thread_depth == 4
    assert strategy.max_total_threads == 20
    assert strategy.listen_token == "["
    assert strategy.return_token == "]"
    assert set(strategy._tool_registry.list_tools()) == {"get_recent_states", "get_action_history"}
