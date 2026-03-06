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


@pytest.mark.asyncio
async def test_agentic_llm_tool_usage_and_final_decision(strategy):
    """Test that AgenticLLMStrategy can use a tool before making a final decision."""
    # First response: use a tool
    tool_response = {"tool": "get_recent_states", "args": {"window_seconds": 300}}
    # Second response: making final decision
    final_response = {
        "final": {
            "needs_adaptation": True,
            "reasoning": "saw stuff",
            "actions": [{"type": "scale_up", "parameters": {}}],
        }
    }

    strategy.llm.generate.side_effect = [
        MockLLMResponse(json.dumps(tool_response)),
        MockLLMResponse(json.dumps(final_response)),
    ]

    # Mock the tool
    strategy.knowledge_store.query_states = AsyncMock(return_value=[])

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
    assert strategy.llm.generate.call_count == 2
    strategy.knowledge_store.query_states.assert_called_once()


@pytest.mark.asyncio
async def test_agentic_llm_step_limit_reached(strategy):
    """Test that AgenticLLMStrategy stops after step limit."""
    strategy.steps_limit = 2
    # Keep returning tool calls
    tool_response = {"tool": "get_recent_states", "args": {}}
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(tool_response))
    strategy.knowledge_store.query_states = AsyncMock(return_value=[])

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 0
    assert strategy.llm.generate.call_count == 2  # Hit limit


@pytest.mark.asyncio
async def test_agentic_llm_malformed_json_response(strategy):
    """Test behavior when LLM returns unparseable JSON."""
    strategy.llm.generate.return_value = MockLLMResponse("not json at all")

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 0


@pytest.mark.asyncio
async def test_agentic_llm_invalid_schema_response(strategy):
    """Test behavior when LLM returns JSON that does not match schema."""
    # missing both tool and final
    invalid_response = {"something_else": True}
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(invalid_response))

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 0


@pytest.mark.asyncio
async def test_agentic_llm_invalid_tool(strategy):
    """Test behavior when LLM tries to call an unallowed tool."""
    tool_response = {"tool": "hack_system", "args": {}}
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(tool_response))

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 0


@pytest.mark.asyncio
async def test_agentic_llm_tool_error(strategy):
    """Test behavior when a tool throws an exception."""
    tool_response = {"tool": "get_world_model_insights", "args": {}}
    final_response = {"final": {"needs_adaptation": False, "reasoning": "error occurred"}}

    strategy.llm.generate.side_effect = [
        MockLLMResponse(json.dumps(tool_response)),
        MockLLMResponse(json.dumps(final_response)),
    ]

    # Mock the tool to throw exception
    strategy.world_model.get_insights = AsyncMock(side_effect=Exception("Database down"))

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 0
    strategy.world_model.get_insights.assert_called_once()


@pytest.mark.asyncio
async def test_agentic_llm_on_action_executed(strategy):
    """Test tracking action execution metrics."""
    # Check initial
    metrics = await strategy.get_performance_metrics()
    assert metrics["success_rate"] == 0.0

    action = AdaptationAction(action_id="1", action_type="scale_up", target_system="sys")

    # Execute a success
    from polaris.core.models import ExecutionResult, ExecutionStatus

    res_success = ExecutionResult(action_id="1", status=ExecutionStatus.SUCCESS, result_data={})
    await strategy.on_action_executed(action, res_success)

    assert strategy._adaptation_count == 1
    assert strategy._success_count == 1

    # Execute a failure
    res_fail = ExecutionResult(action_id="1", status=ExecutionStatus.FAILED, result_data={})
    await strategy.on_action_executed(action, res_fail)

    assert strategy._adaptation_count == 2
    assert strategy._success_count == 1

    metrics = await strategy.get_performance_metrics()
    assert metrics["success_rate"] == 0.5
    assert metrics["total_adaptations"] == 2.0


@pytest.mark.asyncio
async def test_agentic_llm_update_parameter(strategy):
    """Test updating primitive parameters."""
    res = await strategy.update_parameter("temperature", 0.5)
    assert res is True
    assert strategy.temperature == 0.5

    res = await strategy.update_parameter("steps_limit", 5)
    assert res is True
    assert strategy.steps_limit == 5

    res = await strategy.update_parameter("unknown_param", "value")
    assert res is False


@pytest.mark.asyncio
async def test_agentic_llm_apply_config_update(strategy):
    """Test applying a broader configuration update."""
    config = {
        "temperature": 0.8,
        "steps_limit": 8,
        "system_prompt": "new system prompt",
        "per_system_prompts": {"sys1": "val"},
    }
    await strategy.apply_config_update(config)

    assert strategy.temperature == 0.8
    assert strategy.steps_limit == 8
    assert strategy._system_prompt_template == "new system prompt"
    assert strategy._per_system_prompts == {"sys1": "val"}
