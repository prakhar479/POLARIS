"""Tests for LLMReasoningStrategy parsing logic."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

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
from polaris.strategies.llm_reasoning import LLMReasoningStrategy


@pytest.fixture
def strategy():
    llm = Mock()
    llm.generate = AsyncMock()
    return LLMReasoningStrategy(llm_client=llm)


def test_parse_response_single_action(strategy):
    """Test parsing a single 'action' for backward compatibility."""
    response = json.dumps(
        {
            "needs_adaptation": True,
            "reasoning": "High load",
            "action": {"type": "scale_up", "parameters": {"instances": 2}},
        }
    )

    actions = strategy._parse_response(response, "test-system")

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"
    assert actions[0].parameters["instances"] == 2
    assert actions[0].parameters["llm_reasoning"] == "High load"


def test_parse_response_multiple_actions(strategy):
    """Test parsing 'actions' list."""
    response = json.dumps(
        {
            "needs_adaptation": True,
            "reasoning": "Multiple issues",
            "actions": [
                {"type": "scale_up", "parameters": {"instances": 1}},
                {"type": "adjust_qos", "parameters": {"level": "low"}},
            ],
        }
    )

    actions = strategy._parse_response(response, "test-system")

    assert len(actions) == 2
    assert actions[0].action_type == "scale_up"
    assert actions[1].action_type == "adjust_qos"
    assert actions[0].parameters["llm_reasoning"] == "Multiple issues"
    assert actions[1].parameters["llm_reasoning"] == "Multiple issues"


def test_parse_response_no_adaptation(strategy):
    """Test parsing when no adaptation is needed."""
    response = json.dumps({"needs_adaptation": False, "reasoning": "Everything fine"})

    actions = strategy._parse_response(response, "test-system")

    assert len(actions) == 0


def test_parse_response_malformed_json(strategy):
    """Test robustness against malformed JSON."""
    response = "This is not JSON"
    actions = strategy._parse_response(response, "test-system")
    assert len(actions) == 0


def test_parse_response_json_in_markdown(strategy):
    """Test parsing JSON inside markdown blocks."""
    response = (
        'Here is the plan: ```json\n{"needs_adaptation": true, "actions": [{"type": "test"}]}\n```'
    )
    actions = strategy._parse_response(response, "test-system")
    assert len(actions) == 1
    assert actions[0].action_type == "test"


@pytest.mark.asyncio
async def test_assess_success_single_action(strategy):
    """Test assess method when LLM successfully returns a single action."""
    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue(name="cpu", value=90.0, unit="%")},
        health_status=HealthStatus.UNHEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    response_content = json.dumps(
        {
            "needs_adaptation": True,
            "reasoning": "High CPU",
            "action": {"type": "scale_up", "parameters": {"instances": 2}},
        }
    )

    mock_response = Mock()
    mock_response.content = response_content
    strategy.llm.generate.return_value = mock_response

    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"
    assert actions[0].parameters["instances"] == 2
    assert actions[0].parameters["llm_reasoning"] == "High CPU"
    strategy.llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_assess_success_no_adaptation(strategy):
    """Test assess method when LLM decides no adaptation is needed."""
    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue(name="cpu", value=50.0, unit="%")},
        health_status=HealthStatus.HEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    response_content = json.dumps(
        {
            "needs_adaptation": False,
            "reasoning": "System is healthy",
        }
    )

    mock_response = Mock()
    mock_response.content = response_content
    strategy.llm.generate.return_value = mock_response

    actions = await strategy.assess(state, context)

    assert len(actions) == 0
    strategy.llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_assess_llm_error(strategy):
    """Test assess method when LLM raises an exception."""
    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue(name="cpu", value=90.0, unit="%")},
        health_status=HealthStatus.UNHEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])

    strategy.llm.generate.side_effect = Exception("LLM connection failed")

    # The method should catch the exception and return an empty list
    actions = await strategy.assess(state, context)

    assert len(actions) == 0
    strategy.llm.generate.assert_called_once()


def test_get_system_prompt_overrides(strategy):
    """Test system prompt overrides."""
    strategy._per_system_prompts = {"test-system": "Custom per-system prompt"}
    assert strategy._get_system_prompt("test-system") == "Custom per-system prompt"

    strategy._per_system_prompts = {}
    strategy._system_prompt_template = "Custom template for {system_id}"
    assert strategy._get_system_prompt("test-system") == "Custom template for test-system"


@pytest.mark.asyncio
async def test_on_action_executed_success(strategy):
    """Test tracking successful action execution."""
    action = AdaptationAction(
        action_id="test-action", action_type="scale_up", target_system="test-system"
    )
    result = ExecutionResult(
        action_id="test-action", status=ExecutionStatus.SUCCESS, result_data={}
    )

    await strategy.on_action_executed(action, result)

    assert strategy._adaptation_count == 1
    assert strategy._success_count == 1


@pytest.mark.asyncio
async def test_on_action_executed_failure(strategy):
    """Test tracking failed action execution."""
    action = AdaptationAction(
        action_id="test-action", action_type="scale_up", target_system="test-system"
    )
    result = ExecutionResult(action_id="test-action", status=ExecutionStatus.FAILED, result_data={})

    await strategy.on_action_executed(action, result)

    assert strategy._adaptation_count == 1
    assert strategy._success_count == 0


@pytest.mark.asyncio
async def test_get_performance_metrics(strategy):
    """Test getting performance metrics."""
    metrics_initial = await strategy.get_performance_metrics()
    assert metrics_initial["success_rate"] == 0.0

    # Simulate success
    strategy._adaptation_count = 2
    strategy._success_count = 1

    metrics = await strategy.get_performance_metrics()
    assert metrics["success_rate"] == 0.5
    assert metrics["total_adaptations"] == 2.0


@pytest.mark.asyncio
async def test_parameter_updates(strategy):
    """Test updating strategy parameters."""
    # Test temperature update
    updated = await strategy.update_parameter("temperature", 0.5)
    assert updated is True
    assert strategy.temperature == 0.5

    # Test description update
    updated = await strategy.update_parameter("system_description", "New desc")
    assert updated is True
    assert strategy.system_description == "New desc"

    # Test unknown parameter
    updated = await strategy.update_parameter("unknown", "value")
    assert updated is False


@pytest.mark.asyncio
async def test_apply_config_update(strategy):
    """Test applying a batch config update."""
    config = {
        "temperature": 0.8,
        "system_description": "Config desc",
        "system_prompt": "New system prompt",
        "per_system_prompts": {"sys1": "val1"},
    }

    await strategy.apply_config_update(config)

    assert strategy.temperature == 0.8
    assert strategy.system_description == "Config desc"
    assert strategy._system_prompt_template == "New system prompt"
    assert strategy._per_system_prompts == {"sys1": "val1"}
