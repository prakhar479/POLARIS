"""Tests for strict AgenticLLMStrategy behavior."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext
from polaris.abstractions.system_contract import SystemContract
from polaris.abstractions.world_model import WorldModel
from polaris.core.models import AdaptationAction, HealthStatus, MetricValue, SystemState
from polaris.infrastructure.llm import LLMClient
from polaris.strategies.action_resolution import StrictContractViolation
from polaris.strategies.agentic_llm import AgenticLLMStrategy


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.fixture
def strategy():
    llm = Mock()
    llm.generate = AsyncMock()
    ks = Mock()
    wm = Mock()
    return AgenticLLMStrategy(llm_client=llm, knowledge_store=ks, world_model=wm)


@pytest.fixture
def state():
    return SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue("cpu", 90)},
        health_status=HealthStatus.HEALTHY,
    )


@pytest.fixture
def base_deps():
    return {
        "llm_client": AsyncMock(spec=LLMClient),
        "knowledge_store": Mock(spec=KnowledgeStore),
        "world_model": Mock(spec=WorldModel),
        "logger": Mock(spec=Logger),
        "metrics": Mock(spec=MetricsCollector),
    }


@pytest.fixture
def context():
    contract = SystemContract(
        system_id="test-system",
        connector_type="test",
        supported_action_types=("scale_up", "scale_down", "test_action"),
        action_aliases={"add_server": "scale_up"},
    )
    return AdaptationContext(
        system_id="test-system", historical_states=[], system_contract=contract
    )


@pytest.mark.asyncio
async def test_agentic_llm_returns_multiple_actions(strategy, state, context):
    final_response = {
        "final": {
            "needs_adaptation": True,
            "reasoning": "High load",
            "actions": [
                {"type": "scale_up", "parameters": {"instances": 1}},
                {"type": "scale_down", "parameters": {"instances": 1}},
            ],
        }
    }
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(final_response))

    actions = await strategy.assess(state, context)

    assert len(actions) == 2
    assert actions[0].action_type == "scale_up"
    assert actions[1].action_type == "scale_down"


@pytest.mark.asyncio
async def test_agentic_llm_accepts_explicit_contract_aliases(strategy, state, context):
    final_response = {
        "final": {
            "needs_adaptation": True,
            "reasoning": "Need capacity",
            "actions": [{"type": "add_server", "parameters": {"instances": 1}}],
        }
    }
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(final_response))

    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"


@pytest.mark.asyncio
async def test_agentic_llm_no_adaptation(strategy, state, context):
    final_response = {"final": {"needs_adaptation": False, "reasoning": "Stable"}}
    strategy.llm.generate.return_value = MockLLMResponse(json.dumps(final_response))

    actions = await strategy.assess(state, context)

    assert actions == []


@pytest.mark.asyncio
async def test_agentic_llm_uses_tool_then_final(strategy, state, context):
    strategy.llm.generate.side_effect = [
        MockLLMResponse(json.dumps({"tool": "get_recent_states", "args": {"window_seconds": 300}})),
        MockLLMResponse(
            json.dumps(
                {
                    "final": {
                        "needs_adaptation": True,
                        "reasoning": "Need change",
                        "actions": [{"type": "scale_up", "parameters": {}}],
                    }
                }
            )
        ),
    ]
    strategy.knowledge_store.query_states = AsyncMock(return_value=[])

    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    strategy.knowledge_store.query_states.assert_called_once()


@pytest.mark.asyncio
async def test_agentic_llm_rejects_malformed_json(strategy, state, context):
    strategy.llm.generate.return_value = MockLLMResponse("not json")

    with pytest.raises(StrictContractViolation, match="valid JSON"):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_agentic_llm_rejects_disallowed_tool(strategy, state, context):
    strategy.llm.generate.return_value = MockLLMResponse(
        json.dumps({"tool": "hack_system", "args": {}})
    )

    with pytest.raises(StrictContractViolation, match="not in allowed tool list"):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_agentic_llm_requires_contract(strategy, state):
    strategy.llm.generate.return_value = MockLLMResponse(
        json.dumps({"final": {"needs_adaptation": False, "reasoning": "Stable"}})
    )
    context_without_contract = AdaptationContext(system_id="test-system", historical_states=[])

    with pytest.raises(
        StrictContractViolation, match="Missing connector-supported action contract"
    ):
        await strategy.assess(state, context_without_contract)


@pytest.mark.asyncio
async def test_agentic_llm_step_limit_raises(strategy, state, context):
    strategy.steps_limit = 1
    strategy.llm.generate.return_value = MockLLMResponse(
        json.dumps({"tool": "get_recent_states", "args": {}})
    )
    strategy.knowledge_store.query_states = AsyncMock(return_value=[])

    with pytest.raises(StrictContractViolation, match="step limit"):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_agentic_llm_on_action_executed(strategy):
    action = AdaptationAction(action_id="1", action_type="scale_up", target_system="sys")
    from polaris.core.models import ExecutionResult, ExecutionStatus

    res_success = ExecutionResult(action_id="1", status=ExecutionStatus.SUCCESS, result_data={})
    await strategy.on_action_executed(action, res_success)

    assert strategy._adaptation_count == 1
    assert strategy._success_count == 1


@pytest.mark.asyncio
async def test_assess_schema_validation_fails(base_deps, state, context):
    base_deps["llm_client"].generate.return_value = Mock(content='{"final": "not an object"}')
    strategy = AgenticLLMStrategy(**base_deps)

    with pytest.raises(StrictContractViolation, match="Agentic response failed schema validation"):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_assess_final_empty_reasoning(base_deps, state, context):
    resp = '{"final": {"needs_adaptation": true, "reasoning": "   ", "actions": []}}'
    base_deps["llm_client"].generate.return_value = Mock(content=resp)
    strategy = AgenticLLMStrategy(**base_deps)

    with pytest.raises(
        StrictContractViolation, match="Agentic final response requires non-empty 'reasoning'"
    ):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_assess_final_needs_adaptation_no_actions(base_deps, state, context):
    resp = '{"final": {"needs_adaptation": true, "reasoning": "test", "actions": []}}'
    base_deps["llm_client"].generate.return_value = Mock(content=resp)
    strategy = AgenticLLMStrategy(**base_deps)

    with pytest.raises(StrictContractViolation, match="requires non-empty 'actions'"):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_assess_action_missing_type(base_deps, state, context):
    resp = '{"final": {"needs_adaptation": true, "reasoning": "test", "actions": [{"type": "", "parameters": {}}]}}'
    base_deps["llm_client"].generate.return_value = Mock(content=resp)
    strategy = AgenticLLMStrategy(**base_deps)

    with pytest.raises(StrictContractViolation, match="Agentic action requires non-empty 'type'"):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_assess_action_invalid_type(base_deps, state, context):
    resp = '{"final": {"needs_adaptation": true, "reasoning": "test", "actions": [{"type": "bad", "parameters": {}}]}}'
    base_deps["llm_client"].generate.return_value = Mock(content=resp)
    strategy = AgenticLLMStrategy(**base_deps)

    with pytest.raises(StrictContractViolation, match="Unsupported action type 'bad'"):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_assess_tool_error(base_deps, state, context):
    # Simulate first tool call throwing an error -> next step returns final to avoid timeout
    resp1 = '{"tool": "get_system_status", "args": {}}'
    resp2 = '{"final": {"needs_adaptation": false, "reasoning": "test", "actions": []}}'
    base_deps["llm_client"].generate.side_effect = [Mock(content=resp1), Mock(content=resp2)]

    strategy = AgenticLLMStrategy(**base_deps, allowed_tools=["get_system_status"])
    # mock _tool_registry to raise an exception
    strategy._tool_registry.execute = AsyncMock(side_effect=Exception("Tool failure"))

    # should recover and log error, then return []
    res = await strategy.assess(state, context)
    assert res == []
    base_deps["logger"].error.assert_called_with(
        "Agentic tool execution error", tool="get_system_status", error="Tool failure"
    )


@pytest.mark.asyncio
async def test_parameters(base_deps):
    strategy = AgenticLLMStrategy(**base_deps)
    params = strategy.get_tunable_parameters()
    assert "temperature" in params
    assert "steps_limit" in params

    assert await strategy.update_parameter("temperature", 1.0)
    assert strategy.temperature == 1.0

    assert await strategy.update_parameter("steps_limit", 5)
    assert strategy.steps_limit == 5

    assert not await strategy.update_parameter("invalid", "value")


@pytest.mark.asyncio
async def test_apply_config_update(base_deps):
    strategy = AgenticLLMStrategy(**base_deps)

    base_deps["llm_client"].update_resilience = Mock()
    config = {
        "temperature": 0.5,
        "steps_limit": 2,
        "system_prompt": "Global: {system_id}",
        "per_system_prompts": {"sys1": "Override: {system_id}"},
        "tools": {"enabled": ["get_system_status"]},
        "resilience": {"retries": 3},
    }

    await strategy.apply_config_update(config)
    assert strategy.temperature == 0.5
    assert strategy.steps_limit == 2
    assert strategy.allowed_tools == ["get_system_status"]
    base_deps["llm_client"].update_resilience.assert_called_with({"retries": 3})

    # testing _system_prompt formatting
    res1 = strategy._system_prompt("sys1", ["test"])
    assert res1 == "Override: sys1"

    res2 = strategy._system_prompt("sys2", ["test"])
    assert res2 == "Global: sys2"


@pytest.mark.asyncio
async def test_get_performance_metrics(base_deps):
    strategy = AgenticLLMStrategy(**base_deps)
    assert (await strategy.get_performance_metrics())["success_rate"] == 0.0

    # test some successful adaptation
    class ActionStatus:
        value = "success"

    class ExecRes:
        status = ActionStatus()

    await strategy.on_action_executed(Mock(action_type="scale_up", target_system="sys"), ExecRes())
    metrics = await strategy.get_performance_metrics()
    assert metrics["success_rate"] == 1.0
    assert metrics["total_adaptations"] == 1.0


def test_maybe_log_llm_response(base_deps, monkeypatch):
    strategy = AgenticLLMStrategy(**base_deps)
    monkeypatch.setenv("POLARIS_LOG_LLM_RAW", "1")
    monkeypatch.setenv("POLARIS_LOG_LLM_RAW_MAX_CHARS", "10")

    long_content = "1234567890_extra_text"
    strategy._maybe_log_llm_response("sys", 1, long_content)

    base_deps["logger"].debug.assert_called_once()
    call_args = base_deps["logger"].debug.call_args[1]
    assert call_args["llm_raw"].startswith("1234567890\n...<truncated")

    # invalid char int
    monkeypatch.setenv("POLARIS_LOG_LLM_RAW_MAX_CHARS", "invalid")
    strategy._maybe_log_llm_response("sys", 2, long_content)
    # Falls back to 4000
    assert "extra_text" in base_deps["logger"].debug.call_args[1]["llm_raw"]
