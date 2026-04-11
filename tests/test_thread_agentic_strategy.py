"""Tests for strict THREAD-inspired agentic strategy."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.abstractions.system_contract import SystemContract
from polaris.core.models import HealthStatus, MetricValue, SystemState
from polaris.strategies.action_resolution import StrictContractViolation
from polaris.strategies.thread_agentic import ThreadAgenticStrategy


class MockLLMResponse:
    def __init__(self, content: str):
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
    return AdaptationContext(
        system_id="test-system",
        historical_states=[],
        system_contract=SystemContract(
            system_id="test-system",
            connector_type="SWIMConnector",
            supported_action_types=("scale_up", "scale_down", "set_dimmer"),
            action_aliases={"add_server": "scale_up"},
        ),
    )


@pytest.mark.asyncio
async def test_thread_agentic_returns_action_from_root_final(
    strategy, sample_state, sample_context
):
    strategy.llm.generate.return_value = MockLLMResponse(
        json.dumps(
            {
                "final": {
                    "needs_adaptation": True,
                    "reasoning": "CPU high",
                    "actions": [{"type": "scale_up", "parameters": {"instances": 1}}],
                }
            }
        )
    )

    actions = await strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"


@pytest.mark.asyncio
async def test_thread_agentic_uses_explicit_aliases(strategy, sample_state, sample_context):
    strategy.llm.generate.return_value = MockLLMResponse(
        json.dumps(
            {
                "final": {
                    "needs_adaptation": True,
                    "reasoning": "Need capacity",
                    "actions": [{"type": "add_server", "parameters": {}}],
                }
            }
        )
    )

    actions = await strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"


@pytest.mark.asyncio
async def test_thread_agentic_spawn_join_child_feedback(strategy, sample_state, sample_context):
    strategy.max_thread_depth = 2
    strategy.llm.generate.side_effect = [
        MockLLMResponse(json.dumps({"spawn": {"objective": "investigate cpu trend"}})),
        MockLLMResponse(json.dumps({"final": {"return_payload": "cpu trend rising"}})),
        MockLLMResponse(
            json.dumps(
                {
                    "final": {
                        "needs_adaptation": True,
                        "reasoning": "child found rising trend",
                        "actions": [{"type": "scale_up", "parameters": {"instances": 2}}],
                    }
                }
            )
        ),
    ]

    actions = await strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"
    assert strategy.llm.generate.call_count == 3


@pytest.mark.asyncio
async def test_thread_agentic_step_limit_raises(strategy, sample_state, sample_context):
    strategy.steps_limit = 1
    strategy.llm.generate.return_value = MockLLMResponse(
        json.dumps({"tool": "get_recent_states", "args": {}})
    )
    strategy.knowledge_store.query_states = AsyncMock(return_value=[])

    with pytest.raises(StrictContractViolation, match="step limit"):
        await strategy.assess(sample_state, sample_context)


@pytest.mark.asyncio
async def test_thread_agentic_rejects_malformed_json(strategy, sample_state, sample_context):
    strategy.llm.generate.return_value = MockLLMResponse("not json")

    with pytest.raises(StrictContractViolation, match="valid JSON"):
        await strategy.assess(sample_state, sample_context)


@pytest.mark.asyncio
async def test_thread_agentic_requires_contract(strategy, sample_state):
    strategy.llm.generate.return_value = MockLLMResponse(
        json.dumps({"final": {"needs_adaptation": False, "reasoning": "stable", "actions": []}})
    )
    context_without_contract = AdaptationContext(system_id="test-system", historical_states=[])

    with pytest.raises(
        StrictContractViolation, match="Missing connector-supported action contract"
    ):
        await strategy.assess(sample_state, context_without_contract)


@pytest.mark.asyncio
async def test_thread_agentic_tool_result_payload_is_bounded(
    strategy, sample_state, sample_context
):
    strategy.steps_limit = 2
    strategy.max_tool_result_chars = 100

    captured_messages = []

    async def _fake_generate(messages, **kwargs):
        _ = kwargs
        captured_messages.append(messages)
        if len(captured_messages) == 1:
            return MockLLMResponse(json.dumps({"tool": "get_recent_states", "args": {}}))
        return MockLLMResponse(
            json.dumps(
                {
                    "final": {
                        "needs_adaptation": False,
                        "reasoning": "done",
                        "actions": [],
                    }
                }
            )
        )

    strategy.llm.generate = AsyncMock(side_effect=_fake_generate)
    strategy._tool_registry.execute = AsyncMock(return_value={"blob": "x" * 5000})

    actions = await strategy.assess(sample_state, sample_context)

    assert actions == []
    assert len(captured_messages) == 2
    payload = json.loads(captured_messages[1][-1].content)
    tool_data = payload["tool_result"]["data"]
    assert tool_data["_truncated"] is True
    assert tool_data["original_chars"] > 100


@pytest.mark.asyncio
async def test_thread_agentic_execute_tool_injects_connector(sample_state, sample_context):
    llm = Mock()
    llm.generate = AsyncMock()
    strategy = ThreadAgenticStrategy(llm_client=llm, knowledge_store=Mock(), world_model=Mock())

    connector = object()
    context_with_connector = AdaptationContext(
        system_id=sample_context.system_id,
        historical_states=sample_context.historical_states,
        system_contract=sample_context.system_contract,
        metadata={"connector": connector},
    )
    strategy._tool_registry.execute = AsyncMock(return_value={"ok": True})

    result = await strategy._execute_tool(
        "get_recent_states",
        {},
        sample_state,
        context_with_connector,
    )

    assert result == {"ok": True}
    deps = strategy._tool_registry.execute.await_args.kwargs["deps"]
    assert deps.connector is connector
