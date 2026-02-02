import asyncio
import types
import pytest
from datetime import datetime, timezone

from polaris.strategies.agentic_llm import AgenticLLMStrategy
from polaris.infrastructure.llm import LLMMessage
from polaris.knowledge import InMemoryKnowledgeStore
from polaris.world_model import StatisticalWorldModel
from polaris.core.models import SystemState, MetricValue, HealthStatus, AdaptationAction, ExecutionResult, ExecutionStatus
from polaris.abstractions.strategy import AdaptationContext


class DummyLLM:
    async def generate(self, messages, temperature=0.1, max_tokens=256):
        # Not used in these tests; required by interface
        return types.SimpleNamespace(content="{}", model="dummy")


class DummyConnector:
    def __init__(self, actions):
        self._actions = actions

    async def get_supported_actions(self):
        return self._actions


@pytest.mark.asyncio
async def test_list_supported_actions_uses_connector():
    # Prepare
    ks = InMemoryKnowledgeStore()
    wm = StatisticalWorldModel(ks)
    connector = DummyConnector(actions=[types.SimpleNamespace(action_type="scale_up"), types.SimpleNamespace(action_type="scale_down")])

    strategy = AgenticLLMStrategy(
        llm_client=DummyLLM(),
        knowledge_store=ks,
        world_model=wm,
        connector_getter=lambda sid: connector,
        steps_limit=1,
    )

    state = SystemState(
        system_id="sys",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu_usage": MetricValue(name="cpu_usage", value=75.0)},
        health_status=HealthStatus.HEALTHY,
    )
    ctx = AdaptationContext(system_id="sys", historical_states=[], world_model_insights=None)

    # Act
    result = await strategy._execute_tool("list_supported_actions", {}, state, ctx)

    # Assert
    assert result.get("source") == "connector"
    assert set(result.get("action_types", [])) == {"scale_up", "scale_down"}


@pytest.mark.asyncio
async def test_list_supported_actions_falls_back_to_history():
    # Prepare knowledge store with one historical action
    ks = InMemoryKnowledgeStore()
    wm = StatisticalWorldModel(ks)

    state = SystemState(
        system_id="sys",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu_usage": MetricValue(name="cpu_usage", value=60.0)},
        health_status=HealthStatus.HEALTHY,
    )

    action = AdaptationAction(action_id="a1", action_type="adjust_qos", target_system="sys", parameters={"level": "high"})
    result = ExecutionResult(action_id="a1", status=ExecutionStatus.SUCCESS, result_data={})
    await ks.store_action(action, result)

    strategy = AgenticLLMStrategy(
        llm_client=DummyLLM(),
        knowledge_store=ks,
        world_model=wm,
        connector_getter=lambda sid: None,
        steps_limit=1,
    )

    ctx = AdaptationContext(system_id="sys", historical_states=[], world_model_insights=None)

    # Act
    out = await strategy._execute_tool("list_supported_actions", {}, state, ctx)

    # Assert
    assert out.get("source") == "historical"
    assert "adjust_qos" in set(out.get("action_types", []))
