import types
from datetime import datetime, timezone

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.abstractions.system_contract import SystemContract
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.knowledge import InMemoryKnowledgeStore
from polaris.strategies.agentic_llm import AgenticLLMStrategy
from polaris.world_model import StatisticalWorldModel


class DummyLLM:
    async def generate(self, messages, temperature=0.1, max_tokens=256):
        # Not used in these tests; required by interface
        return types.SimpleNamespace(content="{}", model="dummy")


@pytest.mark.asyncio
async def test_list_supported_actions_uses_contract():
    # Prepare
    ks = InMemoryKnowledgeStore()
    wm = StatisticalWorldModel(ks)

    strategy = AgenticLLMStrategy(
        llm_client=DummyLLM(),
        knowledge_store=ks,
        world_model=wm,
        steps_limit=1,
    )

    state = SystemState(
        system_id="sys",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu_usage": MetricValue(name="cpu_usage", value=75.0)},
        health_status=HealthStatus.HEALTHY,
    )
    ctx = AdaptationContext(
        system_id="sys",
        historical_states=[],
        world_model_insights=None,
        system_contract=SystemContract(
            system_id="sys",
            connector_type="DummyConnector",
            supported_action_types=("scale_up", "scale_down"),
        ),
    )

    # Act
    result = await strategy._execute_tool("list_supported_actions", {}, state, ctx)

    # Assert
    assert result.get("source") == "contract"
    assert set(result.get("action_types", [])) == {"scale_up", "scale_down"}


@pytest.mark.asyncio
async def test_list_supported_actions_requires_contract():
    # Prepare knowledge store with one historical action to verify no fallback is used
    ks = InMemoryKnowledgeStore()
    wm = StatisticalWorldModel(ks)

    state = SystemState(
        system_id="sys",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu_usage": MetricValue(name="cpu_usage", value=60.0)},
        health_status=HealthStatus.HEALTHY,
    )

    action = AdaptationAction(
        action_id="a1", action_type="adjust_qos", target_system="sys", parameters={"level": "high"}
    )
    result = ExecutionResult(action_id="a1", status=ExecutionStatus.SUCCESS, result_data={})
    await ks.store_action(action, result)

    strategy = AgenticLLMStrategy(
        llm_client=DummyLLM(),
        knowledge_store=ks,
        world_model=wm,
        steps_limit=1,
    )

    ctx = AdaptationContext(system_id="sys", historical_states=[], world_model_insights=None)

    # Act
    out = await strategy._execute_tool("list_supported_actions", {}, state, ctx)

    # Assert
    assert out.get("error_code") == "missing_system_contract"


def test_tool_to_json_schema_default_and_override():
    from polaris.tools.base import Tool, ToolDependencies

    class MinimalTool(Tool):
        @property
        def name(self) -> str:
            return "minimal"

        async def execute(self, args, state, context, deps: ToolDependencies):
            return {"ok": True}

    tool = MinimalTool()
    schema = tool.to_json_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "minimal"
    assert schema["function"]["description"] == "Tool: minimal"
    assert schema["function"]["parameters"]["type"] == "object"

    class CustomSchemaTool(Tool):
        @property
        def name(self) -> str:
            return "custom"

        @property
        def description(self) -> str:
            return "Custom description"

        @property
        def parameters_schema(self):
            return {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }

        async def execute(self, args, state, context, deps: ToolDependencies):
            return {"ok": True}

    custom_tool = CustomSchemaTool()
    custom_schema = custom_tool.to_json_schema()
    assert custom_schema["function"]["name"] == "custom"
    assert custom_schema["function"]["description"] == "Custom description"
    assert "count" in custom_schema["function"]["parameters"]["properties"]
    assert custom_schema["function"]["parameters"]["required"] == ["count"]
