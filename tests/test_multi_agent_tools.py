import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import HealthStatus, MetricValue, SystemState
from polaris.infrastructure.llm import LLMResponse
from polaris.strategies.multi_agent import (
    ActionBlock,
    AgentConfig,
    DiagnosticianAgenticResponse,
    DiagnosticianOutput,
    MultiAgentStrategy,
    PlannerAgenticResponse,
    PlannerOutput,
    ValidatorAgenticResponse,
    ValidatorOutput,
)


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def mock_world():
    return AsyncMock()


@pytest.mark.asyncio
async def test_diagnostician_uses_tool(mock_llm, mock_store, mock_world):
    strategy = MultiAgentStrategy(
        llm_client=mock_llm, knowledge_store=mock_store, world_model=mock_world, steps_limit=3
    )

    state = SystemState(
        system_id="test-sys",
        metrics={},
        health_status=HealthStatus.HEALTHY,
        timestamp=datetime.now(timezone.utc),
    )
    context = AdaptationContext(system_id="test-sys", historical_states=[])

    # Round 1: Diagnostician asks for a tool
    resp1 = LLMResponse(
        content=json.dumps({"tool": "summarize_metric_trends", "args": {"metric": "cpu_util"}}),
        model="test-model",
    )
    # Round 2: Diagnostician gives final answer
    resp2 = LLMResponse(
        content=json.dumps(
            {
                "final": {
                    "is_anomaly_detected": True,
                    "issues": ["High CPU observed via tool"],
                    "root_causes": ["Unknown"],
                    "severity": "high",
                }
            }
        ),
        model="test-model",
    )
    # Planner gives final answer immediately
    resp3 = LLMResponse(
        content=json.dumps(
            {
                "final": {
                    "plans": [{"type": "scale_up", "parameters": {"count": 1}}],
                    "rationale": "Tool confirmed high CPU",
                }
            }
        ),
        model="test-model",
    )
    # Validator gives final answer immediately
    resp4 = LLMResponse(
        content=json.dumps(
            {
                "final": {
                    "approved": True,
                    "reasoning": "Safe",
                    "safe_actions": [{"type": "scale_up", "parameters": {"count": 1}}],
                }
            }
        ),
        model="test-model",
    )

    mock_llm.generate.side_effect = [resp1, resp2, resp3, resp4]
    mock_store.query_states.return_value = []

    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"
    assert "High CPU observed via tool" in str(actions[0].parameters["llm_diagnosis"])


@pytest.mark.asyncio
async def test_per_agent_steps_limit(mock_llm, mock_store, mock_world):
    diag_cfg = AgentConfig(steps_limit=1)
    strategy = MultiAgentStrategy(
        llm_client=mock_llm,
        knowledge_store=mock_store,
        world_model=mock_world,
        diagnostician_config=diag_cfg,
        steps_limit=5,
    )

    state = SystemState(
        system_id="test-sys",
        metrics={},
        health_status=HealthStatus.HEALTHY,
        timestamp=datetime.now(timezone.utc),
    )

    mock_llm.generate.return_value = LLMResponse(
        content=json.dumps({"tool": "get_recent_states", "args": {}}), model="test-model"
    )

    actions = await strategy.assess(
        state, AdaptationContext(system_id="test-sys", historical_states=[])
    )
    assert actions == []
    assert mock_llm.generate.call_count == 1


@pytest.mark.asyncio
async def test_per_agent_tools_restriction(mock_llm, mock_store, mock_world):
    diag_cfg = AgentConfig(allowed_tools=["get_recent_states"])
    strategy = MultiAgentStrategy(
        llm_client=mock_llm,
        knowledge_store=mock_store,
        world_model=mock_world,
        diagnostician_config=diag_cfg,
    )

    state = SystemState(
        system_id="test-sys",
        metrics={},
        health_status=HealthStatus.HEALTHY,
        timestamp=datetime.now(timezone.utc),
    )

    resp1 = LLMResponse(
        content=json.dumps(
            {
                "tool": "predict_outcome",
                "args": {"candidate_action": {"type": "foo", "parameters": {}}},
            }
        ),
        model="test-model",
    )
    resp2 = LLMResponse(
        content=json.dumps(
            {
                "final": {
                    "is_anomaly_detected": False,
                    "issues": [],
                    "root_causes": [],
                    "severity": "none",
                }
            }
        ),
        model="test-model",
    )
    mock_llm.generate.side_effect = [resp1, resp2]

    await strategy.assess(state, AdaptationContext(system_id="test-sys", historical_states=[]))

    second_call_msgs = mock_llm.generate.call_args_list[1][0][0]
    tool_result_msg = next(msg for msg in second_call_msgs if "tool_result" in msg.content)
    assert "tool_not_allowed" in tool_result_msg.content
