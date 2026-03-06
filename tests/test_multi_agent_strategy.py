"""Tests for the MultiAgentStrategy."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import AdaptationAction, HealthStatus, MetricValue, SystemState
from polaris.strategies.multi_agent import MultiAgentStrategy


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    return client


@pytest.fixture
def mock_knowledge_store():
    return AsyncMock()


@pytest.fixture
def mock_world_model():
    return AsyncMock()


@pytest.fixture
def multi_agent_strategy(mock_llm_client, mock_knowledge_store, mock_world_model):
    return MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.0,
    )


@pytest.fixture
def sample_state():
    return SystemState(
        system_id="test-sys",
        timestamp=datetime.now(timezone.utc),
        metrics={
            "cpu_utilization": MetricValue(
                name="cpu_utilization",
                value=0.95,
                unit="ratio",
                timestamp=datetime.now(timezone.utc),
            )
        },
        health_status=HealthStatus.WARNING,
    )


@pytest.fixture
def sample_context(sample_state):
    return AdaptationContext(
        system_id="test-sys",
        historical_states=[sample_state],
        world_model_insights="High CPU load predicted",
    )


@pytest.mark.asyncio
async def test_multi_agent_assess_no_anomaly(
    multi_agent_strategy, mock_llm_client, sample_state, sample_context
):
    # Mock Diagnostician to return no anomaly
    diag_resp = MagicMock()
    diag_resp.content = json.dumps(
        {"is_anomaly_detected": False, "issues": [], "root_causes": [], "severity": "none"}
    )
    mock_llm_client.generate.return_value = diag_resp

    actions = await multi_agent_strategy.assess(sample_state, sample_context)

    assert len(actions) == 0
    # LLM should only be called once (for the Diagnostician)
    assert mock_llm_client.generate.call_count == 1


@pytest.mark.asyncio
async def test_multi_agent_assess_full_pipeline(
    multi_agent_strategy, mock_llm_client, sample_state, sample_context
):
    # Mock sequence of responses: Diagnostician -> Planner -> Validator

    # 1. Diagnostician Output
    diag_resp = MagicMock()
    diag_resp.content = json.dumps(
        {
            "is_anomaly_detected": True,
            "issues": ["High CPU"],
            "root_causes": ["Traffic spike"],
            "severity": "high",
        }
    )

    # 2. Planner Output
    planner_resp = MagicMock()
    planner_resp.content = json.dumps(
        {
            "plans": [{"type": "scale_up", "parameters": {"replicas": 5}}],
            "rationale": "Scaling up to handle traffic spike",
        }
    )

    # 3. Validator Output
    validator_resp = MagicMock()
    validator_resp.content = json.dumps(
        {
            "approved": True,
            "reasoning": "Scale up is safe",
            "safe_actions": [{"type": "scale_up", "parameters": {"replicas": 5}}],
        }
    )

    mock_llm_client.generate.side_effect = [diag_resp, planner_resp, validator_resp]

    actions = await multi_agent_strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    action = actions[0]
    assert action.action_type == "scale_up"
    assert action.parameters["replicas"] == 5
    assert "llm_diagnosis" in action.parameters
    assert action.parameters["llm_diagnosis"] == ["High CPU"]

    # Generates called 3 times (Diag, Planner, Validator)
    assert mock_llm_client.generate.call_count == 3


@pytest.mark.asyncio
async def test_multi_agent_assess_validator_rejects(
    multi_agent_strategy, mock_llm_client, sample_state, sample_context
):
    diag_resp = MagicMock()
    diag_resp.content = json.dumps(
        {
            "is_anomaly_detected": True,
            "issues": ["High CPU"],
            "root_causes": ["Traffic spike"],
            "severity": "high",
        }
    )

    planner_resp = MagicMock()
    planner_resp.content = json.dumps(
        {
            "plans": [{"type": "dangerous_action", "parameters": {}}],
            "rationale": "Trying something risky",
        }
    )

    validator_resp = MagicMock()
    validator_resp.content = json.dumps(
        {"approved": False, "reasoning": "Action is unsafe", "safe_actions": []}
    )

    mock_llm_client.generate.side_effect = [diag_resp, planner_resp, validator_resp]

    actions = await multi_agent_strategy.assess(sample_state, sample_context)

    # Validator rejected, so no actions
    assert len(actions) == 0
    assert mock_llm_client.generate.call_count == 3
