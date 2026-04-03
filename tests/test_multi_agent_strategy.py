"""Tests for strict MultiAgentStrategy behavior."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.abstractions.system_contract import SystemContract
from polaris.core.models import HealthStatus, MetricValue, SystemState
from polaris.strategies.action_resolution import StrictContractViolation
from polaris.strategies.multi_agent import AgentConfig, MultiAgentStrategy


@pytest.fixture
def mock_llm_client():
    return AsyncMock()


@pytest.fixture
def mock_knowledge_store():
    return AsyncMock()


@pytest.fixture
def mock_world_model():
    return AsyncMock()


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
        world_model_insights={"note": "high cpu"},
        system_contract=SystemContract(
            system_id="test-sys",
            connector_type="MockConnector",
            supported_action_types=("scale_up", "scale_down"),
            action_aliases={"add_server": "scale_up"},
        ),
    )


def _make_diag_resp(anomaly: bool, severity: str = "high"):
    resp = MagicMock()
    resp.content = json.dumps(
        {
            "final": {
                "is_anomaly_detected": anomaly,
                "issues": ["High CPU"] if anomaly else [],
                "root_causes": ["Traffic spike"] if anomaly else [],
                "severity": severity if anomaly else "none",
            }
        }
    )
    return resp


def _make_planner_resp():
    resp = MagicMock()
    resp.content = json.dumps(
        {
            "final": {
                "plans": [{"type": "scale_up", "parameters": {"replicas": 5}}],
                "rationale": "Scale up",
            }
        }
    )
    return resp


def _make_validator_resp(approved: bool = True, action_type: str = "scale_up"):
    resp = MagicMock()
    safe_actions = [{"type": action_type, "parameters": {"replicas": 5}}] if approved else []
    resp.content = json.dumps(
        {
            "final": {
                "approved": approved,
                "reasoning": "safe" if approved else "unsafe",
                "safe_actions": safe_actions,
            }
        }
    )
    return resp


@pytest.mark.asyncio
async def test_multi_agent_assess_full_pipeline(
    mock_llm_client, mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    mock_llm_client.generate.side_effect = [
        _make_diag_resp(True),
        _make_planner_resp(),
        _make_validator_resp(True),
    ]
    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.0,
    )

    actions = await strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"
    assert mock_llm_client.generate.call_count == 3


@pytest.mark.asyncio
async def test_multi_agent_assess_no_anomaly(
    mock_llm_client, mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    mock_llm_client.generate.return_value = _make_diag_resp(anomaly=False)
    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.0,
    )

    actions = await strategy.assess(sample_state, sample_context)

    assert actions == []
    assert mock_llm_client.generate.call_count == 1


@pytest.mark.asyncio
async def test_multi_agent_requires_contract(
    mock_llm_client, mock_knowledge_store, mock_world_model, sample_state
):
    mock_llm_client.generate.return_value = _make_diag_resp(anomaly=False)
    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.0,
    )
    context_no_contract = AdaptationContext(system_id="test-sys", historical_states=[])

    with pytest.raises(
        StrictContractViolation, match="Missing connector-supported action contract"
    ):
        await strategy.assess(sample_state, context_no_contract)


@pytest.mark.asyncio
async def test_multi_agent_rejects_unsupported_action(
    mock_llm_client, mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    mock_llm_client.generate.side_effect = [
        _make_diag_resp(True),
        _make_planner_resp(),
        _make_validator_resp(True, action_type="restart"),
    ]
    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
    )

    with pytest.raises(StrictContractViolation, match="Unsupported action type"):
        await strategy.assess(sample_state, sample_context)


@pytest.mark.asyncio
async def test_multi_agent_honors_explicit_aliases(
    mock_llm_client, mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    mock_llm_client.generate.side_effect = [
        _make_diag_resp(True),
        _make_planner_resp(),
        _make_validator_resp(True, action_type="add_server"),
    ]
    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
    )

    actions = await strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"


@pytest.mark.asyncio
async def test_per_agent_llm_clients_used(
    mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    shared_client = AsyncMock()
    diag_client = AsyncMock()
    planner_client = AsyncMock()
    validator_client = AsyncMock()

    diag_client.generate = AsyncMock(return_value=_make_diag_resp(True))
    planner_client.generate = AsyncMock(return_value=_make_planner_resp())
    validator_client.generate = AsyncMock(return_value=_make_validator_resp(True))

    strategy = MultiAgentStrategy(
        llm_client=shared_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        diagnostician_config=AgentConfig(llm_client=diag_client),
        planner_config=AgentConfig(llm_client=planner_client),
        validator_config=AgentConfig(llm_client=validator_client),
    )

    actions = await strategy.assess(sample_state, sample_context)

    assert len(actions) == 1
    assert diag_client.generate.call_count == 1
    assert planner_client.generate.call_count == 1
    assert validator_client.generate.call_count == 1
    assert shared_client.generate.call_count == 0
