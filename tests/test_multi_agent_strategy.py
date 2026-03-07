"""Tests for the MultiAgentStrategy, including per-agent configuration."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import AdaptationAction, HealthStatus, MetricValue, SystemState
from polaris.strategies.multi_agent import AgentConfig, MultiAgentStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
                "rationale": "Scaling up to handle traffic spike",
            }
        }
    )
    return resp


def _make_validator_resp(approved: bool = True):
    resp = MagicMock()
    safe_actions = [{"type": "scale_up", "parameters": {"replicas": 5}}] if approved else []
    resp.content = json.dumps(
        {
            "final": {
                "approved": approved,
                "reasoning": "Scale up is safe" if approved else "Action is unsafe",
                "safe_actions": safe_actions,
            }
        }
    )
    return resp


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_agent_assess_no_anomaly(
    multi_agent_strategy, mock_llm_client, sample_state, sample_context
):
    # Mock Diagnostician to return no anomaly
    mock_llm_client.generate.return_value = _make_diag_resp(anomaly=False)

    actions = await multi_agent_strategy.assess(sample_state, sample_context)

    assert len(actions) == 0
    # LLM should only be called once (for the Diagnostician)
    assert mock_llm_client.generate.call_count == 1


@pytest.mark.asyncio
async def test_multi_agent_assess_full_pipeline(
    multi_agent_strategy, mock_llm_client, sample_state, sample_context
):
    mock_llm_client.generate.side_effect = [
        _make_diag_resp(True),
        _make_planner_resp(),
        _make_validator_resp(True),
    ]

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
    mock_llm_client.generate.side_effect = [
        _make_diag_resp(True),
        _make_planner_resp(),
        _make_validator_resp(False),
    ]

    actions = await multi_agent_strategy.assess(sample_state, sample_context)

    # Validator rejected, so no actions
    assert len(actions) == 0
    assert mock_llm_client.generate.call_count == 3


# ---------------------------------------------------------------------------
# New tests: per-agent configuration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_agent_llm_clients_used(
    mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    """Each agent should call its own dedicated LLM client when configured."""
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
        temperature=0.1,
        diagnostician_config=AgentConfig(llm_client=diag_client),
        planner_config=AgentConfig(llm_client=planner_client),
        validator_config=AgentConfig(llm_client=validator_client),
    )

    actions = await strategy.assess(sample_state, sample_context)

    # Per-agent clients were called, not the shared client
    assert diag_client.generate.call_count == 1
    assert planner_client.generate.call_count == 1
    assert validator_client.generate.call_count == 1
    assert shared_client.generate.call_count == 0

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"


@pytest.mark.asyncio
async def test_fallback_to_shared_client(
    mock_llm_client, mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    """Without per-agent client, the shared client should be used for all agents."""
    mock_llm_client.generate.side_effect = [
        _make_diag_resp(True),
        _make_planner_resp(),
        _make_validator_resp(True),
    ]

    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.1,
        # No per-agent config — should fall back to shared
    )

    actions = await strategy.assess(sample_state, sample_context)

    assert mock_llm_client.generate.call_count == 3
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_per_agent_temperature_used(
    mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    """Per-agent temperatures should be forwarded to the LLM generate() call."""
    shared_client = AsyncMock()
    diag_client = AsyncMock()
    diag_client.generate = AsyncMock(return_value=_make_diag_resp(True))
    shared_client.generate.side_effect = [_make_planner_resp(), _make_validator_resp(True)]

    strategy = MultiAgentStrategy(
        llm_client=shared_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.5,  # shared temperature
        diagnostician_config=AgentConfig(llm_client=diag_client, temperature=0.0),
        # planner and validator use shared client+temperature
    )

    await strategy.assess(sample_state, sample_context)

    # Diagnostician called with temperature=0.0
    diag_call_kwargs = diag_client.generate.call_args
    assert (
        diag_call_kwargs.kwargs.get("temperature") == 0.0
        or diag_call_kwargs[1].get("temperature") == 0.0
    )

    # Planner and validator called with shared temperature=0.5
    planner_call_kwargs = shared_client.generate.call_args_list[0]
    assert (
        planner_call_kwargs.kwargs.get("temperature") == 0.5
        or planner_call_kwargs[1].get("temperature") == 0.5
    )


@pytest.mark.asyncio
async def test_per_agent_prompt_override(
    mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    """Custom system prompts should be used when set on AgentConfig."""
    shared_client = AsyncMock()
    shared_client.generate.side_effect = [
        _make_diag_resp(True),
        _make_planner_resp(),
        _make_validator_resp(True),
    ]

    custom_prompt = "You are a custom diagnostician. Do not use default prompt."
    strategy = MultiAgentStrategy(
        llm_client=shared_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.1,
        diagnostician_config=AgentConfig(system_prompt=custom_prompt),
    )

    await strategy.assess(sample_state, sample_context)

    # The first generate() call (Diagnostician) should have the custom prompt in the messages
    first_call_args = shared_client.generate.call_args_list[0]
    messages = first_call_args[0][0] if first_call_args[0] else first_call_args.args[0]
    system_msg_content = messages[0].content
    assert "custom diagnostician" in system_msg_content


@pytest.mark.asyncio
async def test_agent_prompts_dict_override(
    mock_knowledge_store, mock_world_model, sample_state, sample_context
):
    """agent_prompts convenience dict should set per-role system prompts."""
    shared_client = AsyncMock()
    shared_client.generate.side_effect = [
        _make_diag_resp(True),
        _make_planner_resp(),
        _make_validator_resp(True),
    ]

    strategy = MultiAgentStrategy(
        llm_client=shared_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.1,
        agent_prompts={
            "diagnostician": "Custom diag prompt for {system_description}",
            "planner": "Custom planner prompt for {system_description}",
        },
        system_description="SWIM pool",
    )

    await strategy.assess(sample_state, sample_context)

    # Verify the diagnostician message contained the resolved custom prompt
    first_call = shared_client.generate.call_args_list[0]
    messages = first_call[0][0] if first_call[0] else first_call.args[0]
    assert "Custom diag prompt for SWIM pool" in messages[0].content


@pytest.mark.asyncio
async def test_tunable_parameters_per_agent(
    mock_llm_client, mock_knowledge_store, mock_world_model
):
    """Per-agent temperatures should be exposed as tunable parameters."""
    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.1,
        diagnostician_config=AgentConfig(temperature=0.0),
        planner_config=AgentConfig(temperature=0.3),
        validator_config=AgentConfig(temperature=0.0),
    )

    params = strategy.get_tunable_parameters()

    assert "temperature" in params
    assert "diagnostician_temperature" in params
    assert "planner_temperature" in params
    assert "validator_temperature" in params
    assert params["diagnostician_temperature"].current_value == 0.0
    assert params["planner_temperature"].current_value == 0.3
    assert params["validator_temperature"].current_value == 0.0


@pytest.mark.asyncio
async def test_update_per_agent_temperature(
    mock_llm_client, mock_knowledge_store, mock_world_model
):
    """update_parameter should update per-agent temperatures."""
    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.1,
    )

    result = await strategy.update_parameter("diagnostician_temperature", 0.5)
    assert result is True
    assert strategy._diagnostician_cfg.temperature == 0.5

    result = await strategy.update_parameter("planner_temperature", 0.7)
    assert result is True
    assert strategy._planner_cfg.temperature == 0.7

    result = await strategy.update_parameter("unknown_param", 1.0)
    assert result is False


@pytest.mark.asyncio
async def test_apply_config_update_hot_reload(
    mock_llm_client, mock_knowledge_store, mock_world_model
):
    """apply_config_update should update shared and per-agent temperatures and prompts."""
    strategy = MultiAgentStrategy(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        world_model=mock_world_model,
        temperature=0.1,
    )

    await strategy.apply_config_update(
        {
            "temperature": 0.2,
            "system_description": "Updated system description",
            "diagnostician": {
                "temperature": 0.05,
                "system_prompt": "New diagnostician prompt",
            },
            "validator": {
                "max_tokens": 2048,
            },
        }
    )

    assert strategy.temperature == 0.2
    assert strategy.system_description == "Updated system description"
    assert strategy._diagnostician_cfg.temperature == 0.05
    assert strategy._diagnostician_cfg.system_prompt == "New diagnostician prompt"
    assert strategy._validator_cfg.max_tokens == 2048
