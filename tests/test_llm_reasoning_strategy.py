"""Tests for strict LLMReasoningStrategy behavior."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

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
from polaris.strategies.action_resolution import StrictContractViolation
from polaris.strategies.llm_reasoning import LLMReasoningStrategy


@pytest.fixture
def strategy():
    llm = Mock()
    llm.generate = AsyncMock()
    return LLMReasoningStrategy(llm_client=llm)


def _strict_contract() -> SystemContract:
    return SystemContract(
        system_id="test-system",
        connector_type="MockConnector",
        supported_action_types=("scale_up", "scale_down"),
        action_aliases={"add_server": "scale_up"},
    )


def test_parse_response_requires_actions_list_only(strategy):
    response = json.dumps(
        {
            "needs_adaptation": True,
            "reasoning": "High load",
            "action": {"type": "scale_up", "parameters": {"instances": 2}},
        }
    )

    with pytest.raises(StrictContractViolation, match="actions"):
        strategy._parse_response(
            response,
            "test-system",
            supported_action_types=["scale_up", "scale_down"],
            action_aliases={"add_server": "scale_up"},
        )


def test_parse_response_parses_actions_with_contract_aliases(strategy):
    response = json.dumps(
        {
            "needs_adaptation": True,
            "reasoning": "High load",
            "actions": [{"type": "add_server", "parameters": {"instances": 2}}],
        }
    )

    actions = strategy._parse_response(
        response,
        "test-system",
        supported_action_types=["scale_up", "scale_down"],
        action_aliases={"add_server": "scale_up"},
    )

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"
    assert actions[0].parameters["instances"] == 2


def test_parse_response_rejects_malformed_json(strategy):
    with pytest.raises(StrictContractViolation, match="valid JSON"):
        strategy._parse_response(
            "not json",
            "test-system",
            supported_action_types=["scale_up"],
            action_aliases={},
        )


@pytest.mark.asyncio
async def test_assess_success_with_strict_contract(strategy):
    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue(name="cpu", value=90.0, unit="%")},
        health_status=HealthStatus.UNHEALTHY,
    )
    context = AdaptationContext(
        system_id="test-system",
        historical_states=[],
        system_contract=_strict_contract(),
    )
    mock_response = Mock()
    mock_response.content = json.dumps(
        {
            "needs_adaptation": True,
            "reasoning": "High CPU",
            "actions": [{"type": "scale_up", "parameters": {"instances": 2}}],
        }
    )
    strategy.llm.generate.return_value = mock_response

    actions = await strategy.assess(state, context)

    assert len(actions) == 1
    assert actions[0].action_type == "scale_up"


@pytest.mark.asyncio
async def test_assess_raises_when_contract_missing(strategy):
    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue(name="cpu", value=90.0, unit="%")},
        health_status=HealthStatus.UNHEALTHY,
    )
    context = AdaptationContext(system_id="test-system", historical_states=[])
    mock_response = Mock()
    mock_response.content = json.dumps(
        {"needs_adaptation": False, "reasoning": "No change", "actions": []}
    )
    strategy.llm.generate.return_value = mock_response

    with pytest.raises(
        StrictContractViolation, match="Missing connector-supported action contract"
    ):
        await strategy.assess(state, context)


@pytest.mark.asyncio
async def test_on_action_executed_tracks_metrics(strategy):
    action = AdaptationAction(
        action_id="test-action", action_type="scale_up", target_system="test-system"
    )
    result = ExecutionResult(
        action_id="test-action", status=ExecutionStatus.SUCCESS, result_data={}
    )

    await strategy.on_action_executed(action, result)

    assert strategy._adaptation_count == 1
    assert strategy._success_count == 1
