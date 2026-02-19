"""Tests for LLMReasoningStrategy parsing logic."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import AdaptationAction, HealthStatus, MetricValue, SystemState
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
