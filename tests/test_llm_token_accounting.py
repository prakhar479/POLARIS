"""Tests for LLM token extraction and accounting."""

from unittest.mock import Mock

import pytest

from polaris.infrastructure.llm.base import LLMResponse
from polaris.infrastructure.llm.openai_compat import parse_openai_compat_response


def test_parse_openai_compat_response_extracts_all_tokens():
    """Verify that prompt_tokens, completion_tokens, and total_tokens are parsed."""
    mock_choice = Mock()
    mock_choice.message = Mock(content="Test reply", tool_calls=None)
    mock_choice.finish_reason = "stop"

    mock_usage = Mock(
        total_tokens=150,
        prompt_tokens=120,
        completion_tokens=30,
    )

    mock_resp = Mock(
        choices=[mock_choice],
        usage=mock_usage,
    )

    parsed = parse_openai_compat_response(mock_resp, "TestProvider")
    assert parsed["content"] == "Test reply"
    assert parsed["tokens_used"] == 150
    assert parsed["prompt_tokens"] == 120
    assert parsed["completion_tokens"] == 30
    assert parsed["finish_reason"] == "stop"


def test_parse_openai_compat_response_without_usage():
    """Verify handling when usage metadata is omitted."""
    mock_choice = Mock()
    mock_choice.message = Mock(content="Test reply without usage", tool_calls=None)
    mock_choice.finish_reason = "stop"

    mock_resp = Mock(
        choices=[mock_choice],
        usage=None,
    )

    parsed = parse_openai_compat_response(mock_resp, "TestProvider")
    assert parsed["tokens_used"] is None
    assert parsed["prompt_tokens"] is None
    assert parsed["completion_tokens"] is None


def test_llm_response_dataclass_fields():
    """Verify LLMResponse accepts and stores prompt_tokens and completion_tokens."""
    resp = LLMResponse(
        content="hello",
        model="gpt-4",
        tokens_used=10,
        prompt_tokens=7,
        completion_tokens=3,
        finish_reason="stop",
    )
    assert resp.content == "hello"
    assert resp.model == "gpt-4"
    assert resp.tokens_used == 10
    assert resp.prompt_tokens == 7
    assert resp.completion_tokens == 3
