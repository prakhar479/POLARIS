"""Tests for shared OpenAI-compatible response parsing helpers."""

from types import SimpleNamespace

import pytest

from polaris.infrastructure.llm.openai_compat import parse_openai_compat_response


def _response_with_message(content: str = "", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(total_tokens=42)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_parse_openai_compat_response_content_only():
    response = _response_with_message(content="ok")

    parsed = parse_openai_compat_response(response, "OpenAI")

    assert parsed["content"] == "ok"
    assert parsed["tool_calls"] is None
    assert parsed["finish_reason"] == "stop"
    assert parsed["tokens_used"] == 42


def test_parse_openai_compat_response_tool_calls():
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="predict_outcome", arguments='{"horizon": 10}')
    )
    response = _response_with_message(tool_calls=[tool_call])

    parsed = parse_openai_compat_response(response, "OpenAI")

    assert parsed["content"] == ""
    assert parsed["tool_calls"] == [
        {
            "name": "predict_outcome",
            "arguments": {"horizon": 10},
        }
    ]


def test_parse_openai_compat_response_invalid_arguments_fallback():
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="predict_outcome", arguments="{not-json")
    )
    response = _response_with_message(tool_calls=[tool_call])

    parsed = parse_openai_compat_response(response, "OpenAI")

    assert parsed["tool_calls"] == [
        {
            "name": "predict_outcome",
            "arguments": {},
        }
    ]


def test_parse_openai_compat_response_empty_raises():
    response = SimpleNamespace(choices=[])

    with pytest.raises(ValueError, match="Empty response from OpenAI API"):
        parse_openai_compat_response(response, "OpenAI")
