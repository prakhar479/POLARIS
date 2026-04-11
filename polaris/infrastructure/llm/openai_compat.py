"""Shared parsing helpers for OpenAI-compatible chat completion responses."""

import json
from typing import Any, Dict, List, Optional


def _parse_tool_arguments(raw_arguments: Any) -> Dict[str, Any]:
    """Parse function-call arguments into an object dictionary.

    OpenAI-compatible providers typically return function arguments as a JSON string,
    but some SDK wrappers may already expose a dict-like object.
    """
    if isinstance(raw_arguments, dict):
        return raw_arguments

    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    try:
        parsed = dict(raw_arguments)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_tool_calls(message: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize OpenAI-compatible tool_calls payload to a stable structure."""
    raw_tool_calls = getattr(message, "tool_calls", None)
    if not raw_tool_calls:
        return None

    normalized: List[Dict[str, Any]] = []
    for tool_call in raw_tool_calls:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None)
        if not isinstance(name, str) or not name.strip():
            continue

        raw_arguments = getattr(function, "arguments", "{}")
        normalized.append(
            {
                "name": name,
                "arguments": _parse_tool_arguments(raw_arguments),
            }
        )

    return normalized or None


def parse_openai_compat_response(response: Any, provider_name: str) -> Dict[str, Any]:
    """Extract content/tool-calls/metadata from an OpenAI-compatible response."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"Empty response from {provider_name} API")

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise ValueError(f"Empty response from {provider_name} API")

    tool_calls = normalize_tool_calls(message)
    content = getattr(message, "content", "") or ""
    if not content and not tool_calls:
        raise ValueError(f"Empty response from {provider_name} API")

    usage = getattr(response, "usage", None)
    tokens_used = getattr(usage, "total_tokens", None) if usage is not None else None

    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": getattr(choice, "finish_reason", None),
        "tokens_used": tokens_used,
    }
