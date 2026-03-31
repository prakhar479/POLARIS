"""Shared utilities for POLARIS adaptation strategies.

This module provides reusable helpers used across multiple strategy implementations
to avoid code duplication and ensure consistent behaviour.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from polaris.abstractions.strategy import AdaptationContext
    from polaris.core.models import SystemState

# ---------------------------------------------------------------------------
# Default tool list shared by all LLM-based strategies
# ---------------------------------------------------------------------------

DEFAULT_ALLOWED_TOOLS: List[str] = [
    "get_recent_states",
    "summarize_metric_trends",
    "list_metric_fields",
    "compute_metric_math",
    "get_world_model_insights",
    "predict_outcome",
    "get_action_history",
    "list_supported_actions",
]


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


def parse_strict_json(content: str, error_class: type) -> Dict[str, Any]:
    """Parse a strict JSON object from LLM model output.

    Args:
        content: Raw string from the model.
        error_class: Exception class to raise on parse failure.
            Must accept a single string message argument.

    Returns:
        Parsed dictionary.

    Raises:
        error_class: If content is empty, not valid JSON, or not an object.
    """
    payload = (content or "").strip()
    if not payload:
        raise error_class("Strategy received empty model response")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise error_class(f"Strategy response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise error_class("Strategy response must be a JSON object")

    return parsed


# ---------------------------------------------------------------------------
# System state serialization
# ---------------------------------------------------------------------------


def format_system_state_for_llm(
    state: "SystemState",
    context: "AdaptationContext",
) -> str:
    """Serialize a SystemState and AdaptationContext into a compact JSON string.

    Handles MetricValue objects gracefully, falling back to string coercion when
    numeric extraction fails.

    Args:
        state: Current system state with metrics and health information.
        context: Adaptation context containing world model insights.

    Returns:
        JSON-encoded string suitable for injection into an LLM prompt.
    """
    metrics: List[Dict[str, Any]] = []
    for k, v in state.metrics.items():
        try:
            metrics.append({"name": k, "value": v.value, "unit": v.unit})
        except Exception:
            metrics.append(
                {
                    "name": k,
                    "value": str(getattr(v, "value", None)),
                    "unit": getattr(v, "unit", None),
                }
            )

    data = {
        "system_id": state.system_id,
        "health": getattr(state.health_status, "value", "unknown"),
        "timestamp": state.timestamp.isoformat(),
        "metrics": metrics,
        "world_model_insights": context.world_model_insights or {},
    }
    return json.dumps(data)
