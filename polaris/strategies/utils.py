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
        # Fallback for concatenated JSON objects (JSON-L-like output)
        import re

        objects = []
        # Find individual JSON objects by scanning brackets
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(payload):
            # Skip whitespace explicitly
            match = re.search(r"\S", payload[idx:])
            if not match:
                break
            idx += match.start()

            try:
                obj, new_idx = decoder.raw_decode(payload[idx:])
                if isinstance(obj, dict):
                    objects.append(obj)
                idx += new_idx
            except json.JSONDecodeError:
                idx += 1

        if not objects:
            raise error_class(f"Strategy response is not valid JSON: {exc}") from exc

        # Prioritize 'final' payload if any, else take first
        parsed = next((obj for obj in objects if "final" in obj), objects[0])

    if not isinstance(parsed, dict):
        raise error_class("Strategy response must be a JSON object")

    return parsed


def extract_connector_from_context(context: "AdaptationContext") -> Any:
    """Extract active connector from typed field or legacy metadata.

    The new preferred path is ``AdaptationContext.connector``. The metadata
    fallback keeps compatibility with existing tests/callers.
    """
    connector = getattr(context, "connector", None)
    if connector is not None:
        return connector

    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("connector")
    return None


def compact_json(value: Any, max_chars: int) -> str:
    """Serialize payload to JSON and truncate to avoid context bloat."""
    try:
        text = json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def bounded_tool_data(tool_result: Dict[str, Any], max_chars: int) -> Any:
    """Return tool payload unchanged or as a truncated preview object."""
    try:
        full_text = json.dumps(tool_result, ensure_ascii=True, default=str)
    except Exception:
        full_text = str(tool_result)

    if len(full_text) <= max_chars:
        return tool_result

    serialized = full_text[:max_chars] + "..."
    return {
        "_truncated": True,
        "preview": serialized,
        "original_chars": len(full_text),
    }


def build_tool_result_message(
    tool_name: str,
    tool_result: Dict[str, Any],
    max_chars: int,
) -> str:
    """Build bounded tool-result payload for model context."""
    bounded = bounded_tool_data(tool_result, max_chars)
    return json.dumps(
        {"tool_result": {"tool": tool_name, "data": bounded}},
        ensure_ascii=True,
    )


async def execute_strategy_tool(
    *,
    tool_registry: Any,
    tool_name: str,
    args: Dict[str, Any],
    state: "SystemState",
    context: "AdaptationContext",
    knowledge_store: Any,
    world_model: Any,
    logger: Any,
    metrics: Any,
    metric_prefix: str,
    error_log_message: str,
) -> Dict[str, Any]:
    """Execute one strategy tool with shared dependency wiring and error handling."""
    from polaris.tools import ToolDependencies

    deps = ToolDependencies(
        knowledge_store=knowledge_store,
        world_model=world_model,
        connector=extract_connector_from_context(context),
        system_contract=context.system_contract,
        logger=logger,
        metrics=metrics,
    )

    try:
        tool_result = await tool_registry.execute(
            tool_name=tool_name,
            args=args,
            state=state,
            context=context,
            deps=deps,
        )
        if metrics:
            metrics.increment(
                f"{metric_prefix}.tool_called",
                tags={"tool": tool_name, "system_id": state.system_id},
            )
        if isinstance(tool_result, dict):
            return tool_result
        return {"result": tool_result}
    except Exception as exc:
        if metrics:
            metrics.increment(
                f"{metric_prefix}.tool_error",
                tags={"tool": tool_name, "system_id": state.system_id},
            )
        if logger:
            logger.error(error_log_message, tool=tool_name, error=str(exc))
        return {"error": f"tool_error: {type(exc).__name__}: {str(exc)}"}


def create_tool_registry(metrics: Any, allowed_tools: Any) -> Any:
    """Build and populate a ToolRegistry from allowed tool names."""
    from polaris.tools import ToolRegistry, build_registered_tools

    registry = ToolRegistry(metrics=metrics)
    registry.register_all(build_registered_tools(allowed_tools))
    return registry


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
