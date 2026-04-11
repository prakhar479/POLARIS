"""Global tool factory registry for POLARIS.

This module provides a registry-based extension surface for tools, mirroring
connector/strategy registration patterns.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from polaris.tools.base import Tool

ToolFactory = Callable[[], "Tool"]

_TOOL_FACTORIES: Dict[str, ToolFactory] = {}
_factories_registered = False


def _ensure_factories_registered() -> None:
    """Ensure built-in tool factories are registered lazily."""
    global _factories_registered
    if _factories_registered:
        return

    _register_default_tool_factories()
    _factories_registered = True


def register_tool_factory(tool_name: str, factory: ToolFactory) -> None:
    """Register or override a tool factory by canonical tool name."""
    normalized = str(tool_name or "").strip()
    if not normalized:
        raise ValueError("tool_name must be a non-empty string")
    if not callable(factory):
        raise ValueError("factory must be callable")
    _TOOL_FACTORIES[normalized] = factory


def get_tool_factory(tool_name: str) -> Optional[ToolFactory]:
    """Return factory for a tool name, if registered."""
    _ensure_factories_registered()
    normalized = str(tool_name or "").strip()
    return _TOOL_FACTORIES.get(normalized)


def registered_tool_types() -> List[str]:
    """Return sorted list of all registered tool names."""
    _ensure_factories_registered()
    return sorted(_TOOL_FACTORIES.keys())


def build_registered_tools(allowed_tools: Optional[List[str]] = None) -> List["Tool"]:
    """Instantiate tools from registry, optionally filtered by allowed names.

    Raises:
        ValueError: if `allowed_tools` contains unknown tool names.
    """
    _ensure_factories_registered()

    if allowed_tools is None:
        names = sorted(_TOOL_FACTORIES.keys())
    else:
        names = []
        seen = set()
        for name in allowed_tools:
            normalized = name.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            names.append(normalized)
        unknown = sorted({name for name in names if name not in _TOOL_FACTORIES})
        if unknown:
            raise ValueError(
                "Unknown tool name(s): "
                + ", ".join(unknown)
                + ". Register with register_tool_factory before strategy initialization."
            )

    return [_TOOL_FACTORIES[name]() for name in names]


def _register_default_tool_factories() -> None:
    """Register built-in tool factories."""
    from polaris.tools.builtin import (
        ComputeMetricMathTool,
        GetActionHistoryTool,
        GetRecentStatesTool,
        GetSystemStatusTool,
        GetWorldModelInsightsTool,
        ListMetricFieldsTool,
        ListSupportedActionsTool,
        PredictOutcomeTool,
        SleepTool,
        SummarizeMetricTrendsTool,
    )

    defaults = {
        "get_recent_states": GetRecentStatesTool,
        "summarize_metric_trends": SummarizeMetricTrendsTool,
        "list_metric_fields": ListMetricFieldsTool,
        "compute_metric_math": ComputeMetricMathTool,
        "get_world_model_insights": GetWorldModelInsightsTool,
        "predict_outcome": PredictOutcomeTool,
        "get_action_history": GetActionHistoryTool,
        "list_supported_actions": ListSupportedActionsTool,
        "sleep": SleepTool,
        "get_system_status": GetSystemStatusTool,
    }

    # Preserve explicit pre-registered factories (e.g., plugin overrides registered
    # before lazy default initialization) and only backfill missing defaults.
    for tool_name, factory in defaults.items():
        if tool_name not in _TOOL_FACTORIES:
            register_tool_factory(tool_name, factory)
