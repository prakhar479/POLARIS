"""Tool system for POLARIS agentic strategies.

This package provides tools that agentic strategies use to gather information
and make adaptation decisions. Tools are stateless, reusable components
that are registered in a ToolRegistry and executed with injected dependencies.

Example:
    from polaris.tools import ToolRegistry, get_builtin_tools, ToolDependencies

    # Create registry with metrics
    registry = ToolRegistry(metrics=my_metrics)

    # Register all built-in tools
    registry.register_all(get_builtin_tools())

    # Execute a tool
    result = await registry.execute(
        tool_name="get_recent_states",
        args={"window_seconds": 300},
        state=current_state,
        context=adaptation_context,
        deps=tool_dependencies,
    )

Available tools:
    - get_recent_states: Query recent system states
    - summarize_metric_trends: Analyze metric trends
    - get_world_model_insights: Get world model predictions
    - predict_outcome: Predict action outcomes
    - get_action_history: Query historical actions
    - list_supported_actions: List available system actions
"""

from polaris.tools.base import Tool, ToolDependencies, ToolError
from polaris.tools.builtin import (
    GetActionHistoryTool,
    GetRecentStatesTool,
    GetWorldModelInsightsTool,
    ListSupportedActionsTool,
    PredictOutcomeTool,
    SummarizeMetricTrendsTool,
    get_builtin_tools,
)
from polaris.tools.registry import ToolRegistry

__all__ = [
    # Base classes
    "Tool",
    "ToolDependencies",
    "ToolError",
    # Registry
    "ToolRegistry",
    # Tool instances
    "GetRecentStatesTool",
    "SummarizeMetricTrendsTool",
    "GetWorldModelInsightsTool",
    "PredictOutcomeTool",
    "GetActionHistoryTool",
    "ListSupportedActionsTool",
    # Helpers
    "get_builtin_tools",
]
