"""Tool system for POLARIS agentic strategies.

This package provides tools that agentic strategies use to gather information and make
adaptation decisions. Tools are stateless, reusable components that are registered in a
ToolRegistry and executed with injected dependencies.

Examples:
    ```
    from polaris.tools import ToolRegistry, build_registered_tools, ToolDependencies
    ```

    ```
    # Create registry with metrics
    registry = ToolRegistry(metrics=my_metrics)
    ```

    ```
    # Register all globally registered tools (includes built-ins)
    registry.register_all(build_registered_tools())
    ```

    ```
    # Execute a tool
    result = await registry.execute(
        tool_name="get_recent_states",
        args={"window_seconds": 300},
        state=current_state,
        context=adaptation_context,
        deps=tool_dependencies,
    )
    ```

    ```
    Available tools:
        - get_recent_states: Query recent system states
        - summarize_metric_trends: Analyze metric trends
        - list_metric_fields: Discover available metric fields
        - compute_metric_math: Compute stats/expressions over metric series
        - get_world_model_insights: Get world model predictions
        - predict_outcome: Predict action outcomes
        - get_action_history: Query historical actions
        - list_supported_actions: List available system actions
        - sleep: Intentional wait utility
        - get_system_status: Snapshot current instantaneous state
    ```
"""

from polaris.tools.base import Tool, ToolDependencies, ToolError
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
    get_builtin_tools,
)
from polaris.tools.factories import (
    build_registered_tools,
    register_tool_factory,
    registered_tool_types,
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
    "GetSystemStatusTool",
    "ListMetricFieldsTool",
    "ComputeMetricMathTool",
    "SleepTool",
    "ListSupportedActionsTool",
    # Helpers
    "get_builtin_tools",
    "build_registered_tools",
    "register_tool_factory",
    "registered_tool_types",
]
