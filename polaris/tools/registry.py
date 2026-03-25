"""Tool registry for managing and executing POLARIS tools.

This module provides the ToolRegistry class, which manages tool instances and provides
centralized tool execution with metrics and error handling.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from polaris.abstractions.observability import MetricsCollector
    from polaris.abstractions.strategy import AdaptationContext
    from polaris.core.models import SystemState
    from polaris.tools.base import Tool, ToolDependencies


class ToolRegistry:
    """Registry for managing and executing tools.

    The registry maintains a mapping of tool names to tool instances and provides
    methods for registration, lookup, and execution.

    Follows the same pattern as ConnectorRegistry for consistency.

    Examples:
        ```
        registry = ToolRegistry(metrics=metrics_collector)
        registry.register(GetRecentStatesTool())
        registry.register(SummarizeMetricTrendsTool())
        ```

        ```
        # Execute a tool
        result = await registry.execute(
            tool_name="get_recent_states",
            args={"window_seconds": 300},
            state=current_state,
            context=adaptation_context,
            deps=tool_dependencies
        )
        ```
    """

    def __init__(self, metrics: Optional["MetricsCollector"] = None):
        """Initialize the tool registry.

        Args:
            metrics: Optional metrics collector for observability
        """
        self._tools: Dict[str, "Tool"] = {}
        self._metrics = metrics

    def register(self, tool: "Tool") -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register

        Raises:
            `ValueError`: If a tool with the same name is already registered
        """
        name = tool.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tools[name] = tool

        if self._metrics:
            self._metrics.increment("polaris.tool.registry.registered", tags={"tool": name})
            self._metrics.gauge("polaris.tool.registry.total_tools", len(self._tools))

    def register_all(self, tools: List["Tool"]) -> None:
        """Register multiple tools at once.

        Args:
            tools: List of tool instances to register
        """
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional["Tool"]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance if found, None otherwise
        """
        tool = self._tools.get(name)

        if self._metrics:
            self._metrics.increment(
                "polaris.tool.registry.accessed",
                tags={"tool": name, "found": str(tool is not None).lower()},
            )

        return tool

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name to check

        Returns:
            True if tool is registered, False otherwise
        """
        return name in self._tools

    def list_tools(self) -> List[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names
        """
        if self._metrics:
            self._metrics.increment("polaris.tool.registry.listed")

        return list(self._tools.keys())

    def all_tools(self) -> List["Tool"]:
        """Get all registered tool instances.

        Returns:
            List of all registered tools
        """
        if self._metrics:
            self._metrics.increment("polaris.tool.registry.all_accessed")

        return list(self._tools.values())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all registered tools.

        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {name: tool.description for name, tool in self._tools.items()}

    async def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: "ToolDependencies",
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Execute a tool by name with the provided context.

        This is the main entry point for tool execution. It handles:
        - Tool lookup
        - Metrics collection
        - Error handling with timeouts
        - Execution time tracking

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments from the LLM
            state: Current system state
            context: Adaptation context
            deps: Tool dependencies
            timeout: Maximum execution time in seconds

        Returns:
            Tool execution result or error dictionary
        """
        import asyncio
        from datetime import datetime as dt
        from datetime import timezone

        from polaris.tools.base import ToolError

        if self._metrics:
            self._metrics.increment(
                "polaris.tool.execution.attempts",
                tags={"tool": tool_name, "system_id": state.system_id},
            )

        # Look up the tool
        tool = self.get(tool_name)
        if tool is None:
            error_result = ToolError(
                code="unknown_tool",
                message=f"unknown_tool: {tool_name}",
                recoverable=True,
            ).to_dict()
            if self._metrics:
                self._metrics.increment(
                    "polaris.tool.execution.errors",
                    tags={"tool": tool_name, "error_type": "unknown_tool"},
                )
            return error_result

        # Execute the tool with timing and timeout
        start = dt.now(timezone.utc)
        error_type = "unknown_error"
        try:
            result = await asyncio.wait_for(
                tool.execute(args, state, context, deps),
                timeout=timeout,
            )

            if self._metrics:
                duration = (dt.now(timezone.utc) - start).total_seconds()
                self._metrics.histogram(
                    "polaris.tool.execution.duration_seconds",
                    duration,
                    tags={"tool": tool_name, "system_id": state.system_id},
                )
                self._metrics.increment(
                    "polaris.tool.execution.success",
                    tags={"tool": tool_name, "system_id": state.system_id},
                )

            return result

        except asyncio.TimeoutError:
            error_type = "timeout"
            if deps.logger:
                deps.logger.error(
                    f"Tool timeout: {tool_name}",
                    timeout=timeout,
                    system_id=state.system_id,
                )
            error = ToolError(
                code="timeout",
                message=f"Tool execution timed out after {timeout} seconds",
                recoverable=True,
            )

        except ToolError as te:
            error_type = te.code
            if deps.logger:
                deps.logger.warning(
                    f"Tool error: {tool_name}",
                    error=te.message,
                    code=te.code,
                )
            error = te

        except Exception as e:
            error_type = type(e).__name__
            if deps.logger:
                deps.logger.error(
                    f"Unexpected tool error: {tool_name}",
                    error=str(e),
                    error_type=error_type,
                    system_id=state.system_id,
                )
            error = ToolError(
                code="execution_error",
                message=f"{error_type}: {str(e)}",
                recoverable=True,
            )

        # Handle metrics for errors
        if self._metrics:
            duration = (dt.now(timezone.utc) - start).total_seconds()
            self._metrics.histogram(
                "polaris.tool.execution.duration_seconds",
                duration,
                tags={"tool": tool_name, "system_id": state.system_id},
            )
            self._metrics.increment(
                "polaris.tool.execution.errors",
                tags={
                    "tool": tool_name,
                    "system_id": state.system_id,
                    "error_type": error_type,
                },
            )

        return error.to_dict()

    def filter_by_allowed(self, allowed_tools: Optional[List[str]]) -> List["Tool"]:
        """Filter registered tools by allowed list.

        Args:
            allowed_tools: List of allowed tool names, or None for all tools

        Returns:
            List of tools that are in the allowed list (or all if None)
        """
        if allowed_tools is None:
            return self.all_tools()

        allowed_set = set(allowed_tools)
        return [tool for name, tool in self._tools.items() if name in allowed_set]
