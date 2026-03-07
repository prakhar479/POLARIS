"""Built-in tools for POLARIS agentic strategies.

This module implements the six core tools used by AgenticLLMStrategy
and MultiAgentStrategy. Each tool is extracted from the original
hardcoded _execute_tool implementations.
"""

import uuid
from typing import TYPE_CHECKING, Any, Dict, List

from polaris.core.models import AdaptationAction
from polaris.tools.base import Tool, ToolDependencies, ToolError

if TYPE_CHECKING:
    from polaris.abstractions.strategy import AdaptationContext
    from polaris.core.models import SystemState


class GetRecentStatesTool(Tool):
    """Tool to query recent system states from the knowledge store."""

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "get_recent_states"

    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return (
            "Query recent system states from the knowledge store. "
            "Returns timestamped states with their metrics. "
            "Parameters: window_seconds (1-3600, default 600), limit (1-200, default 50)"
        )

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute get_recent_states tool."""
        try:
            # Parse and clamp parameters
            limit = self._clamp_int(args.get("limit"), 1, 200, 50)

            # Calculate time window
            start, end = self._get_time_window(args, 600, 3600)

            # Query states from knowledge store
            states = await deps.knowledge_store.query_states(state.system_id, start, end)

            # Apply limit (most recent)
            if states:
                states = states[-limit:]

            # Format output
            out: List[Dict[str, Any]] = []
            for s in states:
                m: Dict[str, float] = {}
                for name, mv in s.metrics.items():
                    try:
                        m[name] = float(mv.value)
                    except (TypeError, ValueError):
                        pass
                out.append({"timestamp": s.timestamp.isoformat(), "metrics": m})

            if deps.logger:
                deps.logger.debug(
                    "get_recent_states returned",
                    count=len(out),
                    system_id=state.system_id,
                )

            return {"states": out}

        except Exception as e:
            if deps.logger:
                deps.logger.error("get_recent_states failed", error=str(e))
            return ToolError(
                code="query_failed",
                message=f"Failed to query states: {str(e)}",
                recoverable=True,
            ).to_dict()


class SummarizeMetricTrendsTool(Tool):
    """Tool to analyze trends in a specific metric over time."""

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "summarize_metric_trends"

    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return (
            "Analyze trends for a specific metric over a time window. "
            "Returns count, min, max, and average values. "
            "Parameters: metric (required, string), window_seconds (1-3600, default 600)"
        )

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute summarize_metric_trends tool."""
        # Validate required parameter
        metric = str(args.get("metric", "")).strip()
        if not metric:
            return ToolError(
                code="missing_metric",
                message="Required parameter 'metric' is missing or empty",
                recoverable=True,
            ).to_dict()

        try:
            # Calculate time window
            start, end = self._get_time_window(args, 600, 3600)

            # Query states
            states = await deps.knowledge_store.query_states(state.system_id, start, end)

            # Extract metric values
            vals = self._extract_metric_values(states, metric)

            if not vals:
                return {"count": 0, "metric": metric}

            result = {
                "count": len(vals),
                "metric": metric,
                "min": min(vals),
                "max": max(vals),
                "avg": sum(vals) / len(vals),
            }

            if deps.logger:
                deps.logger.debug(
                    "summarize_metric_trends returned",
                    metric=metric,
                    count=len(vals),
                )

            return result

        except Exception as e:
            if deps.logger:
                deps.logger.error("summarize_metric_trends failed", error=str(e))
            return ToolError(
                code="query_failed",
                message=f"Failed to summarize trends: {str(e)}",
                recoverable=True,
            ).to_dict()


class GetWorldModelInsightsTool(Tool):
    """Tool to retrieve insights from the world model."""

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "get_world_model_insights"

    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return (
            "Retrieve insights from the world model about current system behavior, "
            "including predicted trends and regime classifications. No parameters."
        )

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute get_world_model_insights tool."""
        try:
            insights = await deps.world_model.get_insights()

            if deps.logger:
                deps.logger.debug("get_world_model_insights returned")

            return {"insights": insights}

        except Exception as e:
            if deps.logger:
                deps.logger.error("get_world_model_insights failed", error=str(e))
            return ToolError(
                code="model_error",
                message=f"Failed to get world model insights: {str(e)}",
                recoverable=True,
            ).to_dict()


class PredictOutcomeTool(Tool):
    """Tool to predict the outcome of a candidate action."""

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "predict_outcome"

    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return (
            "Predict the outcome of a candidate adaptation action using the world model. "
            "Parameters: candidate_action with 'type' (string) and 'parameters' (object)"
        )

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute predict_outcome tool."""
        # Validate candidate action
        block = args.get("candidate_action") or {}
        if not isinstance(block, dict):
            return ToolError(
                code="invalid_candidate_action",
                message="candidate_action must be an object with 'type' and 'parameters'",
                recoverable=True,
            ).to_dict()

        a_type = block.get("type")
        params = block.get("parameters") or {}

        if not a_type or not isinstance(params, dict):
            return ToolError(
                code="invalid_candidate_action",
                message="candidate_action must have 'type' (string) and 'parameters' (object)",
                recoverable=True,
            ).to_dict()

        try:
            # Create candidate action
            candidate = AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type=str(a_type),
                target_system=state.system_id,
                parameters=params,
            )

            # Get prediction from world model
            pred = await deps.world_model.predict(candidate, state)

            if deps.logger:
                deps.logger.debug(
                    "predict_outcome returned",
                    action_type=a_type,
                    confidence=pred.confidence,
                )

            return {
                "predicted_metrics": pred.predicted_metrics,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning,
            }

        except Exception as e:
            if deps.logger:
                deps.logger.error("predict_outcome failed", error=str(e))
            return ToolError(
                code="prediction_failed",
                message=f"Failed to predict outcome: {str(e)}",
                recoverable=True,
            ).to_dict()


class GetActionHistoryTool(Tool):
    """Tool to query historical adaptation actions."""

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "get_action_history"

    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return (
            "Query historical adaptation actions from the knowledge store. "
            "Parameters: window_seconds (1-2592000, default 86400), limit (1-500, default 50)"
        )

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute get_action_history tool."""
        try:
            # Parse parameters with larger limits for historical data
            window_seconds = self._clamp_int(
                args.get("window_seconds"), 1, 30 * 24 * 3600, 86400  # Max 30 days
            )
            limit = self._clamp_int(args.get("limit"), 1, 500, 50)

            # Calculate time window
            from datetime import datetime as dt
            from datetime import timedelta, timezone

            end = dt.now(timezone.utc)
            start = end - timedelta(seconds=window_seconds)

            # Query action history
            history = await deps.knowledge_store.query_actions(state.system_id, start, end)

            # Format output
            items = []
            for action, result in history[-limit:]:
                completed_at = getattr(result, "completed_at", None)
                items.append(
                    {
                        "action_id": getattr(action, "action_id", None),
                        "type": getattr(action, "action_type", None),
                        "parameters": getattr(action, "parameters", {}),
                        "status": getattr(getattr(result, "status", None), "value", None),
                        "error": getattr(result, "error_message", None),
                        "completed_at": (completed_at.isoformat() if completed_at else None),
                    }
                )

            if deps.logger:
                deps.logger.debug(
                    "get_action_history returned",
                    count=len(items),
                    system_id=state.system_id,
                )

            return {"items": items}

        except Exception as e:
            if deps.logger:
                deps.logger.error("get_action_history failed", error=str(e))
            return ToolError(
                code="query_failed",
                message=f"Failed to query action history: {str(e)}",
                recoverable=True,
            ).to_dict()


class ListSupportedActionsTool(Tool):
    """Tool to list actions supported by the managed system."""

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "list_supported_actions"

    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return (
            "List actions supported by the managed system. First tries the connector, "
            "then falls back to historical inference. "
            "Parameters: window_seconds for historical fallback (1-31536000, default 2592000)"
        )

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute list_supported_actions tool."""
        # First, try to get actions from connector if available
        if deps.connector is not None:
            try:
                if hasattr(deps.connector, "get_supported_actions"):
                    actions = await deps.connector.get_supported_actions()
                    types = sorted(
                        {
                            action_type
                            for a in (actions or [])
                            if (action_type := getattr(a, "action_type", None))
                        }
                    )
                    if types:
                        if deps.logger:
                            deps.logger.debug(
                                "list_supported_actions from connector",
                                count=len(types),
                            )
                        return {"action_types": types, "source": "connector"}
            except Exception as e:
                if deps.logger:
                    deps.logger.warning(
                        "Connector get_supported_actions failed, falling back to history",
                        system_id=state.system_id,
                        error=str(e),
                    )
                if deps.metrics:
                    deps.metrics.increment(
                        "polaris.tool.list_supported_actions.fallback",
                        tags={"reason": "connector_failed"},
                    )

        # Fallback: infer from historical actions
        try:
            window_seconds = self._clamp_int(
                args.get("window_seconds"), 1, 365 * 24 * 3600, 30 * 24 * 3600  # Max 1 year
            )

            from datetime import datetime as dt
            from datetime import timedelta, timezone

            end = dt.now(timezone.utc)
            start = end - timedelta(seconds=window_seconds)

            history = await deps.knowledge_store.query_actions(state.system_id, start, end)

            types = sorted(
                {
                    action_type
                    for a, _ in history
                    if (action_type := getattr(a, "action_type", None))
                }
            )

            if deps.logger:
                deps.logger.debug(
                    "list_supported_actions from history",
                    count=len(types),
                    system_id=state.system_id,
                )

            return {"action_types": types, "source": "historical"}

        except Exception as e:
            if deps.logger:
                deps.logger.error("list_supported_history fallback failed", error=str(e))
            return ToolError(
                code="query_failed",
                message=f"Failed to list supported actions: {str(e)}",
                recoverable=True,
            ).to_dict()


def get_builtin_tools() -> List[Tool]:
    """Get all built-in tools as a list.

    Returns:
        List of all built-in tool instances
    """
    return [
        GetRecentStatesTool(),
        SummarizeMetricTrendsTool(),
        GetWorldModelInsightsTool(),
        PredictOutcomeTool(),
        GetActionHistoryTool(),
        ListSupportedActionsTool(),
    ]
