"""Built-in tools for POLARIS agentic strategies.

This module implements the six core tools used by AgenticLLMStrategy and
MultiAgentStrategy. Each tool is extracted from the original hardcoded _execute_tool
implementations.
"""

import ast
import math
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from polaris.core.models import AdaptationAction
from polaris.tools.base import Tool, ToolDependencies, ToolError
from polaris.tools.utils import clamp_int, extract_metric_values, get_time_window

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
        """Get tool description."""
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
            limit = clamp_int(args.get("limit"), 1, 200, 50)

            # Calculate time window
            start, end = get_time_window(args, 600, 3600)

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
            start, end = get_time_window(args, 600, 3600)

            # Query states
            states = await deps.knowledge_store.query_states(state.system_id, start, end)

            # Extract metric values
            vals = extract_metric_values(states, metric)

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


class ListMetricFieldsTool(Tool):
    """Tool to discover metric fields (optionally numeric-only) from recent states.

    This helps LLMs avoid guessing metric names. It returns the set of metric keys
    observed in the queried window plus a "numeric" subset.
    """

    @property
    def name(self) -> str:
        """Get tool name."""
        return "list_metric_fields"

    @property
    def description(self) -> str:
        """Get tool description."""
        return (
            "List metric field names seen in recent system states, and identify which are numeric. "
            "Parameters: window_seconds (1-3600, default 600), limit (1-500, default 200), "
            "numeric_only (bool, default false)."
        )

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute tool."""
        try:
            limit = clamp_int(args.get("limit"), 1, 500, 200)
            numeric_only = bool(args.get("numeric_only", False))
            start, end = get_time_window(args, 600, 3600)

            states = await deps.knowledge_store.query_states(state.system_id, start, end)
            if states:
                states = states[-limit:]

            all_fields: set[str] = set()
            numeric_fields: set[str] = set()
            stats: Dict[str, Dict[str, float]] = {}

            for s in states or []:
                for k, mv in (getattr(s, "metrics", {}) or {}).items():
                    all_fields.add(k)
                    try:
                        v = float(mv.value)
                    except (TypeError, ValueError):
                        continue
                    numeric_fields.add(k)
                    slot = stats.get(k)
                    if slot is None:
                        stats[k] = {"count": 1.0, "min": v, "max": v, "avg": v}
                    else:
                        c = slot["count"] + 1.0
                        slot["count"] = c
                        slot["min"] = min(slot["min"], v)
                        slot["max"] = max(slot["max"], v)
                        slot["avg"] = slot["avg"] + (v - slot["avg"]) / c

            fields_out = sorted(numeric_fields if numeric_only else all_fields)
            return {
                "window_seconds": int((end - start).total_seconds()),
                "count_states": len(states or []),
                "fields": fields_out,
                "numeric_fields": sorted(numeric_fields),
                "numeric_field_stats": stats,
            }

        except Exception as e:
            if deps.logger:
                deps.logger.error("list_metric_fields failed", error=str(e))
            return ToolError(
                code="query_failed",
                message=f"Failed to list metric fields: {str(e)}",
                recoverable=True,
            ).to_dict()


class ComputeMetricMathTool(Tool):
    """Tool to compute safe math/statistics over arbitrary numeric metric fields.

    Supports two modes:
    - Stats mode: Provide `metric` and `op`.
    - Expression mode: Provide `expression` referencing metric names.

    Examples:
        ```
        - {"metric": "mr1_avg", "op": "avg", "window_seconds": 600}
        - {"expression": "mr1_avg / max(fire_cells_burning_ratio, 1e-6)", "op": "avg"}
        ```
    """

    @property
    def name(self) -> str:
        """Get tool name."""
        return "compute_metric_math"

    @property
    def description(self) -> str:
        """Get tool description."""
        return (
            "Compute math/stats on numeric metrics over a recent time window. "
            "Parameters: window_seconds (1-3600, default 600), limit (1-500, default 200), "
            "op (one of: count|min|max|avg|sum|std|latest|delta|rate_per_s, default avg), "
            "metric (string) OR expression (string). Expression may reference metric names and "
            "uses safe functions: abs, min, max, round, log, log10, exp, sqrt."
        )

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute tool."""
        try:
            limit = clamp_int(args.get("limit"), 1, 500, 200)
            op = str(args.get("op", "avg")).strip().lower()
            metric = str(args.get("metric", "")).strip() or None
            expression = args.get("expression")
            if expression is not None:
                expression = str(expression).strip()
                if not expression:
                    expression = None

            if not metric and not expression:
                return ToolError(
                    code="missing_input",
                    message="Provide either 'metric' or 'expression'",
                    recoverable=True,
                ).to_dict()

            start, end = get_time_window(args, 600, 3600)
            states = await deps.knowledge_store.query_states(state.system_id, start, end)
            if states:
                states = states[-limit:]

            series: List[Tuple[float, float]] = []  # (t_seconds, value)
            for s in states or []:
                t = getattr(s, "timestamp", None)
                if not t:
                    continue
                t_s = t.timestamp()
                if expression:
                    val = self._eval_expression_on_state(expression, s)
                else:
                    val = self._get_metric_value(s, metric)  # type: ignore[arg-type]
                if val is None:
                    continue
                series.append((t_s, float(val)))

            values = [v for _, v in series]

            if op == "count":
                return {"op": op, "count": len(values)}

            if not values:
                return {
                    "op": op,
                    "metric": metric,
                    "expression": expression,
                    "count": 0,
                }

            result: Dict[str, Any] = {
                "op": op,
                "metric": metric,
                "expression": expression,
                "count": len(values),
                "window_seconds": int((end - start).total_seconds()),
            }

            if op == "min":
                result["value"] = min(values)
            elif op == "max":
                result["value"] = max(values)
            elif op == "sum":
                result["value"] = float(sum(values))
            elif op == "avg":
                result["value"] = float(sum(values) / len(values))
            elif op == "std":
                mu = sum(values) / len(values)
                var = sum((x - mu) ** 2 for x in values) / max(1, (len(values) - 1))
                result["value"] = float(math.sqrt(var))
            elif op == "latest":
                result["value"] = float(series[-1][1])
                result["timestamp"] = series[-1][0]
            elif op == "delta":
                result["value"] = float(series[-1][1] - series[0][1])
                result["from"] = series[0][1]
                result["to"] = series[-1][1]
            elif op == "rate_per_s":
                if len(series) < 2:
                    result["value"] = 0.0
                else:
                    dt = series[-1][0] - series[0][0]
                    result["value"] = float((series[-1][1] - series[0][1]) / dt) if dt else 0.0
                    result["dt_s"] = float(dt)
            else:
                return ToolError(
                    code="invalid_op",
                    message=f"Unsupported op '{op}'",
                    recoverable=True,
                ).to_dict()

            return result

        except ToolError as te:
            return te.to_dict()
        except Exception as e:
            if deps.logger:
                deps.logger.error("compute_metric_math failed", error=str(e))
            return ToolError(
                code="execution_error",
                message=f"Failed to compute metric math: {str(e)}",
                recoverable=True,
            ).to_dict()

    def _get_metric_value(self, state_obj: Any, metric: str) -> Optional[float]:
        metrics = getattr(state_obj, "metrics", {}) or {}
        mv = metrics.get(metric)
        if mv is None:
            return None
        try:
            return float(getattr(mv, "value", mv))
        except (TypeError, ValueError):
            return None

    def _eval_expression_on_state(self, expression: str, state_obj: Any) -> Optional[float]:
        metrics = getattr(state_obj, "metrics", {}) or {}
        env: Dict[str, float] = {}
        for k, mv in metrics.items():
            try:
                env[k] = float(mv.value)
            except (TypeError, ValueError):
                continue

        fn_env = {
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "sqrt": math.sqrt,
        }

        node = ast.parse(expression, mode="eval")
        self._assert_safe_expr(node)
        value = eval(
            compile(node, "<metric_expr>", "eval"),
            {"__builtins__": {}},
            {**fn_env, **env},
        )
        return float(value)

    def _assert_safe_expr(self, node: ast.AST) -> None:
        allowed_nodes: Tuple[type, ...] = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Call,
            ast.Load,
            ast.Name,
            ast.Constant,
            ast.Compare,
            ast.Gt,
            ast.GtE,
            ast.Lt,
            ast.LtE,
            ast.Eq,
            ast.NotEq,
            ast.IfExp,
        )

        for sub in ast.walk(node):
            if not isinstance(sub, allowed_nodes):
                raise ToolError(
                    code="unsafe_expression",
                    message=f"Expression contains unsupported syntax: {type(sub).__name__}",
                    recoverable=True,
                )
            if isinstance(sub, ast.Call):
                if not isinstance(sub.func, ast.Name):
                    raise ToolError(
                        code="unsafe_expression",
                        message="Only simple function calls are allowed",
                        recoverable=True,
                    )
                if sub.func.id not in {"abs", "min", "max", "round", "log", "log10", "exp", "sqrt"}:
                    raise ToolError(
                        code="unsafe_expression",
                        message=f"Function '{sub.func.id}' is not allowed",
                        recoverable=True,
                    )


class GetWorldModelInsightsTool(Tool):
    """Tool to retrieve insights from the world model."""

    @property
    def name(self) -> str:
        """Get tool name."""
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
        """Get tool name."""
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
        """Get tool name."""
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
            window_seconds = clamp_int(
                args.get("window_seconds"), 1, 30 * 24 * 3600, 86400  # Max 30 days
            )
            limit = clamp_int(args.get("limit"), 1, 500, 50)

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
        """Get tool name."""
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
            window_seconds = clamp_int(
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


class SleepTool(Tool):
    """Tool to purposefully wait for a specified duration."""

    @property
    def name(self) -> str:
        """Get tool name."""
        return "sleep"

    @property
    def description(self) -> str:
        """Get tool description."""
        return "Wait for a specified number of seconds. Parameters: duration_seconds (1-300, default 5)"

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute tool."""
        import asyncio

        duration = clamp_int(args.get("duration_seconds"), 1, 300, 5)
        if deps.logger:
            deps.logger.debug("sleep tool called", duration=duration)
        await asyncio.sleep(duration)
        return {"slept_for_seconds": duration}


class GetSystemStatusTool(Tool):
    """Tool to retrieve a quick snapshot of the instantaneous system state."""

    @property
    def name(self) -> str:
        """Get tool name."""
        return "get_system_status"

    @property
    def description(self) -> str:
        """Get tool description."""
        return "Retrieve the current instantaneous system state, including metrics, without querying historical data. No parameters."

    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute tool."""
        metrics_dict = {}
        for name, mv in state.metrics.items():
            metrics_dict[name] = mv.value

        return {
            "system_id": state.system_id,
            "timestamp": state.timestamp.isoformat(),
            "metrics": metrics_dict,
            "status": (
                state.health_status.value if getattr(state, "health_status", None) else "UNKNOWN"
            ),
        }


def get_builtin_tools() -> List[Tool]:
    """Get all built-in tools as a list.

    Returns:
        List of all built-in tool instances
    """
    return [
        GetRecentStatesTool(),
        SummarizeMetricTrendsTool(),
        ListMetricFieldsTool(),
        ComputeMetricMathTool(),
        GetWorldModelInsightsTool(),
        PredictOutcomeTool(),
        GetActionHistoryTool(),
        ListSupportedActionsTool(),
        SleepTool(),
        GetSystemStatusTool(),
    ]
