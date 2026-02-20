"""Agentic LLM-based adaptation strategy for POLARIS.

This module implements an adaptation strategy that uses a Large Language Model (LLM)
as an agentic reasoning engine to make adaptation decisions. The strategy employs
a tool-using approach where the LLM can query system state, analyze metrics,
predict outcomes, and ultimately decide whether adaptation is needed.

The strategy follows a step-by-step reasoning process:
1. Analyzes current system state and context
2. Uses available tools to gather additional information
3. Makes a final decision on adaptation needs
4. Proposes specific adaptation actions if needed
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

if TYPE_CHECKING:
    from polaris.abstractions.connector import Connector

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.abstractions.world_model import WorldModel
from polaris.core.models import AdaptationAction, MetricValue, SystemState
from polaris.infrastructure.llm import LLMClient, LLMMessage
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector


def _get_connector_class() -> Type["Connector"]:
    from polaris.abstractions.connector import Connector

    return Connector


class AgenticLLMStrategy(AdaptationStrategy):
    """An adaptation strategy that uses LLM as an agentic reasoning engine.

    This strategy leverages a Large Language Model to make intelligent adaptation
    decisions by using a tool-based approach. The LLM can query system state,
    analyze historical data, predict outcomes, and propose adaptation actions.

    Attributes:
        llm: The LLM client for generating responses
        knowledge_store: Store for querying historical system data
        world_model: World model for predicting action outcomes
        steps_limit: Maximum number of reasoning steps allowed
        temperature: LLM temperature parameter for response randomness
        allowed_tools: List of tools the LLM can use
    """

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        connector_getter: Optional[Callable[[str], Optional["Connector"]]] = None,
        steps_limit: int = 3,
        temperature: float = 0.1,
        allowed_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        per_system_prompts: Optional[Dict[str, str]] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize the AgenticLLMStrategy.

        Args:
            llm_client: LLM client for generating responses
            knowledge_store: Store for querying historical system data
            world_model: World model for predicting action outcomes
            connector_getter: Optional function to get system connectors
            steps_limit: Maximum number of reasoning steps (default: 3)
            temperature: LLM temperature for response randomness (default: 0.1)
            allowed_tools: List of permitted tools for the LLM
            system_prompt: Optional custom system prompt template
            per_system_prompts: Optional per-system prompt overrides keyed by system_id
            logger: Optional logger for debugging
            metrics: Optional metrics collector for monitoring
        """
        self.llm = llm_client
        self.knowledge_store = knowledge_store
        self.world_model = world_model
        self._get_connector = connector_getter
        self.steps_limit = steps_limit
        self.temperature = temperature
        self.allowed_tools = allowed_tools or [
            "get_recent_states",
            "summarize_metric_trends",
            "get_world_model_insights",
            "predict_outcome",
            "get_action_history",
            "list_supported_actions",
        ]
        self._system_prompt_template = system_prompt
        self._per_system_prompts = per_system_prompts or {}
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()
        self._adaptation_count = 0
        self._success_count = 0

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Assess system state and determine if adaptation is needed.

        Uses the LLM to analyze the current system state and context through
        a tool-using reasoning process. The LLM can query historical data,
        analyze trends, and predict outcomes before making a final decision.

        Args:
            state: Current system state with metrics and health information
            context: Adaptation context containing world model insights

        Returns:
            Optional[AdaptationAction]: Proposed adaptation action if needed,
                None if no adaptation is required
        """
        if self.logger:
            self.logger.debug("Agentic assessment started", system_id=state.system_id)
        self.metrics.increment(
            "polaris.strategy.agentic.assessments", tags={"system_id": state.system_id}
        )
        start = datetime.now(timezone.utc)
        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=self._system_prompt(state.system_id)),
            LLMMessage(role="user", content=self._initial_user_prompt(state, context)),
        ]
        try:
            for step in range(self.steps_limit):
                self.metrics.gauge(
                    "polaris.strategy.agentic.step",
                    step + 1,
                    tags={"system_id": state.system_id},
                )
                llm_start = datetime.now(timezone.utc)
                response = await self.llm.generate(
                    messages, temperature=self.temperature, max_tokens=2048
                )
                self.metrics.histogram(
                    "polaris.strategy.agentic.llm_call_duration_seconds",
                    (datetime.now(timezone.utc) - llm_start).total_seconds(),
                    tags={"system_id": state.system_id},
                )
                parsed = self._parse_json(response.content)
                if not isinstance(parsed, dict):
                    if self.logger:
                        self.logger.debug(
                            "AgenticLLMStrategy received non-object JSON",
                            content_preview=response.content[:300],
                        )
                    break
                if "final" in parsed:
                    final = parsed.get("final") or {}
                    if not isinstance(final, dict):
                        break
                    needs = bool(final.get("needs_adaptation", False))
                    if not needs:
                        if self.logger:
                            self.logger.info(
                                "Agentic decision: no adaptation", system_id=state.system_id
                            )
                        return []

                    action_block = final.get("action")
                    actions_list = final.get("actions")

                    raw_actions = []
                    if isinstance(actions_list, list):
                        raw_actions = actions_list
                    elif isinstance(action_block, dict):
                        raw_actions = [action_block]

                    if not raw_actions:
                        if self.logger:
                            self.logger.warning(
                                "Agentic decision: needs_adaptation=true but no actions provided",
                                system_id=state.system_id,
                            )
                        return []

                    reasoning = str(final.get("reasoning", ""))
                    proposed_actions = []

                    for ab in raw_actions:
                        if not isinstance(ab, dict):
                            continue
                        at = ab.get("type")
                        ap = ab.get("parameters") or {}
                        if not at or not isinstance(ap, dict):
                            continue

                        proposed_actions.append(
                            AdaptationAction(
                                action_id=str(uuid.uuid4()),
                                action_type=str(at),
                                target_system=state.system_id,
                                parameters={**ap, "llm_reasoning": reasoning},
                            )
                        )

                    if not proposed_actions:
                        return []

                    if self.logger:
                        self.logger.info(
                            f"Agentic decision: propose {len(proposed_actions)} actions",
                            system_id=state.system_id,
                        )
                    for action in proposed_actions:
                        self.metrics.increment(
                            "polaris.strategy.agentic.actions_proposed",
                            tags={
                                "system_id": state.system_id,
                                "action_type": action.action_type,
                            },
                        )
                    return proposed_actions
                tool = parsed.get("tool")
                args = parsed.get("args") or {}
                if not tool:
                    break
                if tool not in self.allowed_tools:
                    self.metrics.increment(
                        "polaris.strategy.agentic.invalid_tool", tags={"tool": str(tool)}
                    )
                    tool_result = {"error": f"tool_not_allowed: {tool}"}
                else:
                    try:
                        if self.logger:
                            self.logger.debug("Agentic tool requested", tool=tool, args=args)
                        tool_result = await self._execute_tool(tool, args, state, context)
                        self.metrics.increment(
                            "polaris.strategy.agentic.tool_called",
                            tags={"tool": tool, "system_id": state.system_id},
                        )
                    except Exception as e:
                        self.metrics.increment(
                            "polaris.strategy.agentic.tool_error",
                            tags={"tool": tool, "system_id": state.system_id},
                        )
                        if self.logger:
                            self.logger.error(
                                "Agentic tool execution error", tool=tool, error=str(e)
                            )
                        tool_result = {"error": f"tool_error: {type(e).__name__}: {str(e)}"}
                tool_msg = json.dumps({"tool_result": {"tool": tool, "data": tool_result}})
                messages.append(LLMMessage(role="user", content=tool_msg))
                self.metrics.increment(
                    "polaris.strategy.agentic.step_limit_reached",
                    tags={"system_id": state.system_id},
                )
            if self.logger:
                self.logger.debug(
                    "Agentic step limit reached with no final decision", system_id=state.system_id
                )
            return []
        finally:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            self.metrics.histogram(
                "polaris.strategy.agentic.assess_duration_seconds",
                duration,
                tags={"system_id": state.system_id},
            )

    async def on_action_executed(self, action: AdaptationAction, result: Any) -> None:
        """Handle callback when an adaptation action is executed.

        Updates internal metrics tracking adaptation success rates and
        publishes execution metrics for monitoring.

        Args:
            action: The adaptation action that was executed
            result: The result of the action execution
        """
        self._adaptation_count += 1
        ok = hasattr(result, "status") and getattr(result.status, "value", None) == "success"
        if ok:
            self._success_count += 1
        self.metrics.increment(
            "polaris.strategy.agentic.actions_executed",
            tags={
                "action_type": action.action_type,
                "system_id": action.target_system,
                "status": getattr(getattr(result, "status", None), "value", "unknown"),
            },
        )

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Get specification of tunable parameters for this strategy.

        Returns:
            Dict[str, ParameterSpec]: Mapping of parameter names to their specifications
                including current values, types, bounds, and descriptions
        """
        return {
            "temperature": ParameterSpec(
                current_value=self.temperature,
                type=float,
                min_value=0.0,
                max_value=2.0,
                description="",
                kind="llm_temperature",
            ),
            "steps_limit": ParameterSpec(
                current_value=self.steps_limit,
                type=int,
                min_value=1,
                max_value=10,
                description="",
                kind="agent_steps_limit",
            ),
        }

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a tunable parameter value.

        Args:
            parameter_path: Path to the parameter (e.g., 'temperature')
            new_value: New value for the parameter

        Returns:
            bool: True if parameter was updated successfully, False otherwise
        """
        if parameter_path == "temperature":
            self.temperature = float(new_value)
            return True
        if parameter_path == "steps_limit":
            self.steps_limit = int(new_value)
            return True
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply configuration updates to the strategy.

        Updates parameters, tool availability, and resilience settings based
        on the provided configuration dictionary.

        Args:
            config: Configuration dictionary with updates to apply
        """
        if not isinstance(config, dict):
            return  # type: ignore[unreachable]

        if "temperature" in config:
            await self.update_parameter("temperature", config["temperature"])
        if "steps_limit" in config:
            await self.update_parameter("steps_limit", config["steps_limit"])

        if "system_prompt" in config:
            self._system_prompt_template = config["system_prompt"]
        if "per_system_prompts" in config and isinstance(config["per_system_prompts"], dict):
            self._per_system_prompts = config["per_system_prompts"]

        tools_cfg = config.get("tools")
        if isinstance(tools_cfg, dict):
            enabled = tools_cfg.get("enabled")
            if isinstance(enabled, list):
                self.allowed_tools = enabled

        resil = config.get("resilience")
        if resil and hasattr(self.llm, "update_resilience"):
            try:
                self.llm.update_resilience(resil)
            except Exception as e:
                if self.logger:
                    self.logger.warning("AgenticLLMStrategy resilience update failed", error=str(e))

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the strategy.

        Returns:
            Dict[str, float]: Performance metrics including success rate
                and total adaptations count
        """
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}
        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }

    def _system_prompt(self, system_id: Optional[str] = None) -> str:
        # Per-system override if provided
        if system_id and self._per_system_prompts:
            override = self._per_system_prompts.get(system_id)
            if override:
                return override

        tools = ", ".join(self.allowed_tools)

        # Global template override, optionally formatted
        if self._system_prompt_template:
            try:
                return self._system_prompt_template.format(
                    system_id=system_id or "",
                    allowed_tools=tools,
                )
            except Exception:
                return self._system_prompt_template

        return (
            "You are an adaptation controller. Use a short tool-using loop "
            "to reason about the system and then decide.\n"
            "Always reply as strict JSON. Two possible forms:\n"
            '1) {"tool": "name", "args": {...}} to request a tool.\n'
            '2) {"final": {"needs_adaptation": true|false, "reasoning": "...", '
            '"actions": [{"type": "...", "parameters": {...}}]}} to finish.\n'
            "IMPORTANT: You can propose MULTIPLE actions in the 'actions' list if "
            "it helps achieve system goals more effectively.\n"
            f"Allowed tools: {tools}. Keep steps minimal."
        )

    def _initial_user_prompt(self, state: SystemState, context: AdaptationContext) -> str:
        metrics = []
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
        return json.dumps({"current_state": data})

    def _parse_json(self, content: str) -> Any:
        s = content.strip()
        if not s:
            return {}
        if "```json" in s:
            part = s.split("```json", 1)[1]
            s = part.split("```", 1)[0].strip()
        elif "```" in s:
            part = s.split("```", 1)[1]
            s = part.split("```", 1)[0].strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            import logging

            logging.getLogger(__name__).warning(
                "LLM returned malformed JSON (truncated to 500 chars): %.500s", s
            )
            return {}

    async def _execute_tool(
        self, tool: str, args: Dict[str, Any], state: SystemState, context: AdaptationContext
    ) -> Dict[str, Any]:
        if tool == "get_recent_states":
            window_seconds = int(max(1, min(int(args.get("window_seconds", 600)), 3600)))
            limit = int(max(1, min(int(args.get("limit", 50)), 200)))
            end = datetime.now(timezone.utc)
            start = end - timedelta(seconds=window_seconds)
            states = await self.knowledge_store.query_states(state.system_id, start, end)
            if states:
                states = states[-limit:]
            out: List[Dict[str, Any]] = []
            for s in states:
                m = {}
                for name, mv in s.metrics.items():
                    try:
                        m[name] = float(mv.value)
                    except Exception:
                        pass
                out.append({"timestamp": s.timestamp.isoformat(), "metrics": m})
            return {"states": out}
        if tool == "summarize_metric_trends":
            metric = str(args.get("metric", "")).strip()
            if not metric:
                return {"error": "missing_metric"}
            window_seconds = int(max(1, min(int(args.get("window_seconds", 600)), 3600)))
            end = datetime.now(timezone.utc)
            start = end - timedelta(seconds=window_seconds)
            states = await self.knowledge_store.query_states(state.system_id, start, end)
            vals: List[float] = []
            for s in states:
                metric_value: Optional[MetricValue] = s.metrics.get(metric)
                if metric_value is None or metric_value.value is None:
                    continue
                try:
                    vals.append(float(metric_value.value))
                except Exception:
                    continue
            if not vals:
                return {"count": 0}
            return {
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "avg": sum(vals) / len(vals),
            }
        if tool == "get_world_model_insights":
            insights = await self.world_model.get_insights()
            return {"insights": insights}
        if tool == "predict_outcome":
            block = args.get("candidate_action") or {}
            if not isinstance(block, dict):
                return {"error": "invalid_candidate_action"}
            a_type = block.get("type")
            params = block.get("parameters") or {}
            if not a_type or not isinstance(params, dict):
                return {"error": "invalid_candidate_action"}
            candidate = AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type=str(a_type),
                target_system=state.system_id,
                parameters=params,
            )
            pred = await self.world_model.predict(candidate, state)
            return {
                "predicted_metrics": pred.predicted_metrics,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning,
            }
        if tool == "get_action_history":
            window_seconds = int(
                max(1, min(int(args.get("window_seconds", 86400)), 30 * 24 * 3600))
            )
            limit = int(max(1, min(int(args.get("limit", 50)), 500)))
            end = datetime.now(timezone.utc)
            start = end - timedelta(seconds=window_seconds)
            history = await self.knowledge_store.query_actions(state.system_id, start, end)
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
                        "completed_at": completed_at.isoformat() if completed_at else None,
                    }
                )
            return {"items": items}
        if tool == "list_supported_actions":
            # Prefer connector-reported supported actions if available
            if callable(self._get_connector):
                try:
                    connector = self._get_connector(state.system_id)
                except Exception:
                    connector = None
                if connector is not None and hasattr(connector, "get_supported_actions"):
                    try:
                        actions = await connector.get_supported_actions()
                        types = sorted(
                            {
                                action_type
                                for a in (actions or [])
                                if (action_type := getattr(a, "action_type", None))
                            }
                        )
                        if types:
                            return {"action_types": types, "source": "connector"}
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(
                                "Connector get_supported_actions failed, falling back to history",
                                system_id=state.system_id,
                                error=str(e),
                            )
                        self.metrics.increment(
                            "polaris.strategy.agentic.tool_fallback",
                            tags={
                                "tool": "list_supported_actions",
                                "reason": "connector_failed",
                            },
                        )
            # Fallback to historical inference
            window_seconds = int(
                max(1, min(int(args.get("window_seconds", 30 * 24 * 3600)), 365 * 24 * 3600))
            )
            end = datetime.now(timezone.utc)
            start = end - timedelta(seconds=window_seconds)
            history = await self.knowledge_store.query_actions(state.system_id, start, end)
            types = sorted(
                {
                    action_type
                    for a, _ in history
                    if (action_type := getattr(a, "action_type", None))
                }
            )
            return {"action_types": types, "source": "historical"}
        return {"error": "unknown_tool"}
