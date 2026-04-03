"""Agentic LLM-based adaptation strategy for POLARIS.

This module implements an adaptation strategy that uses a Large Language Model (LLM) as
an agentic reasoning engine to make adaptation decisions. The strategy employs a tool-
using approach where the LLM can query system state, analyze metrics, predict outcomes,
and ultimately decide whether adaptation is needed.

The strategy follows a step-by-step reasoning process: 1. Analyzes current system state
and context 2. Uses available tools to gather additional information 3. Makes a final
decision on adaptation needs 4. Proposes specific adaptation actions if needed
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.abstractions.world_model import WorldModel
from polaris.core.models import AdaptationAction, SystemState
from polaris.infrastructure.constants import DEFAULT_MAX_TOKENS_REASONING
from polaris.infrastructure.llm import LLMClient, LLMMessage
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector
from polaris.strategies.action_resolution import (
    ConnectorActionResolver,
    StrictContractViolation,
    require_supported_action_contract,
    resolve_strict_action_payload,
)
from polaris.strategies.utils import (
    DEFAULT_ALLOWED_TOOLS,
    format_system_state_for_llm,
    parse_strict_json,
)
from polaris.tools import ToolRegistry, get_builtin_tools


class ActionBlock(BaseModel):
    """A block defining an action to be executed."""

    type: str = Field(description="The name or type of the action to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters required for the action"
    )


class FinalDecisionBlock(BaseModel):
    """A block containing the final adaptation decision."""

    needs_adaptation: bool = Field(description="True if adaptation is needed, False otherwise")
    reasoning: str = Field(description="Explanation of why this decision was made")
    actions: List[ActionBlock] = Field(
        default_factory=list, description="List of actions to execute"
    )


class AgenticResponseSchema(BaseModel):
    """Schema for responses from the agentic LLM reasoning engine."""

    tool: Optional[str] = Field(
        None, description="Name of the tool to call. Leave null if making a final decision."
    )
    args: Optional[Dict[str, Any]] = Field(
        None, description="Arguments to pass to the tool. Leave null if making a final decision."
    )
    final: Optional[FinalDecisionBlock] = Field(
        None,
        description="Provide this block only when you are ready to make a final adaptation decision.",
    )


class AgenticLLMStrategy(AdaptationStrategy):
    """An adaptation strategy that uses LLM as an agentic reasoning engine.

    This strategy leverages a Large Language Model to make intelligent adaptation
    decisions by using a tool-based approach. The LLM can query system state, analyze
    historical data, predict outcomes, and propose adaptation actions.

    Attributes:
        llm: The LLM client for generating responses
        knowledge_store: Store for querying historical system data
        world_model: World model for predicting action outcomes
        steps_limit: Maximum number of reasoning steps allowed
        temperature: LLM temperature parameter for response randomness
        allowed_tools: List of tools the LLM can use
    """

    requires_system_contract: bool = True

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        steps_limit: int = 3,
        temperature: float = 0.1,
        decision_cooldown_seconds: float = 60.0,
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
            steps_limit: Maximum number of reasoning steps (default: 3)
            temperature: LLM temperature for response randomness (default: 0.1)
            decision_cooldown_seconds: Minimum seconds between consecutive
                adaptation decisions (default: 60.0)
            allowed_tools: List of permitted tools for the LLM
            system_prompt: Optional custom system prompt template
            per_system_prompts: Optional per-system prompt overrides keyed by system_id
            logger: Optional logger for debugging
            metrics: Optional metrics collector for monitoring
        """
        self.llm = llm_client
        self.knowledge_store = knowledge_store
        self.world_model = world_model
        self.steps_limit = steps_limit
        self.temperature = temperature
        self.decision_cooldown_seconds = max(0.0, float(decision_cooldown_seconds))
        self.allowed_tools = allowed_tools or list(DEFAULT_ALLOWED_TOOLS)
        self._system_prompt_template = system_prompt
        self._per_system_prompts = per_system_prompts or {}
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()
        self._action_resolver = ConnectorActionResolver()
        # Initialize tool registry with built-in tools
        self._tool_registry = ToolRegistry(metrics=self.metrics)
        all_tools = get_builtin_tools()
        if self.allowed_tools:
            # Only register allowed tools
            for tool in all_tools:
                if tool.name in self.allowed_tools:
                    self._tool_registry.register(tool)
        else:
            self._tool_registry.register_all(all_tools)

        self._adaptation_count = 0
        self._success_count = 0
        self._last_decision_time: Optional[datetime] = None

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Assess system state and determine if adaptation is needed.

        Uses the LLM to analyze the current system state and context through a tool-
        using reasoning process. The LLM can query historical data, analyze trends, and
        predict outcomes before making a final decision.

        Args:
            state: Current system state with metrics and health information
            context: Adaptation context containing world model insights

        Returns:
            Optional[AdaptationAction]: Proposed adaptation action if needed, None if no
                adaptation is required
        """
        now = datetime.now(timezone.utc)
        if self.decision_cooldown_seconds > 0 and self._last_decision_time is not None:
            elapsed = (now - self._last_decision_time).total_seconds()
            if elapsed < self.decision_cooldown_seconds:
                if self.logger:
                    self.logger.debug(
                        "Agentic decision cooldown active",
                        system_id=state.system_id,
                        remaining_seconds=round(self.decision_cooldown_seconds - elapsed, 1),
                    )
                self.metrics.increment(
                    "polaris.strategy.agentic.decision_cooldown_skips",
                    tags={"system_id": state.system_id},
                )
                return []

        if self.logger:
            self.logger.debug("Agentic assessment started", system_id=state.system_id)
        self.metrics.increment(
            "polaris.strategy.agentic.assessments", tags={"system_id": state.system_id}
        )
        _contract, supported_action_types, action_aliases = require_supported_action_contract(
            context,
            strategy_name="agentic",
        )
        start = datetime.now(timezone.utc)
        messages: List[LLMMessage] = [
            LLMMessage(
                role="system",
                content=self._system_prompt(state.system_id, supported_action_types),
            ),
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
                    messages,
                    temperature=self.temperature,
                    max_tokens=DEFAULT_MAX_TOKENS_REASONING,
                    response_schema=AgenticResponseSchema,
                )

                # Optional deep debug to help diagnose provider-specific formatting issues.
                # Enabled via env var to avoid logging large model outputs by default.
                self._maybe_log_llm_response(
                    system_id=state.system_id,
                    step=step + 1,
                    content=getattr(response, "content", ""),
                )
                self.metrics.histogram(
                    "polaris.strategy.agentic.llm_call_duration_seconds",
                    (datetime.now(timezone.utc) - llm_start).total_seconds(),
                    tags={"system_id": state.system_id},
                )
                parsed = self._parse_json_object(response.content)

                try:
                    structured_response = AgenticResponseSchema.model_validate(parsed)
                except Exception as exc:
                    raise StrictContractViolation(
                        f"Agentic response failed schema validation: {exc}"
                    ) from exc

                if structured_response.final is not None:
                    final = structured_response.final
                    if not isinstance(final.reasoning, str) or not final.reasoning.strip():
                        raise StrictContractViolation(
                            "Agentic final response requires non-empty 'reasoning'"
                        )
                    if not final.needs_adaptation:
                        if self.logger:
                            self.logger.info(
                                "Agentic decision: no adaptation", system_id=state.system_id
                            )
                        return []

                    if not final.actions:
                        raise StrictContractViolation(
                            "Agentic final response with needs_adaptation=true requires non-empty 'actions'"
                        )

                    proposed_actions: List[AdaptationAction] = []
                    for ab in final.actions:
                        resolved_action_type, resolved_parameters = resolve_strict_action_payload(
                            resolver=self._action_resolver,
                            action_type=ab.type,
                            parameters=ab.parameters,
                            supported_action_types=supported_action_types,
                            action_aliases=action_aliases,
                            system_id=state.system_id,
                            missing_type_error="Agentic action requires non-empty 'type'",
                            invalid_parameters_error="Agentic action requires object 'parameters'",
                        )

                        proposed_actions.append(
                            AdaptationAction(
                                action_id=str(uuid.uuid4()),
                                action_type=resolved_action_type,
                                target_system=state.system_id,
                                parameters={
                                    **resolved_parameters,
                                    "llm_reasoning": final.reasoning,
                                },
                            )
                        )

                    if self.logger:
                        self.logger.info(
                            "Agentic decision: propose actions",
                            system_id=state.system_id,
                            action_count=len(proposed_actions),
                        )
                    for action in proposed_actions:
                        self.metrics.increment(
                            "polaris.strategy.agentic.actions_proposed",
                            tags={
                                "system_id": state.system_id,
                                "action_type": action.action_type,
                            },
                        )
                    self._last_decision_time = datetime.now(timezone.utc)
                    return proposed_actions

                tool = structured_response.tool
                args = structured_response.args or {}
                if not tool:
                    raise StrictContractViolation(
                        "Agentic response must include either a 'final' block or a valid 'tool'"
                    )
                if tool not in self.allowed_tools:
                    self.metrics.increment(
                        "polaris.strategy.agentic.invalid_tool", tags={"tool": str(tool)}
                    )
                    raise StrictContractViolation(f"Tool '{tool}' is not in allowed tool list")
                else:
                    tool_result = await self._execute_tool(tool, args, state, context)
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
            raise StrictContractViolation(
                "Agentic strategy reached step limit without producing a final decision"
            )
        finally:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            self.metrics.histogram(
                "polaris.strategy.agentic.assess_duration_seconds",
                duration,
                tags={"system_id": state.system_id},
            )

    async def _execute_tool(
        self,
        tool: str,
        args: Dict[str, Any],
        state: SystemState,
        context: AdaptationContext,
    ) -> Dict[str, Any]:
        """Execute a strategy tool with connector/world/knowledge dependencies."""
        from polaris.tools import ToolDependencies

        deps = ToolDependencies(
            knowledge_store=self.knowledge_store,
            world_model=self.world_model,
            system_contract=context.system_contract,
            logger=self.logger,
            metrics=self.metrics,
        )

        try:
            tool_result = await self._tool_registry.execute(
                tool_name=tool,
                args=args,
                state=state,
                context=context,
                deps=deps,
            )
            self.metrics.increment(
                "polaris.strategy.agentic.tool_called",
                tags={"tool": tool, "system_id": state.system_id},
            )
            return tool_result
        except Exception as exc:
            self.metrics.increment(
                "polaris.strategy.agentic.tool_error",
                tags={"tool": tool, "system_id": state.system_id},
            )
            if self.logger:
                self.logger.error("Agentic tool execution error", tool=tool, error=str(exc))
            return {"error": f"tool_error: {type(exc).__name__}: {str(exc)}"}

    async def on_action_executed(self, action: AdaptationAction, result: Any) -> None:
        """Handle callback when an adaptation action is executed.

        Updates internal metrics tracking adaptation success rates and publishes
        execution metrics for monitoring.

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
            "decision_cooldown_seconds": ParameterSpec(
                current_value=self.decision_cooldown_seconds,
                type=float,
                min_value=0.0,
                max_value=3600.0,
                description="",
                kind="cooldown",
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
        if parameter_path == "decision_cooldown_seconds":
            self.decision_cooldown_seconds = max(0.0, float(new_value))
            return True
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply configuration updates to the strategy.

        Updates parameters, tool availability, and resilience settings based on the
        provided configuration dictionary.

        Args:
            config: Configuration dictionary with updates to apply
        """
        if not isinstance(config, dict):
            return  # type: ignore[unreachable]

        if "temperature" in config:
            await self.update_parameter("temperature", config["temperature"])
        if "steps_limit" in config:
            await self.update_parameter("steps_limit", config["steps_limit"])
        if "decision_cooldown_seconds" in config:
            await self.update_parameter(
                "decision_cooldown_seconds", config["decision_cooldown_seconds"]
            )

        if "system_prompt" in config:
            self._system_prompt_template = config["system_prompt"]
        if "per_system_prompts" in config and isinstance(config["per_system_prompts"], dict):
            self._per_system_prompts = config["per_system_prompts"]

        # Update allowed tools in registry if changed
        if "tools" in config:
            tools_cfg = config["tools"]
            if isinstance(tools_cfg, dict):
                enabled = tools_cfg.get("enabled")
                if isinstance(enabled, list):
                    self.allowed_tools = enabled
                    # Re-initialize registry with new allowed tools
                    self._tool_registry = ToolRegistry(metrics=self.metrics)
                    all_tools = get_builtin_tools()
                    for tool in all_tools:
                        if tool.name in self.allowed_tools:
                            self._tool_registry.register(tool)

        resil = config.get("resilience")
        if resil and hasattr(self.llm, "update_resilience"):
            try:
                self.llm.update_resilience(resil)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "AgenticLLMStrategy resilience update failed", error=str(exc)
                    )

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the strategy.

        Returns:
            Dict[str, float]: Performance metrics including success rate and total
                adaptations count
        """
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}
        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }

    def _system_prompt(
        self,
        system_id: Optional[str] = None,
        supported_action_types: Optional[List[str]] = None,
    ) -> str:
        supported_actions_text = (
            ", ".join(supported_action_types)
            if supported_action_types
            else "unknown (use connector-supported canonical action names)"
        )

        # Per-system override if provided
        if system_id and self._per_system_prompts:
            override = self._per_system_prompts.get(system_id)
            if override:
                try:
                    return override.format(
                        system_id=system_id,
                        allowed_tools=", ".join(self.allowed_tools),
                        supported_actions=supported_actions_text,
                    )
                except (KeyError, IndexError, ValueError):
                    return override

        tools = ", ".join(self.allowed_tools)

        # Global template override, optionally formatted
        if self._system_prompt_template:
            try:
                return self._system_prompt_template.format(
                    system_id=system_id or "",
                    allowed_tools=tools,
                    supported_actions=supported_actions_text,
                )
            except (KeyError, IndexError, ValueError):
                return self._system_prompt_template

        tool_descriptions = self._get_tool_descriptions()

        return (
            "You are an adaptation controller. Use a short tool-using loop "
            "to reason about the system and then decide.\n"
            "Always reply as strict JSON. Two possible forms:\n"
            '1) {"tool": "name", "args": {...}} to request a tool.\n'
            '2) {"final": {"needs_adaptation": true|false, "reasoning": "...", '
            '"actions": [{"type": "...", "parameters": {...}}]}} to finish.\n'
            "IMPORTANT: You can propose MULTIPLE actions in the 'actions' list if "
            "it helps achieve system goals more effectively.\n"
            f"Connector-supported action types: {supported_actions_text}.\n"
            f"Allowed tools: {tools}.\n"
            f"Tool descriptions:\n{tool_descriptions}\n"
            "Keep steps minimal."
        )

    def _initial_user_prompt(self, state: SystemState, context: AdaptationContext) -> str:
        return json.dumps(
            {"current_state": json.loads(format_system_state_for_llm(state, context))}
        )

    def _parse_json_object(self, content: str) -> Dict[str, Any]:
        """Parse strict JSON object from model output."""
        return parse_strict_json(content, StrictContractViolation)

    def _maybe_log_llm_response(self, system_id: str, step: int, content: str) -> None:
        """Optionally log raw LLM output for debugging parsing issues.

        Controlled by env:
        - POLARIS_LOG_LLM_RAW=1 to enable
        - POLARIS_LOG_LLM_RAW_MAX_CHARS (default 4000)
        """
        import os

        if not self.logger:
            return
        if os.getenv("POLARIS_LOG_LLM_RAW", "").strip() not in {"1", "true", "TRUE", "yes", "YES"}:
            return

        try:
            max_chars = int(os.getenv("POLARIS_LOG_LLM_RAW_MAX_CHARS", "4000"))
        except (TypeError, ValueError):
            max_chars = 4000

        safe = content or ""
        if len(safe) > max_chars:
            safe = safe[:max_chars] + f"\n...<truncated {len(content) - max_chars} chars>"

        self.logger.debug(
            "LLM raw response",
            system_id=system_id,
            step=step,
            llm_raw=safe,
        )

    def _get_tool_descriptions(self) -> str:
        """Get descriptions of available tools for the system prompt."""
        descriptions = []
        for name, desc in self._tool_registry.get_tool_descriptions().items():
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)
