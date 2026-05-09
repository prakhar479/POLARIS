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
    bounded_tool_data,
    build_tool_result_message,
    compact_json,
    create_tool_registry,
    execute_strategy_tool,
    extract_connector_from_context,
    format_system_state_for_llm,
    parse_strict_json,
)
from polaris.tools import ToolRegistry


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
    _SUPPORTED_NATIVE_TOOLS_UNSUPPORTED_POLICIES = {
        "skip_cycle",
        "json_fallback",
        "strict_fail",
    }

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
        system_prompt_suffix: Optional[str] = None,
        per_system_prompts: Optional[Dict[str, str]] = None,
        native_tools: Optional[List[Dict[str, Any]]] = None,
        max_tool_result_chars: int = 1200,
        native_tools_unsupported_policy: str = "skip_cycle",
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
            allowed_tools: List of permitted Polaris built-in tools for the LLM
            system_prompt: Optional custom system prompt template
            system_prompt_suffix: Optional meta-learner observations appended to the prompt
            per_system_prompts: Optional per-system prompt overrides keyed by system_id
            native_tools: Optional list of OpenAI-format function definitions. When
                provided, the strategy uses native provider tool calling instead of
                the JSON text response flow. The list must contain dicts of the form
                ``{"type": "function", "function": {"name": ..., "description": ...,
                "parameters": {...}}}``.
            max_tool_result_chars: Maximum serialized tool result size injected into
                model context before truncation metadata is applied.
            native_tools_unsupported_policy: Behavior when provider does not support
                native tool calling. One of: ``skip_cycle`` (default),
                ``json_fallback`` (retry via JSON text mode), ``strict_fail``.
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
        self._native_tools: List[Dict[str, Any]] = list(native_tools) if native_tools else []
        self.max_tool_result_chars = max(200, int(max_tool_result_chars))
        self.native_tools_unsupported_policy = self._normalize_native_tools_unsupported_policy(
            native_tools_unsupported_policy
        )
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()
        self._action_resolver = ConnectorActionResolver()
        self._tool_registry = ToolRegistry(metrics=self.metrics)
        self._rebuild_tool_registry()

        self._adaptation_count = 0
        self._success_count = 0
        self._last_decision_time: Optional[datetime] = None
        # Meta-learner injectable suffix — appended to the resolved system prompt each cycle.
        # The meta-learner writes to this via update_parameter("system_prompt_suffix", ...).
        self._system_prompt_suffix: str = (
            str(system_prompt_suffix).strip() if system_prompt_suffix else ""
        )

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Assess system state and determine if adaptation is needed.

        Dispatches to either the native tool-calling path (when ``native_tools``
        are configured) or the JSON text reasoning loop (legacy default).

        Args:
            state: Current system state with metrics and health information
            context: Adaptation context containing world model insights

        Returns:
            List of AdaptationActions to execute, or an empty list when no
            adaptation is required.
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

        if self._native_tools:
            return await self._assess_with_native_tools(state, context)
        return await self._assess_with_json_text(state, context)

    async def _assess_with_native_tools(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Adaptation assessment via native provider tool calling.

        Passes ``native_tools`` to the LLM, then maps the returned ``tool_calls``
        either to internal Polaris tool execution (for analytical helper tools)
        or directly to ``AdaptationAction`` objects (for connector actions).

        If the model returns no ``tool_calls`` (e.g. it replied with plain text),
        this falls back to attempting JSON text parsing so results are still
        meaningful during a transition period or on providers that ignore tools.
        """
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
                llm_start = datetime.now(timezone.utc)
                try:
                    response = await self.llm.generate_with_tools(
                        messages,
                        tools=self._native_tools,
                        tool_choice="auto",
                        temperature=self.temperature,
                        max_tokens=DEFAULT_MAX_TOKENS_REASONING,
                    )
                except (NotImplementedError, AttributeError) as exc:
                    if self.logger:
                        self.logger.error(
                            "Native tool calling is not implemented for configured provider",
                            system_id=state.system_id,
                            error=str(exc),
                            policy=self.native_tools_unsupported_policy,
                        )
                    self.metrics.increment(
                        "polaris.strategy.agentic.native_tools_not_implemented",
                        tags={
                            "system_id": state.system_id,
                            "policy": self.native_tools_unsupported_policy,
                        },
                    )
                    if self.native_tools_unsupported_policy == "strict_fail":
                        raise StrictContractViolation(
                            "Native tool calling is not supported by the configured provider"
                        ) from exc

                    if self.native_tools_unsupported_policy == "json_fallback":
                        self.metrics.increment(
                            "polaris.strategy.agentic.native_tools_fallback",
                            tags={
                                "system_id": state.system_id,
                                "reason": "not_implemented",
                            },
                        )
                        return await self._assess_with_json_text(state, context)

                    # Default policy: skip current cycle on hard capability failure.
                    return []
                self.metrics.histogram(
                    "polaris.strategy.agentic.llm_call_duration_seconds",
                    (datetime.now(timezone.utc) - llm_start).total_seconds(),
                    tags={"system_id": state.system_id},
                )
                self._maybe_log_llm_response(
                    system_id=state.system_id,
                    step=step + 1,
                    content=getattr(response, "content", ""),
                )

                if response.tool_calls is None:
                    # Soft failure: provider returned text path instead of native tool calls.
                    if self.logger:
                        self.logger.warning(
                            "Native tool calling: provider returned tool_calls=None, "
                            "falling back to JSON text parsing",
                            system_id=state.system_id,
                        )
                    self.metrics.increment(
                        "polaris.strategy.agentic.native_tools_fallback",
                        tags={"system_id": state.system_id},
                    )
                    return await self._parse_json_text_response(
                        response.content,
                        state,
                        context,
                        supported_action_types,
                        action_aliases,
                    )

                if not response.tool_calls:
                    if self.logger:
                        self.logger.warning(
                            "Native tool calling: provider returned empty tool_calls, "
                            "falling back to JSON text parsing",
                            system_id=state.system_id,
                        )
                    self.metrics.increment(
                        "polaris.strategy.agentic.native_tools_fallback",
                        tags={"system_id": state.system_id},
                    )
                    return await self._parse_json_text_response(
                        response.content,
                        state,
                        context,
                        supported_action_types,
                        action_aliases,
                    )

                self.metrics.increment(
                    "polaris.strategy.agentic.native_tools_used",
                    tags={"system_id": state.system_id},
                )

                # Sentinel: model chose not to adapt
                if any(tc["name"] == "no_adaptation" for tc in response.tool_calls):
                    if self.logger:
                        reasoning = next(
                            (
                                tc["arguments"].get("reasoning", "")
                                for tc in response.tool_calls
                                if tc["name"] == "no_adaptation"
                            ),
                            "",
                        )
                        self.logger.info(
                            "Agentic decision: no adaptation (native tool)",
                            system_id=state.system_id,
                            reasoning=reasoning,
                        )
                    self._last_decision_time = datetime.now(timezone.utc)
                    return []

                strategy_calls = [
                    tc for tc in response.tool_calls if tc["name"] in self.allowed_tools
                ]
                action_calls = [
                    tc for tc in response.tool_calls if tc["name"] not in self.allowed_tools
                ]

                # If a Polaris analytical tool is called, execute exactly one and continue.
                if strategy_calls:
                    if len(strategy_calls) > 1:
                        raise StrictContractViolation(
                            "Native tool response must call at most one Polaris tool per step"
                        )
                    if action_calls:
                        raise StrictContractViolation(
                            "Native tool response must not mix Polaris tool calls and action calls in the same step"
                        )

                    tc = strategy_calls[0]
                    tool_name = tc["name"]
                    tool_args = tc.get("arguments") or {}
                    if not isinstance(tool_args, dict):
                        raise StrictContractViolation(
                            "Native Polaris tool call requires object arguments"
                        )

                    tool_result = await self._execute_tool(tool_name, tool_args, state, context)
                    tool_msg = self._build_tool_result_message(tool_name, tool_result)
                    messages.append(LLMMessage(role="user", content=tool_msg))
                    continue

                # Map all returned action calls to AdaptationActions.
                proposed_actions: List[AdaptationAction] = []
                for tc in action_calls:
                    tool_name = tc["name"]
                    tool_args = tc.get("arguments")

                    resolved_action_type, resolved_parameters = resolve_strict_action_payload(
                        resolver=self._action_resolver,
                        action_type=tool_name,
                        parameters=tool_args,
                        supported_action_types=supported_action_types,
                        action_aliases=action_aliases,
                        system_id=state.system_id,
                        missing_type_error="Native tool call requires a non-empty function name",
                        invalid_parameters_error="Native tool call requires object arguments",
                    )
                    proposed_actions.append(
                        AdaptationAction(
                            action_id=str(uuid.uuid4()),
                            action_type=resolved_action_type,
                            target_system=state.system_id,
                            parameters=resolved_parameters,
                        )
                    )

                if self.logger:
                    self.logger.info(
                        "Agentic decision: propose actions (native tool)",
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

            self.metrics.increment(
                "polaris.strategy.agentic.step_limit_reached",
                tags={"system_id": state.system_id},
            )
            raise StrictContractViolation(
                "Agentic native tool strategy reached step limit without producing a final decision"
            )

        finally:
            self.metrics.histogram(
                "polaris.strategy.agentic.assess_duration_seconds",
                (datetime.now(timezone.utc) - start).total_seconds(),
                tags={"system_id": state.system_id},
            )

    async def _parse_json_text_response(
        self,
        content: str,
        state: SystemState,
        context: AdaptationContext,
        supported_action_types: List[str],
        action_aliases: Dict[str, str],
    ) -> List[AdaptationAction]:
        """Parse a JSON-text response into AdaptationActions (used as fallback)."""
        parsed = self._parse_json_object(content)
        try:
            structured_response = AgenticResponseSchema.model_validate(parsed)
        except Exception as exc:
            raise StrictContractViolation(
                f"Agentic response failed schema validation: {exc}"
            ) from exc

        if structured_response.final is None:
            raise StrictContractViolation(
                "Agentic JSON fallback response must include a 'final' block"
            )
        final = structured_response.final
        if not isinstance(final.reasoning, str) or not final.reasoning.strip():
            raise StrictContractViolation("Agentic final response requires non-empty 'reasoning'")
        if not final.needs_adaptation:
            if self.logger:
                self.logger.info(
                    "Agentic decision: no adaptation (JSON fallback)",
                    system_id=state.system_id,
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
        self._last_decision_time = datetime.now(timezone.utc)
        return proposed_actions

    async def _assess_with_json_text(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Original multi-step JSON text reasoning loop (used when native_tools is empty)."""
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
                tool_msg = self._build_tool_result_message(tool, tool_result)
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
        return await execute_strategy_tool(
            tool_registry=self._tool_registry,
            tool_name=tool,
            args=args,
            state=state,
            context=context,
            knowledge_store=self.knowledge_store,
            world_model=self.world_model,
            logger=self.logger,
            metrics=self.metrics,
            metric_prefix="polaris.strategy.agentic",
            error_log_message="Agentic tool execution error",
        )

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
                description="LLM sampling temperature",
                kind="llm_temperature",
            ),
            "steps_limit": ParameterSpec(
                current_value=self.steps_limit,
                type=int,
                min_value=1,
                max_value=10,
                description="Maximum tool-use steps per cycle",
                kind="agent_steps_limit",
            ),
            "decision_cooldown_seconds": ParameterSpec(
                current_value=self.decision_cooldown_seconds,
                type=float,
                min_value=0.0,
                max_value=3600.0,
                description="Minimum seconds between consecutive LLM decisions",
                kind="cooldown",
            ),
            "system_prompt_suffix": ParameterSpec(
                current_value=self._system_prompt_suffix,
                type=str,
                description=(
                    "Learnings appended to the system prompt each cycle. "
                    "Set by the meta-learner to inject observed patterns without "
                    "overwriting the base prompt."
                ),
                kind="system_prompt_suffix",
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
        if parameter_path == "system_prompt_suffix":
            # Strip and store; an empty string clears any previous learnings.
            self._system_prompt_suffix = str(new_value).strip()
            if self.logger:
                self.logger.info(
                    "AgenticLLMStrategy system_prompt_suffix updated by meta-learner",
                    suffix_length=len(self._system_prompt_suffix),
                )
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
        if "system_prompt_suffix" in config:
            await self.update_parameter("system_prompt_suffix", config["system_prompt_suffix"])
        if "per_system_prompts" in config and isinstance(config["per_system_prompts"], dict):
            self._per_system_prompts = config["per_system_prompts"]

        # Update native tools if provided
        if "native_tools" in config:
            nt = config["native_tools"]
            self._native_tools = list(nt) if isinstance(nt, list) else []

        if "max_tool_result_chars" in config:
            self.max_tool_result_chars = max(200, int(config["max_tool_result_chars"]))

        if "native_tools_unsupported_policy" in config:
            try:
                self.native_tools_unsupported_policy = (
                    self._normalize_native_tools_unsupported_policy(
                        config["native_tools_unsupported_policy"]
                    )
                )
            except ValueError as exc:
                if self.logger:
                    self.logger.warning(
                        "AgenticLLMStrategy invalid native_tools_unsupported_policy update ignored",
                        error=str(exc),
                    )

        # Update allowed tools in registry if changed
        if "tools" in config:
            tools_cfg = config["tools"]
            if isinstance(tools_cfg, list):
                self.allowed_tools = tools_cfg
                self._rebuild_tool_registry()
            elif isinstance(tools_cfg, dict):
                enabled = tools_cfg.get("enabled")
                if isinstance(enabled, list):
                    self.allowed_tools = enabled
                    self._rebuild_tool_registry()

        resil = config.get("resilience")
        if resil:
            update_resilience = getattr(self.llm, "update_resilience", None)
            if callable(update_resilience):
                try:
                    update_resilience(resil)
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
                    base = override.format(
                        system_id=system_id,
                        allowed_tools=", ".join(self.allowed_tools),
                        supported_actions=supported_actions_text,
                    )
                except (KeyError, IndexError, ValueError):
                    base = override
                if self._system_prompt_suffix:
                    return (
                        base.rstrip()
                        + "\n\n## Meta-learner observations\n"
                        + self._system_prompt_suffix
                    )
                return base

        tools = ", ".join(self.allowed_tools)

        # Global template override, optionally formatted
        if self._system_prompt_template:
            try:
                base = self._system_prompt_template.format(
                    system_id=system_id or "",
                    allowed_tools=tools,
                    supported_actions=supported_actions_text,
                )
            except (KeyError, IndexError, ValueError):
                base = self._system_prompt_template
            if self._system_prompt_suffix:
                return (
                    base.rstrip()
                    + "\n\n## Meta-learner observations\n"
                    + self._system_prompt_suffix
                )
            return base

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

    def _rebuild_tool_registry(self) -> None:
        """Rebuild tool registry from globally registered tool factories."""
        allowed = self.allowed_tools if self.allowed_tools else None
        self._tool_registry = create_tool_registry(self.metrics, allowed)

    def _normalize_native_tools_unsupported_policy(self, value: Any) -> str:
        """Normalize and validate unsupported-native-tools behavior policy."""
        policy = str(value or "skip_cycle").strip().lower()
        if policy not in self._SUPPORTED_NATIVE_TOOLS_UNSUPPORTED_POLICIES:
            supported = sorted(self._SUPPORTED_NATIVE_TOOLS_UNSUPPORTED_POLICIES)
            raise ValueError(
                "native_tools_unsupported_policy must be one of " f"{supported}, got {value!r}"
            )
        return policy

    def _extract_connector_from_context(self, context: AdaptationContext) -> Any:
        """Extract the active connector from adaptation context metadata if present."""
        return extract_connector_from_context(context)

    def _compact_json(self, value: Any, max_chars: int) -> str:
        """Serialize payload to JSON and truncate to avoid unbounded context growth."""
        return compact_json(value, max_chars)

    def _bounded_tool_data(self, tool_result: Dict[str, Any]) -> Any:
        """Return either original tool result or a truncated representation."""
        return bounded_tool_data(tool_result, self.max_tool_result_chars)

    def _build_tool_result_message(self, tool_name: str, tool_result: Dict[str, Any]) -> str:
        """Build a bounded tool result message for model context."""
        return build_tool_result_message(
            tool_name,
            tool_result,
            self.max_tool_result_chars,
        )

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
