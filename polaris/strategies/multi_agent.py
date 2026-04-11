"""Multi-agent LLM-based adaptation strategy for POLARIS.

This module implements an advanced adaptation strategy that uses a committee of
specialized Large Language Model (LLM) agents working together to make robust adaptation
decisions.

The decision process flows through three specialized agents:

1. **Diagnostician** — Analyses current system metrics to detect anomalies, identify
root causes, and assign a severity level. 2. **Planner** — Given the diagnosis, proposes
a concrete sequence of adaptation actions to resolve the identified issues. 3.
**SafetyValidator** — Reviews the plan for safety, approves or rejects it, and may
return a safer subset of actions.

Each agent can be independently configured with its own LLM client, temperature, and
system-prompt override, enabling autonomous multi-agent setups where different agents
can use different models or providers optimised for their role (e.g., a cheap fast model
for diagnosis, a stronger model for validation).
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.abstractions.world_model import WorldModel
from polaris.core.models import AdaptationAction, SystemState
from polaris.infrastructure.constants import (
    DEFAULT_MAX_TOKENS_DIAGNOSTICIAN,
    DEFAULT_MAX_TOKENS_PLANNER,
    DEFAULT_MAX_TOKENS_VALIDATOR,
)
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
    extract_connector_from_context,
    format_system_state_for_llm,
    parse_strict_json,
)
from polaris.tools import ToolDependencies, ToolRegistry

# ---------------------------------------------------------------------------
# Agent output models
# ---------------------------------------------------------------------------


class DiagnosticianOutput(BaseModel):
    """Output from the Diagnostician agent."""

    is_anomaly_detected: bool = Field(description="Whether an anomaly or issue is present")
    issues: List[str] = Field(description="List of identified issues (if any)")
    root_causes: List[str] = Field(description="List of potential root causes for the issues")
    severity: str = Field(description="Severity of the situation: none, low, medium, high")


class ActionBlock(BaseModel):
    """A block defining an action to be executed."""

    type: str = Field(description="The name or type of the action to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters required for the action"
    )


class PlannerOutput(BaseModel):
    """Output from the Planner agent."""

    plans: List[ActionBlock] = Field(
        description="Proposed sequence of actions to resolve the issues (empty if no action needed)"
    )
    rationale: str = Field(description="Reasoning behind the proposed plan")


class ValidatorOutput(BaseModel):
    """Output from the SafetyValidator agent."""

    approved: bool = Field(
        description="Whether the proposed plan is safe and approved for execution"
    )
    reasoning: str = Field(description="Reasoning for approval or rejection")
    safe_actions: List[ActionBlock] = Field(
        description="The finalized, safe list of actions to execute"
    )


# ---------------------------------------------------------------------------
# Tool-using (Agentic) response schemas
# ---------------------------------------------------------------------------


class AgenticResponseBase(BaseModel):
    """Base class for all agentic response schemas.

    Defines the common structure for tool-using agent responses: optional tool call or
    final output.
    """

    tool: Optional[str] = Field(None, description="Name of the tool to call")
    args: Optional[Dict[str, Any]] = Field(None, description="Arguments for the tool")
    final: Optional[Any] = Field(None, description="Final output from the agent")


class DiagnosticianAgenticResponse(AgenticResponseBase):
    """Agentic response schema for the Diagnostician."""

    final: Optional[DiagnosticianOutput] = Field(None, description="Final diagnosis")


class PlannerAgenticResponse(AgenticResponseBase):
    """Agentic response schema for the Planner."""

    final: Optional[PlannerOutput] = Field(None, description="Final plan proposal")


class ValidatorAgenticResponse(AgenticResponseBase):
    """Agentic response schema for the SafetyValidator."""

    final: Optional[ValidatorOutput] = Field(None, description="Final safety validation")


# ---------------------------------------------------------------------------
# Per-agent configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for a single agent within the multi-agent committee.

    Allows each agent (Diagnostician, Planner, SafetyValidator) to use a different LLM
    client, temperature, and/or system prompt.  Any field left as ``None`` falls back to
    the shared ``MultiAgentStrategy`` defaults.

    Attributes:
        llm_client: LLM client to use for this agent. Falls back to the strategy-level
            shared client when ``None``.
        temperature: Sampling temperature for this agent. Falls back to the
            strategy-level ``temperature`` when ``None``.
        system_prompt: Full system-prompt string for this agent. Falls back to the
            built-in default prompt for the role when ``None``.
        max_tokens: Maximum tokens to request. Falls back to role default.
    """

    llm_client: Optional[LLMClient] = field(default=None)
    temperature: Optional[float] = field(default=None)
    system_prompt: Optional[str] = field(default=None)
    max_tokens: Optional[int] = field(default=None)
    steps_limit: Optional[int] = field(default=None)
    allowed_tools: Optional[List[str]] = field(default=None)


# ---------------------------------------------------------------------------
# Default role-prompts
# ---------------------------------------------------------------------------

_DEFAULT_DIAGNOSTICIAN_PROMPT_TMPL = (
    "You are the Diagnostician agent for {system_description}.\n"
    "Your goal is to analyze system metrics to detect anomalies. "
    "You have access to tools to query history and trends. "
    "Review the metrics, use tools if needed, then provide your final diagnosis.\n"
    "Connector-supported action types (for context): {supported_actions}\n"
    "Available tools: {allowed_tools}\n\n"
    "Reply as strict JSON:\n"
    '1) {{"tool": "name", "args": {{...}}}} to request info\n'
    '2) {{"final": {{...}}}} to provide final diagnosis'
)

_DEFAULT_PLANNER_PROMPT_TMPL = (
    "You are the Planner agent for {system_description}.\n"
    "The Diagnostician has identified issues. Review the context, "
    "possibly use tools to predict outcomes of actions, and propose a plan.\n"
    "Action type constraint: use connector-supported names only: {supported_actions}\n"
    "Available tools: {allowed_tools}\n\n"
    "Reply as strict JSON:\n"
    '1) {{"tool": "name", "args": {{...}}}} to request info\n'
    '2) {{"final": {{...}}}} to provide final plan'
)

_DEFAULT_VALIDATOR_PROMPT_TMPL = (
    "You are the Safety Validator agent for {system_description}.\n"
    "Review the diagnosis and the proposed plan. Evaluate if the actions are safe. "
    "You may use tools to check system history or predict stability impact.\n"
    "Action type constraint: validator output must use connector-supported names only: {supported_actions}\n"
    "Available tools: {allowed_tools}\n\n"
    "Reply as strict JSON:\n"
    '1) {{"tool": "name", "args": {{...}}}} to request info\n'
    '2) {{"final": {{...}}}} to provide final validation'
)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class MultiAgentStrategy(AdaptationStrategy):
    """An adaptation strategy that uses a committee of LLM agents.

    The decision process flows through three specialized agents:

    1. **Diagnostician** — Detects anomalies, lists issues and root causes. 2.
    **Planner** — Proposes adaptation actions to mitigate the diagnosis. 3.
    **SafetyValidator** — Reviews the plan for safety and approves/rejects it.

    Each agent can be independently configured via an :class:`AgentConfig` object,
    enabling fully autonomous multi-agent setups where each role uses a different model,
    provider, temperature, or system prompt.

    Attributes:
        llm: Shared default LLM client.
        knowledge_store: Store for querying historical system data.
        world_model: World model for predicting action outcomes.
        temperature: Shared default LLM sampling temperature.
        system_description: Human-readable description used in default prompts.
        steps_limit: Default max reasoning steps for each agent.
        allowed_tools: Default enabled tools for each agent.
        diagnostician_config: Per-agent config for the Diagnostician (optional).
        planner_config: Per-agent config for the Planner (optional).
        validator_config: Per-agent config for the SafetyValidator (optional).
    """

    requires_system_contract: bool = True

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        temperature: float = 0.1,
        system_description: str = "Managed system",
        steps_limit: int = 3,
        max_tool_result_chars: int = 1200,
        allowed_tools: Optional[List[str]] = None,
        # Per-agent overrides
        diagnostician_config: Optional[AgentConfig] = None,
        planner_config: Optional[AgentConfig] = None,
        validator_config: Optional[AgentConfig] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialise the MultiAgentStrategy.

        Args:
            llm_client: Shared LLM client used when no per-agent client is
                configured.
            knowledge_store: Store for querying historical system data.
            world_model: World model for predicting action outcomes.
            temperature: Shared sampling temperature (default: 0.1). Each agent can
                override this via its :class:`AgentConfig`.
            system_description: Description of the managed system embedded in default
                agent prompts (default: "A generic managed cloud system").
            steps_limit: Default max reasoning steps for each agent stage (default: 3).
            max_tool_result_chars: Maximum serialized tool result size injected into
                model context before truncation metadata is applied.
            allowed_tools: List of enabled tools. If None, all built-in tools are
                enabled.
            diagnostician_config: Optional per-agent config for the Diagnostician.
            planner_config: Optional per-agent config for the Planner.
            validator_config: Optional per-agent config for the SafetyValidator.
            logger: Optional structured logger.
            metrics: Optional metrics collector (falls back to NullMetricsCollector).
        """
        self.llm = llm_client
        self.knowledge_store = knowledge_store
        self.world_model = world_model
        self.temperature = temperature
        self.system_description = system_description
        self.steps_limit = steps_limit
        self.max_tool_result_chars = max(200, int(max_tool_result_chars))
        self.allowed_tools = allowed_tools or list(DEFAULT_ALLOWED_TOOLS)

        # Per-agent configs — normalise to AgentConfig objects
        self._diagnostician_cfg = diagnostician_config or AgentConfig()
        self._planner_cfg = planner_config or AgentConfig()
        self._validator_cfg = validator_config or AgentConfig()

        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()
        self._action_resolver = ConnectorActionResolver()
        self._tool_registry = ToolRegistry(metrics=self.metrics)
        self._rebuild_tool_registry()
        self._adaptation_count = 0
        self._success_count = 0

    # ------------------------------------------------------------------
    # Helper: resolve per-agent settings
    # ------------------------------------------------------------------

    def _agent_llm(self, cfg: AgentConfig) -> LLMClient:
        return cfg.llm_client if cfg.llm_client is not None else self.llm

    def _agent_temperature(self, cfg: AgentConfig) -> float:
        return cfg.temperature if cfg.temperature is not None else self.temperature

    def _agent_prompt(
        self,
        cfg: AgentConfig,
        default_tmpl: str,
        supported_action_types: Optional[List[str]] = None,
    ) -> str:
        tools = ", ".join(cfg.allowed_tools or self.allowed_tools)
        supported_actions_text = (
            ", ".join(supported_action_types)
            if supported_action_types
            else "unknown (use connector-supported canonical action names)"
        )
        if cfg.system_prompt is not None:
            try:
                return cfg.system_prompt.format(
                    system_description=self.system_description,
                    allowed_tools=tools,
                    supported_actions=supported_actions_text,
                )
            except (KeyError, IndexError, ValueError):
                return cfg.system_prompt
        return default_tmpl.format(
            system_description=self.system_description,
            allowed_tools=tools,
            supported_actions=supported_actions_text,
        )

    def _agent_steps_limit(self, cfg: AgentConfig) -> int:
        return cfg.steps_limit if cfg.steps_limit is not None else self.steps_limit

    def _agent_allowed_tools(self, cfg: AgentConfig) -> List[str]:
        return cfg.allowed_tools if cfg.allowed_tools is not None else self.allowed_tools

    # ------------------------------------------------------------------
    # Core assessment pipeline
    # ------------------------------------------------------------------

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Assess system state using the multi-agent committee.

        Runs the Diagnostician → Planner → SafetyValidator pipeline. Each agent may use
        its own LLM client, temperature, and system prompt if configured via
        :class:`AgentConfig`.

        Args:
            state: Current system state with metrics and health information.
            context: Adaptation context containing world-model insights.

        Returns:
            List of approved: class:`AdaptationAction` objects, or an empty list if no
                adaptation is required / the plan was rejected.
        """
        if self.logger:
            self.logger.debug("MultiAgent assessment started", system_id=state.system_id)

        self.metrics.increment(
            "polaris.strategy.multi_agent.assessments", tags={"system_id": state.system_id}
        )

        start_time = datetime.now(timezone.utc)
        system_context_str = self._format_system_context(state, context)
        _contract, supported_action_types, action_aliases = require_supported_action_contract(
            context,
            strategy_name="multi-agent",
        )

        try:
            # ----------------------------------------------------------
            # Stage 1: Diagnostician
            # ----------------------------------------------------------
            diagnosis = await self._run_agentic_loop(
                role="diagnostician",
                cfg=self._diagnostician_cfg,
                initial_input=f"Current System State:\n{system_context_str}",
                response_schema=DiagnosticianAgenticResponse,
                default_prompt_tmpl=_DEFAULT_DIAGNOSTICIAN_PROMPT_TMPL,
                state=state,
                context=context,
                supported_action_types=supported_action_types,
            )

            if not diagnosis:
                return []

            if not diagnosis.is_anomaly_detected or diagnosis.severity.lower() == "none":
                return []

            # ----------------------------------------------------------
            # Stage 2: Planner
            # ----------------------------------------------------------
            planner_input = (
                f"--- SYSTEM CONTEXT ---\n{system_context_str}\n\n"
                f"--- DIAGNOSIS ---\n"
                f"Issues: {diagnosis.issues}\n"
                f"Root Causes: {diagnosis.root_causes}\n"
                f"Severity: {diagnosis.severity}"
            )
            plan = await self._run_agentic_loop(
                role="planner",
                cfg=self._planner_cfg,
                initial_input=planner_input,
                response_schema=PlannerAgenticResponse,
                default_prompt_tmpl=_DEFAULT_PLANNER_PROMPT_TMPL,
                state=state,
                context=context,
                supported_action_types=supported_action_types,
            )

            if not plan or not plan.plans:
                return []

            # ----------------------------------------------------------
            # Stage 3: SafetyValidator
            # ----------------------------------------------------------
            actions_str = json.dumps([a.model_dump() for a in plan.plans])
            validator_input = (
                f"--- DIAGNOSIS ---\n"
                f"Severity: {diagnosis.severity}\n"
                f"Root Causes: {diagnosis.root_causes}\n\n"
                f"--- PROPOSED PLAN ---\n"
                f"Rationale: {plan.rationale}\n"
                f"Actions: {actions_str}"
            )
            validation = await self._run_agentic_loop(
                role="validator",
                cfg=self._validator_cfg,
                initial_input=validator_input,
                response_schema=ValidatorAgenticResponse,
                default_prompt_tmpl=_DEFAULT_VALIDATOR_PROMPT_TMPL,
                state=state,
                context=context,
                supported_action_types=supported_action_types,
            )

            if not validation or not validation.approved or not validation.safe_actions:
                return []

            # Convert approved actions to AdaptationAction objects
            final_actions: List[AdaptationAction] = []
            for action_block in validation.safe_actions:
                resolved_action_type, resolved_parameters = resolve_strict_action_payload(
                    resolver=self._action_resolver,
                    action_type=action_block.type,
                    parameters=action_block.parameters,
                    supported_action_types=supported_action_types,
                    action_aliases=action_aliases,
                    system_id=state.system_id,
                    missing_type_error="Validator action requires non-empty 'type'",
                    invalid_parameters_error="Validator action requires object 'parameters'",
                )
                final_actions.append(
                    AdaptationAction(
                        action_id=str(uuid.uuid4()),
                        action_type=resolved_action_type,
                        target_system=state.system_id,
                        parameters={
                            **resolved_parameters,
                            "llm_diagnosis": diagnosis.issues,
                            "llm_rationale": plan.rationale,
                            "llm_validator_reasoning": validation.reasoning,
                        },
                    )
                )

            for a in final_actions:
                self.metrics.increment(
                    "polaris.strategy.multi_agent.actions_proposed",
                    tags={"system_id": state.system_id, "action_type": a.action_type},
                )

            return final_actions

        finally:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics.histogram(
                "polaris.strategy.multi_agent.assess_duration_seconds",
                duration,
                tags={"system_id": state.system_id},
            )

    # ------------------------------------------------------------------
    # Callbacks & meta-learner interface
    # ------------------------------------------------------------------

    async def on_action_executed(self, action: AdaptationAction, result: Any) -> None:
        """Handle callback when an adaptation action is executed."""
        self._adaptation_count += 1
        ok = hasattr(result, "status") and getattr(result.status, "value", None) == "success"
        if ok:
            self._success_count += 1

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Get specification of tunable parameters for this strategy."""
        params: Dict[str, ParameterSpec] = {
            "temperature": ParameterSpec(
                current_value=self.temperature,
                type=float,
                min_value=0.0,
                max_value=2.0,
                description="Shared default LLM sampling temperature",
                kind="llm_temperature",
            ),
            "steps_limit": ParameterSpec(
                current_value=self.steps_limit,
                type=int,
                min_value=1,
                max_value=10,
                description="Shared default max reasoning steps",
                kind="agent_steps_limit",
            ),
        }
        # Per-agent temperatures (only those without a dedicated client override)
        for role, cfg in [
            ("diagnostician", self._diagnostician_cfg),
            ("planner", self._planner_cfg),
            ("validator", self._validator_cfg),
        ]:
            effective_temp = cfg.temperature if cfg.temperature is not None else self.temperature
            params[f"{role}_temperature"] = ParameterSpec(
                current_value=effective_temp,
                type=float,
                min_value=0.0,
                max_value=2.0,
                description=f"Sampling temperature for the {role.capitalize()} agent",
                kind="llm_temperature",
            )
        return params

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a tunable parameter value.

        Supports both the shared ``temperature`` and per-agent temperatures
        (``diagnostician_temperature``, ``planner_temperature``,
        ``validator_temperature``).

        Args:
            parameter_path: Parameter name / path.
            new_value: New value to apply.

        Returns:
            ``True`` if the parameter was recognised and updated.
        """
        if parameter_path == "temperature":
            self.temperature = float(new_value)
            return True
        if parameter_path == "steps_limit":
            self.steps_limit = int(new_value)
            return True

        mapping = {
            "diagnostician_temperature": self._diagnostician_cfg,
            "planner_temperature": self._planner_cfg,
            "validator_temperature": self._validator_cfg,
        }
        if parameter_path in mapping:
            mapping[parameter_path].temperature = float(new_value)
            return True
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply hot-reloaded configuration updates.

        Handles updates to shared and per-agent temperatures, ``system_description``,
        per-agent system prompts, and LLM resilience (when the underlying client
        supports it).

        Args:
            config: Configuration dictionary with updates to apply.
        """
        if "temperature" in config:
            await self.update_parameter("temperature", config["temperature"])

        if "steps_limit" in config:
            await self.update_parameter("steps_limit", config["steps_limit"])

        if "max_tool_result_chars" in config:
            self.max_tool_result_chars = max(200, int(config["max_tool_result_chars"]))

        if "tools" in config:
            tools_cfg = config["tools"]
            if isinstance(tools_cfg, list):
                self.allowed_tools = tools_cfg
            elif isinstance(tools_cfg, dict):
                enabled = tools_cfg.get("enabled")
                if isinstance(enabled, list):
                    self.allowed_tools = enabled

        if "system_description" in config:
            self.system_description = config["system_description"]

        # Per-agent config blocks
        for role, cfg_obj in [
            ("diagnostician", self._diagnostician_cfg),
            ("planner", self._planner_cfg),
            ("validator", self._validator_cfg),
        ]:
            role_cfg = config.get(role)
            if not isinstance(role_cfg, dict):
                continue
            if "temperature" in role_cfg:
                cfg_obj.temperature = float(role_cfg["temperature"])
            if "system_prompt" in role_cfg:
                cfg_obj.system_prompt = role_cfg["system_prompt"]
            if "max_tokens" in role_cfg:
                cfg_obj.max_tokens = int(role_cfg["max_tokens"])
            if "steps_limit" in role_cfg:
                cfg_obj.steps_limit = int(role_cfg["steps_limit"])
            if "tools" in role_cfg:
                role_tools = role_cfg["tools"]
                if isinstance(role_tools, list):
                    cfg_obj.allowed_tools = role_tools
                elif isinstance(role_tools, dict):
                    enabled = role_tools.get("enabled")
                    if isinstance(enabled, list):
                        cfg_obj.allowed_tools = enabled
            # Resilience on per-agent client
            if "resilience" in role_cfg and cfg_obj.llm_client is not None:
                if hasattr(cfg_obj.llm_client, "update_resilience"):
                    try:
                        cfg_obj.llm_client.update_resilience(role_cfg["resilience"])
                    except Exception as exc:
                        if self.logger:
                            self.logger.warning(
                                "Multi-agent resilience update failed",
                                role=role,
                                error=str(exc),
                            )

        self._rebuild_tool_registry()

        # Shared resilience
        if "resilience" in config and hasattr(self.llm, "update_resilience"):
            try:
                self.llm.update_resilience(config["resilience"])
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "MultiAgentStrategy shared resilience update failed", error=str(exc)
                    )

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the multi-agent strategy.

        Returns:
            Dict with ``success_rate`` and ``total_adaptations`` keys.
        """
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}
        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_agentic_loop(
        self,
        role: str,
        cfg: AgentConfig,
        initial_input: str,
        response_schema: Type[AgenticResponseBase],
        default_prompt_tmpl: str,
        state: SystemState,
        context: AdaptationContext,
        supported_action_types: Optional[List[str]] = None,
    ) -> Any:
        """Run an iterative tool-using loop for a specific agent role."""
        llm = self._agent_llm(cfg)
        temp = self._agent_temperature(cfg)
        prompt = self._agent_prompt(cfg, default_prompt_tmpl, supported_action_types)
        max_tokens = cfg.max_tokens or (
            DEFAULT_MAX_TOKENS_DIAGNOSTICIAN
            if role == "diagnostician"
            else DEFAULT_MAX_TOKENS_VALIDATOR if role == "validator" else DEFAULT_MAX_TOKENS_PLANNER
        )
        steps_limit = self._agent_steps_limit(cfg)
        allowed_tools = self._agent_allowed_tools(cfg)

        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content=initial_input),
        ]

        for _step in range(steps_limit):
            self.metrics.increment(
                "polaris.strategy.multi_agent.step",
                tags={"system_id": state.system_id, "role": role},
            )
            response = await llm.generate(
                messages,
                temperature=temp,
                max_tokens=max_tokens,
                response_schema=response_schema,
            )

            parsed = self._parse_json_object(response.content)

            try:
                structured = response_schema.model_validate(parsed)
            except Exception as exc:
                raise StrictContractViolation(
                    f"MultiAgent {role} response failed schema validation: {exc}"
                ) from exc

            if getattr(structured, "final", None) is not None:
                return structured.final

            tool = getattr(structured, "tool", None)
            args = getattr(structured, "args", {}) or {}

            if not tool:
                raise StrictContractViolation(
                    f"MultiAgent {role} response must include either 'final' or 'tool'"
                )

            if tool not in allowed_tools:
                raise StrictContractViolation(
                    f"MultiAgent {role} requested disallowed tool '{tool}'"
                )
            else:
                try:
                    # Build tool dependencies
                    deps = ToolDependencies(
                        knowledge_store=self.knowledge_store,
                        world_model=self.world_model,
                        connector=self._extract_connector_from_context(context),
                        system_contract=context.system_contract,
                        logger=self.logger,
                        metrics=self.metrics,
                    )
                    tool_result = await self._tool_registry.execute(
                        tool_name=tool,
                        args=args,
                        state=state,
                        context=context,
                        deps=deps,
                    )
                except Exception as exc:
                    if self.logger:
                        self.logger.error(
                            "MultiAgent tool execution failed",
                            role=role,
                            tool=tool,
                            error=str(exc),
                        )
                    from polaris.tools import ToolError

                    tool_result = ToolError(
                        code="tool_error",
                        message=f"tool_error: {type(exc).__name__}: {str(exc)}",
                        recoverable=True,
                    ).to_dict()

            tool_msg = self._build_tool_result_message(tool, tool_result)
            messages.append(LLMMessage(role="user", content=tool_msg))

        raise StrictContractViolation(
            f"MultiAgent {role} reached step limit without producing a final decision"
        )

    def _format_system_context(self, state: SystemState, context: AdaptationContext) -> str:
        """Format system state and context into a JSON string."""
        return format_system_state_for_llm(state, context)

    def _parse_json_object(self, content: str) -> Dict[str, Any]:
        """Parse strict JSON object from model output."""
        return parse_strict_json(content, StrictContractViolation)

    def _registry_allowed_tools(self) -> Optional[List[str]]:
        """Compute effective allowed tool list for registry population.

        Includes shared allowed tools and per-agent overrides so role-specific tools
        are available to execute when selected.
        """
        merged: set[str] = set(self.allowed_tools or [])
        for cfg in (self._diagnostician_cfg, self._planner_cfg, self._validator_cfg):
            if cfg.allowed_tools:
                merged.update(cfg.allowed_tools)

        if not merged:
            return None
        return sorted(merged)

    def _rebuild_tool_registry(self) -> None:
        """Rebuild tool registry from globally registered tool factories."""
        self._tool_registry = create_tool_registry(self.metrics, self._registry_allowed_tools())

    def _extract_connector_from_context(self, context: AdaptationContext) -> Any:
        """Extract active connector from adaptation context metadata if available."""
        return extract_connector_from_context(context)

    def _compact_json(self, value: Any, max_chars: int) -> str:
        """Serialize payload to JSON and truncate to cap model context growth."""
        return compact_json(value, max_chars)

    def _bounded_tool_data(self, tool_result: Dict[str, Any]) -> Any:
        """Return either original tool result or a truncated representation."""
        return bounded_tool_data(tool_result, self.max_tool_result_chars)

    def _build_tool_result_message(self, tool_name: str, tool_result: Dict[str, Any]) -> str:
        """Build bounded tool-result payload for model context."""
        return build_tool_result_message(
            tool_name,
            tool_result,
            self.max_tool_result_chars,
        )
