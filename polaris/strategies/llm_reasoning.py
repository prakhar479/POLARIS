"""LLM-powered reasoning strategy.

Uses LLM to analyze system state and decide on adaptations.
"""

import json
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.core.models import AdaptationAction, ExecutionResult, SystemState
from polaris.infrastructure.constants import DEFAULT_JSON_INDENT, DEFAULT_MAX_TOKENS_REASONING
from polaris.infrastructure.llm import LLMClient, LLMMessage
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector
from polaris.strategies.action_resolution import (
    ConnectorActionResolver,
    StrictContractViolation,
    require_supported_action_contract,
    resolve_strict_action_payload,
)


class LLMReasoningStrategy(AdaptationStrategy):
    """LLM-powered adaptation strategy.

    Uses an LLM to analyze system state and decide on adaptations based on natural
    language reasoning.
    """

    requires_system_contract: bool = True

    def __init__(
        self,
        llm_client: LLMClient,
        system_description: str = "Managed system",
        adaptation_goals: str = "Maintain reliability, performance, and policy objectives",
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        per_system_prompts: Optional[Dict[str, str]] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize LLM reasoning strategy.

        Args:
            llm_client: LLM client to use for reasoning
            system_description: Description of the managed system
            adaptation_goals: Goals for adaptation
            temperature: LLM temperature (lower = more deterministic)
            logger: Optional logger for observability
            metrics: Optional metrics collector
            system_prompt: Optional system prompt template
            per_system_prompts: Optional per-system prompts
        """
        self.llm = llm_client
        self.system_description = system_description
        self.adaptation_goals = adaptation_goals
        self.temperature = temperature
        self._system_prompt_template = system_prompt
        self._per_system_prompts = per_system_prompts or {}
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()
        self._action_resolver = ConnectorActionResolver()
        self._adaptation_count = 0
        self._success_count = 0

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Use LLM to assess if adaptation is needed."""
        self.metrics.increment(
            "polaris.strategy.llm.assessments",
            tags={"system_id": state.system_id},
        )
        if self.logger:
            self.logger.debug(
                "LLM reasoning assessment started",
                system_id=state.system_id,
                health_status=state.health_status.value,
                metric_count=len(state.metrics),
            )
            for metric_name, metric_value in state.metrics.items():
                self.logger.debug(
                    "LLM reasoning metric snapshot",
                    system_id=state.system_id,
                    metric=metric_name,
                    value=getattr(metric_value, "value", None),
                    unit=getattr(metric_value, "unit", None),
                )

        # Build prompt with system state
        prompt = self._build_prompt(state, context)
        if self.logger:
            self.logger.debug(
                "LLM reasoning prompt built",
                system_id=state.system_id,
                prompt_length=len(prompt),
            )

        # Call LLM
        _contract, supported_action_types, action_aliases = require_supported_action_contract(
            context,
            strategy_name="LLM",
        )

        messages = [
            LLMMessage(
                role="system",
                content=self._get_system_prompt(state.system_id, supported_action_types),
            ),
            LLMMessage(role="user", content=prompt),
        ]

        if self.logger:
            self.logger.debug(
                "LLM reasoning request prepared",
                system_id=state.system_id,
                temperature=self.temperature,
                system_prompt_length=len(messages[0].content),
                user_prompt_length=len(messages[1].content),
            )

        try:
            if self.logger:
                self.logger.info("LLM reasoning request started", system_id=state.system_id)

            llm_start = datetime.now(timezone.utc)
            response = await self.llm.generate(
                messages, temperature=self.temperature, max_tokens=DEFAULT_MAX_TOKENS_REASONING
            )
            llm_duration = (datetime.now(timezone.utc) - llm_start).total_seconds()

            self.metrics.histogram(
                "polaris.strategy.llm.llm_call_duration_seconds",
                llm_duration,
                tags={"system_id": state.system_id},
            )

            if self.logger:
                self.logger.debug(
                    "LLM reasoning response received",
                    system_id=state.system_id,
                    response_length=len(response.content),
                )
                self.logger.debug("LLM reasoning response lines", system_id=state.system_id)
                for line in response.content.split("\n"):
                    self.logger.debug(
                        "LLM reasoning response line", system_id=state.system_id, line=line
                    )

            # Parse LLM response
            actions = self._parse_response(
                response.content,
                state.system_id,
                supported_action_types=supported_action_types,
                action_aliases=action_aliases,
            )

            if actions:
                if self.logger:
                    self.logger.info(
                        "LLM reasoning adaptation decision",
                        system_id=state.system_id,
                        needs_adaptation=True,
                        action_count=len(actions),
                    )
                    for action in actions:
                        self.logger.info(
                            "LLM reasoning action proposed",
                            system_id=state.system_id,
                            action_type=action.action_type,
                        )
                        self.logger.debug(
                            "LLM reasoning action parameters",
                            system_id=state.system_id,
                            action_parameters=json.dumps(
                                action.parameters,
                                indent=DEFAULT_JSON_INDENT,
                            ),
                        )
                for action in actions:
                    self.metrics.increment(
                        "polaris.strategy.llm.actions_proposed",
                        tags={"system_id": state.system_id, "action_type": action.action_type},
                    )
            else:
                if self.logger:
                    self.logger.info(
                        "LLM reasoning adaptation decision",
                        system_id=state.system_id,
                        needs_adaptation=False,
                        action_count=0,
                    )
                self.metrics.increment(
                    "polaris.strategy.llm.no_action_needed",
                    tags={"system_id": state.system_id},
                )

            return actions

        except Exception as exc:
            if self.logger:
                self.logger.error(
                    "LLM reasoning assessment failed",
                    system_id=state.system_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                self.logger.debug(
                    "LLM reasoning traceback",
                    system_id=state.system_id,
                    traceback=traceback.format_exc(),
                )
            self.metrics.increment(
                "polaris.strategy.llm.assessment_errors",
                tags={"system_id": state.system_id},
            )
            raise

    def _get_system_prompt(
        self,
        system_id: Optional[str] = None,
        supported_action_types: Optional[List[str]] = None,
    ) -> str:
        """Get system prompt for LLM, with optional system-specific overrides."""
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
                        system_description=self.system_description,
                        adaptation_goals=self.adaptation_goals,
                        supported_actions=supported_actions_text,
                    )
                except (KeyError, IndexError, ValueError):
                    return override

        # Global template override, optionally formatted
        if self._system_prompt_template:
            try:
                return self._system_prompt_template.format(
                    system_id=system_id or "",
                    system_description=self.system_description,
                    adaptation_goals=self.adaptation_goals,
                    supported_actions=supported_actions_text,
                )
            except (KeyError, IndexError, ValueError):
                # If formatting fails, fall back to the raw template
                return self._system_prompt_template

        # Default generic prompt
        return f"""You are an intelligent adaptation controller for a self-adaptive system.

System Description: {self.system_description}
Adaptation Goals: {self.adaptation_goals}

Your task is to analyze the current system state and decide if an adaptation action is needed.

IMPORTANT: You can propose MULTIPLE actions in the "actions" list if a compound adaptation is more effective.

Respond in JSON format:{{
    "needs_adaptation": "true" or "false",
    "reasoning": "explanation of your decision",
    "actions":[  # provide a list of actions - can contain multiple elements
        {{
            "type": "connector action type name",
            "parameters":{{"key": "value"}}
        }},
    ]
}}

Connector-supported action types: {supported_actions_text}

Be conservative - only adapt when there's a clear need. Consider:
- Current metric values vs normal ranges
- Trends and patterns
- Potential impact of adaptation
"""

    def _build_prompt(self, state: SystemState, context: AdaptationContext) -> str:
        """Build prompt with current state."""
        # Format metrics
        metrics_str = "\n".join(
            [
                f" - {name}: {metric.value} {metric.unit or ''}"
                for name, metric in state.metrics.items()
            ]
        )

        # Get world model insights if available
        insights_str = ""
        if context.world_model_insights:
            insights_str = "\nWorld Model Insights:\n" + json.dumps(
                context.world_model_insights, indent=DEFAULT_JSON_INDENT
            )

        return f"""Current System State:

System ID:  {state.system_id}
Health Status: {state.health_status.value}
Timestamp: {state.timestamp.isoformat()}

Metrics:
{metrics_str}
{insights_str}

Should this system be adapted right now? Analyze the state and provide your decision.
"""

    def _parse_response(
        self,
        response: str,
        system_id: str,
        supported_action_types: Optional[List[str]] = None,
        action_aliases: Optional[Dict[str, str]] = None,
    ) -> List[AdaptationAction]:
        """Parse strict JSON response into adaptation actions."""
        if not supported_action_types:
            raise StrictContractViolation(
                "Missing connector-supported action contract for strict LLM strategy"
            )

        payload = (response or "").strip()
        if not payload:
            raise StrictContractViolation("LLM response is empty")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StrictContractViolation(f"LLM response is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise StrictContractViolation("LLM response must be a JSON object")

        needs_adaptation = data.get("needs_adaptation")
        if not isinstance(needs_adaptation, bool):
            raise StrictContractViolation("LLM response requires boolean 'needs_adaptation'")

        reasoning = data.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise StrictContractViolation("LLM response requires non-empty string 'reasoning'")

        if not needs_adaptation:
            return []

        raw_actions = data.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise StrictContractViolation(
                "LLM response with needs_adaptation=true requires non-empty 'actions' list"
            )

        adaptation_actions: List[AdaptationAction] = []
        for action_data in raw_actions:
            if not isinstance(action_data, dict):
                raise StrictContractViolation("Each action entry must be a JSON object")

            resolved_action_type, resolved_parameters = resolve_strict_action_payload(
                resolver=self._action_resolver,
                action_type=action_data.get("type"),
                parameters=action_data.get("parameters", {}),
                supported_action_types=supported_action_types,
                action_aliases=action_aliases,
                system_id=system_id,
                missing_type_error="Each action requires non-empty string 'type'",
                invalid_parameters_error="Each action requires object 'parameters'",
            )

            adaptation_actions.append(
                AdaptationAction(
                    action_id=str(uuid.uuid4()),
                    action_type=resolved_action_type,
                    target_system=system_id,
                    parameters={**resolved_parameters, "llm_reasoning": reasoning},
                )
            )

        return adaptation_actions

    async def on_action_executed(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Track adaptation success."""
        self._adaptation_count += 1

        is_success = hasattr(result, "status") and result.status.value == "success"
        if is_success:
            self._success_count += 1
        self.metrics.increment(
            "polaris.strategy.llm.actions_executed",
            tags={
                "action_type": action.action_type,
                "system_id": action.target_system,
                "status": result.status.value if hasattr(result, "status") else "unknown",
            },
        )
        self.metrics.gauge(
            "polaris.strategy.llm.success_rate",
            self._success_count / self._adaptation_count if self._adaptation_count > 0 else 0.0,
        )

        if self.logger:
            self.logger.info(
                "LLM reasoning action execution result",
                action_id=action.action_id,
                action_type=action.action_type,
                status="SUCCESS" if is_success else "FAILED",
                total_adaptations=self._adaptation_count,
                successful_adaptations=self._success_count,
            )
            if hasattr(result, "error_message"):
                self.logger.debug(
                    "LLM reasoning action execution error message",
                    action_id=action.action_id,
                    error_message=result.error_message,
                )

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """LLM strategy parameters."""
        return {
            "temperature": ParameterSpec(
                current_value=self.temperature,
                type=float,
                min_value=0.0,
                max_value=2.0,
                description="LLM temperature for reasoning",
                kind="llm_temperature",
            ),
            "system_description": ParameterSpec(
                current_value=self.system_description,
                type=str,
                description="Description of the managed system",
                kind="llm_system_description",
            ),
        }

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update strategy parameters."""
        if parameter_path == "temperature":
            old_value = self.temperature
            self.temperature = float(new_value)
            if self.logger:
                self.logger.info(
                    "LLM reasoning parameter updated",
                    parameter="temperature",
                    old_value=old_value,
                    new_value=self.temperature,
                )
            return True
        elif parameter_path == "system_description":
            old_desc = self.system_description
            self.system_description = str(new_value)
            if self.logger:
                self.logger.info(
                    "LLM reasoning parameter updated",
                    parameter="system_description",
                    old_value=old_desc,
                    new_value=self.system_description,
                )
            return True
        if self.logger:
            self.logger.warning(
                "LLM reasoning unknown parameter update",
                parameter=parameter_path,
            )
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply configuration updates to the LLM reasoning strategy."""
        if "temperature" in config:
            await self.update_parameter("temperature", config["temperature"])
        if "system_description" in config:
            await self.update_parameter("system_description", config["system_description"])

        if "system_prompt" in config:
            self._system_prompt_template = config["system_prompt"]
        if "per_system_prompts" in config and isinstance(config["per_system_prompts"], dict):
            self._per_system_prompts = config["per_system_prompts"]

        resil = config.get("resilience")
        if resil and hasattr(self.llm, "update_resilience"):
            try:
                self.llm.update_resilience(resil)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                if self.logger:
                    self.logger.warning(
                        "LLM reasoning resilience hot-update failed",
                        error=str(exc),
                    )

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy performance metrics."""
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}

        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }
