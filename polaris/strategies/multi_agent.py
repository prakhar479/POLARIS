"""Multi-agent LLM-based adaptation strategy for POLARIS.

This module implements an advanced adaptation strategy that uses a committee
of specialized Large Language Model (LLM) agents working together to make robust
adaptation decisions.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.abstractions.world_model import WorldModel
from polaris.core.models import AdaptationAction, SystemState
from polaris.infrastructure.llm import LLMClient, LLMMessage
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector

# Models for Agent Interfaces


class DiagnosticianOutput(BaseModel):
    """Output from the diagnostician agent."""

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
    """Output from the planner agent."""

    plans: List[ActionBlock] = Field(
        description="Proposed sequence of actions to resolve the issues (empty if no action needed)"
    )
    rationale: str = Field(description="Reasoning behind the proposed plan")


class ValidatorOutput(BaseModel):
    """Output from the validator agent."""

    approved: bool = Field(
        description="Whether the proposed plan is safe and approved for execution"
    )
    reasoning: str = Field(description="Reasoning for approval or rejection")
    safe_actions: List[ActionBlock] = Field(
        description="The finalized, safe list of actions to execute"
    )


class MultiAgentStrategy(AdaptationStrategy):
    """An adaptation strategy that uses a committee of LLM agents.

    The decision process flows through three specialized agents:
    1. Diagnostician: Analyzes metrics to identify issues and root causes.
    2. Planner: Suggests adaptation actions to mitigate the diagnosed issues.
    3. SafetyValidator: Reviews the plan for safety and approves/rejects it.

    Attributes:
        llm: The LLM client for generating responses
        knowledge_store: Store for querying historical system data
        world_model: World model for predicting action outcomes
        temperature: LLM temperature parameter
    """

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        temperature: float = 0.1,
        system_description: str = "A generic managed cloud system",
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize the MultiAgentStrategy."""
        self.llm = llm_client
        self.knowledge_store = knowledge_store
        self.world_model = world_model
        self.temperature = temperature
        self.system_description = system_description
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()
        self._adaptation_count = 0
        self._success_count = 0

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Assess system state using the multi-agent committee."""
        if self.logger:
            self.logger.debug("MultiAgent assessment started", system_id=state.system_id)

        self.metrics.increment(
            "polaris.strategy.multi_agent.assessments", tags={"system_id": state.system_id}
        )

        start_time = datetime.now(timezone.utc)
        system_context_str = self._format_system_context(state, context)

        try:
            # -------------------------------------------------------------
            # Stage 1: Diagnostician
            # -------------------------------------------------------------
            diag_prompt = (
                f"You are the Diagnostician agent for {self.system_description}.\n"
                "Review the following system metrics and context. "
                "Determine if there is an anomaly, list any issues, possible root causes, and severity."
            )
            diag_msgs = [
                LLMMessage(role="system", content=diag_prompt),
                LLMMessage(role="user", content=system_context_str),
            ]

            diag_resp = await self.llm.generate(
                diag_msgs,
                temperature=self.temperature,
                max_tokens=1024,
                response_schema=DiagnosticianOutput,
            )

            try:
                parsed_diag = self._parse_json(diag_resp.content)
                diagnosis = DiagnosticianOutput.model_validate(parsed_diag)
            except Exception as e:
                if self.logger:
                    self.logger.error("Diagnostician failed to produce valid schema", error=str(e))
                self.metrics.increment(
                    "polaris.strategy.multi_agent.errors", tags={"agent": "diagnostician"}
                )
                return []

            if self.logger:
                self.logger.info(
                    "Diagnostician finished",
                    anomaly=diagnosis.is_anomaly_detected,
                    severity=diagnosis.severity,
                )

            if not diagnosis.is_anomaly_detected or diagnosis.severity.lower() == "none":
                return []

            # -------------------------------------------------------------
            # Stage 2: Planner
            # -------------------------------------------------------------
            planner_prompt = (
                f"You are the Planner agent for {self.system_description}.\n"
                "The Diagnostician has identified issues. Given the system context and the diagnosis, "
                "propose a sequence of adaptation actions (e.g., scale_up, scale_down, set_dimmer) to resolve them."
            )
            planner_input = (
                f"--- SYSTEM CONTEXT ---\n{system_context_str}\n\n"
                f"--- DIAGNOSIS ---\n"
                f"Issues: {diagnosis.issues}\n"
                f"Root Causes: {diagnosis.root_causes}\n"
                f"Severity: {diagnosis.severity}"
            )
            planner_msgs = [
                LLMMessage(role="system", content=planner_prompt),
                LLMMessage(role="user", content=planner_input),
            ]

            planner_resp = await self.llm.generate(
                planner_msgs,
                temperature=self.temperature,
                max_tokens=1500,
                response_schema=PlannerOutput,
            )

            try:
                parsed_plan = self._parse_json(planner_resp.content)
                plan = PlannerOutput.model_validate(parsed_plan)
            except Exception as e:
                if self.logger:
                    self.logger.error("Planner failed to produce valid schema", error=str(e))
                self.metrics.increment(
                    "polaris.strategy.multi_agent.errors", tags={"agent": "planner"}
                )
                return []

            if self.logger:
                self.logger.info("Planner finished", num_actions=len(plan.plans))

            if not plan.plans:
                return []

            # -------------------------------------------------------------
            # Stage 3: Safety Validator
            # -------------------------------------------------------------
            validator_prompt = (
                f"You are the Safety Validator agent for {self.system_description}.\n"
                "Review the diagnosis and the proposed plan. Evaluate if the actions are safe, "
                "appropriate, and will not destabilize the system further. You may approve the plan, "
                "modify it by returning a safer subset of actions, or reject it entirely (return empty safe_actions)."
            )

            actions_str = json.dumps([a.model_dump() for a in plan.plans])
            validator_input = (
                f"--- DIAGNOSIS ---\n"
                f"Severity: {diagnosis.severity}\n"
                f"Root Causes: {diagnosis.root_causes}\n\n"
                f"--- PROPOSED PLAN ---\n"
                f"Rationale: {plan.rationale}\n"
                f"Actions: {actions_str}"
            )
            validator_msgs = [
                LLMMessage(role="system", content=validator_prompt),
                LLMMessage(role="user", content=validator_input),
            ]

            valid_resp = await self.llm.generate(
                validator_msgs,
                temperature=self.temperature,
                max_tokens=1500,
                response_schema=ValidatorOutput,
            )

            try:
                parsed_valid = self._parse_json(valid_resp.content)
                validation = ValidatorOutput.model_validate(parsed_valid)
            except Exception as e:
                if self.logger:
                    self.logger.error("Validator failed to produce valid schema", error=str(e))
                self.metrics.increment(
                    "polaris.strategy.multi_agent.errors", tags={"agent": "validator"}
                )
                return []

            if self.logger:
                self.logger.info(
                    "Validator finished",
                    approved=validation.approved,
                    final_actions=len(validation.safe_actions),
                )

            if not validation.approved or not validation.safe_actions:
                return []

            # Convert approved actions to AdaptationAction objects
            final_actions: List[AdaptationAction] = []
            for action_block in validation.safe_actions:
                final_actions.append(
                    AdaptationAction(
                        action_id=str(uuid.uuid4()),
                        action_type=action_block.type,
                        target_system=state.system_id,
                        parameters={
                            **action_block.parameters,
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

    async def on_action_executed(self, action: AdaptationAction, result: Any) -> None:
        """Handle callback when an adaptation action is executed."""
        self._adaptation_count += 1
        ok = hasattr(result, "status") and getattr(result.status, "value", None) == "success"
        if ok:
            self._success_count += 1

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Get specification of tunable parameters for this strategy."""
        return {
            "temperature": ParameterSpec(
                current_value=self.temperature,
                type=float,
                min_value=0.0,
                max_value=2.0,
                description="LLM temperature tuning",
                kind="llm_temperature",
            ),
        }

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a tunable parameter value."""
        if parameter_path == "temperature":
            self.temperature = float(new_value)
            return True
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply configuration updates to the multi-agent strategy."""
        if isinstance(config, dict):
            if "temperature" in config:
                await self.update_parameter("temperature", config["temperature"])

            if "system_description" in config:
                self.system_description = config["system_description"]

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the multi-agent strategy."""
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}
        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }

    def _format_system_context(self, state: SystemState, context: AdaptationContext) -> str:
        """Format system state and context into a string representation."""
        metrics = []
        for k, v in state.metrics.items():
            try:
                metrics.append({"name": k, "value": v.value, "unit": v.unit})
            except Exception:
                metrics.append({"name": k, "value": str(getattr(v, "value", None))})

        data = {
            "system_id": state.system_id,
            "health": getattr(state.health_status, "value", "unknown"),
            "timestamp": state.timestamp.isoformat(),
            "metrics": metrics,
            "world_model_insights": context.world_model_insights or {},
        }
        return json.dumps(data)

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

            logging.getLogger(__name__).warning("LLM returned malformed JSON: %.500s", s)
            return {}
