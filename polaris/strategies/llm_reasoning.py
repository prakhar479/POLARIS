"""
LLM-powered reasoning strategy.

Uses LLM to analyze system state and decide on adaptations.
"""

from typing import Optional, Dict, Any
import json
import uuid

from polaris.abstractions.strategy import AdaptationStrategy, AdaptationContext, ParameterSpec
from polaris.core.models import SystemState, AdaptationAction
from polaris.infrastructure.llm import LLMClient, LLMMessage


class LLMReasoningStrategy(AdaptationStrategy):
    """
    LLM-powered adaptation strategy.

    Uses an LLM to analyze system state and decide on adaptations
    based on natural language reasoning.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_description: str = "A web application server",
        adaptation_goals: str = "Maintain performance and availability",
        temperature: float = 0.1
    ):
        """
        Initialize LLM reasoning strategy.

        Args:
            llm_client: LLM client to use for reasoning
            system_description: Description of the managed system
            adaptation_goals: Goals for adaptation
            temperature: LLM temperature (lower = more deterministic)
        """
        self.llm = llm_client
        self.system_description = system_description
        self.adaptation_goals = adaptation_goals
        self.temperature = temperature
        self._adaptation_count = 0
        self._success_count = 0

    async def assess(
        self,
        state: SystemState,
        context: AdaptationContext
    ) -> Optional[AdaptationAction]:
        """Use LLM to assess if adaptation is needed."""

        # Build prompt with system state
        prompt = self._build_prompt(state, context)

        # Call LLM
        messages = [
            LLMMessage(role="system", content=self._get_system_prompt()),
            LLMMessage(role="user", content=prompt)
        ]

        try:
            response = await self.llm.generate(
                messages,
                temperature=self.temperature,
                max_tokens=512
            )

            # Parse LLM response
            action = self._parse_response(response.content, state.system_id)
            return action

        except Exception as e:
            # Fall back to no action on error
            return None

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return f"""You are an intelligent adaptation controller for a self-adaptive system.

System Description: {self.system_description}
Adaptation Goals: {self.adaptation_goals}

Your task is to analyze the current system state and decide if an adaptation action is needed.

Respond in JSON format:
{{
    "needs_adaptation": true/false,
    "reasoning": "explanation of your decision",
    "action": {{  // only if needs_adaptation is true
        "type": "scale_up" or "scale_down" or "adjust_qos",
        "parameters": {{key: value}}
    }}
}}

Be conservative - only adapt when there's a clear need. Consider:
- Current metric values vs normal ranges
- Trends and patterns
- Potential impact of adaptation
"""

    def _build_prompt(self, state: SystemState, context: AdaptationContext) -> str:
        """Build prompt with current state."""

        # Format metrics
        metrics_str = "\n".join([
            f"  - {name}: {metric.value} {metric.unit or ''}"
            for name, metric in state.metrics.items()
        ])

        # Get world model insights if available
        insights_str = ""
        if context.world_model_insights:
            insights_str = "\nWorld Model Insights:\n" + json.dumps(
                context.world_model_insights, indent=2
            )

        return f"""Current System State:

System ID: {state.system_id}
Health Status: {state.health_status.value}
Timestamp: {state.timestamp.isoformat()}

Metrics:
{metrics_str}
{insights_str}

Should this system be adapted right now? Analyze the state and provide your decision.
"""

    def _parse_response(self, response: str, system_id: str) -> Optional[AdaptationAction]:
        """Parse LLM response into adaptation action."""

        try:
            # Extract JSON from response with improved robustness
            response = response.strip()
            if not response:
                return None
            
            json_content = response
            
            # Handle ```json blocks
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0].strip()
                    if json_part:
                        json_content = json_part
            
            # Handle generic ``` blocks  
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 3:
                    json_part = parts[1].strip()
                    if json_part:
                        json_content = json_part

            data = json.loads(json_content)
            
            # Validate response structure
            if not isinstance(data, dict):
                return None

            if not data.get("needs_adaptation", False):
                return None

            action_data = data.get("action", {})
            if not isinstance(action_data, dict):
                return None
                
            action_type = action_data.get("type")
            parameters = action_data.get("parameters", {})

            if not action_type or not isinstance(parameters, dict):
                return None

            return AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type=action_type,
                target_system=system_id,
                parameters={
                    **parameters,
                    "llm_reasoning": data.get("reasoning", "")
                }
            )

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            # Failed to parse - no action
            return None

    async def on_action_executed(self, action: AdaptationAction, result) -> None:
        """Track adaptation success."""
        self._adaptation_count += 1
        if hasattr(result, 'status') and result.status.value == 'success':
            self._success_count += 1

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """LLM strategy parameters."""
        return {
            "temperature": ParameterSpec(
                current_value=self.temperature,
                type=float,
                min_value=0.0,
                max_value=2.0,
                description="LLM temperature for reasoning"
            ),
            "system_description": ParameterSpec(
                current_value=self.system_description,
                type=str,
                description="Description of the managed system"
            )
        }

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update strategy parameters."""
        if parameter_path == "temperature":
            self.temperature = float(new_value)
            return True
        elif parameter_path == "system_description":
            self.system_description = str(new_value)
            return True
        return False

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy performance metrics."""
        if self._adaptation_count == 0:
            return {'success_rate': 0.0}

        return {
            'success_rate': self._success_count / self._adaptation_count,
            'total_adaptations': float(self._adaptation_count)
        }
