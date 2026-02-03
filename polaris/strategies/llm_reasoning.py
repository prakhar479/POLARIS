"""
LLM-powered reasoning strategy.

Uses LLM to analyze system state and decide on adaptations.
"""

from typing import Optional, Dict, Any
import json
import uuid
from datetime import datetime, timezone

from polaris.abstractions.strategy import AdaptationStrategy, AdaptationContext, ParameterSpec
from polaris.abstractions.observability import Logger, MetricsCollector
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
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        per_system_prompts: Optional[Dict[str, str]] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None
    ):
        """
        Initialize LLM reasoning strategy.

        Args:
            llm_client: LLM client to use for reasoning
            system_description: Description of the managed system
            adaptation_goals: Goals for adaptation
            temperature: LLM temperature (lower = more deterministic)
            logger: Optional logger for observability
            metrics: Optional metrics collector
        """
        self.llm = llm_client
        self.system_description = system_description
        self.adaptation_goals = adaptation_goals
        self.temperature = temperature
        self._system_prompt_template = system_prompt
        self._per_system_prompts = per_system_prompts or {}
        self.logger = logger
        self.metrics = metrics
        self._adaptation_count = 0
        self._success_count = 0

    async def assess(
        self,
        state: SystemState,
        context: AdaptationContext
    ) -> Optional[AdaptationAction]:
        """Use LLM to assess if adaptation is needed."""
        if self.metrics:
            self.metrics.increment(
                "polaris.strategy.llm.assessments",
                tags={"system_id": state.system_id},
            )
        if self.logger:
            self.logger.debug(f"[LLM Reasoner] Starting assessment for system: {state.system_id}")
            self.logger.debug(f"[LLM Reasoner] System health: {state.health_status.value}")
            self.logger.debug(f"[LLM Reasoner] Metrics count: {len(state.metrics)}")
            for metric_name, metric_value in state.metrics.items():
                self.logger.debug(f"[LLM Reasoner]   - {metric_name}: {metric_value.value} {metric_value.unit or ''}")

        # Build prompt with system state
        prompt = self._build_prompt(state, context)
        if self.logger:
            self.logger.debug(f"[LLM Reasoner] Built assessment prompt (length: {len(prompt)} chars)")

        # Call LLM
        messages = [
            LLMMessage(role="system", content=self._get_system_prompt(state.system_id)),
            LLMMessage(role="user", content=prompt)
        ]

        if self.logger:
            self.logger.debug(f"[LLM Reasoner] Sending request to LLM with temperature={self.temperature}")
            self.logger.debug(f"[LLM Reasoner] System prompt length: {len(messages[0].content)} chars")
            self.logger.debug(f"[LLM Reasoner] User prompt length: {len(messages[1].content)} chars")

        try:
            if self.logger:
                self.logger.info(f"[LLM Reasoner] Calling LLM API for system {state.system_id}...")
            
            llm_start = datetime.now(timezone.utc)
            response = await self.llm.generate(
                messages,
                temperature=self.temperature,
                max_tokens=2048
            )
            llm_duration = (datetime.now(timezone.utc) - llm_start).total_seconds()

            if self.metrics:
                self.metrics.histogram(
                    "polaris.strategy.llm.llm_call_duration_seconds",
                    llm_duration,
                    tags={"system_id": state.system_id},
                )

            if self.logger:
                self.logger.debug(f"[LLM Reasoner] Received LLM response (length: {len(response.content)} chars)")
                self.logger.debug(f"[LLM Reasoner] LLM Response content:")
                for line in response.content.split('\n'):
                    self.logger.debug(f"[LLM Reasoner] {line}")

            # Parse LLM response
            action = self._parse_response(response.content, state.system_id)
            
            if action:
                if self.logger:
                    self.logger.info(f"[LLM Reasoner] Adaptation decision: YES")
                    self.logger.info(f"[LLM Reasoner] Action type: {action.action_type}")
                    self.logger.debug(f"[LLM Reasoner] Action parameters: {json.dumps(action.parameters, indent=2)}")
                if self.metrics:
                    self.metrics.increment(
                        "polaris.strategy.llm.actions_proposed",
                        tags={"system_id": state.system_id, "action_type": action.action_type},
                    )
            else:
                if self.logger:
                    self.logger.info(f"[LLM Reasoner] Adaptation decision: NO")
                if self.metrics:
                    self.metrics.increment(
                        "polaris.strategy.llm.no_action_needed",
                        tags={"system_id": state.system_id},
                    )
            
            return action

        except Exception as e:
            # Fall back to no action on error
            if self.logger:
                self.logger.error(f"[LLM Reasoner] Error during LLM assessment: {type(e).__name__}: {str(e)}")
                import traceback
                self.logger.debug(f"[LLM Reasoner] Traceback: {traceback.format_exc()}")
            if self.metrics:
                self.metrics.increment(
                    "polaris.strategy.llm.errors",
                    tags={"system_id": state.system_id},
                )
            return None

    def _get_system_prompt(self, system_id: Optional[str] = None) -> str:
        """Get system prompt for LLM, with optional system-specific overrides."""

        # Per-system override if provided
        if system_id and self._per_system_prompts:
            override = self._per_system_prompts.get(system_id)
            if override:
                return override

        # Global template override, optionally formatted
        if self._system_prompt_template:
            try:
                return self._system_prompt_template.format(
                    system_id=system_id or "",
                    system_description=self.system_description,
                    adaptation_goals=self.adaptation_goals,
                )
            except Exception:
                # If formatting fails, fall back to the raw template
                return self._system_prompt_template

        # Default generic prompt
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
                if self.logger:
                    self.logger.warn("[LLM Reasoner] Empty response from LLM")
                return None
            
            json_content = response
            extraction_method = "direct"
            
            # Handle ```json blocks
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0].strip()
                    if json_part:
                        json_content = json_part
                        extraction_method = "json_block"
            
            # Handle generic ``` blocks  
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 3:
                    json_part = parts[1].strip()
                    if json_part:
                        json_content = json_part
                        extraction_method = "code_block"

            if self.logger:
                self.logger.debug(f"[LLM Reasoner] JSON extraction method: {extraction_method}")
                self.logger.debug(f"[LLM Reasoner] Extracted JSON content (length: {len(json_content)} chars):")
                for line in json_content.split('\n')[:20]:  # Log first 20 lines
                    self.logger.debug(f"[LLM Reasoner] {line}")

            # Try to fix incomplete JSON by checking for unterminated strings
            if extraction_method != "direct":
                # Check if JSON appears incomplete
                if json_content.rstrip().endswith(('", ', '"')):
                    if self.logger:
                        self.logger.warn("[LLM Reasoner] Response appears incomplete - attempting to repair")
                    # Try to auto-complete the JSON structure
                    try:
                        # Count unclosed braces and brackets
                        open_braces = json_content.count('{') - json_content.count('}')
                        open_brackets = json_content.count('[') - json_content.count(']')
                        
                        # Complete the JSON
                        if '"reasoning": "' in json_content and not json_content.strip().endswith('"'):
                            # Complete unterminated reasoning field
                            json_content = json_content.rstrip() + '"'
                        
                        # Close all open structures
                        json_content += '}' * open_braces
                        json_content += ']' * open_brackets
                        
                        if self.logger:
                            self.logger.debug(f"[LLM Reasoner] Auto-repaired JSON by adding closing braces/brackets")
                    except Exception as repair_error:
                        if self.logger:
                            self.logger.warn(f"[LLM Reasoner] Could not auto-repair JSON: {repair_error}")

            data = json.loads(json_content)
            
            if self.logger:
                self.logger.debug(f"[LLM Reasoner] Successfully parsed JSON structure")
                self.logger.debug(f"[LLM Reasoner] Parsed JSON structure: {json.dumps(data, indent=2)}")
            
            # Validate response structure
            if not isinstance(data, dict):
                if self.logger:
                    self.logger.warn("[LLM Reasoner] Response is not a JSON object")
                return None

            needs_adaptation = data.get("needs_adaptation", False)
            reasoning = data.get("reasoning", "")
            
            if self.logger:
                self.logger.debug(f"[LLM Reasoner] needs_adaptation: {needs_adaptation}")
                self.logger.debug(f"[LLM Reasoner] reasoning: {reasoning}")

            if not needs_adaptation:
                if self.logger:
                    self.logger.debug(f"[LLM Reasoner] LLM determined no adaptation needed")
                return None

            action_data = data.get("action", {})
            if not isinstance(action_data, dict):
                if self.logger:
                    self.logger.warn("[LLM Reasoner] Action data is not a dictionary")
                return None
                
            action_type = action_data.get("type")
            parameters = action_data.get("parameters", {})

            if self.logger:
                self.logger.debug(f"[LLM Reasoner] Parsed action_type: {action_type}")
                self.logger.debug(f"[LLM Reasoner] Parsed parameters: {json.dumps(parameters, indent=2)}")

            if not action_type or not isinstance(parameters, dict):
                if self.logger:
                    self.logger.warn(f"[LLM Reasoner] Invalid action structure - type: {action_type}, params type: {type(parameters)}")
                return None

            adaptation_action = AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type=action_type,
                target_system=system_id,
                parameters={
                    **parameters,
                    "llm_reasoning": reasoning
                }
            )
            
            if self.logger:
                self.logger.info(f"[LLM Reasoner] Successfully created adaptation action: {adaptation_action.action_id}")
            
            return adaptation_action

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.error(f"[LLM Reasoner] JSON parsing error: {str(e)}")
                self.logger.debug(f"[LLM Reasoner] Failed to parse content (first 300 chars): {response[:300]}...")
            return None
        except (KeyError, TypeError, AttributeError) as e:
            if self.logger:
                self.logger.error(f"[LLM Reasoner] Error extracting adaptation data: {type(e).__name__}: {str(e)}")
            return None

    async def on_action_executed(self, action: AdaptationAction, result) -> None:
        """Track adaptation success."""
        self._adaptation_count += 1
        
        is_success = hasattr(result, 'status') and result.status.value == 'success'
        if is_success:
            self._success_count += 1
        if self.metrics:
            self.metrics.increment(
                "polaris.strategy.llm.actions_executed",
                tags={
                    "action_type": action.action_type,
                    "system_id": action.target_system,
                    "status": result.status.value if hasattr(result, 'status') else 'unknown',
                },
            )
            self.metrics.gauge(
                "polaris.strategy.llm.success_rate",
                self._success_count / self._adaptation_count if self._adaptation_count > 0 else 0.0,
            )
        
        if self.logger:
            status_str = "SUCCESS" if is_success else "FAILED"
            self.logger.info(f"[LLM Reasoner] Action execution result: {status_str}")
            self.logger.debug(f"[LLM Reasoner] Action ID: {action.action_id}")
            self.logger.debug(f"[LLM Reasoner] Action type: {action.action_type}")
            self.logger.debug(f"[LLM Reasoner] Total adaptations: {self._adaptation_count}")
            self.logger.debug(f"[LLM Reasoner] Successful adaptations: {self._success_count}")
            if hasattr(result, 'error_message'):
                self.logger.debug(f"[LLM Reasoner] Error message: {result.error_message}")

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
            )
        }

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update strategy parameters."""
        if parameter_path == "temperature":
            old_value = self.temperature
            self.temperature = float(new_value)
            if self.logger:
                self.logger.info(f"[LLM Reasoner] Updated temperature: {old_value} -> {self.temperature}")
            return True
        elif parameter_path == "system_description":
            old_value = self.system_description
            self.system_description = str(new_value)
            if self.logger:
                self.logger.info(f"[LLM Reasoner] Updated system_description: {old_value} -> {self.system_description}")
            return True
        if self.logger:
            self.logger.warn(f"[LLM Reasoner] Unknown parameter: {parameter_path}")
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            return

        if 'temperature' in config:
            await self.update_parameter("temperature", config['temperature'])
        if 'system_description' in config:
            await self.update_parameter("system_description", config['system_description'])

        if 'system_prompt' in config:
            self._system_prompt_template = config['system_prompt']
        if 'per_system_prompts' in config and isinstance(config['per_system_prompts'], dict):
            self._per_system_prompts = config['per_system_prompts']

        resil = config.get('resilience')
        if resil and hasattr(self.llm, "update_resilience"):
            try:
                self.llm.update_resilience(resil)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[LLM Reasoner] Failed to hot-update LLM resilience: {e}")

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy performance metrics."""
        if self._adaptation_count == 0:
            return {'success_rate': 0.0}

        return {
            'success_rate': self._success_count / self._adaptation_count,
            'total_adaptations': float(self._adaptation_count)
        }
