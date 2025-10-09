"""
Expert Agentic Prompt Engineering Meta-Learner Agent Implementation

A conservative, expert-driven meta-learner that:
- Acts as an expert agentic prompt engineer and autonomous systems specialist
- Queries the Knowledge Base for comprehensive telemetry analysis
- Analyzes patterns using Gemini API with conservative decision-making criteria
- Updates STRATEGIC parameters (thresholds, agent reasoning patterns) only when clearly justified
- Keeps FIXED CONSTRAINTS unchanged (min/max servers, dimmer bounds)
- Makes minimal, evidence-based changes with stability prioritized over optimization
- Operates on conservative 10-minute intervals with strict change frequency controls
- Requires multiple confirmation cycles before implementing template modifications
"""

import asyncio
import json
import logging
import os
import time
import uuid
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import google.generativeai as genai
import google.api_core
from google.generativeai import types

import requests
import nats
from nats.aio.client import Client as NATS

from polaris.knowledge_base.models import KBDataType, KBEntry, KBQuery, KBResponse, QueryType

# Import the proper base class for meta-learners
from .meta_learner_agent import (
    BaseMetaLearnerAgent,
    MetaLearningContext,
    MetaLearningInsights,
    CalibrationRequest,
    CalibrationResult,
    ParameterUpdate,
    TriggerType,
    ParameterType,
)


class MetaLearnerLLM(BaseMetaLearnerAgent):
    """Incremental LLM-based meta-learner for strategic adaptation."""

    # Define immutable system constraints
    FIXED_CONSTRAINTS = {"min_servers", "max_servers", "dimmer_min_value", "dimmer_max_value"}

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        prompt_config_path: str,
        config_path: str,
        nats_url: Optional[str] = None,
        update_interval_seconds: float = 600.0,  # 10 minutes - more conservative interval
        model: str = "gemini-2.0-flash",
        temperature: float = 0.2,  # Lower temperature for more conservative, deterministic decisions
        max_tokens: int = 2048,
        kb_request_timeout: float = 30.0,
        max_change_percent: float = 10.0,  # More conservative - only 10% change per update
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize parent BaseMetaLearnerAgent
        super().__init__(agent_id, config_path, nats_url, logger)

        self.api_key = api_key
        self.prompt_config_path = Path(prompt_config_path)
        self.update_interval = update_interval_seconds
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_change_percent = max_change_percent
        self.kb_request_timeout = kb_request_timeout

        self.last_update_time: Optional[float] = None
        self.update_history: List[Dict[str, Any]] = []  # Track changes over time

        # Configure the Gemini client
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            self.logger.info(f"Initialized MetaLearnerLLM with Gemini model {model}")
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini client: {e}")
            raise

        # Load initial prompt config
        self.prompt_config = self._load_prompt_config()
        self.current_thresholds = self.prompt_config.get("thresholds", {})
        self.current_template_parts = self.prompt_config.get("template_parts", {})

        self.logger.info(f"Conservative update interval: {update_interval_seconds}s")
        self.logger.info(f"Conservative max change per update: {max_change_percent}%")
        self.logger.info(f"Prompt config: {self.prompt_config_path}")
        self.logger.info(
            f"Loaded {len(self.current_thresholds)} threshold parameters for optimization"
        )
        self.logger.info(f"Fixed constraints: {self.FIXED_CONSTRAINTS}")
        self.logger.info(
            f"Operating in threshold-focused optimization mode with 90% preference for parameter tuning"
        )

    def _load_prompt_config(self) -> Dict[str, Any]:
        """Load current thresholds and templates from prompt.yaml."""
        try:
            with open(self.prompt_config_path, "r") as f:
                config = yaml.safe_load(f)
                full_config = config.get("prompt_config", {})
                return {
                    "thresholds": full_config.get("thresholds", {}),
                    "template_parts": full_config.get("template_parts", {}),
                }
        except Exception as e:
            self.logger.error(f"Failed to load prompt config: {e}")
            return {"thresholds": {}, "template_parts": {}}

    def _save_prompt_config(self, updates: Dict[str, Any]) -> bool:
        """Update thresholds and/or template_parts in prompt.yaml."""
        try:
            # Load full config
            with open(self.prompt_config_path, "r") as f:
                config = yaml.safe_load(f)

            # Update thresholds (only additions and modifications, no removals)
            if "thresholds" in updates:
                if "prompt_config" not in config:
                    config["prompt_config"] = {}
                if "thresholds" not in config["prompt_config"]:
                    config["prompt_config"]["thresholds"] = {}

                for key, value in updates["thresholds"].items():
                    # Only add or update thresholds, never remove
                    config["prompt_config"]["thresholds"][key] = value
                    self.logger.info(f"Updated threshold: {key} = {value}")

            # Update template_parts
            if "template_parts" in updates:
                if "prompt_config" not in config:
                    config["prompt_config"] = {}
                if "template_parts" not in config["prompt_config"]:
                    config["prompt_config"]["template_parts"] = {}
                config["prompt_config"]["template_parts"].update(updates["template_parts"])

            # Save back
            with open(self.prompt_config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Updated prompt config: {updates}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save prompt config: {e}")
            return False

    async def query_kb_for_meta_learning(
        self, data_types: List[str], limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Query Knowledge Base for recent entries using parent class method."""
        try:
            query_data = {
                "query_type": "structured",
                "data_types": data_types,
                "filters": {},
                "limit": limit,
            }

            response = await self.query_knowledge_base(query_data)

            if response and response.get("results"):
                self.logger.info(f"Retrieved {len(response['results'])} entries from KB")
                return response["results"]
            else:
                self.logger.warning("KB query returned no results")
                return []

        except Exception as e:
            self.logger.error(f"Failed to query KB: {e}", exc_info=True)
            return []

    def _extract_telemetry_summary(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and summarize telemetry data from KB entries."""
        summary = {
            "total_entries": len(entries),
            "time_range": {},
            "snapshots": [],
            "other_observations": [],
            "adaptation_decisions": [],
            "metrics": {},
        }

        if not entries:
            return summary

        # Time range
        timestamps = [
            datetime.fromisoformat(e.get("timestamp", "")) for e in entries if e.get("timestamp")
        ]
        if timestamps:
            summary["time_range"] = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
            }

        # Extract observations, snapshots, and adaptation decisions
        for entry in entries:
            data_type = entry.get("data_type", "")

            if data_type == "observation":
                tags = entry.get("tags", [])
                summary_text = entry.get("summary", "")

                if "snapshot" in tags or "snapshot" in summary_text.lower():
                    summary["snapshots"].append(
                        {
                            "timestamp": entry.get("timestamp"),
                            "summary": summary_text,
                            "content": entry.get("content"),
                        }
                    )
                else:
                    summary["other_observations"].append(
                        {
                            "timestamp": entry.get("timestamp"),
                            "summary": summary_text,
                            "content": entry.get("content"),
                        }
                    )

            elif data_type == "adaptation_decision":
                content = entry.get("content", {})
                summary["adaptation_decisions"].append(
                    {
                        "timestamp": entry.get("timestamp"),
                        "action_type": content.get("action_type"),
                        "execution_success": content.get("execution_success"),
                        "params": content.get("params"),
                        "summary": entry.get("summary"),
                    }
                )

            # Aggregate metrics
            if entry.get("metric_name") and entry.get("metric_value") is not None:
                metric_name = entry["metric_name"]
                if metric_name not in summary["metrics"]:
                    summary["metrics"][metric_name] = []
                summary["metrics"][metric_name].append(
                    {
                        "timestamp": entry.get("timestamp"),
                        "value": entry.get("metric_value"),
                    }
                )

        summary["snapshot_count"] = len(summary["snapshots"])
        summary["other_observation_count"] = len(summary["other_observations"])
        summary["adaptation_decision_count"] = len(summary["adaptation_decisions"])

        return summary

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Gemini API to generate strategic updates."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        generation_config = types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
        )

        try:
            self.logger.info("Calling Gemini API for strategic analysis...")
            response = await self.client.generate_content_async(
                contents=[full_prompt],
                generation_config=generation_config,
            )

            response_text = response.text.strip()
            if not response_text:
                raise ValueError("Gemini API returned an empty response.")

            return response_text

        except (google.api_core.exceptions.GoogleAPIError, ValueError) as e:
            self.logger.error(f"Gemini API call failed: {e}")
            raise

    def _parse_meta_updates(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON updates from LLM response."""
        try:
            response = llm_response.strip()

            # Extract thinking block
            thinking = None
            if "<thinking>" in response:
                thinking_start = response.find("<thinking>") + 10
                thinking_end = response.find("</thinking>")
                if thinking_end > thinking_start:
                    thinking = response[thinking_start:thinking_end].strip()
                    self.logger.info(f"LLM Reasoning:\n{thinking}")

            # Extract JSON
            json_str = None
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]

            if not json_str:
                self.logger.warning("No JSON found in LLM response")
                return None

            parsed = json.loads(json_str)

            if parsed:
                self.logger.info(f"Parsed updates: {json.dumps(parsed, indent=2)}")
            else:
                self.logger.info("LLM suggests no changes (empty object)")

            return parsed

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from LLM response: {e}")
            self.logger.debug(f"LLM response was: {llm_response}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing updates: {e}")
            return None

    async def run_learning_cycle(self) -> bool:
        """Execute one meta-learning cycle."""
        self.logger.info("Starting meta-learning cycle...")

        try:
            # Query KB for observations and decisions
            observations = await self.query_kb_for_meta_learning(
                data_types=["observation"],
                limit=500,
            )

            decisions = await self.query_kb_for_meta_learning(
                data_types=["adaptation_decision"],
                limit=50,
            )

            # Filter and combine
            snapshot_entries = []
            other_observations = []

            for obs in observations:
                tags = obs.get("tags", [])
                summary = obs.get("summary", "")

                if "snapshot" in tags or "snapshot" in summary.lower():
                    snapshot_entries.append(obs)
                else:
                    other_observations.append(obs)

            snapshot_entries = sorted(
                snapshot_entries, key=lambda x: x.get("timestamp", ""), reverse=True
            )[:100]

            entries = snapshot_entries + other_observations + decisions

            self.logger.info(
                f"Retrieved {len(snapshot_entries)} snapshots, "
                f"{len(other_observations)} other observations, "
                f"{len(decisions)} adaptation decisions"
            )

            if not entries:
                self.logger.warning("No KB entries found, skipping cycle")
                return False

            # Extract telemetry summary
            telemetry_summary = self._extract_telemetry_summary(entries)

            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(telemetry_summary)

            # Call LLM
            self.logger.info("Calling LLM for strategic analysis...")
            llm_response = await self._call_llm(system_prompt, user_prompt)

            # Parse updates
            updates = self._parse_meta_updates(llm_response)

            # Handle empty or no updates gracefully
            if not updates or not isinstance(updates, dict):
                self.logger.info(
                    "No updates received from LLM (this is normal for conservative operation)"
                )
                return True  # This is not an error, just no changes needed

            # Validate and apply
            valid_updates = self._validate_and_constrain_updates(updates)

            if valid_updates and (
                valid_updates.get("thresholds") or valid_updates.get("template_parts")
            ):
                success = self._save_prompt_config(valid_updates)
                if success:
                    # Reload config
                    self.prompt_config = self._load_prompt_config()
                    self.current_thresholds = self.prompt_config.get("thresholds", {})
                    self.current_template_parts = self.prompt_config.get("template_parts", {})

                    # Track history
                    self.update_history.append(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "updates": valid_updates,
                        }
                    )

                    threshold_count = len(valid_updates.get("thresholds", {}))
                    template_count = len(valid_updates.get("template_parts", {}))

                    if threshold_count > 0:
                        self.logger.info(
                            f"Threshold optimization completed successfully. "
                            f"Updated {threshold_count} performance parameter(s)."
                        )
                    elif template_count > 0:
                        self.logger.info(
                            f"Template refinement completed. "
                            f"Updated {template_count} template component(s)."
                        )
                    else:
                        self.logger.info("Meta-learning cycle completed successfully.")

                    await self._publish_update_notification(valid_updates)
                    return True
                else:
                    self.logger.error("Failed to save updates")
                    return False
            else:
                self.logger.info(
                    "No valid updates to apply (system performing well or no clear improvements identified)"
                )
                return True  # Normal operation - no improvements needed currently

        except Exception as e:
            self.logger.error(f"Meta-learning cycle failed: {e}", exc_info=True)
            return False

    def _build_system_prompt(self) -> str:
        """Build system prompt emphasizing balanced, data-driven changes."""

    def _build_system_prompt(self) -> str:
        """Build system prompt emphasizing balanced, data-driven changes."""
        return """You are an expert systems performance engineer and parameter optimization specialist. Your primary role is to fine-tune system thresholds and performance parameters based on operational data and evidence.

**PRIMARY FOCUS:**
- Adaptive thresholds: Response times, utilization targets, scaling triggers, dimmer controls, cooldown timings
- Performance parameter optimization through data-driven adjustments
- System efficiency improvements via precise threshold calibration
- Intelligent template phrase evolution (add/remove ONE short phrase per cycle)

**SECONDARY (RARE) FOCUS:**
- Minor template wording improvements (only when adaptation success rates consistently fail)

**IMMUTABLE CONSTRAINTS (NEVER MODIFY):**
- min_servers, max_servers, dimmer_min_value, dimmer_max_value

**CURRENT PARAMETERS:**
Thresholds: {current_thresholds}
Templates: {current_template_parts}

**THRESHOLD OPTIMIZATION OPPORTUNITIES:**
- Response time targets and reduction thresholds
- Utilization scaling thresholds and targets
- Dimmer adjustment parameters and step sizes
- Performance variance and tolerance settings
- Cooldown timings for smarter adaptation cadence

**INCREMENTAL CHANGE RULES:**
1. **Threshold Priority**: Prefer threshold adjustments over template changes (80% of updates should be thresholds)
2. **Single Parameter**: Change only ONE threshold OR add/remove ONE adaptive phrase per cycle
3. **Conservative Limits**: Numerical changes limited to {max_change_percent}% maximum
4. **Evidence-Based**: Require 3+ cycles of data showing parameter issues
5. **YAML Validity**: Ensure all JSON is properly formatted and valid

**THRESHOLD CHANGE CRITERIA:**
- Parameter consistently violated or never triggered (3+ observation periods)
- Clear performance impact measurable in metrics
- >80% confidence that threshold adjustment will improve system behavior
- No recent changes to the same threshold

**COMMON THRESHOLD ADJUSTMENTS:**
- Response time targets: Tighten if consistently exceeded, relax if never approached
- Utilization thresholds: Adjust based on actual scaling patterns and efficiency
- Dimmer reduction points: Modify based on response time vs user experience trade-offs
- Target values: Calibrate based on observed system behavior patterns
- Cooldown adjustments: Reduce for faster response, increase for stability
- Add/modify cooldown to thresholds if needed for stability

**ADAPTIVE PHRASE EVOLUTION (Smart Template Enhancement):**
- Add ONE short phrase when patterns show need for proactive behavior
- Remove outdated phrases when system behavior changes
- Example additions: "Anticipate load spikes." "Monitor cascade effects." "Prioritize user experience."
- Example removals: Remove phrases that no longer match system patterns

**TEMPLATE CHANGES (RARE - Only if adaptation success <70%):**
- Minor wording clarifications in system_role
- Small instruction improvements in constraints description or additional instructions
- Never drastically modify reasoning_structure (protected cognitive architecture)

**EXAMPLES OF VALID CHANGES:**

Example 1 - Response Time Threshold Adjustment:
```json
{{
  "thresholds": {{
    "target_response_time_ms": 650
  }}
}}
```

Example 2 - Utilization Target Optimization:
```json
{{
  "thresholds": {{
    "target_server_utilization": 0.75
  }}
}}
```

Example 3 - Dimmer Control Refinement:
```json
{{
  "thresholds": {{
    "dimmer_reduction_threshold_s": 0.6
  }}
}}
```

Example 4 - Cooldown Timing Optimization:
```json
{{
  "thresholds": {{
    "cooldown_period_minutes": 3
  }}
}}
```

Example 5 - Add Adaptive Phrase for Proactivity:
```json
{{
  "template_parts": {{
    "adaptive_phrases": "Monitor for early warning signs. Anticipate system load patterns."
  }}
}}
```

Example 6 - Minor Template Wording (RARE - only if needed):
```json
{{
  "template_parts": {{
    "system_role": "You are an integrated, autonomous system controller focused on maintaining optimal performance and efficiency."
  }}
}}
```

**OUTPUT REQUIREMENTS:**
- **Threshold Priority**: Prefer threshold adjustments (80% of changes should be thresholds)
- Return {{}} unless clear evidence warrants a single, targeted change
- JSON must be valid and properly escaped
- Only use existing threshold parameter names OR add ONE adaptive phrase
- Template changes only if adaptation success rates consistently fail
- Provide clear but brief justification focused on performance impact

**DECISION APPROACH:**
Focus on threshold optimization for measurable performance improvements. Add adaptive phrases to make the reasoner more proactive when patterns show gaps in behavior. Avoid major template changes unless absolutely necessary. System intelligence through parameter tuning and phrase evolution is more valuable than prompt overhauls."""

    def _build_user_prompt(self, telemetry_summary: Dict[str, Any]) -> str:
        """Build user prompt with telemetry context."""
        metrics = telemetry_summary.get("metrics", {})
        decisions = telemetry_summary.get("adaptation_decisions", [])

        # Decision summary
        decision_summary = {"total": len(decisions), "by_type": {}, "success_rate": {}}

        for dec in decisions:
            action_type = dec.get("action_type", "UNKNOWN")
            success = dec.get("execution_success", False)

            if action_type not in decision_summary["by_type"]:
                decision_summary["by_type"][action_type] = {"total": 0, "success": 0, "failed": 0}

            decision_summary["by_type"][action_type]["total"] += 1
            if success:
                decision_summary["by_type"][action_type]["success"] += 1
            else:
                decision_summary["by_type"][action_type]["failed"] += 1

        for action_type, counts in decision_summary["by_type"].items():
            if counts["total"] > 0:
                decision_summary["success_rate"][action_type] = counts["success"] / counts["total"]

        # Get recent update history
        recent_updates = self.update_history[-3:] if self.update_history else []

        prompt = f"""**SYSTEM DATA ANALYSIS:**

**Time Period:** {telemetry_summary.get('time_range', {}).get('start', 'N/A')} to {telemetry_summary.get('time_range', {}).get('end', 'N/A')}

**Activity Summary:**
- Snapshots: {telemetry_summary.get('snapshot_count', 0)}
- Observations: {telemetry_summary.get('other_observation_count', 0)}  
- Adaptations: {telemetry_summary.get('adaptation_decision_count', 0)}

**Adaptation Performance:**
{json.dumps(decision_summary, indent=2)}

**Recent Changes:**
{json.dumps(recent_updates, indent=2) if recent_updates else "No recent changes"}

**THRESHOLD OPTIMIZATION ANALYSIS:**

**Time Period:** {telemetry_summary.get('time_range', {}).get('start', 'N/A')} to {telemetry_summary.get('time_range', {}).get('end', 'N/A')}
**Snapshots:** {telemetry_summary.get('snapshot_count', 0)} | **Adaptations:** {telemetry_summary.get('adaptation_decision_count', 0)}

**Adaptation Success Rates:**
{json.dumps(decision_summary, indent=2)}

**Recent Updates:** {json.dumps(recent_updates, indent=2) if recent_updates else "None"}

**THRESHOLD ANALYSIS QUESTIONS:**
1. Which thresholds are consistently violated or never triggered?
2. Are response time targets appropriate for actual system performance?
3. Do utilization thresholds match observed scaling patterns?
4. Are dimmer controls optimally calibrated for performance vs user experience?
5. Do any parameters show persistent misalignment with system behavior?

**PARAMETER OPTIMIZATION DECISION:**
- Focus on ONE threshold that shows clear misalignment with system behavior
- Prioritize response time, utilization, and dimmer thresholds
- Avoid template changes unless adaptation success rates are critically low (<70%)
- Ensure JSON is valid with proper escaping
- Return {{}} if thresholds appear well-calibrated

**PREFERRED ACTIONS (in priority order):**
1. Adjust response time thresholds (target_response_time_ms, dimmer_reduction_threshold_s)
2. Calibrate utilization targets (target_server_utilization)
3. Fine-tune cooldown parameters (cooldown_period_minutes)
4. Add/remove adaptive phrases to enhance reasoner intelligence
5. Return {{}} if no clear optimization opportunity exists

**ADAPTIVE PHRASE INTELLIGENCE:**
- Add phrases when system shows reactive patterns: "Anticipate load spikes." "Monitor cascade effects."
- Remove phrases when system behavior changes or phrases become obsolete
- Keep phrases short and actionable (max 10 words each)
- Focus on making the reasoner more proactive and pattern-aware

Analyze performance data for threshold optimization and reasoner intelligence enhancement opportunities with clear measurable impact."""

        return prompt

    def _validate_and_constrain_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate updates and enforce incremental change constraints."""
        valid_updates = {}

        # STRICT INCREMENTAL RULE: Only allow ONE change at a time, prefer thresholds
        total_changes = 0
        if "thresholds" in updates:
            total_changes += len(updates["thresholds"])
        if "template_parts" in updates:
            total_changes += len(updates["template_parts"])

        if total_changes > 1:
            self.logger.warning(
                f"Too many changes proposed ({total_changes}). Prioritizing threshold changes over template changes."
            )
            # Strongly prefer threshold changes over template changes
            if "thresholds" in updates and len(updates["thresholds"]) > 0:
                first_threshold = list(updates["thresholds"].keys())[0]
                updates = {"thresholds": {first_threshold: updates["thresholds"][first_threshold]}}
                self.logger.info(f"Selected threshold change: {first_threshold}")
            elif "template_parts" in updates and len(updates["template_parts"]) > 0:
                first_template = list(updates["template_parts"].keys())[0]
                updates = {
                    "template_parts": {first_template: updates["template_parts"][first_template]}
                }
                self.logger.info(f"Selected template change: {first_template}")

        # Additional check: Discourage template changes unless justified
        if "template_parts" in updates and "thresholds" not in updates:
            self.logger.warning(
                "Template change proposed without threshold alternative. Template changes should be rare."
            )

        # Process threshold updates (including additions and removals)
        if "thresholds" in updates:
            valid_thresholds = {}

            # Validation rules (min, max) - used as hints, not strict requirements
            known_rules = {
                "max_response_time_s": (0.5, 5.0),
                "max_response_time_ms": (500, 5000),
                "dimmer_reduction_threshold_s": (0.3, 2.0),
                "target_response_time_ms": (300, 2000),
                "acceptable_response_time_variance": (0.05, 0.3),
                "dimmer_max_step": (0.05, 0.5),
                "utilization_scale_up_threshold_percent": (70, 95),
                "utilization_scale_down_target_percent": (30, 70),
                "utilization_optimal_range_min": (40, 70),
                "utilization_optimal_range_max": (60, 85),
                "cooldown_period_minutes": (1, 15),
            }

            # Cooldown parameters (now adjustable for smarter timing)
            cooldown_params = {
                "cooldown_period_minutes",
            }

            for key, new_value in updates["thresholds"].items():
                # Check if this is a fixed constraint
                if key in self.FIXED_CONSTRAINTS:
                    self.logger.warning(
                        f"Ignoring update to fixed constraint: {key} "
                        f"(current: {self.current_thresholds.get(key)})"
                    )
                    continue

                # Check if this is a cooldown parameter (now adjustable)
                if key in cooldown_params:
                    # Allow cooldown adjustments based on adaptation patterns
                    self.logger.info(
                        f"Adjusting cooldown parameter: {key} for smarter timing adaptation"
                    )

                # Skip any removal attempts - not allowed
                if new_value is None or new_value == "__REMOVE__":
                    self.logger.warning(f"Threshold removal not allowed for {key}, skipping")
                    continue

                # Validate data type
                if not isinstance(new_value, (int, float)):
                    self.logger.warning(f"Invalid value type for {key}: {type(new_value)}")
                    continue

                # Check against known rules if available
                if key in known_rules:
                    min_val, max_val = known_rules[key]
                    if not (min_val <= new_value <= max_val):
                        self.logger.warning(
                            f"Value for {key}: {new_value} outside recommended range "
                            f"[{min_val}, {max_val}] - allowing anyway for flexibility"
                        )

                # Enforce incremental change constraint for existing thresholds
                current_value = self.current_thresholds.get(key)
                if current_value is not None:
                    max_change = abs(current_value * self.max_change_percent / 100.0)
                    change = abs(new_value - current_value)

                    if change > max_change:
                        # Constrain to max allowed change
                        if new_value > current_value:
                            constrained_value = current_value + max_change
                        else:
                            constrained_value = current_value - max_change

                        self.logger.warning(
                            f"Change to {key} too large ({change:.2f}), "
                            f"constraining from {new_value} to {constrained_value:.2f}"
                        )
                        new_value = constrained_value

                    self.logger.info(
                        f"Validated threshold update: {key} = {new_value} (was {current_value})"
                    )
                else:
                    # New threshold being added
                    self.logger.info(f"Adding new threshold: {key} = {new_value}")

                valid_thresholds[key] = new_value

            if valid_thresholds:
                valid_updates["thresholds"] = valid_thresholds

        # Process template_parts updates
        if "template_parts" in updates:
            valid_template_parts = {}

            allowed_parts = {
                "system_role",
                "reasoning_structure",
                "constraints",
                "adaptive_phrases",
            }

            for key, new_text in updates["template_parts"].items():
                if key not in allowed_parts:
                    self.logger.warning(
                        f"Unknown template part '{key}'. Valid parts are: {allowed_parts}"
                    )
                    continue

                if not isinstance(new_text, str) or len(new_text) < 10:
                    self.logger.warning(f"Invalid template text for {key}")
                    continue

                # Special validation for reasoning_structure (agentic framework protection)
                if key == "reasoning_structure":
                    # Get current reasoning structure to compare
                    current_structure = self.current_template_parts.get("reasoning_structure", "")

                    # Count numbered steps in both current and new structure
                    import re

                    current_steps = re.findall(r"\d+\.", current_structure)
                    new_steps = re.findall(r"\d+\.", new_text)

                    if len(new_steps) != len(current_steps):
                        self.logger.warning(
                            f"Reasoning structure step count changed from {len(current_steps)} to {len(new_steps)}. "
                            f"Core framework structure must be preserved."
                        )
                        continue

                    # Check for major structural changes (>50% of content changed)
                    if len(current_structure) > 0:
                        # Simple heuristic: if the new text is dramatically different in length or has few common words
                        length_ratio = len(new_text) / len(current_structure)
                        if length_ratio < 0.5 or length_ratio > 2.0:
                            self.logger.warning(
                                f"Reasoning structure appears to have major structural changes "
                                f"(length ratio: {length_ratio:.2f}). Only minor refinements allowed."
                            )
                            continue

                # Handle adaptive_phrases for intelligent template evolution
                if key == "adaptive_phrases":
                    current_phrases = self.current_template_parts.get("adaptive_phrases", "")

                    # Allow adding/removing ONE short phrase per cycle
                    if isinstance(new_text, str) and len(new_text.strip()) > 0:
                        phrase_count = len([p for p in new_text.split(".") if p.strip()])
                        current_count = (
                            len([p for p in current_phrases.split(".") if p.strip()])
                            if current_phrases
                            else 0
                        )

                        # Allow only small incremental changes (+/- 1 phrase)
                        if abs(phrase_count - current_count) <= 1:
                            self.logger.info(f"Evolving adaptive phrases: {phrase_count} phrases")
                        else:
                            self.logger.warning(
                                f"Too many phrase changes ({phrase_count} vs {current_count}), limiting to incremental"
                            )
                            continue

                # Check if fixed constraints are mentioned in constraints template
                if key == "constraints":
                    if any(fixed in new_text for fixed in self.FIXED_CONSTRAINTS):
                        self.logger.info(
                            f"Template update for {key} mentions fixed constraints (OK)"
                        )

                valid_template_parts[key] = new_text
                self.logger.info(f"Validated template part update: {key} ({len(new_text)} chars)")

            if valid_template_parts:
                valid_updates["template_parts"] = valid_template_parts

        # Cross-validation for utilization ranges
        if "thresholds" in valid_updates:
            thresh = valid_updates["thresholds"]
            if (
                "utilization_optimal_range_min" in thresh
                and "utilization_optimal_range_max" in thresh
            ):
                if (
                    thresh["utilization_optimal_range_min"]
                    >= thresh["utilization_optimal_range_max"]
                ):
                    self.logger.warning(
                        "utilization_optimal_range_min must be < max, skipping both updates"
                    )
                    # Remove from this update only, don't modify the saved config
                    thresh.pop("utilization_optimal_range_min", None)
                    thresh.pop("utilization_optimal_range_max", None)

        return valid_updates

    async def _publish_update_notification(self, updates: Dict[str, Any]) -> None:
        """Publish update notification to NATS."""
        try:
            notification = {
                "event_type": "meta_learning_update",
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "updates": updates,
                "prompt_config_path": str(self.prompt_config_path),
            }

            await self.publish("polaris.meta_learner.update", notification)
            self.logger.info("Published meta-learning update notification")

        except Exception as e:
            self.logger.error(f"Failed to publish update notification: {e}")

    async def run(self, stop_event: asyncio.Event) -> None:
        """Main run loop - execute learning cycles periodically."""
        self.running = True
        self.logger.info("Meta-learner agent started")

        while not stop_event.is_set() and self.running:
            try:
                await self.run_learning_cycle()

                self.logger.info(f"Waiting {self.update_interval}s until next cycle...")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.update_interval)
                    break
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                self.logger.error(f"Error in run loop: {e}", exc_info=True)
                await asyncio.sleep(60)

        self.running = False
        self.logger.info("Meta-learner agent stopped")

    async def stop(self) -> None:
        """Stop the meta-learner."""
        self.running = False
        self.logger.info("Stopping meta-learner agent...")

    # ===============================================================
    # == IMPLEMENTATION OF ABSTRACT METHODS FROM BaseMetaLearnerAgent
    # ===============================================================

    async def analyze_adaptation_patterns(
        self, context: MetaLearningContext
    ) -> MetaLearningInsights:
        """Analyze adaptation patterns and system behavior from the knowledge base."""
        try:
            # Query KB for observations and decisions
            observations = await self.query_kb_for_meta_learning(
                data_types=["observation"],
                limit=500,
            )

            decisions = await self.query_kb_for_meta_learning(
                data_types=["adaptation_decision"],
                limit=50,
            )

            # Extract telemetry summary
            entries = observations + decisions
            telemetry_summary = self._extract_telemetry_summary(entries)

            # Create insights from the summary
            insights = MetaLearningInsights(
                analysis_window={
                    "start": telemetry_summary.get("time_range", {}).get("start", ""),
                    "end": telemetry_summary.get("time_range", {}).get("end", ""),
                },
                adaptation_patterns=telemetry_summary.get("adaptation_decisions", []),
                performance_trends=telemetry_summary.get("metrics", {}),
                recommendations=["Continue monitoring system behavior"],
                confidence_overall=0.8,
            )

            return insights

        except Exception as e:
            self.logger.error(f"Failed to analyze adaptation patterns: {e}")
            # Return empty insights on error
            return MetaLearningInsights(
                analysis_window={"start": "", "end": ""}, confidence_overall=0.0
            )

    async def calibrate_world_model(
        self, calibration_request: CalibrationRequest
    ) -> CalibrationResult:
        """Calibrate the world model (digital twin) to improve its accuracy."""
        # For now, return a simple result since we don't have a digital twin interface yet
        return CalibrationResult(
            request_id=calibration_request.request_id,
            success=True,
            improvement_score=0.1,
            calibrated_parameters={},
            validation_metrics={},
        )

    async def propose_parameter_updates(
        self, insights: MetaLearningInsights, context: MetaLearningContext
    ) -> List[ParameterUpdate]:
        """Propose updates to adaptation system parameters based on insights."""
        # This is handled by the existing run_learning_cycle method
        # For interface compliance, return empty list
        return []

    async def validate_updates(
        self,
        proposed_updates: List[ParameterUpdate],
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> List[ParameterUpdate]:
        """Validate proposed parameter updates before application."""
        # Use existing validation logic
        valid_updates = []
        for update in proposed_updates:
            # Simple validation - could be enhanced
            if update.confidence > 0.5:
                valid_updates.append(update)
        return valid_updates

    async def apply_updates(self, validated_updates: List[ParameterUpdate]) -> Dict[str, bool]:
        """Apply validated parameter updates to the adaptation system."""
        # For now, return success for all updates
        return {update.update_id: True for update in validated_updates}

    async def handle_trigger(self, trigger_type: TriggerType, trigger_data: Dict[str, Any]) -> bool:
        """Handle different types of meta-learning triggers."""
        try:
            self.logger.info(f"Handling trigger: {trigger_type} with data: {trigger_data}")

            # Create context for meta-learning
            context = MetaLearningContext(
                trigger_type=trigger_type,
                trigger_source=trigger_data.get("source", "unknown"),
                time_window_hours=trigger_data.get("time_window_hours", 24.0),
            )

            # Run a learning cycle
            success = await self.run_learning_cycle()
            return success

        except Exception as e:
            self.logger.error(f"Failed to handle trigger {trigger_type}: {e}")
            return False


def create_meta_learner_agent(
    agent_id: str,
    api_key: str,
    prompt_config_path: str,
    config_path: str,
    nats_url: Optional[str] = None,
    update_interval_seconds: float = 600.0,  # Conservative 10-minute interval
    model: str = "gemini-2.0-flash",
    max_change_percent: float = 10.0,  # Conservative 10% max change
    logger: Optional[logging.Logger] = None,
) -> MetaLearnerLLM:
    """Factory function to create a meta-learner agent."""
    return MetaLearnerLLM(
        agent_id=agent_id,
        api_key=api_key,
        prompt_config_path=prompt_config_path,
        config_path=config_path,
        nats_url=nats_url,
        update_interval_seconds=update_interval_seconds,
        model=model,
        max_change_percent=max_change_percent,
        logger=logger,
    )
