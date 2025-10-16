"""
Expert Agentic Meta-Learner Agent Implementation

A balanced, intelligent meta-learner that:
- Acts as an expert systems engineer and pattern recognition specialist
- Queries the Knowledge Base for comprehensive telemetry analysis
- Analyzes patterns using Gemini API for balanced optimization decisions
- Updates BOTH threshold parameters AND stores valuable observations in observed_learnings
- Keeps FIXED CONSTRAINTS unchanged (min/max servers, dimmer bounds)
- Makes evidence-based changes with equal focus on optimization and pattern storage
- Operates on 10-minute intervals with intelligent change management
- Stores system insights, patterns, and behaviors to make the reasoner more intelligent
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
    """Balanced LLM-based meta-learner for threshold optimization and observation storage."""

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

        self.logger.info(f"Update interval: {update_interval_seconds}s")
        self.logger.info(f"Max change per update: {max_change_percent}%")
        self.logger.info(f"Prompt config: {self.prompt_config_path}")
        self.logger.info(
            f"Loaded {len(self.current_thresholds)} threshold parameters for optimization"
        )
        self.logger.info(f"Fixed constraints: {self.FIXED_CONSTRAINTS}")
        self.logger.info(
            f"Operating in balanced mode: 50% threshold optimization + 50% observation storage"
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

            # Custom representer for all strings - use | for everything
            def str_presenter(dumper, data):
                if isinstance(data, str):
                    data = data.replace("\\n", "\n")
                    # Use literal block style if multiline OR contains backslashes
                    style = "|" if ("\n" in data or "\\" in data) else None
                    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)
                return dumper.represent_scalar("tag:yaml.org,2002:str", data)

            # Register the custom representer
            yaml.add_representer(str, str_presenter)

            # Save back with proper formatting
            with open(self.prompt_config_path, "w") as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=4096,  # Prevent line wrapping
                )

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
            print(f"LLM full response:\n{response}")

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

                    if threshold_count > 0 and template_count > 0:
                        self.logger.info(
                            f"System optimization completed: {threshold_count} threshold(s) + {template_count} observation(s) stored."
                        )
                    elif threshold_count > 0:
                        self.logger.info(
                            f"Threshold optimization completed: Updated {threshold_count} performance parameter(s)."
                        )
                    elif template_count > 0:
                        observed_learnings_updated = "observed_learnings" in valid_updates.get(
                            "template_parts", {}
                        )
                        if observed_learnings_updated:
                            self.logger.info(
                                f"Observation storage completed: Stored system patterns for reasoner intelligence."
                            )
                        else:
                            self.logger.info(
                                f"Template refinement completed: Updated {template_count} template component(s)."
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
        """Build system prompt emphasizing balanced optimization and observation storage."""
        return f"""You are an expert systems engineer and prompt engineer specializing in adaptive system optimization and pattern recognition. Your role is to improve system performance through both parameter tuning and intelligent observation storage.

**DECISION FRAMEWORK:**
1. **OBSERVATION STORAGE** - When you identify new patterns or insights that aren't captured yet
2. **THRESHOLD TUNING** - When numerical parameters are clearly misaligned with actual system behavior
3. **TEMPLATE REFINEMENT** - When reasoning structure or constraints need minor improvements for better performance

**CURRENT STATE:**
- Thresholds: {self.current_thresholds}
- Templates: {self.current_template_parts}

**OBSERVATION STORAGE (when new patterns emerge):**
Store brief insights in `observed_learnings` when you discover:
- Performance correlations: "How CPU, memory, and response time relate"
- Operational patterns: "What kinds of loads cause spikes"  
- Adaptation learnings: "What kind of actions worked well under what conditions"

**THRESHOLD TUNING (when parameters are clearly wrong):**
Look for these signs that thresholds need adjustment:
- Response time targets consistently exceeded or never approached
- Utilization thresholds causing unnecessary scaling or failing to scale
- Dimmer controls not matching performance vs user experience trade-offs
- Cooldown timings causing delays or instability
- Change by max {self.max_change_percent}% per cycle

**TEMPLATE REFINEMENT (only for significant performance improvements):**
You may make LIMITED modifications to `reasoning_structure` or `constraints` ONLY when:
- Current reasoning steps consistently lead to poor decisions
- Evidence shows a specific reasoning gap that could be filled

TEMPLATE MODIFICATION RULES:
- Keep the same overall structure
- Only add clarifications, don't remove core logic
- For constraints: only add clarifying details, never remove safety constraints
- Maximum 1-2 sentence additions per modification then give final updated text
- Must be directly supported by strong performance evidence
- You CANNOT change tool calling examples or logic

**FIXED CONSTRAINTS (NEVER CHANGE):**
{', '.join(self.FIXED_CONSTRAINTS)}

**EXAMPLES:**
Both Threshold Adjustment AND Pattern Storage (data shows misaligned parameter + new insight):
```json
{{
  "thresholds": {{
    "utilization_scale_up_threshold_percent": 75
  }},
  "template_parts": {{
    "observed_learnings": "Scaling at 85% CPU causes 3s delay - early scaling at 75% improves user experience."
  }}
}}
```
Threshold Too High (Response time target 800ms but system averages 600ms):
```json
{{
  "thresholds": {{
    "target_response_time_ms": 650
  }}
}}
```json
{{
  "template_parts": {{
    "some other template part": "old contents + new clarification/slight refinement"
  }}
}}

### TASK
Analyze the provided decision data and determine whether to **tune thresholds**, **refine templates**, **apply both**, or **make no change**.

---

### DECISION CRITERIA
1. **Tune Thresholds** Only if current parameters clearly mismatch system behavior (too high, too low, or never triggered).
2. **Refine Templates** Only if reasoning or structure shows repeated logical gaps or misaligned decisions.
3. **Combine Both** If both conditions above are independently supported by strong evidence.
4. **Return {{}}** When:
   - Thresholds are well-calibrated
   - No new behavior patterns are observed
   - Reasoning is already sound and consistent

### OUTPUT REQUIREMENTS
- Output must be **valid JSON**, properly formatted.
- Evaluate **thresholds**, **patterns**, and **templates** with equal rigor.
- Only propose changes when **clearly supported by evidence**.
- For **template changes**, include the **entire revised text**, not just fragments.
- Keep the response **concise and non-redundant** — suitable for direct use as a prompt.
- Provide a **brief justification** for each decision (1 sentence).
"""

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
**Data Points:** {telemetry_summary.get('snapshot_count', 0)} snapshots, {telemetry_summary.get('other_observation_count', 0)} observations, {telemetry_summary.get('adaptation_decision_count', 0)} adaptations

**Adaptation Performance:**
{json.dumps(decision_summary, indent=2)}

**Recent Changes:** {json.dumps(recent_updates, indent=2) if recent_updates else "None"}

**ANALYSIS FOCUS:**

**PATTERN DISCOVERY CHECK:**  
Look for new insights not yet captured in observed_learnings:
- Performance correlations (CPU, memory, response time relationships)
- User behavior patterns (dimmer thresholds)
- Operational patterns (time-based load, cascade effects)
- Change the current content to add new observation and given final, summarized and brief text 
- Successful adaptation strategies under specific conditions

**THRESHOLD MISALIGNMENT CHECK:**
Examine if current thresholds match actual system behavior:
- Are response time targets appropriate for observed performance?
- Do utilization thresholds trigger scaling at right times?  
- Are dimmer controls calibrated for performance vs user experience?
- Do cooldown periods cause delays or allow instability?

**TEMPLATE REFINEMENT CHECK:**
Analyze if reasoning structure or constraints need minor improvements:
- Are decision failures due to missing reasoning steps or unclear logic?
- Would small clarifications significantly improve decision quality?

**DECISION PRIORITY:**
1. **Capture New Patterns**: If you discover insights not yet stored, add them to observed_learnings  
2. **Fix Parameter Misalignment**: If thresholds clearly don't match system behavior, adjust them
3. **Refine Decision Templates**: If reasoning structure or constraints have clear gaps causing repeated failures
You should make ANY COMBINATION of changes if strongly supported by data
4. **No Changes**: Return {{}} only if parameters are well-aligned AND no new patterns are evident AND reasoning is sound

- "Append and give final phrase (old+new) to store in observed_learnings"
"""

        return prompt

    def _validate_and_constrain_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate updates and enforce incremental change constraints."""
        valid_updates = {}

        # INCREMENTAL RULE: Allow up to 1 change in each category (thresholds and template_parts)
        threshold_changes = len(updates.get("thresholds", {}))
        template_changes = len(updates.get("template_parts", {}))

        # Limit thresholds to maximum 1 change
        if threshold_changes > 1:
            self.logger.warning(
                f"Too many threshold changes proposed ({threshold_changes}). Limiting to 1 change."
            )
            first_threshold = list(updates["thresholds"].keys())[0]
            updates["thresholds"] = {first_threshold: updates["thresholds"][first_threshold]}
            self.logger.info(f"Selected threshold change: {first_threshold}")

        # Limit template_parts to maximum 1 change
        if template_changes > 1:
            self.logger.warning(
                f"Too many template changes proposed ({template_changes}). Limiting to 1 change."
            )
            first_template = list(updates["template_parts"].keys())[0]
            updates["template_parts"] = {first_template: updates["template_parts"][first_template]}
            self.logger.info(f"Selected template change: {first_template}")

        # Log the final change summary
        final_threshold_changes = len(updates.get("thresholds", {}))
        final_template_changes = len(updates.get("template_parts", {}))
        if final_threshold_changes > 0 and final_template_changes > 0:
            self.logger.info(f"Allowing 1 threshold change and 1 template change in this cycle")
        elif final_threshold_changes > 0:
            self.logger.info(f"Allowing 1 threshold change in this cycle")
        elif final_template_changes > 0:
            self.logger.info(f"Allowing 1 template change (observation storage) in this cycle")

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
                "observed_learnings",
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
                        if length_ratio < 0.7 or length_ratio > 1.5:
                            self.logger.warning(
                                f"Reasoning structure appears to have major structural changes "
                                f"(length ratio: {length_ratio:.2f}). Only minor refinements allowed."
                            )
                            continue

                    # Ensure core step names are preserved
                    required_steps = [
                        "Data Observer",
                        "Trend Analyst",
                        "Performance Reviewer",
                        "Strategy Planner",
                        "Action Sanity Check",
                        "Command Generator",
                    ]
                    missing_steps = [step for step in required_steps if step not in new_text]
                    if missing_steps:
                        self.logger.warning(
                            f"Reasoning structure missing required steps: {missing_steps}. "
                            f"Core framework must be preserved."
                        )
                        continue

                # Special validation for constraints (safety constraint protection)
                if key == "constraints":
                    current_constraints = self.current_template_parts.get("constraints", "")

                    # Check that core safety constraints are preserved
                    required_safety_elements = [
                        "Primary Goal",
                        "NON-NEGOTIABLE SLA",
                        "Response time must never exceed",
                        "Server Limits",
                        "cannot exceed its maximum server count",
                        "Dimmer Range",
                        "must be a float between",
                    ]

                    missing_safety = [
                        elem for elem in required_safety_elements if elem not in new_text
                    ]
                    if missing_safety:
                        self.logger.warning(
                            f"Constraints update missing critical safety elements: {missing_safety}. "
                            f"Safety constraints cannot be removed."
                        )
                        continue

                    # Check that fixed constraint values are still referenced
                    fixed_constraint_refs = [f"{{{const}}}" for const in self.FIXED_CONSTRAINTS]
                    missing_refs = [ref for ref in fixed_constraint_refs if ref not in new_text]
                    if missing_refs:
                        self.logger.warning(
                            f"Constraints update missing fixed constraint references: {missing_refs}. "
                            f"Fixed constraints must remain referenced."
                        )
                        continue

                    # Ensure length doesn't change dramatically (no major rewrites)
                    if len(current_constraints) > 0:
                        length_ratio = len(new_text) / len(current_constraints)
                        if length_ratio < 0.8 or length_ratio > 1.3:
                            self.logger.warning(
                                f"Constraints appears to have major changes "
                                f"(length ratio: {length_ratio:.2f}). Only minor clarifications allowed."
                            )
                            continue

                # Handle observed_learnings for intelligent template evolution
                if key == "observed_learnings":
                    current_phrases = self.current_template_parts.get("observed_learnings", "")

                    # Allow adding/removing ONE short phrase per cycle
                    if isinstance(new_text, str) and len(new_text.strip()) > 0:
                        phrase_count = len([p for p in new_text.split(".") if p.strip()])
                        current_count = (
                            len([p for p in current_phrases.split(".") if p.strip()])
                            if current_phrases
                            else 0
                        )

                        # Allow reasonable changes for observation storage (up to 5 new insights)
                        if phrase_count <= current_count + 5:
                            self.logger.info(
                                f"Storing system observations: {phrase_count} insights total"
                            )
                        else:
                            self.logger.info(
                                f"Large observation update ({phrase_count} vs {current_count}) - allowing for pattern storage"
                            )
                            # Still allow it - observations are valuable

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
