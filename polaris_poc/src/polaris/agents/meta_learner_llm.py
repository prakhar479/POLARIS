"""
LLM-based Meta-Learner Agent Implementation

A simple meta-learner that:
- Queries the Knowledge Base for recent telemetry snapshots and observations
- Analyzes patterns using Gemini API
- Updates prompt.yaml thresholds based on learned insights
- Runs periodically every 5 minutes
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

# Import the base class that handles NATS communication
from .reasoner_agent import NATSReasonerBase


class MetaLearnerLLM(NATSReasonerBase):
    """Simple LLM-based meta-learner for threshold adaptation."""

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        prompt_config_path: str,
        config_path: str,
        nats_url: Optional[str] = None,
        update_interval_seconds: float = 300.0,  # 5 minutes
        model: str = "gemini-2.5-flash",  # Updated model for Gemini
        temperature: float = 0.3,
        max_tokens: int = 1024,
        kb_request_timeout: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize parent NATSReasonerBase
        super().__init__(agent_id, config_path, nats_url, kb_request_timeout, logger)

        self.api_key = api_key
        self.prompt_config_path = Path(prompt_config_path)
        self.update_interval = update_interval_seconds
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.running = False
        self.last_update_time: Optional[float] = None

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
        self.logger.info(f"Update interval: {update_interval_seconds}s")
        self.logger.info(f"Prompt config: {self.prompt_config_path}")
        self.logger.info(f"Loaded {len(self.current_thresholds)} threshold parameters")

    async def perform_reasoning(self, context) -> Any:
        """
        Stub implementation of abstract method from NATSReasonerBase.
        Meta-learner doesn't use the standard reasoning flow.
        """
        self.logger.warning("perform_reasoning called on MetaLearner (not used)")
        return {"error": "MetaLearner does not implement standard reasoning flow"}

    def _load_prompt_config(self) -> Dict[str, Any]:
        """Load current thresholds from prompt.yaml."""
        try:
            with open(self.prompt_config_path, "r") as f:
                config = yaml.safe_load(f)
                full_config = config.get("prompt_config", {})
                # Return both thresholds and template parts for reference
                return {
                    "thresholds": full_config.get("thresholds", {}),
                    "template_parts": full_config.get("template_parts", {}),
                }
        except Exception as e:
            self.logger.error(f"Failed to load prompt config: {e}")
            return {"thresholds": {}, "template_parts": {}}

    def _save_prompt_config(self, new_thresholds: Dict[str, Any]) -> bool:
        """Update thresholds in prompt.yaml."""
        try:
            # Load full config
            with open(self.prompt_config_path, "r") as f:
                config = yaml.safe_load(f)

            # Update thresholds
            if "prompt_config" not in config:
                config["prompt_config"] = {}
            if "thresholds" not in config["prompt_config"]:
                config["prompt_config"]["thresholds"] = {}

            config["prompt_config"]["thresholds"].update(new_thresholds)

            # Save back
            with open(self.prompt_config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Updated prompt config with new thresholds: {new_thresholds}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save prompt config: {e}")
            return False

    async def query_kb_for_meta_learning(
        self, data_types: List[str], limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Query Knowledge Base for recent entries using parent class method."""
        try:
            # Use the parent class's query_knowledge_base method
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
                # Check if this is a snapshot
                tags = entry.get("tags", [])
                summary_text = entry.get("summary", "")

                if "snapshot" in tags or "snapshot" in summary_text.lower():
                    # This is a snapshot observation
                    summary["snapshots"].append(
                        {
                            "timestamp": entry.get("timestamp"),
                            "summary": summary_text,
                            "content": entry.get("content"),
                        }
                    )
                else:
                    # This is another type of observation (aggregated metrics, etc.)
                    summary["other_observations"].append(
                        {
                            "timestamp": entry.get("timestamp"),
                            "summary": summary_text,
                            "content": entry.get("content"),
                        }
                    )

            elif data_type == "adaptation_decision":
                # Extract adaptation decision information
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

            # Aggregate metrics from all entry types
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

        # Add counts for easy reference
        summary["snapshot_count"] = len(summary["snapshots"])
        summary["other_observation_count"] = len(summary["other_observations"])
        summary["adaptation_decision_count"] = len(summary["adaptation_decisions"])

        return summary

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Gemini API to generate threshold updates."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Configure the model to output JSON directly
        generation_config = types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
        )

        try:
            self.logger.info("Calling Gemini API for threshold analysis...")
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

    def _parse_threshold_updates(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON threshold updates from LLM response."""
        try:
            response = llm_response.strip()

            # Extract thinking block if present (for logging)
            thinking = None
            if "<thinking>" in response:
                thinking_start = response.find("<thinking>") + 10
                thinking_end = response.find("</thinking>")
                if thinking_end > thinking_start:
                    thinking = response[thinking_start:thinking_end].strip()
                    self.logger.info(f"LLM Reasoning: {thinking}")

            # Try to find JSON between ```json and ``` or just raw JSON
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
                # Try to find JSON object (look for outermost braces)
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]

            if not json_str:
                self.logger.warning("No JSON found in LLM response")
                return None

            parsed = json.loads(json_str)

            if parsed:
                self.logger.info(f"Parsed threshold updates: {json.dumps(parsed, indent=2)}")
            else:
                self.logger.info("LLM suggests no threshold changes (empty object)")

            return parsed

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from LLM response: {e}")
            self.logger.debug(f"LLM response was: {llm_response}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing threshold updates: {e}")
            return None

    async def run_learning_cycle(self) -> bool:
        """Execute one meta-learning cycle."""
        self.logger.info("Starting meta-learning cycle...")

        try:
            # 1. Query KB for observations (includes snapshots and aggregated metrics)
            observations = await self.query_kb_for_meta_learning(
                data_types=["observation"],
                limit=500,  # Get more to filter
            )

            # 2. Query KB for adaptation decisions
            decisions = await self.query_kb_for_meta_learning(
                data_types=["adaptation_decision"],
                limit=50,
            )

            # 3. Filter observations to separate snapshots from other observations
            snapshot_entries = []
            other_observations = []

            for obs in observations:
                # Check if this is a snapshot entry (has "snapshot" in tags or summary)
                tags = obs.get("tags", [])
                summary = obs.get("summary", "")

                if "snapshot" in tags or "snapshot" in summary.lower():
                    snapshot_entries.append(obs)
                else:
                    other_observations.append(obs)

            # Take only last 100 snapshot entries (most recent)
            snapshot_entries = sorted(
                snapshot_entries, key=lambda x: x.get("timestamp", ""), reverse=True
            )[:100]

            # Combine: snapshots + other observations + decisions
            entries = snapshot_entries + other_observations + decisions

            self.logger.info(
                f"Retrieved {len(snapshot_entries)} snapshots, "
                f"{len(other_observations)} other observations, "
                f"{len(decisions)} adaptation decisions"
            )

            if not entries:
                self.logger.warning("No KB entries found, skipping cycle")
                return False

            # 4. Extract telemetry summary
            telemetry_summary = self._extract_telemetry_summary(entries)

            # 3. Build prompts for LLM
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(telemetry_summary)

            # 4. Call LLM
            self.logger.info("Calling LLM for threshold analysis...")
            llm_response = await self._call_llm(system_prompt, user_prompt)

            # 5. Parse threshold updates
            threshold_updates = self._parse_threshold_updates(llm_response)

            if not threshold_updates:
                self.logger.warning("No valid threshold updates received")
                return False

            # 6. Validate and apply updates
            valid_updates = self._validate_threshold_updates(threshold_updates)

            if valid_updates:
                success = self._save_prompt_config(valid_updates)
                if success:
                    # Reload config to ensure consistency
                    self.prompt_config = self._load_prompt_config()
                    self.current_thresholds = self.prompt_config.get("thresholds", {})
                    self.logger.info(
                        f"Meta-learning cycle completed successfully. Updated {len(valid_updates)} thresholds."
                    )

                    # Publish update notification
                    await self._publish_update_notification(valid_updates)
                    return True
                else:
                    self.logger.error("Failed to save threshold updates")
                    return False
            else:
                self.logger.warning("No valid threshold updates to apply")
                return False

        except Exception as e:
            self.logger.error(f"Meta-learning cycle failed: {e}", exc_info=True)
            return False

    def _build_system_prompt(self) -> str:
        """Build system prompt for meta-learning with domain-specific information."""
        return """You are a meta-learning agent for POLARIS, a self-adaptive web service management system.

**DOMAIN CONTEXT:**
POLARIS controls a web service with the following characteristics:
- **Server Scaling**: The system can dynamically add or remove servers to handle varying load
- **Content Dimmer**: A quality control mechanism (0.0-1.0) that reduces content fidelity to improve response times
  * 1.0 = Full quality (all features enabled, highest user experience)
  * Lower values = Reduced quality (simplified rendering, fewer features, faster responses)
  * Trade-off: User experience vs. performance
- **Performance Metrics**: 
  * Response Time: How quickly the system responds to user requests
  * Utilization: Percentage of server capacity being used
  * Server Count: Number of active servers (impacts cost)
- **System Goals**:
  1. Maintain response times below critical thresholds (primary constraint)
  2. Maximize utilization efficiency (reduce idle capacity)
  3. Maximize user experience (keep dimmer high when possible)
  4. Minimize operational costs (reduce server count when safe)

**YOUR TASK:**
Analyze recent system telemetry, snapshots, and adaptation decisions to learn patterns and 
propose improved threshold values that better balance these competing goals.

**Current Threshold Configuration:**
{current_thresholds}

**Available Threshold Parameters:**
Response Time Controls:
- max_response_time_s/ms: Hard limit for response times (critical constraint)
- dimmer_reduction_threshold_s: When to reduce content quality to improve speed
- target_response_time_ms: Ideal response time target
- acceptable_response_time_variance: Acceptable deviation from target

Dimmer Controls:
- dimmer_max_step: Maximum dimmer adjustment per action (prevents drastic quality changes)

Utilization Controls:
- utilization_scale_up_threshold_percent: When to add servers (high load)
- utilization_scale_down_target_percent: When to remove servers (low load)
- utilization_optimal_range_min/max: Target efficiency range

Server Management:
- server_action_cooldown_seconds: Minimum time between server changes (prevents thrashing)
- min_servers/max_servers: System capacity bounds

**Data You'll Receive:**
1. **System Snapshots**: Recent state observations (utilization, response times, server counts, dimmer values)
2. **Aggregated Observations**: Statistical summaries showing trends over time
3. **Adaptation Decisions**: Controller actions (ADD_SERVER, REMOVE_SERVER, SET_DIMMER) with success/failure

**Analysis Guidelines:**
- Are response times consistently above/below thresholds? (Performance vs. over-provisioning)
- Are adaptation actions effective or failing? (Controller effectiveness)
- Do utilization patterns suggest threshold adjustments? (Efficiency improvements)
- Is the system stable or oscillating? (Stability analysis)
- Are server changes too frequent? (Cooldown tuning)
- Is the dimmer being used effectively? (Quality vs. performance balance)
- Is the system meeting goals? (Multi-objective optimization)

**Output Format:**
Return ONLY a JSON object with threshold fields you want to update, followed by a <thinking> block 
explaining your reasoning in 2-3 concise sentences.

Example:
```json
{{
  "max_response_time_s": 1.2,
  "dimmer_reduction_threshold_s": 0.8,
  "utilization_scale_down_target_percent": 55,
  "server_action_cooldown_seconds": 90,
  "target_response_time_ms": 750
}}
```

<thinking>
Response times are consistently 200ms below threshold, indicating over-provisioning.
Increasing scale-down target and cooldown will reduce unnecessary server churn.
Adjusting dimmer threshold gives more headroom before quality reduction.
</thinking>

**Important Rules:**
- Only include thresholds you want to change
- Values must be within safe, reasonable bounds
- If current thresholds are working well, return empty object: {{}}
- Prioritize stability over aggressive optimization
- Consider interactions between thresholds (e.g., scaling thresholds vs. cooldown)
- Explain your reasoning clearly in the <thinking> block
""".format(
            current_thresholds=json.dumps(self.current_thresholds, indent=2)
        )

    def _build_user_prompt(self, telemetry_summary: Dict[str, Any]) -> str:
        """Build user prompt with telemetry context."""
        # Extract key statistics for easier analysis
        metrics = telemetry_summary.get("metrics", {})
        decisions = telemetry_summary.get("adaptation_decisions", [])

        # Count decision outcomes
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

        # Calculate success rates
        for action_type, counts in decision_summary["by_type"].items():
            if counts["total"] > 0:
                decision_summary["success_rate"][action_type] = counts["success"] / counts["total"]

        prompt = f"""Analyze the following system data and propose threshold adjustments:

**Time Period:**
- Start: {telemetry_summary.get('time_range', {}).get('start', 'N/A')}
- End: {telemetry_summary.get('time_range', {}).get('end', 'N/A')}

**Data Overview:**
- System Snapshots: {telemetry_summary.get('snapshot_count', 0)} observations
- Metric Aggregations: {telemetry_summary.get('other_observation_count', 0)} summaries
- Adaptation Actions: {telemetry_summary.get('adaptation_decision_count', 0)} decisions

**Adaptation Decision Analysis:**
{json.dumps(decision_summary, indent=2)}

**Key Metrics Summary:**
Available metrics: {list(metrics.keys())}
(Each metric contains timestamped values showing trends over the period)

**Complete Detailed Data:**
{json.dumps(telemetry_summary, indent=2)}

**Your Analysis Task:**
1. Examine response time patterns vs. current thresholds
2. Assess utilization trends and scaling behavior
3. Evaluate adaptation decision effectiveness (success rates, timing)
4. Identify system stability issues (oscillations, thrashing)
5. Consider dimmer usage patterns
6. Propose specific threshold adjustments with clear justification

Return the JSON object with your proposed threshold updates and explain your reasoning in the <thinking> block."""

        return prompt

    def _validate_threshold_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate threshold updates are within safe bounds."""
        valid_updates = {}

        # Define validation rules (min, max)
        rules = {
            # Response time thresholds
            "max_response_time_s": (0.5, 5.0),
            "max_response_time_ms": (500, 5000),
            "dimmer_reduction_threshold_s": (0.3, 2.0),
            "target_response_time_ms": (300, 2000),
            "acceptable_response_time_variance": (0.05, 0.3),
            # Dimmer control parameters
            "dimmer_max_step": (0.05, 0.5),
            "dimmer_min_value": (0.0, 0.0),  # Fixed value
            "dimmer_max_value": (1.0, 1.0),  # Fixed value
            # Utilization thresholds
            "utilization_scale_up_threshold_percent": (70, 95),
            "utilization_scale_down_target_percent": (30, 70),
            "utilization_optimal_range_min": (40, 70),
            "utilization_optimal_range_max": (60, 85),
            # Server management parameters
            "server_action_cooldown_seconds": (30, 300),
            "min_servers": (1, 5),
            "max_servers": (3, 50),
        }

        for key, value in updates.items():
            if key in rules:
                min_val, max_val = rules[key]
                if isinstance(value, (int, float)) and min_val <= value <= max_val:
                    valid_updates[key] = value
                    self.logger.info(f"Validated update: {key} = {value}")
                else:
                    self.logger.warning(
                        f"Invalid value for {key}: {value} (must be between {min_val} and {max_val})"
                    )
            else:
                self.logger.warning(f"Unknown threshold key: {key}")

        # Additional cross-validation
        if (
            "utilization_optimal_range_min" in valid_updates
            and "utilization_optimal_range_max" in valid_updates
        ):
            if (
                valid_updates["utilization_optimal_range_min"]
                >= valid_updates["utilization_optimal_range_max"]
            ):
                self.logger.warning(
                    "utilization_optimal_range_min must be less than max, skipping both"
                )
                del valid_updates["utilization_optimal_range_min"]
                del valid_updates["utilization_optimal_range_max"]

        return valid_updates

    async def _publish_update_notification(self, updates: Dict[str, Any]) -> None:
        """Publish threshold update notification to NATS."""
        try:
            notification = {
                "event_type": "threshold_update",
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "updates": updates,
                "prompt_config_path": str(self.prompt_config_path),
            }

            # Use parent class's publish method
            await self.publish("polaris.meta_learner.threshold_update", notification)
            self.logger.info(
                f"Published threshold update notification to polaris.meta_learner.threshold_update"
            )

        except Exception as e:
            self.logger.error(f"Failed to publish update notification: {e}")

    async def run(self, stop_event: asyncio.Event) -> None:
        """Main run loop - execute learning cycles periodically."""
        self.running = True
        self.logger.info("Meta-learner agent started")

        while not stop_event.is_set() and self.running:
            try:
                # Run learning cycle
                await self.run_learning_cycle()

                # Wait for next cycle
                self.logger.info(f"Waiting {self.update_interval}s until next cycle...")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.update_interval)
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    continue  # Timeout reached, run next cycle

            except Exception as e:
                self.logger.error(f"Error in run loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a bit before retrying

        self.running = False
        self.logger.info("Meta-learner agent stopped")

    async def stop(self) -> None:
        """Stop the meta-learner."""
        self.running = False
        self.logger.info("Stopping meta-learner agent...")


def create_meta_learner_agent(
    agent_id: str,
    api_key: str,
    prompt_config_path: str,
    config_path: str,
    nats_url: Optional[str] = None,
    update_interval_seconds: float = 300.0,
    model: str = "gemini-2.0-flash",
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
        logger=logger,
    )
