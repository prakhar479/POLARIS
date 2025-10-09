"""
LLM-based Reasoner Implementation

Single class that adds LLM-powered reasoning implementation to process
telemetry data and generate action outputs without relying on KB or Digital Twin.
Includes convenient prompt modification capabilities.
"""

import json
import asyncio
import time
from typing import Any, Dict, List, Optional, Union
import logging
from google import genai
import google
from pathlib import Path
import random
import uuid  # Import uuid for generating action_id
import httpx
import yaml

# Add these imports to the existing reasoner_agent.py file
from .reasoner_core import (
    ReasoningInterface,
    ReasoningContext,
    ReasoningResult,
    ReasoningType,
)

from .reasoner_agent import KnowledgeQueryInterface
from .reasoner_agent import ReasonerAgent
from google.genai import types


class LLMReasoningImplementation(ReasoningInterface):
    """LLM-powered reasoning implementation with customizable prompts."""

    def __init__(
        self,
        api_key: str,
        reasoning_type: ReasoningType,
        prompt_config: Dict[str, Any],  # <-- New parameter for the config
        prompt_config_path: Optional[str] = None,  # <-- Add path to reload config
        kb_query_interface: Optional[KnowledgeQueryInterface] = None,
        model: str = "gpt-oss:20b",
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 600.0,
        base_url: str = "http://10.10.16.46:11435",
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = api_key
        self.reasoning_type = reasoning_type
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = 5
        self.timeout = timeout
        self.base_url = base_url
        self.prompt_config_path = prompt_config_path  # Store for reloading
        self.last_run_time: Optional[float] = None
        self.initial_backoff = 2.0
        self.logger = logger or logging.getLogger(f"LLMReasoner.{reasoning_type.value}")
        self.logger.info(
            f"Initializing LLMReasoningImplementation with model {model} at endpoint {self.base_url}"
        )

        self.last_action_memory: Optional[Dict[str, Any]] = None
        self.kb_query = kb_query_interface
        self.context_builder = ContextBuilder(self.logger, prompt_config)

        # Load prompt template and thresholds from the config dictionary
        try:
            self.prompt_thresholds = prompt_config["thresholds"]
            self.prompt_template_parts = prompt_config.get("template_parts", {})
            self.prompt_examples = prompt_config.get("examples", "")
            self.raw_prompt_template = prompt_config["template"]

            # Combine thresholds, template parts, and examples for formatting
            self.prompt_format_vars = {
                **self.prompt_thresholds,
                **self.prompt_template_parts,
                "examples": self.prompt_examples,
            }

            self.logger.info(
                "Successfully loaded prompt template, thresholds, template parts, and examples from config."
            )
        except KeyError as e:
            self.logger.error(f"Prompt configuration is missing required key: {e}")
            raise ValueError(f"Invalid prompt_config dictionary: missing {e}") from e

        # The user prompt remains static for now
        self.user_prompt_template = """
---
**Now, perform your analysis on the following data:**

{context_data}
"""
        # Customizable fields can still be used if needed
        self.custom_instructions = ""
        self.domain_context = ""
        self.safety_constraints = []
        self.call_history: List[Dict[str, Any]] = []

    def reload_prompt_config(self) -> bool:
        """Reload prompt configuration from file."""
        if not self.prompt_config_path:
            self.logger.warning("No prompt_config_path set, cannot reload config")
            return False

        try:
            self.logger.info(f"Reloading prompt config from {self.prompt_config_path}")
            with open(self.prompt_config_path, "r") as f:
                config = yaml.safe_load(f)
                prompt_config = config["prompt_config"]

            old_thresholds = self.prompt_thresholds.copy()
            old_template_parts = self.prompt_template_parts.copy()

            self.prompt_thresholds = prompt_config["thresholds"]
            self.prompt_template_parts = prompt_config.get("template_parts", {})
            self.prompt_examples = prompt_config.get("examples", "")
            self.raw_prompt_template = prompt_config["template"]

            # Combine thresholds, template parts, and examples for formatting
            self.prompt_format_vars = {
                **self.prompt_thresholds,
                **self.prompt_template_parts,
                "examples": self.prompt_examples,
            }

            self.logger.info(f"✅ Prompt config reloaded successfully")
            self.logger.info(f"   Old thresholds: {old_thresholds}")
            self.logger.info(f"   New thresholds: {self.prompt_thresholds}")
            self.logger.info(
                f"   Template parts updated: {list(self.prompt_template_parts.keys())}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to reload prompt config: {e}", exc_info=True)
            return False

    def _build_system_prompt(self, context: ReasoningContext) -> str:
        """Build the system prompt by formatting the template with configured thresholds and template parts."""
        try:
            # Use the stored template and combined format variables to create the final prompt
            return self.raw_prompt_template.format(**self.prompt_format_vars)
        except KeyError as e:
            self.logger.error(f"Failed to format prompt template. Missing key: {e}")
            self.logger.error(f"Available keys: {list(self.prompt_format_vars.keys())}")
            # Fallback to the raw template to avoid a hard crash
            return self.raw_prompt_template

    async def reason(
        self, context: ReasoningContext, knowledge: Optional[List[Dict[str, Any]]] = None
    ) -> ReasoningResult:
        """Execute LLM-based reasoning on the telemetry data."""
        start_time = time.time()
        reasoning_steps = ["Starting single-LLM agentic reasoning"]
        now = time.time()
        if self.last_run_time and (now - self.last_run_time) < 60:
            self.logger.info("Skipping reasoning : cooldown not reached (2 mins)")
            return ReasoningResult(
                result={"action_type": "NO_ACTION", "source": "rate_limiter"},
                confidence=1.0,
                reasoning_steps=["Rate-limited: too soon since last run"],
                context=context,
                execution_time=0.01,
            )

        self.last_run_time = now
        try:
            # 1. Query KB for necessary data
            reasoning_steps.append("Querying KB for recent monitor snapshots.")
            snapshots = await self.kb_query.query_structured(
                data_types=["observation"],
                filters={"source": "swim_snapshotter", "tags": ["snapshot"]},
                limit=120,
            )
            reasoning_steps.append("Retrieving last action from internal memory to critique.")
            last_action = self.last_action_memory

            # 2. Build the rich context
            reasoning_steps.append("Processing KB results with ContextBuilder.")
            enriched_context_data = self.context_builder.build_context_from_kb_results(
                snapshots or [], last_action
            )

            # 3. Prepare prompts
            system_prompt = self._build_system_prompt(context)
            user_prompt = self._build_user_prompt(context, enriched_context_data)
            reasoning_steps.append("Built LLM prompts with enriched context")

            # 4. Call the LLM
            llm_response = await self._call_llm(system_prompt, user_prompt)
            reasoning_steps.append("Received LLM response")

            # 5. Parse and return the result
            action_result = self._parse_llm_response(llm_response)
            reasoning_steps.append("Parsed and validated LLM output")

            if "error" not in action_result and action_result.get("action_type") != "ALERT":
                reasoning_steps.append(
                    "Storing generated action in internal memory for future reflection."
                )
                action_to_store = {"content": action_result.copy()}
                action_to_store["content"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                self.last_action_memory = action_to_store
                self.logger.info(
                    f"Successfully stored action '{action_result['action_type']}' in internal memory."
                )

            self._store_call_history(
                context, system_prompt, user_prompt, llm_response, action_result
            )

            execution_time = time.time() - start_time
            return ReasoningResult(
                result=action_result,
                confidence=action_result.get("confidence", 0.9),
                reasoning_steps=reasoning_steps,
                context=context,
                execution_time=execution_time,
                kb_queries_made=1,
            )

        except Exception as e:
            self.logger.error(f"LLM reasoning failed: {e}", exc_info=True)
            execution_time = time.time() - start_time

            return ReasoningResult(
                result={
                    "error": str(e),
                    "action_type": "ALERT",
                    "target": "system",
                    "params": {"message": f"LLM reasoning failed: {e}"},
                    "reasoning": "Fallback due to LLM failure",
                },
                confidence=0.1,
                reasoning_steps=reasoning_steps + [f"Error: {str(e)}"],
                context=context,
                execution_time=execution_time,
            )

    def _build_user_prompt(self, context: ReasoningContext, telemetry_data: Dict[str, Any]) -> str:
        """Build the user prompt with the dynamic context data."""
        return self.user_prompt_template.format(context_data=json.dumps(telemetry_data, indent=2))

    # ======================================================================
    # == MODIFIED RESPONSE PARSER                                         ==
    # ======================================================================
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM JSON output from within the <json_output> block and extract thinking."""
        try:
            # Extract and log the thinking process first
            thinking_content = None
            if "<thinking>" in response and "</thinking>" in response:
                thinking_start = response.find("<thinking>") + len("<thinking>")
                thinking_end = response.find("</thinking>", thinking_start)
                if thinking_end != -1:
                    thinking_content = response[thinking_start:thinking_end].strip()
                    self.logger.info(
                        "LLM Reasoning Process:\n" + thinking_content,
                        extra={"component": "llm_thinking"},
                    )

                # # Look for the <json_output> block
                # if "<json_output>" in response:
                #     json_start = response.find("<json_output>") + len("<json_output>")
                #     json_end = response.find("</json_output>", json_start)
                #     if json_end == -1:
                #         raise ValueError("Found <json_output> but no closing </json_output> tag.")
                #     json_str = response[json_start:json_end].strip()
                # else:
                # Fallback for cases where the model might forget the tags
            self.logger.warning(
                "LLM response did not contain <json_output> tags. Falling back to extracting first JSON object."
            )
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")
            json_str = response[json_start:json_end]

            action = json.loads(json_str)

            # Validate required fields
            required_fields = ["action_type", "source", "params"]
            for field in required_fields:
                if field not in action:
                    raise ValueError(f"Missing required field in parsed JSON: {field}")

            # Overwrite action_id with a real UUID for system use
            action["action_id"] = str(uuid.uuid4())

            # Add the thinking content to the action result for debugging/logging
            # if thinking_content:
            #     action["_thinking_process"] = thinking_content

            return action

        except Exception as e:
            self.logger.error(f"Failed to parse LLM JSON response: {e}. Response was: {response}")
            return {
                "action_type": "ALERT",
                "source": "llm_parser",
                "params": {"message": f"Failed to parse LLM JSON response: {e}"},
                "action_id": f"fallback_{int(time.time())}",
            }

    # ==== Unchanged Private & Public Methods ====
    # The following methods (_call_llm, _store_call_history, create_llm_reasoner_agent, etc.)
    # and classes (ContextBuilder) remain the same as in your original file.
    # They are included here for completeness of the single file.

    # ==== Private Methods ====

    def _extract_telemetry_data(self, context: ReasoningContext) -> Dict[str, Any]:
        """
        Extract and format telemetry data from context, with specific handling
        for the SWIM 'events' list structure.
        """
        telemetry = {}

        # --- NEW LOGIC TO PARSE THE 'events' LIST ---
        # Check if 'events' key exists and is a list in the input data
        if "events" in context.input_data and isinstance(context.input_data["events"], list):
            # Iterate through each event dictionary in the list
            for event in context.input_data["events"]:
                # Check if the event has a 'name' and 'value'
                if "name" in event and "value" in event:
                    # Add the metric to our telemetry dictionary.
                    # If a metric appears multiple times, this will keep the latest one.
                    telemetry[event["name"]] = event["value"]

        # --- Keep old logic as a fallback for other data formats ---
        # Direct telemetry data
        if "telemetry" in context.input_data:
            telemetry.update(context.input_data["telemetry"])

        # Any other top-level numerical measurements
        for key, value in context.input_data.items():
            if key not in telemetry and isinstance(value, (int, float)):
                telemetry[key] = value

        return telemetry

    async def validate_input(self, context: ReasoningContext) -> bool:
        return True

    def get_required_knowledge_types(self, context: ReasoningContext) -> List[str]:
        return []

    def extract_search_terms(self, context: ReasoningContext) -> List[str]:
        return []

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Calls the local LLM API endpoint."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        endpoint_url = f"{self.base_url}/api/generate"

        # Payload structure for Ollama-like APIs
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "max_tokens": self.max_tokens,
        }

        headers = {"Content-Type": "application/json"}

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    self.logger.info(f"Sending request to local LLM at {endpoint_url}")
                    response = await client.post(endpoint_url, json=payload, headers=headers)
                    response.raise_for_status()  # Raise an exception for 4xx/5xx errors

                    self.logger.debug(f"Raw response: {response.text[:500]}")

                    response_data = response.json()
                    response_text = response_data.get("response", "").strip()

                    if not response_text:
                        raise ValueError("LLM response JSON was empty or malformed.")

                    self.logger.info(f"LLM response received on attempt {attempt + 1}")
                    return response_text

            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
                ValueError,
            ) as e:
                self.logger.warning(
                    f"Local LLM API call failed with error: {e}. Attempt {attempt + 1}/{self.max_retries}."
                )
                if attempt + 1 == self.max_retries:
                    self.logger.error("Max retries reached. Failing the LLM operation.")
                    raise

                backoff_time = self.initial_backoff * (2**attempt)
                jitter = random.uniform(0, backoff_time * 0.1)
                delay = backoff_time + jitter
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        raise Exception("LLM call failed after all retries.")

    def _store_call_history(
        self,
        context: ReasoningContext,
        system_prompt: str,
        user_prompt: str,
        llm_response: str,
        parsed_result: Dict[str, Any],
    ):
        call_record = {
            "timestamp": context.timestamp,
            "session_id": context.session_id,
            "reasoning_type": self.reasoning_type.value,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_response": llm_response,
            "parsed_result": parsed_result,
            "model": self.model,
        }
        self.call_history.append(call_record)
        if len(self.call_history) > 100:
            self.call_history.pop(0)

    # ==== Prompt Customization Methods ====

    def set_system_prompt(self, template: str):
        """Update the system prompt template."""
        self.system_prompt_template = template
        return self

    def set_user_prompt(self, template: str):
        """Update the user prompt template."""
        self.user_prompt_template = template
        return self

    def add_custom_instructions(self, instructions: str):
        """Add custom instructions to the system prompt."""
        self.custom_instructions = instructions
        return self

    def set_domain_context(self, context: str):
        """Set domain-specific context."""
        self.domain_context = context
        return self

    def add_safety_constraint(self, constraint: str):
        """Add a safety constraint."""
        self.safety_constraints.append(constraint)
        return self

    def set_safety_constraints(self, constraints: List[str]):
        """Replace all safety constraints."""
        self.safety_constraints = constraints
        return self

    def configure_basic(self):
        """Generic safe defaults."""
        self.set_domain_context(
            """Proactive adaptive control of SWIM exemplar.
                                
                                About SWIM: The system manages servers and content delivery.

                                Actions: Add/remove servers, adjust a dimmer (0.0–1.0 for optional content), or take no action.

                                Metrics: Track dimmer level, active/max servers, utilization, response times (overall, basic, optional), arrival rate, and throughput (basic/optional).

                                Behavior: High utilization → exponential rise in response time; adding servers lowers per-server load but raises cost; lowering dimmer reduces load/response time.

                                Thresholds: Critical utilization = 1.0, normal response time = 1.0s, recommended servers ≈ arrival_rate/10.
                                
                                """
        )
        self.set_safety_constraints(
            [
                "Only use: ADD_SERVER, REMOVE_SERVER, SET_DIMMER",
                "Dimmer values must be between 0.0 and 1.0",
                "Validate parameter ranges",
                "Prefer minimal, reversible actions",
            ]
        )
        self.temperature = 0.3
        return self


def create_llm_reasoner_agent(
    agent_id: str,
    config_path: str,
    llm_api_key: str,
    llm_config_path: str,  # <-- Add path to the new llm_config.yaml
    nats_url: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    mode="llm",
) -> "ReasonerAgent":
    """
    Create a reasoner agent with LLM-based reasoning implementations.
    """
    from .reasoner_agent import ReasonerAgent, ReasoningType
    from .llm_reasoner import LLMReasoningImplementation
    from .multi_agent_reasoner import MultiAgentReasoner

    # Load the new LLM prompt configuration from the YAML file
    try:
        with open(llm_config_path, "r") as f:
            llm_config = yaml.safe_load(f)
            prompt_config = llm_config["prompt_config"]
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        if logger:
            logger.error(f"Failed to load or parse LLM config from {llm_config_path}: {e}")
        raise ValueError(f"Could not load LLM configuration from {llm_config_path}") from e

    agent = ReasonerAgent(
        agent_id=agent_id,
        reasoning_implementations={},
        config_path=config_path,
        nats_url=nats_url,
        logger=logger,
    )

    for reasoning_type in ReasoningType:
        if mode == "multi":
            llm_reasoner = MultiAgentReasoner(
                api_key=llm_api_key,
                reasoning_type=reasoning_type,
                kb_query_interface=agent.kb_query,
                logger=logger,
            )
            # Note: You might need a similar config system for the MultiAgentReasoner
        else:
            llm_reasoner = LLMReasoningImplementation(
                api_key=llm_api_key,
                reasoning_type=reasoning_type,
                prompt_config=prompt_config,  # <-- Pass the loaded config here
                prompt_config_path=llm_config_path,  # <-- Pass path for reloading
                kb_query_interface=agent.kb_query,
                logger=logger,
            )

        llm_reasoner.configure_basic()
        agent.add_reasoning_implementation(reasoning_type, llm_reasoner)

    return agent


from datetime import datetime, timezone


class ContextBuilder:
    """
    Processes lists of KB entries (snapshots, actions) to build the
    rich context payload required by the LLM.
    """

    def __init__(self, logger, prompt_config: Optional[Dict[str, Any]] = None) -> None:
        self.logger = logger or logging.getLogger("ContextBuilder")
        self.prompt_config = prompt_config or {}

    def calculate_ewma(self, data: List[float], alpha: float = 0.5) -> Optional[float]:
        if not data:
            return None
        ewma = data[0]
        for value in data[1:]:
            ewma = alpha * value + (1 - alpha) * ewma
        return ewma

    def calculate_rate_of_change(
        self, sorted_snapshots: List[Dict[str, Any]], metric: str, time_window_seconds: int = 300
    ) -> Optional[str]:
        if len(sorted_snapshots) < 2:
            return None

        # We need to extract the value and timestamp for the given metric
        # from each snapshot's 'content' field.
        time_series = []
        for s in sorted_snapshots:
            if metric in s["content"]["content"]["current_state"]:
                time_series.append(
                    {
                        "timestamp": s["timestamp"],
                        "value": s["content"]["content"]["current_state"][metric],
                    }
                )

        if len(time_series) < 2:
            return None

        latest_point = time_series[-1]
        # self.logger.info(f"Calculating rate of change for {metric} using latest point: {latest_point}")
        past_timestamp_limit = (
            datetime.fromisoformat(latest_point["timestamp"]).timestamp() - time_window_seconds
        )

        reference_point = time_series[0]
        for point in time_series:
            if datetime.fromisoformat(point["timestamp"]).timestamp() >= past_timestamp_limit:
                reference_point = point
                break

        time_diff_seconds = (
            datetime.fromisoformat(latest_point["timestamp"]).timestamp()
            - datetime.fromisoformat(reference_point["timestamp"]).timestamp()
        )
        if time_diff_seconds < 1:
            return "+0.00/min"

        value_diff = latest_point["value"] - reference_point["value"]
        rate_per_minute = (value_diff / time_diff_seconds) * 60
        return f"{rate_per_minute:+.2f}/min"

    def build_context_from_kb_results(
        self, snapshots: List[Dict[str, Any]], last_action: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        The main method to transform raw KB results into the final context payload.
        """
        if not snapshots:
            return {"error": "No recent snapshots found in the Knowledge Base."}

        # Sort snapshots by timestamp just in case they are out of order
        sorted_snapshots = sorted(snapshots, key=lambda s: s["timestamp"])
        latest_snapshot = sorted_snapshots[-1]["content"]["content"]

        # 1. Get the Current State from the latest snapshot
        current_state = latest_snapshot.get("current_state", {})

        # 2. Calculate Historical Trends across all snapshots
        historical_trends = {}

        # Identify all unique metrics across the snapshots to calculate trends for

        all_metrics = set(
            k
            for s in sorted_snapshots
            for k in s.get("content", {}).get("content", {}).get("current_state", {}).keys()
        )

        for metric in all_metrics:
            # Create a time series of values for the current metric
            values = [
                s["content"]["content"]["current_state"][metric]
                for s in sorted_snapshots
                if "content" in s
                and "content" in s["content"]
                and metric in s["content"]["content"].get("current_state", {})
            ]

            ewma = self.calculate_ewma(values)
            roc = self.calculate_rate_of_change(sorted_snapshots, metric)

            historical_trends[metric] = {
                "5min_avg_ewma": round(ewma, 4) if ewma is not None else None,
                "5min_rate_of_change": roc,
            }

        # 3. Process the Controller State from the last action
        controller_state = {}
        if last_action:
            action_content = last_action["content"]
            action_time = datetime.fromisoformat(action_content["timestamp"])
            minutes_since = (datetime.now(timezone.utc) - action_time).total_seconds() / 60
            controller_state = {
                "last_action": {
                    "timestamp": action_content["timestamp"],
                    "action_type": action_content["action_type"],
                },
                "minutes_since_last_action": round(minutes_since, 2),
            }
        else:
            controller_state = {"last_action": None, "minutes_since_last_action": 9999}

        # NEW: Feedback loop logic
        feedback_on_last_action = {"evaluation": "No recent action to evaluate."}
        if last_action and len(sorted_snapshots) > 1:
            action_time_ts = datetime.fromisoformat(last_action["content"]["timestamp"]).timestamp()
            pre_action_snapshot = next(
                (
                    s
                    for s in reversed(sorted_snapshots)
                    if datetime.fromisoformat(s["timestamp"]).timestamp() < action_time_ts
                ),
                None,
            )

            if pre_action_snapshot:
                pre_state = pre_action_snapshot["content"]["content"]["current_state"]

                rt_before = pre_state.get("average_response_time", 0)
                rt_after = current_state.get("average_response_time", 0)
                # convert to ms
                rt_before = rt_before * 1000
                rt_after = rt_after * 1000
                rt_change = rt_after - rt_before

                # Define simple evaluation criteria
                evaluation = "NEUTRAL"
                rt_improvement_threshold = -10.0  # Must decrease by at least 10ms
                if rt_change < rt_improvement_threshold:
                    evaluation = "SUCCESSFUL"
                elif rt_change > abs(rt_improvement_threshold):
                    evaluation = "FAILED"

                feedback_on_last_action = {
                    "action_taken": last_action["content"]["action_type"],
                    "params": last_action["content"].get("params", {}),
                    "evaluation": evaluation,
                    "outcome_details": f"Response time changed by {rt_change:+.2f}ms (from {rt_before:.2f} to {rt_after:.2f}).",
                }
                self.logger.info(f"Generated feedback for last action: {feedback_on_last_action}")

        # 4. Define Goals and Constraints from prompt configuration
        fixed_constraints = self.prompt_config.get("fixed_constraints", {})
        thresholds = self.prompt_config.get("thresholds", {})

        system_goals_and_constraints = {
            "goals": {
                "target_response_time_ms_weighted": thresholds.get("target_response_time_ms", 742),
                "target_utilization": thresholds.get("target_server_utilization", 0.7),
            },
            "constraints": {
                "max_servers": fixed_constraints.get("max_servers", 3),
                "min_servers": fixed_constraints.get("min_servers", 1),
                "cooldown_period_minutes": thresholds.get("cooldown_period_minutes", 2),
            },
        }

        return {
            "current_state": current_state,
            "historical_trends": historical_trends,
            "controller_state": controller_state,
            "system_goals_and_constraints": system_goals_and_constraints,
        }
