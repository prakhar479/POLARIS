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
from collections import deque
from datetime import datetime, timezone, timedelta

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
        model: str = "gemini-2.5-flash",
        max_tokens: int = 2048,
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

        # In-memory storage for the action history using a deque
        self.action_history: deque = deque(maxlen=20)  # Store the last 20 actions

        # self.last_action_memory: Optional[Dict[str, Any]] = None
        self.kb_query = kb_query_interface
        self.context_builder = ContextBuilder(self.logger, prompt_config)
        self.last_inference_snapshots: Optional[List[Dict[str, Any]]] = None
        self.last_inference_action: Optional[Dict[str, Any]] = None
        self.last_inference_timestamp: Optional[str] = None

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
            reasoning_steps.append(f"Fetched {len(snapshots)} KB snapshots")
            # === Filter to only snapshots newer than last inference ===
            filtered_snapshots = snapshots

            # === Compute feedback on previous action ===
            if self.last_inference_action and self.last_inference_action.get("action_type") not in [
                "NO_ACTION",
                "NO_OP",
            ]:
                feedback = self._evaluate_action_from_snapshot_shift(
                    self.last_inference_action,
                    self.last_inference_snapshots or [],
                    filtered_snapshots or [],
                )
                if feedback:
                    reasoning_steps.append(f"Evaluated previous action feedback: {feedback}")
                    # Only append feedback if there's an action history
                    if self.action_history:
                        self.action_history[-1]["content"]["feedback"] = feedback
            # === Use filtered snapshots for current reasoning ===
            current_snapshots = filtered_snapshots or snapshots
            action_history_list = list(self.action_history)
            enriched_context_data = self.context_builder.build_context_from_kb_results(
                current_snapshots, action_history_list
            )

            # 3. Prepare prompts
            system_prompt = self._build_system_prompt(context)
            user_prompt = self._build_user_prompt(context, enriched_context_data)
            reasoning_steps.append("Built LLM prompts with enriched context")

            # 4. Call the LLM
            llm_response = await self._call_llm(system_prompt, user_prompt)
            reasoning_steps.append("Received LLM response")
            with open("./llm_responses.txt", "a") as f:
                f.write(
                    f"---\nTimestamp: {datetime.now(timezone.utc).isoformat()}\nSystem Prompt:\n{system_prompt}\nUser Prompt:\n{user_prompt}\nLLM Response:\n{llm_response}\n"
                )

            # 5. Parse and return the result
            action_result = self._parse_llm_response(llm_response)
            reasoning_steps.append("Parsed and validated LLM output")

            self.last_inference_snapshots = snapshots
            self.last_inference_action = action_result
            self.last_inference_timestamp = datetime.now(timezone.utc).isoformat()

            # Append to history if it's a significant action
            if "error" not in action_result and action_result.get("action_type") not in [
                "ALERT",
                "NO_ACTION",
            ]:
                reasoning_steps.append(
                    "Storing generated action in internal memory for future reflection."
                )
                action_to_store = {"content": action_result.copy()}
                action_to_store["content"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                self.action_history.append(action_to_store)
                self.logger.info(
                    f"Successfully stored action '{action_result['action_type']}' in history. History size: {len(self.action_history)}."
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

    def _evaluate_action_from_snapshot_shift(
        self,
        last_action: Dict[str, Any],
        prev_snapshots: List[Dict[str, Any]],
        new_snapshots: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Compare average_response_time and server_utilization between consecutive inference snapshots."""

        if not prev_snapshots or not new_snapshots:
            return {
                "evaluation": "UNKNOWN",
                "details": "Missing snapshot data from one of the runs.",
            }

        def extract_metric(snapshot, metric):
            """Safely extract a metric from nested snapshot structure."""
            try:
                return snapshot["content"]["content"]["current_state"].get(metric)
            except (KeyError, TypeError):
                return None

        def avg_metric(snapshots, metric):
            vals = []
            for s in snapshots:
                val = extract_metric(s, metric)
                if isinstance(val, (int, float)):
                    vals.append(val)
            return sum(vals) / len(vals) if vals else None

            # DEBUG: inspect one example snapshot to confirm structure

        if new_snapshots and "DEBUG_PRINTED" not in self.__dict__:
            self.__dict__["DEBUG_PRINTED"] = True
            example = new_snapshots[-1]["content"]["content"]["current_state"]
            # logging.warning(f"[DEBUG] Example snapshot current_state = {json.dumps(example, indent=2)}")

        # logging.warning(f"[DEBUG] Prev snapshots count={len(prev_snapshots)}, New snapshots count={len(new_snapshots)}")
        # logging.warning(f"[DEBUG] First prev timestamp={prev_snapshots[0]['content']['timestamp']} | First new timestamp={new_snapshots[0]['content']['timestamp']}")
        # logging.warning(f"[DEBUG] Last prev timestamp={prev_snapshots[-1]['content']['timestamp']} | Last new timestamp={new_snapshots[-1]['content']['timestamp']}")

        prev_rt = avg_metric(prev_snapshots, "average_response_time")
        curr_rt = avg_metric(new_snapshots, "average_response_time")
        prev_util = avg_metric(prev_snapshots, "server_utilization")
        curr_util = avg_metric(new_snapshots, "server_utilization")

        self.logger.info(
            f"Evaluating action feedback: Prev RT={prev_rt}, Curr RT={curr_rt}, Prev Util={prev_util}, Curr Util={curr_util}"
        )

        if prev_rt is None or curr_rt is None:
            return {
                "evaluation": "UNKNOWN",
                "details": "Metrics missing in one or both snapshot sets.",
            }

        # Convert RT from seconds to milliseconds for readability
        rt_change_ms = (curr_rt - prev_rt) * 1000
        util_change = (
            curr_util - prev_util if (prev_util is not None and curr_util is not None) else 0.0
        )

        self.logger.info(
            f"Action feedback computed: ΔRT={rt_change_ms:+.2f} ms, ΔUtil={util_change:+.4f}"
        )

        # --- Evaluation rules ---
        eval_result = "NEUTRAL"
        action_type = last_action.get("action_type")

        if action_type == "ADD_SERVER":
            if rt_change_ms < -50:
                eval_result = "SUCCESSFUL"
            elif rt_change_ms > 20:
                eval_result = "FAILED"
        elif action_type == "REMOVE_SERVER":
            if util_change > 0.05 and rt_change_ms < 100:
                eval_result = "SUCCESSFUL"
            elif rt_change_ms > 200:
                eval_result = "FAILED"
        elif action_type == "SET_DIMMER":
            new_val = last_action.get("params", {}).get("value", 1.0)
            if rt_change_ms < -10 and new_val < 1.0:
                eval_result = "SUCCESSFUL"
            elif rt_change_ms > 10:
                eval_result = "FAILED"

        return {
            "action_taken": action_type,
            "evaluation": eval_result,
            "details": f"ΔRT={rt_change_ms:+.2f} ms, ΔUtil={util_change:+.4f}",
        }

    def _build_user_prompt(self, context: ReasoningContext, telemetry_data: Dict[str, Any]) -> str:
        """Build the user prompt with the dynamic context data."""
        return self.user_prompt_template.format(context_data=json.dumps(telemetry_data, indent=2))

    # ======================================================================
    # == MODIFIED RESPONSE PARSER                                         ==
    # ======================================================================
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM JSON output from within the <json_output> block and extract thinking."""
        try:
            thinking_content = "No <thinking> block found."
            if "<thinking>" in response and "</thinking>" in response:
                thinking_start = response.find("<thinking>") + len("<thinking>")
                thinking_end = response.find("</thinking>", thinking_start)
                if thinking_end != -1:
                    thinking_content = response[thinking_start:thinking_end].strip()
            self.logger.info(
                f"LLM Reasoning Process:\n{thinking_content}", extra={"component": "llm_thinking"}
            )

            json_str = ""
            if "<json_output>" in response and "</json_output>" in response:
                json_start = response.find("<json_output>") + len("<json_output>")
                json_end = response.find("</json_output>", json_start)
                if json_end != -1:
                    json_str = response[json_start:json_end].strip()
            else:
                self.logger.warning(
                    "LLM response did not contain <json_output> tags. Falling back to extracting first JSON object."
                )
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start != -1 and json_end != 0:
                    json_str = response[json_start:json_end]

            if not json_str:
                raise ValueError("No valid JSON object found in response.")

            action = json.loads(json_str)

            required_fields = ["action_type", "source"]
            for field in required_fields:
                if field not in action:
                    raise ValueError(f"Missing required field in parsed JSON: {field}")

            action["action_id"] = str(uuid.uuid4())
            action["_thinking_process"] = thinking_content
            return action

        except Exception as e:
            self.logger.error(f"Failed to parse LLM JSON response: {e}. Response was: {response}")
            return {
                "action_type": "ALERT",
                "source": "llm_parser",
                "params": {"message": f"Failed to parse LLM JSON response: {e}"},
                "action_id": f"fallback_{int(time.time())}",
            }

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
        self.set_domain_context("Proactive adaptive control of SWIM exemplar.")
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
        self, snapshots: List[Dict[str, Any]], action_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """The main method to transform raw KB results into the final context payload."""
        if not snapshots:
            return {"error": "No recent snapshots found in the Knowledge Base."}

        sorted_snapshots = sorted(snapshots, key=lambda s: s["timestamp"])
        latest_snapshot = sorted_snapshots[-1]["content"]["content"]
        current_state = latest_snapshot.get("current_state", {})

        historical_trends = {}
        all_metrics = set(
            k
            for s in sorted_snapshots
            for k in s.get("content", {}).get("content", {}).get("current_state", {}).keys()
        )
        for metric in all_metrics:
            values = [
                s["content"]["content"]["current_state"][metric]
                for s in sorted_snapshots
                if metric in s.get("content", {}).get("content", {}).get("current_state", {})
            ]
            ewma = self.calculate_ewma(values)
            roc = self.calculate_rate_of_change(sorted_snapshots, metric)
            historical_trends[metric] = {
                "5min_avg_ewma": round(ewma, 4) if ewma is not None else None,
                "5min_rate_of_change": roc,
            }

        historical_actions_context = self.build_historical_actions_context(
            action_history, sorted_snapshots
        )

        system_goals_and_constraints = {
            "goals": {"target_utilization": 1, "target_response_time_ms_weighted": 900.0},
            "constraints": {"max_servers": 3, "min_servers": 1, "cooldown_period_minutes": 2},
        }

        return {
            "current_state": current_state,
            "historical_trends": historical_trends,
            "historical_actions": historical_actions_context,
            "system_goals_and_constraints": system_goals_and_constraints,
        }

    def build_historical_actions_context(
        self, action_history: List[Dict[str, Any]], sorted_snapshots: List[Dict[str, Any]]
    ):
        """Generates summary view of the last 10 actions and feedback for the most recent ones."""
        if not action_history:
            return {"summary_last_10": "No actions recorded.", "recent_actions_with_feedback": []}

        # Consider only the last 10 actions
        last_10_actions = action_history[-10:] if len(action_history) >= 10 else action_history

        # Initialize counters for known action types
        summary = {"ADD_SERVER": 0, "REMOVE_SERVER": 0, "SET_DIMMER": 0}

        for action_record in last_10_actions:
            action_type = action_record["content"].get("action_type")
            if action_type in summary:
                summary[action_type] += 1

        # Build readable summary text
        summary_text = (
            ", ".join([f"{k}: {v} times" for k, v in summary.items() if v > 0])
            or "No recognized control actions in the last 10."
        )

        # Generate feedback for the most recent 3 actions (or fewer if less available)
        recent_actions_with_feedback = []
        # Take the last 3 actions from the history for detailed feedback display
        for action_record in action_history[-3:]:
            # The feedback is already attached to the content dictionary
            feedback = action_record["content"].get("feedback")

            if feedback:
                recent_actions_with_feedback.append(feedback)
            else:
                # Fallback if feedback is missing for some reason
                recent_actions_with_feedback.append(
                    {
                        "action_taken": action_record["content"].get(
                            "action_type", "UNKNOWN_ACTION"
                        ),
                        "evaluation": "UNKNOWN",
                        "details": "Feedback not calculated for this past action.",
                    }
                )

        return {
            "summary_last_10": summary_text,
            "recent_actions_with_feedback": recent_actions_with_feedback,
            "current_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def generate_feedback_for_action(
        self, action_record: Dict[str, Any], sorted_snapshots: List[Dict[str, Any]]
    ):
        """
        Analyzes snapshots approximately 10 seconds before and after an action
        to evaluate its outcome.
        """
        action_content = action_record["content"]
        action_timestamp = datetime.fromisoformat(action_content["timestamp"])

        # --- REVISED LOGIC (10-second window) ---

        # 1. Define the target timestamps for our comparison window
        target_pre_ts = action_timestamp - timedelta(seconds=10)
        target_post_ts = action_timestamp + timedelta(seconds=10)

        pre_action_snapshot = None
        post_action_snapshot = None

        # 2. Find the snapshot closest to the T-10s mark from all snapshots BEFORE the action
        # First, filter to only include valid candidates
        pre_candidates = [
            s for s in sorted_snapshots if datetime.fromisoformat(s["timestamp"]) < action_timestamp
        ]
        if pre_candidates:
            pre_action_snapshot = min(
                pre_candidates,
                key=lambda s: abs(datetime.fromisoformat(s["timestamp"]) - target_pre_ts),
            )

        # 3. Find the snapshot closest to the T+10s mark from all snapshots ON OR AFTER the action
        # First, filter to only include valid candidates
        post_candidates = [
            s
            for s in sorted_snapshots
            if datetime.fromisoformat(s["timestamp"]) >= action_timestamp
        ]
        if post_candidates:
            post_action_snapshot = min(
                post_candidates,
                key=lambda s: abs(datetime.fromisoformat(s["timestamp"]) - target_post_ts),
            )

        # --- END REVISED LOGIC ---

        # 4. If we couldn't find a valid before/after pair, we cannot evaluate the action.
        if not pre_action_snapshot or not post_action_snapshot:
            return {
                "action_taken": action_content["action_type"],
                "timestamp": action_content["timestamp"],
                "evaluation": "UNKNOWN",
                "outcome_details": "Insufficient data to evaluate outcome (could not find snapshots in the T+/-10s window).",
            }

        # It's possible we found the same snapshot if data is sparse, so check for that.
        if pre_action_snapshot["timestamp"] == post_action_snapshot["timestamp"]:
            return {
                "action_taken": action_content["action_type"],
                "timestamp": action_content["timestamp"],
                "evaluation": "UNKNOWN",
                "outcome_details": "Pre and post action snapshots are identical. Cannot determine outcome.",
            }

        pre_state = pre_action_snapshot["content"]["content"]["current_state"]
        post_state = post_action_snapshot["content"]["content"]["current_state"]

        rt_before = pre_state.get("response_time_ms_weighted", 0)
        rt_after = post_state.get("response_time_ms_weighted", 0)
        rt_change = rt_after - rt_before

        util_before = pre_state.get("utilization", 0)
        util_after = post_state.get("utilization", 0)
        util_change = util_after - util_before

        evaluation = "NEUTRAL"
        action_type = action_content["action_type"]

        # Log the actual timestamps used for evaluation for better debugging
        pre_ts_used = pre_action_snapshot["timestamp"]
        post_ts_used = post_action_snapshot["timestamp"]
        self.logger.info(
            f"Evaluating action {action_type} at {action_content['timestamp']}. "
            f"Comparing snapshot at {pre_ts_used} with {post_ts_used}. "
            f"RT change: {rt_change:.2f}, Util change: {util_change:.2f}"
        )

        # This evaluation logic remains the same
        if action_type == "ADD_SERVER" and rt_change < -50:
            evaluation = "SUCCESSFUL"
        elif action_type == "ADD_SERVER" and rt_change > 20:
            evaluation = "FAILED"
        elif action_type == "REMOVE_SERVER" and util_change > 0.05 and rt_change < 100:
            evaluation = "SUCCESSFUL"
        elif (
            action_type == "SET_DIMMER"
            and action_content.get("params", {}).get("value", 1.0) < pre_state.get("dimmer", 1.0)
            and rt_change < -20
        ):
            evaluation = "SUCCESSFUL"
        elif action_type == "SET_DIMMER" and rt_change > 20:
            evaluation = "FAILED"

        return {
            "action_taken": action_type,
            "params": action_content.get("params", {}),
            "timestamp": action_content["timestamp"],
            "evaluation": evaluation,
            "outcome_details": f"Response time changed by {rt_change:+.2f}ms. Utilization changed by {util_change:+.2f}.",
        }
