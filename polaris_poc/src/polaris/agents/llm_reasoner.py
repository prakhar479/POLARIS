"""
LLM-based Reasoner Implementation

Single class that adds LLM-powered reasoning implementation to process
telemetry data and generate action outputs without relying on KB or Digital Twin.
Includes convenient prompt modification capabilities and advanced historical context management.
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
import uuid
import httpx
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
    """LLM-powered reasoning implementation with customizable prompts and action history."""

    def __init__(
        self,
        api_key: str,
        reasoning_type: ReasoningType,
        kb_query_interface: Optional[KnowledgeQueryInterface] = None,
        model: str = "gpt-oss:20b",  # Default model for your local API
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout: float = 600.0,
        base_url: str = "http://10.10.16.46:11435",  # Make the URL configurable
        logger: Optional[logging.Logger] = None,
    ):

        self.api_key = api_key
        self.reasoning_type = reasoning_type
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = 5
        self.timeout = timeout
        self.base_url = base_url  # Store the base URL
        self.last_run_time: Optional[float] = None
        self.initial_backoff = 2.0  # seconds
        self.logger = logger or logging.getLogger(f"LLMReasoner.{reasoning_type.value}")
        self.logger.info(
            f"Initializing LLMReasoningImplementation with model {model} at endpoint {self.base_url}"
        )

        # In-memory storage for the action history using a deque
        self.action_history: deque = deque(maxlen=20) # Store the last 20 actions

        self.kb_query = kb_query_interface
        self.context_builder = ContextBuilder(self.logger)
        
        
        self.last_inference_snapshots: Optional[List[Dict[str, Any]]] = None
        self.last_inference_action: Optional[Dict[str, Any]] = None
        self.last_inference_timestamp: Optional[str] = None

        # ======================================================================
        # == NEW PROMPT TEMPLATE WITH HISTORICAL CONTEXT AWARENESS            ==
        # ======================================================================

        self.system_prompt_template = """
You are an integrated, proactive autonomous system controller for a web service. Your objective is to maintain optimal performance by analyzing telemetry data and executing control actions. You have to predict and prevent the future state of the system based on current and historical data and take actions accordingly.

You MUST follow this structured, sequential reasoning process, embodying a different expert persona at each step.

**System Context & Rules:**
-  **Overarching Goal:** Maximize cumulative utility over the long term. This means balancing three factors: 1) Strictly meeting the response time SLA, 2) Maximizing system utilization to be cost-effective, and 3) Maximizing user experience (dimmer value).
- **Primary Goal (NON-NEGOTIABLE SLA):** Response time must never exceed 1s (1000ms). Treat sustained >900ms and rising as an impending breach and act preemptively. First try a modest dimmer reduction (step <= 0.15). If dimmer is already low (<0.7) or utilization is high, add a server instead. Only when stable well below 900ms may you cautiously raise dimmer or remove capacity. Optimize utilization & user experience while minimizing servers—never at the expense of avoiding a breach.
- **Server Limits:** The system cannot exceed its maximum server count or go below 1 server. Goal is to minimize server count while meeting performance targets. Server count should be low unless there is an impending or current breach.
- **Dimmer Range:** The dimmer value must be a float between 0.0 and 1.0. Changes should be gradual (less than 0.2 step) and not frequent.
- **Action Cooldown:** Avoid adding/removing servers if a server action was taken recently.

**Reasoning Steps (Chain of Thought):**

1.  **Data Observer:**
    - **Task:** Concisely summarize the current system state from the input data and predict its near-future trajectory.
    - **Check:** Does the state violate any thresholds or predict a violation soon? Is utility really low (<0.7?) (e.g., "Response time is 2.5s, exceeding the 1.0s target", "Rate of change is +50ms/min and climbing and currently at 850ms, likely to breach soon.","Utility is at 0.65, which is quite low. I should consider adding a server if response time is stable otherwise I will adjust dimmer.")

2.  **Trend Analyst:**
    - **Task:** Analyze the `historical_trends` data.
    - **Check:** Are key metrics like `utilization` or `response_time_ms_weighted` trending in a problematic direction?

3.  **Performance Reviewer:**
    - **Task:** Critically evaluate the effectiveness of your most recent actions using the `historical_actions` data.
    - **Check:** Did your recent actions succeed or fail? Are you repeating failed actions? Does the summary show a pattern (e.g., constantly adding servers)? (e.g., "My last action to `ADD_SERVER` was `SUCCESSFUL`, reducing response time. However, the summary shows I've done this 3 times in an hour, indicating a sustained load increase.")

4.  **Strategy Planner:**
    - **Task:** Based on the observation, trend, and your past performance, devise a plan. Your priorities are: 1) prevent or fix any current or impending response time breach ( >1000ms or >900ms and rising ), 2) increase utilization while lowering/maintaining response time, 3) maximize user experience (dimmer) without risking a future breach.
    - **Action:** Propose one action from [`add_server`, `remove_server`, `set_dimmer [value]`, `no_action`] with a brief justification that refers to past performance.

5.  **Action Sanity Check:**
    - **Task:** Validate the proposed action against all System Context & Rules using the provided data.
    - **Check:** If the action is invalid (e.g., adding a server at max capacity), propose a valid alternative.

6.  **Command Generator:**
    - **Task:** Convert the final, validated action into a complete **JSON object**.
    - **Output Rules:** The JSON must strictly adhere to the following structure:
        - `action_type`: (String) Must be one of `"ADD_SERVER"`, `"REMOVE_SERVER"`, `"SET_DIMMER"`, or `"NO_ACTION"`.
        - `source`: (String) Must be `"single_llm_controller"`.
        - `action_id`: (String) Provide a placeholder string like `"generated-uuid"`.
        - `params`: (Object) Varies by `action_type`. E.g., `{"value": 0.8}` for `SET_DIMMER`.
        - `priority`: (String) Choose from `"low"`, `"medium"`, `"high"`.

**Output Format:**
First, provide your entire step-by-step reasoning within a `<thinking>` block. Keep it in one short sentence per agent.
After the thinking block, provide ONLY the final **JSON object** in a `<json_output>` block.

---
**Example 1 (Overloaded System):**
**Input Data:**
{
    "current_state": { "utilization": 0.95, "response_time_ms_weighted": 2100, "servers": 2, "dimmer": 1.0 },
    "historical_actions": { "summary_last_hour": "SET_DIMMER: 2 times", "recent_actions_with_feedback": [{ "action_taken": "SET_DIMMER", "evaluation": "FAILED", "outcome_details": "Response time continued to climb."}]},
    "system_goals_and_constraints": { "constraints": { "max_servers": 3 }}
}

**Your Output:**
<thinking>
1.  **Data Observer:** The system is overloaded. Utilization is 0.95 and response time is 2100ms (exceeds 1000ms target).
2.  **Trend Analyst:** Data indicates metrics are worsening, suggesting increasing load.
3.  **Performance Reviewer:** My last action to `SET_DIMMER` failed to control the rising response time, so I must try a different strategy.
4.  **Strategy Planner:** The priority is to reduce response time immediately. Since adjusting the dimmer failed, the best action is to increase capacity by adding a server.
5.  **Action Sanity Check:** The action `add_server` is valid. Current server count is 2, which is below the max of 3.
6.  **Command Generator:** The command is to add a server. I will now generate the JSON object with high priority.
</thinking>
<json_output>
{
    "action_type": "ADD_SERVER",
    "source": "single_llm_controller",
    "action_id": "generated-uuid-1",
    "params": {"server_type": "compute", "count": 1},
    "priority": "high"
}
</json_output>
"""
        self.user_prompt_template = """
---
**Now, perform your analysis on the following data:**

{context_data}

"""
        # Customizable fields
        self.custom_instructions = ""
        self.domain_context = ""
        self.safety_constraints = []
        self.call_history: List[Dict[str, Any]] = []

    async def reason(
        self, context: ReasoningContext, knowledge: Optional[List[Dict[str, Any]]] = None
    ) -> ReasoningResult:
        """Execute LLM-based reasoning on the telemetry data."""
        start_time = time.time()
        reasoning_steps = ["Starting single-LLM agentic reasoning"]
        now = time.time()
        if self.last_run_time and (now - self.last_run_time) < -1: # Cooldown set to 2 minutes
            self.logger.info("Skipping reasoning: cooldown not reached (2 mins)")
            return ReasoningResult(
                result={"action_type": "NO_ACTION", "source": "rate_limiter"},
                confidence=1.0,
                reasoning_steps=["Rate-limited: too soon since last run"],
                context=context,
                execution_time=0.01,
            )
            
        # self.logger.info(f"Starting LLM reasoning for context: {context}")

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
            if self.last_inference_action and self.last_inference_action.get("action_type") not in ["NO_ACTION", "NO_OP"]:
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


            # Store response for analysis
            with open("./llm_responses.txt", "a") as f:
                f.write(f"---\nTimestamp: {datetime.now(timezone.utc).isoformat()}\nSystem Prompt:\n{system_prompt}\nUser Prompt:\n{user_prompt}\nLLM Response:\n{llm_response}\n")

            # 5. Parse and return the result
            action_result = self._parse_llm_response(llm_response)
            reasoning_steps.append("Parsed and validated LLM output")


            self.last_inference_snapshots = snapshots
            self.last_inference_action = action_result
            self.last_inference_timestamp = datetime.now(timezone.utc).isoformat()

            # Append to history if it's a significant action
            if "error" not in action_result and action_result.get("action_type") not in ["ALERT", "NO_ACTION"]:
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
            return {"evaluation": "UNKNOWN", "details": "Missing snapshot data from one of the runs."}

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
        
        self.logger.info(f"Evaluating action feedback: Prev RT={prev_rt}, Curr RT={curr_rt}, Prev Util={prev_util}, Curr Util={curr_util}")

        if prev_rt is None or curr_rt is None:
            return {"evaluation": "UNKNOWN", "details": "Metrics missing in one or both snapshot sets."}

        # Convert RT from seconds to milliseconds for readability
        rt_change_ms = (curr_rt - prev_rt) * 1000
        util_change = curr_util - prev_util if (prev_util is not None and curr_util is not None) else 0.0

        self.logger.info(f"Action feedback computed: ΔRT={rt_change_ms:+.2f} ms, ΔUtil={util_change:+.4f}")
        
        # --- Evaluation rules ---
        eval_result = "NEUTRAL"
        action_type = last_action.get("action_type")

        if action_type == "ADD_SERVER":
            if rt_change_ms < -50: eval_result = "SUCCESSFUL"
            elif rt_change_ms > 20: eval_result = "FAILED"
        elif action_type == "REMOVE_SERVER":
            if util_change > 0.05 and rt_change_ms < 100: eval_result = "SUCCESSFUL"
            elif rt_change_ms > 200: eval_result = "FAILED"
        elif action_type == "SET_DIMMER":
            new_val = last_action.get("params", {}).get("value", 1.0)
            if rt_change_ms < -10 and new_val < 1.0: eval_result = "SUCCESSFUL"
            elif rt_change_ms > 10: eval_result = "FAILED"

        return {
            "action_taken": action_type,
            "evaluation": eval_result,
            "details": f"ΔRT={rt_change_ms:+.2f} ms, ΔUtil={util_change:+.4f}",
        }


    def _build_system_prompt(self, context: ReasoningContext) -> str:
        """Build the system prompt."""
        return self.system_prompt_template

    def _build_user_prompt(self, context: ReasoningContext, telemetry_data: Dict[str, Any]) -> str:
        """Build the user prompt with the dynamic context data."""
        return self.user_prompt_template.format(context_data=json.dumps(telemetry_data, indent=2))

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM JSON output from within the <json_output> block and extract thinking."""
        try:
            thinking_content = "No <thinking> block found."
            if "<thinking>" in response and "</thinking>" in response:
                thinking_start = response.find("<thinking>") + len("<thinking>")
                thinking_end = response.find("</thinking>", thinking_start)
                if thinking_end != -1:
                    thinking_content = response[thinking_start:thinking_end].strip()
            self.logger.info(f"LLM Reasoning Process:\n{thinking_content}", extra={"component": "llm_thinking"})

            json_str = ""
            if "<json_output>" in response and "</json_output>" in response:
                json_start = response.find("<json_output>") + len("<json_output>")
                json_end = response.find("</json_output>", json_start)
                if json_end != -1:
                    json_str = response[json_start:json_end].strip()
            else:
                self.logger.warning("LLM response did not contain <json_output> tags. Falling back to extracting first JSON object.")
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

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Calls the local LLM API endpoint."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        endpoint_url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": full_prompt, "stream": False, "options": {"temperature": self.temperature, "num_predict": self.max_tokens}}
        headers = {"Content-Type": "application/json"}

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    self.logger.info(f"Sending request to local LLM at {endpoint_url}")
                    response = await client.post(endpoint_url, json=payload, headers=headers)
                    response.raise_for_status()
                    self.logger.debug(f"Raw response: {response.text[:500]}")
                    response_data = response.json()
                    response_text = response_data.get("response", "").strip()
                    if not response_text:
                        raise ValueError("LLM response JSON was empty or malformed.")
                    self.logger.info(f"LLM response received on attempt {attempt + 1}")
                    return response_text
            except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Local LLM API call failed with error: {e}. Attempt {attempt + 1}/{self.max_retries}.")
                if attempt + 1 == self.max_retries:
                    self.logger.error("Max retries reached. Failing the LLM operation.")
                    raise
                backoff_time = self.initial_backoff * (2**attempt)
                jitter = random.uniform(0, backoff_time * 0.1)
                delay = backoff_time + jitter
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        raise Exception("LLM call failed after all retries.")

    def _store_call_history(self, context: ReasoningContext, system_prompt: str, user_prompt: str, llm_response: str, parsed_result: Dict[str, Any]):
        call_record = {"timestamp": context.timestamp, "session_id": context.session_id, "reasoning_type": self.reasoning_type.value, "system_prompt": system_prompt, "user_prompt": user_prompt, "llm_response": llm_response, "parsed_result": parsed_result, "model": self.model}
        self.call_history.append(call_record)
        if len(self.call_history) > 100: self.call_history.pop(0)

    # ==== Prompt Customization Methods ====
    def set_system_prompt(self, template: str): self.system_prompt_template = template; return self
    def set_user_prompt(self, template: str): self.user_prompt_template = template; return self
    def add_custom_instructions(self, instructions: str): self.custom_instructions = instructions; return self
    def set_domain_context(self, context: str): self.domain_context = context; return self
    def add_safety_constraint(self, constraint: str): self.safety_constraints.append(constraint); return self
    def set_safety_constraints(self, constraints: List[str]): self.safety_constraints = constraints; return self

    def configure_basic(self):
        """Generic safe defaults."""
        self.set_domain_context("Proactive adaptive control of SWIM exemplar...")
        self.set_safety_constraints(["Only use: ADD_SERVER, REMOVE_SERVER, SET_DIMMER", "Dimmer values must be between 0.0 and 1.0"])
        self.temperature = 0.3
        return self
    
    async def validate_input(self, context: ReasoningContext) -> bool:
        """This implementation assumes input is always valid for now."""
        return True

    def get_required_knowledge_types(self, context: ReasoningContext) -> List[str]:
        """This reasoner fetches its own knowledge inside the reason() method, so it doesn't pre-request types."""
        return []

    def extract_search_terms(self, context: ReasoningContext) -> List[str]:
        """This reasoner does not extract generic search terms from the initial context."""
        return []


class ContextBuilder:
    """
    Processes lists of KB entries (snapshots, actions) to build the
    rich context payload required by the LLM.
    """

    def __init__(self, logger) -> None:
        self.logger = logger or logging.getLogger("ContextBuilder")

    def calculate_ewma(self, data: List[float], alpha: float = 0.5) -> Optional[float]:
        if not data: return None
        ewma = data[0]
        for value in data[1:]: ewma = alpha * value + (1 - alpha) * ewma
        return ewma

    def calculate_rate_of_change(self, sorted_snapshots: List[Dict[str, Any]], metric: str, time_window_seconds: int = 300) -> Optional[str]:
        if len(sorted_snapshots) < 2: return None
        time_series = []
        for s in sorted_snapshots:
            if metric in s["content"]["content"]["current_state"]:
                time_series.append({"timestamp": s["timestamp"], "value": s["content"]["content"]["current_state"][metric]})
        if len(time_series) < 2: return None
        latest_point = time_series[-1]
        past_timestamp_limit = datetime.fromisoformat(latest_point["timestamp"]).timestamp() - time_window_seconds
        reference_point = time_series[0]
        for point in time_series:
            if datetime.fromisoformat(point["timestamp"]).timestamp() >= past_timestamp_limit:
                reference_point = point
                break
        time_diff_seconds = datetime.fromisoformat(latest_point["timestamp"]).timestamp() - datetime.fromisoformat(reference_point["timestamp"]).timestamp()
        if time_diff_seconds < 1: return "+0.00/min"
        value_diff = latest_point["value"] - reference_point["value"]
        rate_per_minute = (value_diff / time_diff_seconds) * 60
        return f"{rate_per_minute:+.2f}/min"

    def build_context_from_kb_results(self, snapshots: List[Dict[str, Any]], action_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The main method to transform raw KB results into the final context payload."""
        if not snapshots: return {"error": "No recent snapshots found in the Knowledge Base."}
        
        sorted_snapshots = sorted(snapshots, key=lambda s: s["timestamp"])
        latest_snapshot = sorted_snapshots[-1]["content"]["content"]
        current_state = latest_snapshot.get("current_state", {})
        
        historical_trends = {}
        all_metrics = set(k for s in sorted_snapshots for k in s.get("content", {}).get("content", {}).get("current_state", {}).keys())
        for metric in all_metrics:
            values = [s["content"]["content"]["current_state"][metric] for s in sorted_snapshots if metric in s.get("content", {}).get("content", {}).get("current_state", {})]
            ewma = self.calculate_ewma(values)
            roc = self.calculate_rate_of_change(sorted_snapshots, metric)
            historical_trends[metric] = {"5min_avg_ewma": round(ewma, 4) if ewma is not None else None, "5min_rate_of_change": roc}

        historical_actions_context = self.build_historical_actions_context(action_history, sorted_snapshots)

        system_goals_and_constraints = {"goals": {"target_utilization": 1, "target_response_time_ms_weighted": 900.0}, "constraints": {"max_servers": 3, "min_servers": 1, "cooldown_period_minutes": 2}}

        return {
            "current_state": current_state,
            "historical_trends": historical_trends,
            "historical_actions": historical_actions_context,
            "system_goals_and_constraints": system_goals_and_constraints,
        }

    def build_historical_actions_context(self, action_history: List[Dict[str, Any]], sorted_snapshots: List[Dict[str, Any]]):
        """Generates summary view of the last 10 actions and feedback for the most recent ones."""
        if not action_history:
            return {
                "summary_last_10": "No actions recorded.",
                "recent_actions_with_feedback": []
            }

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
                recent_actions_with_feedback.append({
                    "action_taken": action_record["content"].get("action_type", "UNKNOWN_ACTION"),
                    "evaluation": "UNKNOWN",
                    "details": "Feedback not calculated for this past action.",
                })

        return {
            "summary_last_10": summary_text,
            "recent_actions_with_feedback": recent_actions_with_feedback,
            "current_timestamp": datetime.now(timezone.utc).isoformat()
        }


    def generate_feedback_for_action(self, action_record: Dict[str, Any], sorted_snapshots: List[Dict[str, Any]]):
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
        pre_candidates = [s for s in sorted_snapshots if datetime.fromisoformat(s["timestamp"]) < action_timestamp]
        if pre_candidates:
            pre_action_snapshot = min(
                pre_candidates,
                key=lambda s: abs(datetime.fromisoformat(s["timestamp"]) - target_pre_ts)
            )

        # 3. Find the snapshot closest to the T+10s mark from all snapshots ON OR AFTER the action
        # First, filter to only include valid candidates
        post_candidates = [s for s in sorted_snapshots if datetime.fromisoformat(s["timestamp"]) >= action_timestamp]
        if post_candidates:
            post_action_snapshot = min(
                post_candidates,
                key=lambda s: abs(datetime.fromisoformat(s["timestamp"]) - target_post_ts)
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
        if pre_action_snapshot['timestamp'] == post_action_snapshot['timestamp']:
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
        pre_ts_used = pre_action_snapshot['timestamp']
        post_ts_used = post_action_snapshot['timestamp']
        self.logger.info(
            f"Evaluating action {action_type} at {action_content['timestamp']}. "
            f"Comparing snapshot at {pre_ts_used} with {post_ts_used}. "
            f"RT change: {rt_change:.2f}, Util change: {util_change:.2f}"
        )
        
        # This evaluation logic remains the same
        if action_type == "ADD_SERVER" and rt_change < -50: evaluation = "SUCCESSFUL"
        elif action_type == "ADD_SERVER" and rt_change > 20: evaluation = "FAILED"
        elif action_type == "REMOVE_SERVER" and util_change > 0.05 and rt_change < 100: evaluation = "SUCCESSFUL"
        elif action_type == "SET_DIMMER" and action_content.get("params", {}).get("value", 1.0) < pre_state.get("dimmer", 1.0) and rt_change < -20: evaluation = "SUCCESSFUL"
        elif action_type == "SET_DIMMER" and rt_change > 20: evaluation = "FAILED"

        return {
            "action_taken": action_type,
            "params": action_content.get("params", {}),
            "timestamp": action_content["timestamp"],
            "evaluation": evaluation,
            "outcome_details": f"Response time changed by {rt_change:+.2f}ms. Utilization changed by {util_change:+.2f}."
        }

def create_llm_reasoner_agent(
    agent_id: str,
    config_path: str,
    llm_api_key: str,
    nats_url: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    mode="llm",
) -> "ReasonerAgent":
    """
    Create a reasoner agent with LLM-based reasoning implementations.
    This factory correctly injects the agent's knowledge base query interface
    into the LLM reasoner for its own specialized data processing.
    """
    from .reasoner_agent import ReasonerAgent, ReasoningType
    from .llm_reasoner import LLMReasoningImplementation
    from .multi_agent_reasoner import MultiAgentReasoner

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
        else:
            llm_reasoner = LLMReasoningImplementation(
                api_key=llm_api_key,
                reasoning_type=reasoning_type,
                kb_query_interface=agent.kb_query,
                logger=logger,
            )

        llm_reasoner.configure_basic()
        agent.add_reasoning_implementation(reasoning_type, llm_reasoner)

    return agent