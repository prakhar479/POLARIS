"""
Multi-Agent Reasoner Implementation

Instead of one giant LLMReasoningImplementation, this version splits logic
into Analyst, Planner, and Executor agents. The coordinator orchestrates them.

Key differences:
- AnalystAgent: Summarizes telemetry and trends from KB.
- PlannerAgent: Uses LLM to propose high-level intent ("increase_capacity", "reduce_load").
- ExecutorAgent: Deterministic. Validates, enforces safety, cooldown, and converts intent to schema.
- Last action handling remains in this coordinator.
"""

import json
import time
import uuid
import logging
import asyncio
import random
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timezone

import google
from google import genai
from google.genai import types

from .reasoner_core import (
    ReasoningInterface, 
    ReasoningContext, 
    ReasoningResult, 
    ReasoningType,
)
from .reasoner_agent import KnowledgeQueryInterface


# ==========================
# AGENTS
# ==========================

async def _call_llm( system_prompt: str, user_prompt: str) -> str:
        """
        Make a direct Gemini API call with retry logic, without using any tools.
        """
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # --- Gemini API Client Setup ---
        client = genai.Client(api_key='')

        # self.logger.info(f"Making LLM call with prompt {full_prompt}")

        # self.logger.info(f"Making LLM call with prompt: {full_prompt}")
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)])]

        # Simplified generation config without tool-related settings
        generate_content_config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096
        )

        for attempt in range(5):
            try:
                # Direct API call without the 'tools' parameter
                response = client.models.generate_content(
                    model='gemini-2.5-flash',  # Ensure self.model is set to "gemini-1.5-flash"
                    contents=contents,
                    config=generate_content_config,
                )

                response_text = response.text.strip()
                # self.logger.info(f"LLM response: {response_text}")

                # --- Success Case: Log and return ---
                # log_entry = {
                #     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                #     "model": self.model,
                #     "system_prompt": str(system_prompt),
                #     "user_prompt": str(user_prompt),
                #     "response": str(response_text),
                #     "reasoning_type": getattr(self.reasoning_type, "value", None),
                #     "status": "success",
                #     "attempt": attempt + 1
                # }
                # log_file = Path("llm_calls.jsonl")
                # with log_file.open("a", encoding="utf-8") as f:
                #     f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
                return response_text

            except google.genai.errors.ServerError as e:
                # --- Failure Case (retry logic remains unchanged) ---
                # self.logger.warning(f"Gemini API call failed with ServerError: {e}. This is attempt {attempt + 1}/{self.max_retries}.")
                if attempt + 1 == 5:
                    # self.logger.error("Max retries reached. Failing the operation.")
                    # ... (logging code for final failure) ...
                    raise
                
                # backoff_time = self.initial_backoff * (2 ** attempt)
                # jitter = random.uniform(0, backoff_time * 0.1)
                # delay = backoff_time + jitter
                # # self.logger.info(f"Model is overloaded. Retrying in {delay:.2f} seconds...")
                # await asyncio.sleep(delay)

        raise Exception("LLM call failed after all retries.")
    

class AnalystAgent:
    """Extracts and summarizes telemetry & trends from KB snapshots."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("AnalystAgent")

    def summarize(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not snapshots:
            return {"error": "No telemetry available"}
        
        sorted_snapshots = sorted(snapshots, key=lambda s: s['timestamp'])
        latest_snapshot = sorted_snapshots[-1]['content']['content']
        current_state = latest_snapshot.get('current_state', {})

        return {
            "current_state": current_state,
            "historical_trends": self._calculate_trends(sorted_snapshots),
        }

    def _calculate_trends(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        def ewma(values, alpha=0.5):
            if not values: return None
            out = values[0]
            for v in values[1:]:
                out = alpha * v + (1 - alpha) * out
            return out

        metrics = set(
            k
            for s in snapshots
            for k in s.get("content", {}).get("content", {}).get("current_state", {}).keys()
        )

        trends = {}
        for m in metrics:
            values = [
                s["content"]["content"]["current_state"][m]
                for s in snapshots
                if m in s["content"]["content"].get("current_state", {})
            ]
            trends[m] = {"ewma": ewma(values)}
        return trends


class PlannerAgent:
    """LLM-powered: Given state summary, propose high-level control intent."""

    def __init__(self, api_key: str, model="gemini-2.5-flash", logger=None):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.logger = logger or logging.getLogger("PlannerAgent")

    async def plan(self, summary: Dict[str, Any], system_prompt: str) -> str:
        contents = [types.Content(
            role="user", 
            parts=[types.Part.from_text(
                text=f"{system_prompt}\n\nSystem Summary:\n{json.dumps(summary, indent=2)}"
            )]
        )]
        config = types.GenerateContentConfig(temperature=0.3, max_output_tokens=4096)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        self.logger.info(f"PlannerAgent LLM response: {response.text}")
        return response.text.strip()


class ExecutorAgent:
    """Deterministic: enforce safety, schema, cooldown, last action logic."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("ExecutorAgent")

    async def execute(self, planner_output: str, last_action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executor that asks the LLM to directly generate a canonical action JSON.
        Code enforces safety minimally (fallbacks, clamping).
        """
        try:
            # --- Step 1: Build refinement prompt for Executor LLM ---
            refinement_prompt = f"""
            You are the ExecutorAgent for a fast controller.

            Input from planner:
            {planner_output}

            Last action taken:
            {json.dumps(last_action or {}, indent=2)}

            You must return ONE action as valid JSON with this schema:
            {{
                "action_type": "SET_DIMMER" | "ADD_SERVER" | "REMOVE_SERVER" | "NO_OP" | "ALERT",
                "source": "fast_controller",
                "action_id": <a new UUID>,
                "params": {{
                    "value": <float between 0.0 and 1.0>   # for SET_DIMMER
                    "server_type": "compute", "count": <1-3>  # for server actions
                }},
                "priority": "low"
            }}

            Rules:
            - Never repeat the same action_type as the last action unless strictly necessary.
            - Keep dimmer between 0.0 and 1.0.
            - Keep server count between 1 and 3.
            - If planner output is unclear → return NO_OP.
            """

            # --- Step 2: Call LLM to generate action ---
            llm_response = await _call_llm(
                system_prompt="You are a careful executor that outputs only valid JSON.",
                user_prompt=refinement_prompt
            )

            # --- Step 3: Parse JSON fr
            # om LLM output ---
            candidate = json.loads(llm_response[llm_response.find("{"): llm_response.rfind("}")+1])

            # --- Step 4: Minimal validation ---
            # Cooldown check
            if last_action and candidate.get("action_type") == last_action.get("action_type"):
                candidate["action_type"] = "NO_OP"
                candidate["params"] = {}

            # Clamp values if needed
            if candidate["action_type"] == "SET_DIMMER":
                v = float(candidate["params"].get("value", 0.5))
                candidate["params"]["value"] = max(0.0, min(1.0, v))
            elif candidate["action_type"] in {"ADD_SERVER", "REMOVE_SERVER"}:
                c = int(candidate["params"].get("count", 1))
                candidate["params"]["count"] = max(1, min(3, c))

            return candidate

        except Exception as e:
            self.logger.error(f"Executor failed: {e}")
            return {
                "action_type": "ALERT",
                "source": "fast_controller",
                "action_id": str(uuid.uuid4()),
                "params": {"message": f"Executor failed: {e}"},
                "priority": "high"
            }



# ==========================
# COORDINATOR (MAIN ENTRY)
# ==========================

class MultiAgentReasoner(ReasoningInterface):
    """Coordinates Analyst → Planner → Executor. Keeps last_action tracking."""

    def __init__(self, api_key: str, reasoning_type: ReasoningType,
                 kb_query_interface: Optional[KnowledgeQueryInterface] = None,
                 logger=None):
        self.logger = logger or logging.getLogger("MultiAgentReasoner")
        self.api_key = api_key
        self.reasoning_type = reasoning_type
        self.kb_query = kb_query_interface

        # agents
        self.analyst = AnalystAgent(logger=self.logger)
        self.planner = PlannerAgent(api_key, logger=self.logger)
        self.executor = ExecutorAgent(logger=self.logger)

    async def reason(self, context: ReasoningContext, knowledge=None) -> ReasoningResult:
        start = time.time()
        steps = ["Starting multi-agent reasoning"]

        # === KB queries ===
        snapshots = await self.kb_query.query_structured(
            data_types=["observation"],
            filters={"source": "swim_snapshotter", "tags": ["snapshot"]},
            limit=120
        )
        last_action_list = await self.kb_query.query_structured(
            data_types=["observation"],
            filters={"source": "polaris.reasoning_engine", "tags": ["control_action"]},
            limit=1
        )
        last_action = last_action_list[0] if last_action_list else None
        steps.append(f"Retrieved {len(snapshots)} snapshots, last_action={last_action}")

        # === Analyst ===
        summary = self.analyst.summarize(snapshots)
        steps.append("Analyst produced summary")

        # === Planner (LLM) ===

        context_description = """Terminology:

Response Time: How long requests take to complete (lower is better).

Throughput: Amount of basic vs. optional work being handled.

Utilization: How much capacity the servers are using.

Dimmer: A control between 0.0 and 1.0 that determines how much optional content is served (lower reduces load, higher offers more content).

Servers: Active servers can be added or removed, up to a maximum."""


        system_prompt = f"""
            You are the PlannerAgent for an adaptive system.

            Context:
            - System Description: {context_description}
            - Reasoning Type: {self.reasoning_type.value}
            - Current Timestamp: {datetime.now(timezone.utc).isoformat()}

            Your job:
            - Look at the system summary below.
            - Suggest a **high-level intent**: ["increase_capacity", "reduce_load", "hold_steady"].
            - Do not output JSON schema here — just the intent as plain text.
            - Keep reasoning concise.

            System Summary:
            {json.dumps(summary, indent=2)}
            """
        
        llm_response = await self.planner.plan(summary, system_prompt)
        steps.append("Planner proposed intent")

        # === Executor ===
        action = await self.executor.execute(llm_response, last_action)
        steps.append("Executor produced validated action")

        exec_time = time.time() - start
        return ReasoningResult(
            result=action,
            confidence=0.8,
            reasoning_steps=steps,
            context=context,
            execution_time=exec_time,
            kb_queries_made=2
        )
    
    async def validate_input(self, context: ReasoningContext) -> bool:
        """Validate that we have sufficient telemetry data for LLM reasoning."""
        
        
        return True
    
    def get_required_knowledge_types(self, context: ReasoningContext) -> List[str]:
        """LLM reasoner doesn't require KB knowledge."""
        return []
    
    def extract_search_terms(self, context: ReasoningContext) -> List[str]:
        """LLM reasoner doesn't require KB search."""
        return []
    
    def configure_basic(self):
        pass
