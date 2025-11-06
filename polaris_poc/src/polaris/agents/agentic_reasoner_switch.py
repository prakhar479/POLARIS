"""
Agentic LLM Reasoner Implementation with Google Gemini Flash 2.5

An autonomous agentic reasoner that can dynamically decide which tools to use
(Knowledge Base queries, Digital Twin interactions) and make adaptation decisions
based on its analysis. Uses Google Gemini Flash 2.5 as the reasoning engine.
"""

import json
import asyncio
import time
import uuid
import os
from typing import Any, Dict, List, Optional, Union, Callable
import logging
from datetime import datetime, timezone, timedelta
import tempfile
import yaml
import sys
from pathlib import Path

from google import genai
from google.genai import types

from .reasoner_core import (
    ReasoningInterface,
    ReasoningContext,
    ReasoningResult,
    ReasoningType,
)

from collections import deque

from collections import deque

# Add config/prompts to path for dynamic prompt loading
config_prompts_path = Path(__file__).parent.parent.parent / "config" / "prompts"
if str(config_prompts_path) not in sys.path:
    sys.path.insert(0, str(config_prompts_path))

from .reasoner_agent import (
    KnowledgeQueryInterface,
    DigitalTwinInterface,
    DTQuery,
    DTSimulation,
    DTDiagnosis,
    DTResponse,
    GRPCDigitalTwinClient,
    ReasonerAgent,
)

from .reasoner_core import ReasoningType
from .improved_grpc_client import ImprovedGRPCDigitalTwinClient


class PerformanceLogger:
    """Simple performance logger for overhead metrics."""

    def __init__(self, log_dir: str = "logs/overhead"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "performance_metrics.jsonl"

    def log_metric(self, metric_type: str, duration_ms: float, details: Dict[str, Any] = None):
        """Log a performance metric to file."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metric_type": metric_type,
                "duration_ms": round(duration_ms, 3),
                "details": details or {},
            }

            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Don't let logging errors break the main flow
            print(f"Performance logging error: {e}")


class AgenticTool:
    """Base class for tools that the agentic reasoner can use."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        raise NotImplementedError


class KnowledgeBaseTool(AgenticTool):
    """Tool for querying the knowledge base."""

    def __init__(
        self,
        kb_query: KnowledgeQueryInterface,
        logger: logging.Logger,
        perf_logger: PerformanceLogger,
    ):
        super().__init__(
            name="query_knowledge_base",
            description="Query the knowledge base for historical data, observations, and telemetry",
            parameters={
                "query_type": {
                    "type": "string",
                    "enum": [
                        "structured",
                        "natural_language",
                    ],
                    "description": "Type of query to perform (use 'structured' with data_types filter for specific data)",
                },
                "data_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of data to query (for structured queries)",
                },
                "filters": {"type": "object", "description": "Filters to apply to the query"},
                "query_text": {"type": "string", "description": "Natural language query text"},
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return",
                },
            },
        )
        self.kb_query = kb_query
        self.logger = logger
        self.perf_logger = perf_logger

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute knowledge base query."""
        start_time = time.time()
        try:
            query_type = kwargs.get("query_type", "structured")
            limit = kwargs.get("limit", 10)

            if query_type == "structured":
                data_types = kwargs.get("data_types", ["observation"])
                filters = kwargs.get("filters", {})
                results = await self.kb_query.query_structured(
                    data_types=data_types, filters=filters, limit=limit
                )
            elif query_type == "natural_language":
                query_text = kwargs.get("query_text", "")
                results = await self.kb_query.query_natural_language(
                    query_text=query_text, limit=limit
                )
            else:
                return {"success": False, "error": f"Unknown query type: {query_type}"}

            duration_ms = (time.time() - start_time) * 1000
            self.perf_logger.log_metric(
                "kb_query",
                duration_ms,
                {
                    "query_type": query_type,
                    "result_count": len(results) if results else 0,
                    "limit": limit,
                },
            )

            return {
                "success": True,
                "results": results or [],
                "count": len(results) if results else 0,
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.perf_logger.log_metric(
                "kb_query_error",
                duration_ms,
                {"error": str(e), "query_type": kwargs.get("query_type", "unknown")},
            )
            self.logger.error(f"Knowledge base query failed: {e}")
            return {"success": False, "error": str(e)}


class DigitalTwinTool(AgenticTool):
    """Tool for interacting with the digital twin."""

    def __init__(
        self,
        dt_interface: DigitalTwinInterface,
        logger: logging.Logger,
        perf_logger: PerformanceLogger,
    ):
        super().__init__(
            name="query_digital_twin",
            description="Query the digital twin for system state, simulations, and diagnostics",
            parameters={
                "operation": {
                    "type": "string",
                    "enum": ["query", "simulate", "diagnose"],
                    "description": "Type of digital twin operation",
                },
                "query_type": {
                    "type": "string",
                    "description": "Type of query (for query operations)",
                },
                "query_content": {"type": "string", "description": "Content of the query"},
                "simulation_type": {
                    "type": "string",
                    "description": "Type of simulation (for simulate operations)",
                },
                "actions": {"type": "array", "description": "Actions to simulate"},
                "horizon_minutes": {
                    "type": "integer",
                    "default": 60,
                    "description": "Simulation horizon in minutes",
                },
                "anomaly_description": {
                    "type": "string",
                    "description": "Description of anomaly (for diagnose operations)",
                },
            },
        )
        self.dt_interface = dt_interface
        self.logger = logger
        self.perf_logger = perf_logger

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute digital twin operation."""
        start_time = time.time()
        operation = kwargs.get("operation", "query")

        try:
            # Check if digital twin interface is available
            if not self.dt_interface:
                duration_ms = (time.time() - start_time) * 1000
                self.perf_logger.log_metric(
                    "dt_error",
                    duration_ms,
                    {"error": "interface_not_available", "operation": operation},
                )
                return {"success": False, "error": "Digital twin interface not available"}

            # Ensure connection is established with better error handling
            if hasattr(self.dt_interface, "stub") and not self.dt_interface.stub:
                self.logger.info("Digital twin not connected, attempting to connect...")
                try:
                    await asyncio.wait_for(self.dt_interface.connect(), timeout=15.0)
                    if not self.dt_interface.stub:
                        duration_ms = (time.time() - start_time) * 1000
                        self.perf_logger.log_metric(
                            "dt_connection_error",
                            duration_ms,
                            {"error": "connection_failed", "operation": operation},
                        )
                        return {"success": False, "error": "Failed to connect to digital twin"}
                except asyncio.TimeoutError:
                    duration_ms = (time.time() - start_time) * 1000
                    self.perf_logger.log_metric(
                        "dt_connection_error",
                        duration_ms,
                        {"error": "connection_timeout", "operation": operation},
                    )
                    return {"success": False, "error": "Digital twin connection timeout"}
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    self.perf_logger.log_metric(
                        "dt_connection_error",
                        duration_ms,
                        {"error": f"connection_exception: {str(e)}", "operation": operation},
                    )
                    return {"success": False, "error": f"Digital twin connection failed: {str(e)}"}

            self.logger.debug(f"Executing digital twin operation: {operation}")
            self.logger.debug(f"Operation parameters: {kwargs}")

            # Log parameter types for debugging
            for key, value in kwargs.items():
                self.logger.debug(f"Parameter {key}: {value} (type: {type(value).__name__})")

            if operation == "query":
                query = DTQuery(
                    query_type=kwargs.get("query_type", "current_state"),
                    query_content=kwargs.get("query_content", "Get system overview"),
                    parameters=kwargs.get("parameters", {}),
                )
                response = await self.dt_interface.query(query)
            elif operation == "simulate":
                # Format actions properly - convert strings to action dictionaries
                raw_actions = kwargs.get("actions", [])
                formatted_actions = []

                self.logger.debug(f"Raw actions received: {raw_actions}")
                self.logger.debug(f"Raw actions types: {[type(a) for a in raw_actions]}")

                for i, action in enumerate(raw_actions):
                    self.logger.debug(f"Processing action {i}: {action} (type: {type(action)})")

                    if isinstance(action, str):
                        # Convert string action to proper action dictionary
                        action_dict = {
                            "action_id": str(uuid.uuid4()),
                            "action_type": action,
                            "target": "system",
                            "params": {},
                        }
                        formatted_actions.append(action_dict)
                        self.logger.debug(f"Converted string action to: {action_dict}")
                    elif isinstance(action, dict):
                        # Already a dictionary, use as-is
                        formatted_actions.append(action)
                        self.logger.debug(f"Using dict action as-is: {action}")
                    else:
                        self.logger.warning(
                            f"Unknown action format: {action} (type: {type(action)})"
                        )

                self.logger.debug(f"Formatted actions: {formatted_actions}")

                simulation = DTSimulation(
                    simulation_type=kwargs.get("simulation_type", "forecast"),
                    actions=formatted_actions,
                    horizon_minutes=kwargs.get("horizon_minutes", 60),
                    parameters=kwargs.get("parameters", {}),
                )
                response = await self.dt_interface.simulate(simulation)
            elif operation == "diagnose":
                diagnosis = DTDiagnosis(
                    anomaly_description=kwargs.get("anomaly_description", ""),
                    context=kwargs.get("context", {}),
                )
                response = await self.dt_interface.diagnose(diagnosis)
            else:
                duration_ms = (time.time() - start_time) * 1000
                self.perf_logger.log_metric(
                    "dt_error",
                    duration_ms,
                    {"error": f"unknown_operation: {operation}", "operation": operation},
                )
                return {"success": False, "error": f"Unknown operation: {operation}"}

            duration_ms = (time.time() - start_time) * 1000

            if response:
                self.perf_logger.log_metric(
                    "dt_operation",
                    duration_ms,
                    {
                        "operation": operation,
                        "success": response.success,
                        "confidence": response.confidence,
                    },
                )

                result = {
                    "success": response.success,
                    "result": response.result,
                    "confidence": response.confidence,
                    "explanation": response.explanation,
                    "metadata": response.metadata or {},
                }

                # Add optional fields if they exist
                if hasattr(response, "future_states") and response.future_states:
                    result["future_states"] = response.future_states
                if hasattr(response, "hypotheses") and response.hypotheses:
                    result["hypotheses"] = response.hypotheses

                self.logger.debug(
                    f"Digital twin response received: success={response.success}, confidence={response.confidence}"
                )
                if not response.success:
                    self.logger.error(f"Digital twin operation failed: {response.explanation}")
                    self.logger.error(f"Digital twin metadata: {response.metadata}")

                return result
            else:
                self.perf_logger.log_metric(
                    "dt_error", duration_ms, {"error": "no_response", "operation": operation}
                )
                return {"success": False, "error": "No response from digital twin"}

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.perf_logger.log_metric(
                "dt_error", duration_ms, {"error": str(e), "operation": operation}
            )
            self.logger.error(f"Digital twin operation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


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
        self,
        snapshots: List[Dict[str, Any]],
        action_history: List[Dict[str, Any]],
        reactive_log,
        decisions,
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
        self.logger.info(f"Current state extracted: {current_state}")

        self.current_servers = current_state.get("servers", -1)

        self.logger.info(f"Current server count updated to: {self.current_servers}")

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
            "reactive_logs": list(reactive_log),
            "recent_decisions_including_reactive_controller": decisions[-5:] if decisions else [],
        }

    def build_historical_actions_context(
        self, action_history: List[Dict[str, Any]], sorted_snapshots: List[Dict[str, Any]]
    ):
        """Generates summary view of the last 10 actions and feedback for the most recent ones."""
        if not action_history:
            return {"summary_last_10": "No actions recorded.", "recent_actions_with_feedback": []}

        # Get current UTC timestamp for time difference calculations
        current_utc = datetime.now(timezone.utc)

        # Consider only the last 10 actions
        last_10_actions = action_history[-10:] if len(action_history) >= 10 else action_history

        # Initialize counters for known action types and track most recent action times
        summary = {"ADD_SERVER": 0, "REMOVE_SERVER": 0, "SET_DIMMER": 0}
        most_recent_action_times = {}
        cooldown_status = {}

        for action_record in last_10_actions:
            action_type = action_record["content"].get("action_type")
            if action_type in summary:
                summary[action_type] += 1

                # Track the most recent timestamp for each action type
                action_timestamp = action_record.get("timestamp")
                if action_timestamp:
                    try:
                        # Parse timestamp and calculate time difference
                        action_time = datetime.fromisoformat(
                            action_timestamp.replace("Z", "+00:00")
                        )
                        if action_type not in most_recent_action_times:
                            most_recent_action_times[action_type] = action_time
                        else:
                            if action_time > most_recent_action_times[action_type]:
                                most_recent_action_times[action_type] = action_time
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Failed to parse timestamp {action_timestamp}: {e}")

        # Calculate cooldown status for server actions (2-minute cooldown)
        cooldown_minutes = 2.0
        for action_type in ["ADD_SERVER", "REMOVE_SERVER"]:
            if action_type in most_recent_action_times:
                time_since_action = current_utc - most_recent_action_times[action_type]
                minutes_since = time_since_action.total_seconds() / 60.0

                cooldown_status[action_type] = {
                    "last_action_time": most_recent_action_times[action_type].isoformat(),
                    "minutes_since_last_action": round(minutes_since, 2),
                    "cooldown_remaining_minutes": max(
                        0, round(cooldown_minutes - minutes_since, 2)
                    ),
                    "cooldown_expired": minutes_since >= cooldown_minutes,
                }

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
            "current_timestamp": current_utc.isoformat(),
            "cooldown_status": cooldown_status,
            "most_recent_actions": {
                action_type: time.isoformat()
                for action_type, time in most_recent_action_times.items()
            },
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


class AgenticLLMReasoner(ReasoningInterface):
    """
    Agentic LLM-based reasoner using Google Gemini Flash 2.5.

    This reasoner can autonomously decide when to use tools (Knowledge Base, Digital Twin)
    and make adaptation decisions based on its analysis.

    System-agnostic implementation: Loads prompts dynamically based on managed system type.
    """

    def __init__(
        self,
        api_key: str,
        config: Dict[str, Any],
        kb_query_interface: Optional[KnowledgeQueryInterface] = None,
        dt_interface: Optional[DigitalTwinInterface] = None,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 8192,
        temperature: float = 0.3,
        max_tool_calls: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = api_key
        self.reasoning_type = reasoning_type
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_tool_calls = max_tool_calls
        self.max_retries = 3
        self.logger = logger or logging.getLogger(f"AgenticReasoner.{reasoning_type.value}")

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # Initialize tools
        self.tools = {}
        if kb_query_interface:
            self.tools["query_knowledge_base"] = KnowledgeBaseTool(kb_query_interface, self.logger)
            self.logger.info("Knowledge Base tool initialized")
        if dt_interface:
            self.tools["query_digital_twin"] = DigitalTwinTool(dt_interface, self.logger)
            self.logger.info("Digital Twin tool initialized")

        if self.tools:
            self.logger.info(
                f"Agentic reasoner initialized with {len(self.tools)} tools: {list(self.tools.keys())}"
            )

        # Load system-specific prompts dynamically
        self.system_name = system_name or config.get("system_name", "generic")
        self._load_system_prompts()

        # System prompt for the agentic reasoner
        self.prompt_config = self._load_prompt_config()
        self.system_prompt = self._build_system_prompt()

        self.logger.info(
            f"Initialized AgenticLLMReasoner with Gemini Flash 2.5 and {len(self.tools)} tools available"
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agentic reasoner."""
        tools_description = ""
        if self.tools:
            tools_description = "\n\nAvailable Tools:\n"
            for tool_name, tool in self.tools.items():
                tools_description += f"- {tool_name}: {tool.description}\n"
                tools_description += f"  Parameters: {json.dumps(tool.parameters, indent=2)}\n"
        else:
            tools_description = "\n\nNote: No external tools are currently available. You must make decisions based solely on the provided input data.\n"

        return f"""You are an autonomous adaptive system controller{"" if self.tools else " operating in standalone mode"} for making adaptation decisions.

Your role is to:
1. Analyze the current system situation
2. Use available tools to gather additional information as needed
3. Make informed adaptation decisions based on your analysis
4. Generate appropriate control actions

{tools_description}

KNOWLEDGE BASE QUERY GUIDELINES:
When using query_knowledge_base tool, follow these patterns:

1. **For recent observations/aggregated telemetry**:
   - query_type: "structured"
   - data_types: ["observation"]
   - Optional filters for specific metrics

2. **For raw telemetry events**:
   - query_type: "structured" 
   - data_types: ["raw_telemetry_event"]
   - Use filters to specify metric_name if needed

3. **For adaptation decisions history**:
   - query_type: "structured"
   - data_types: ["adaptation_decision"]

4. **For natural language search**:
   - query_type: "natural_language"
   - query_text: "your search terms"

Valid data_types: ["raw_telemetry_event", "adaptation_decision", "system_goal", "learned_pattern", "observation", "system_info", "generic_fact"]

IMPORTANT INSTRUCTIONS FOR TOOL USAGE:
- If you need additional information to make a good decision, you MUST use the available tools
- Always try to gather more context before making decisions when tools are available
- Use tools to check historical patterns, system state, or run simulations
- Tool usage is STRONGLY ENCOURAGED when tools are available

Decision Making Process:
1. First, analyze the input data to understand the current situation
2. Determine what additional information you need (historical data, simulations, diagnostics)
3. Use tools to gather that information (this step is CRITICAL)
4. Wait for tool results and analyze them
5. Synthesize all information to make an adaptation decision
6. Generate the appropriate control action
7. You cannot make server decisions too frequently

Control Actions Available:
- ADD_SERVER: Add a server to handle increased load
- REMOVE_SERVER: Remove a server to reduce costs
- SET_DIMMER: Adjust the dimmer value (0.0-1.0) to control optional content
- NO_ACTION: Take no action if system is operating optimally

System Constraints:
- Response time should stay below 1000ms
- Utilization should be between 50-85%
- Server count should be between 1-10
- Dimmer value should be between 0.0-1.0

CRITICAL WORKFLOW RULES:
1. **NEVER provide both tool calls AND action in the same response**
2. **If you need tools, ONLY provide tool calls - NO action**
3. **Wait for tool results, then provide your final action**
4. **Tool calls are processed iteratively - you can make up to 5 tool calls total**
5. **Each tool call response will be provided back to you before you can make more calls**

MANDATORY OUTPUT FORMAT:

**PHASE 1 - INFORMATION GATHERING (if tools needed):**
Your response should contain:

1. **Analysis**: Brief analysis of the current situation

2. **Tool Usage Decision**: 
   - State what information you need and which tools you will use
   - Be specific about what you hope to learn

3. **Tool Calls**: Use this EXACT format for each tool call:
   ```
   TOOL_CALL: tool_name
   PARAMETERS: {{"param1": "value1", "param2": "value2"}}
   ```

**PHASE 2 - FINAL DECISION (after tool results received):**
Your response should contain:

1. **Analysis**: Analysis of tool results and current situation

2. **Final Decision**: Your decision and reasoning based on all available information

3. **Action**: The control action in JSON format:
   ```json
   {{
     "action_type": "ACTION_TYPE",
     "source": "agentic_reasoner",
     "action_id": "generated-uuid",
     "params": {{}},
     "priority": "low|medium|high",
     "reasoning": "Brief explanation of the decision"
   }}
   ```

EXAMPLES:

**Example 1 - PHASE 1 (Information Gathering):**
```
**Analysis**: System utilization is at 95% with response time at 850ms, approaching the 1000ms threshold.

**Tool Usage Decision**: I need historical patterns and system diagnostics to make an informed decision about whether to reduce the dimmer.

**Tool Calls**:
TOOL_CALL: query_knowledge_base
PARAMETERS: {{"query_type": "structured", "data_types": ["observation"], "limit": 5}}

TOOL_CALL: query_digital_twin
PARAMETERS: {{"operation": "simulate", "simulation_type": "dimmer_reduction", "actions": [{{"action_type": "SET_DIMMER", "params": {{"value": 0.9}}}}], "horizon_minutes": 30}}
```

**Example 1 - PHASE 2 (After receiving tool results):**
```
**Analysis**: Tool results show consistent high utilization trend in recent observations. Simulation indicates reducing dimmer to 0.7 will bring response time to 650ms while maintaining acceptable service levels.

**Final Decision**: Based on simulation results showing dimmer reduction will effectively address the performance issue without significant service impact.

**Action**:
{{
  "action_type": "SET_DIMMER",
  "source": "agentic_reasoner", 
  "action_id": "dimmer-reduction-001",
  "params": {{"value": 0.7}},
  "priority": "medium",
  "reasoning": "Reducing dimmer from 0.9 to 0.7 to prevent response time breach while maintaining service quality"
}}
```

**Example 2 - Direct Action (when sufficient information available):**
```
**Analysis**: System has 1 server with 99% utilization and response time of 1200ms, clearly exceeding the 1000ms threshold.

**Final Decision**: This is a clear capacity bottleneck requiring immediate scaling. No additional tools needed as the situation is unambiguous.

**Action**:
{{
  "action_type": "ADD_SERVER",
  "source": "agentic_reasoner",
  "action_id": "emergency-scale-001", 
  "params": {{"server_type": "compute", "count": 1}},
  "priority": "high",
  "reasoning": "Emergency scaling due to severe capacity bottleneck with response time far exceeding threshold"
}}
```

CRITICAL REMINDERS:
1. **NEVER provide tool calls AND action in the same response**
2. **If you use tools, wait for results before providing action**
3. **Tool results will be provided back to you in subsequent messages**
4. **You can make multiple rounds of tool calls (up to 5 total) before final action**
"""

    async def reason(
        self, context: ReasoningContext, knowledge: Optional[List[Dict[str, Any]]] = None
    ) -> ReasoningResult:
        """Execute agentic reasoning with dynamic tool usage."""
        overall_start_time = time.time()
        reasoning_steps = ["Starting agentic LLM reasoning with Gemini Flash 2.5"]
        tool_calls_made = 0

        # Performance tracking
        tool_call_timings = []
        llm_call_timings = []

        try:
            # Query for historical actions and snapshots to build enhanced context
            hist_start_time = time.time()
            await self._gather_historical_context()
            hist_duration_ms = (time.time() - hist_start_time) * 1000
            self.perf_logger.log_metric("historical_context_gathering", hist_duration_ms)
            reasoning_steps.append("Gathered historical actions context")

            # Initial analysis
            user_prompt = self._build_initial_prompt(context)
            reasoning_steps.append("Built initial analysis prompt")

            # Start the reasoning loop with chat history
            chat_history = [
                types.Content(role="user", parts=[types.Part(text=self.system_prompt)]),
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text="Understood. I will analyze system contexts and make adaptation decisions following the structured format with tool usage when needed."
                        )
                    ],
                ),
                types.Content(role="user", parts=[types.Part(text=user_prompt)]),
            ]

            final_action = None

            for iteration in range(self.max_tool_calls + 1):  # +1 for final decision
                reasoning_steps.append(f"Reasoning iteration {iteration + 1}")

                # Get LLM response
                llm_start_time = time.time()
                llm_response = await self._call_gemini(chat_history)
                llm_duration_ms = (time.time() - llm_start_time) * 1000
                llm_call_timings.append(llm_duration_ms)
                self.perf_logger.log_metric(
                    "llm_call",
                    llm_duration_ms,
                    {
                        "iteration": iteration + 1,
                        "model": self.model,
                        "response_length": len(llm_response),
                    },
                )
                reasoning_steps.append(f"Received Gemini response (iteration {iteration + 1})")
                self.logger.debug(f"Gemini Response (iteration {iteration + 1}): {llm_response}")

                # Add assistant response to history
                chat_history.append(
                    types.Content(role="model", parts=[types.Part(text=llm_response)])
                )

                # Parse response for tool calls or final decision
                tool_calls, action = self._parse_llm_response(llm_response)
                self.logger.debug(f"Parsed tool calls: {tool_calls}")
                self.logger.debug(f"Parsed action: {action}")

                # Check if we should execute tool calls first
                self.logger.info(
                    f"Tool calls made: {tool_calls_made}, Max tool calls: {self.max_tool_calls}"
                )
                if tool_calls and tool_calls_made < self.max_tool_calls:
                    # Execute tool calls (prioritize over action)
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("tool_name")
                        parameters = tool_call.get("parameters", {})

                        if tool_name in self.tools:
                            reasoning_steps.append(f"Executing tool: {tool_name}")
                            self.logger.info(
                                f"Executing tool '{tool_name}' with parameters: {parameters}"
                            )
                            tool_start_time = time.time()
                            result = await self.tools[tool_name].execute(**parameters)
                            tool_duration_ms = (time.time() - tool_start_time) * 1000
                            tool_call_timings.append(tool_duration_ms)
                            self.perf_logger.log_metric(
                                "tool_call_total",
                                tool_duration_ms,
                                {
                                    "tool_name": tool_name,
                                    "iteration": iteration + 1,
                                    "parameters": str(parameters)[:100],  # Truncate for logging
                                },
                            )

                            tool_results.append(
                                {"tool_name": tool_name, "parameters": parameters, "result": result}
                            )
                            # self.logger.debug(f"Tool result: {result}")
                            tool_calls_made += 1
                        else:
                            reasoning_steps.append(f"Unknown tool requested: {tool_name}")
                            self.logger.warning(f"Unknown tool requested: {tool_name}")

                    # Add tool results to conversation
                    if tool_results:
                        tool_results_text = "Tool Results:\n" + json.dumps(tool_results, indent=2)
                        chat_history.append(
                            types.Content(role="user", parts=[types.Part(text=tool_results_text)])
                        )
                        reasoning_steps.append(
                            f"Added {len(tool_results)} tool results to conversation"
                        )
                        self.logger.info(
                            f"Feeding tool results back to LLM: {len(tool_results)} results"
                        )
                        self.logger.debug(f"Tool Results Text: {tool_results_text}")
                        # Continue to next iteration to let LLM process the tool results
                        continue
                    else:
                        # No valid tool calls were executed
                        reasoning_steps.append("No valid tool calls executed")
                        self.logger.warning("Tool calls were parsed but none were valid")

                if action:
                    # Final decision reached (only if no tool calls were executed)
                    final_action = action
                    reasoning_steps.append("Final decision reached")
                    self.logger.info(f"Final action determined: {action}")
                    break

                # No tool calls and no action - force a decision
                if not tool_calls and not action:
                    chat_history.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    text="Please provide your final decision and action in the required JSON format."
                                )
                            ],
                        )
                    )
                    llm_start_time = time.time()
                    final_response = await self._call_gemini(chat_history)
                    llm_duration_ms = (time.time() - llm_start_time) * 1000
                    llm_call_timings.append(llm_duration_ms)
                    self.perf_logger.log_metric(
                        "llm_call_final",
                        llm_duration_ms,
                        {"model": self.model, "forced_decision": True},
                    )

                    _, final_action = self._parse_llm_response(final_response)
                    reasoning_steps.append("Forced final decision")
                    break

            # Ensure we have a valid action
            # Normalize or fallback to NO_ACTION
            if final_action:
                final_action = self._normalize_action(final_action)
            else:
                final_action = self._normalize_action(
                    {
                        "action_type": "NO_ACTION",
                        "priority": "low",
                        "reasoning": "Unable to determine appropriate action",
                    }
                )
                reasoning_steps.append("Fallback to NO_ACTION")

            overall_duration_ms = (time.time() - overall_start_time) * 1000

            # Log comprehensive overhead metrics
            self.perf_logger.log_metric(
                "end_to_end_reasoning",
                overall_duration_ms,
                {
                    "tool_calls_made": tool_calls_made,
                    "llm_calls_made": len(llm_call_timings),
                    "final_action_type": final_action.get("action_type"),
                    "total_tool_time_ms": sum(tool_call_timings),
                    "total_llm_time_ms": sum(llm_call_timings),
                    "avg_tool_time_ms": (
                        sum(tool_call_timings) / len(tool_call_timings) if tool_call_timings else 0
                    ),
                    "avg_llm_time_ms": (
                        sum(llm_call_timings) / len(llm_call_timings) if llm_call_timings else 0
                    ),
                },
            )

            execution_time = time.time() - overall_start_time

            return ReasoningResult(
                result=final_action,
                confidence=0.8,  # Could be dynamic based on tool results
                reasoning_steps=reasoning_steps,
                context=context,
                execution_time=execution_time,
                kb_queries_made=tool_calls_made,  # Approximate
                dt_queries_made=0,  # Could track separately
            )

        except Exception as e:
            overall_duration_ms = (time.time() - overall_start_time) * 1000
            self.perf_logger.log_metric(
                "reasoning_error",
                overall_duration_ms,
                {
                    "error": str(e),
                    "tool_calls_made": tool_calls_made,
                    "llm_calls_made": len(llm_call_timings),
                },
            )
            self.logger.error(f"Agentic reasoning failed: {e}", exc_info=True)
            execution_time = time.time() - overall_start_time

            return ReasoningResult(
                result={
                    "action_type": "ALERT",
                    "source": "agentic_reasoner",
                    "action_id": str(uuid.uuid4()),
                    "params": {"message": f"Agentic reasoning failed: {e}"},
                    "priority": "high",
                    "reasoning": "Fallback due to reasoning failure",
                },
                confidence=0.1,
                reasoning_steps=reasoning_steps + [f"Error: {str(e)}"],
                context=context,
                execution_time=execution_time,
            )

    def _build_initial_prompt(self, context: ReasoningContext) -> str:
        """Build the initial prompt for the reasoning session with historical actions context."""
        tools_reminder = ""
        if self.tools:
            tools_reminder = f"""
IMPORTANT: You have access to the following tools that can provide additional information:
{', '.join(self.tools.keys())}

You should use these tools to gather more information before making decisions. For example:
- Use 'query_knowledge_base' to get historical data and trends
- Use 'query_digital_twin' to run simulations or get current system state
- Always try to gather more context before acting

START BY USING TOOLS TO GATHER MORE INFORMATION!
"""

        # Build enhanced context with historical actions if available
        enhanced_context = context.input_data.copy()

        # Add historical actions context if available
        if hasattr(self, "last_historical_context") and self.last_historical_context:
            enhanced_context["historical_actions_summary"] = self.last_historical_context

        return f"""Current System Context:
{json.dumps(enhanced_context, indent=2)}

Session ID: {context.session_id}
Reasoning Type: {context.reasoning_type.value}
Timestamp: {datetime.fromtimestamp(context.timestamp).isoformat()}

{tools_reminder}

Please analyze this situation and determine what actions, if any, should be taken to maintain optimal system performance.

Remember to follow the structured output format with tool calls if you need more information!
"""

    def _normalize_action(self, raw_action: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the action matches the expected format."""
        normalized = {
            "action_type": raw_action.get("action_type", "NO_ACTION").upper(),
            "source": "agentic_reasoner",
            "action_id": raw_action.get("action_id", str(uuid.uuid4())),
            "params": raw_action.get("params", {}),
            "priority": raw_action.get("priority", "medium"),
            "reasoning": raw_action.get("reasoning", "No explicit reasoning provided by model"),
        }

        # Extra cleanup: remove stray quotes, handle nested JSON strings
        if isinstance(normalized["params"], str):
            try:
                normalized["params"] = json.loads(normalized["params"])
            except json.JSONDecodeError:
                normalized["params"] = {}

        return normalized

    async def _call_gemini(self, chat_history: List[types.Content]) -> str:
        """Call Gemini API with chat history."""
        for attempt in range(self.max_retries):
            call_start_time = time.time()
            try:
                # Generate content with the chat history
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=chat_history,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )

                call_duration_ms = (time.time() - call_start_time) * 1000

                # Extract text from response
                if response and response.text:
                    self.perf_logger.log_metric(
                        "gemini_api_call",
                        call_duration_ms,
                        {
                            "attempt": attempt + 1,
                            "success": True,
                            "response_length": len(response.text),
                            "model": self.model,
                        },
                    )
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")

            except Exception as e:
                call_duration_ms = (time.time() - call_start_time) * 1000
                self.perf_logger.log_metric(
                    "gemini_api_call",
                    call_duration_ms,
                    {
                        "attempt": attempt + 1,
                        "success": False,
                        "error": str(e),
                        "model": self.model,
                    },
                )
                self.logger.warning(f"Gemini call attempt {attempt + 1} failed: {e}")
                if attempt + 1 == self.max_retries:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

        raise Exception("Gemini call failed after all retries")

    def _parse_llm_response(
        self, response: str
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Parse LLM response for tool calls and final actions."""
        tool_calls = []
        action = None

        lines = response.split("\n")
        i = 0

        # First pass: look for tool calls
        while i < len(lines):
            line = lines[i].strip()

            # Look for tool calls with more flexible matching
            if line.startswith("TOOL_CALL:") or "TOOL_CALL:" in line:
                tool_name = line.split("TOOL_CALL:")[-1].strip()
                i += 1

                # Look for parameters on the next line or few lines
                parameters = {}
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith("PARAMETERS:") or "PARAMETERS:" in next_line:
                        params_line = next_line.split("PARAMETERS:")[-1].strip()
                        try:
                            parameters = json.loads(params_line)
                        except json.JSONDecodeError:
                            # Try to find JSON in the next few lines
                            json_candidates = []
                            j = i
                            while j < min(i + 3, len(lines)):
                                candidate = lines[j].strip()
                                if candidate.startswith("{") and candidate.endswith("}"):
                                    try:
                                        parameters = json.loads(candidate)
                                        break
                                    except json.JSONDecodeError:
                                        pass
                                j += 1
                        break
                    elif next_line.startswith("{"):
                        # Found JSON directly without PARAMETERS: prefix
                        try:
                            parameters = json.loads(next_line)
                            break
                        except json.JSONDecodeError:
                            pass
                    i += 1

                if tool_name:
                    tool_calls.append({"tool_name": tool_name, "parameters": parameters})

            i += 1

        # Second pass: look for final action JSON
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for JSON action (more robust detection)
            if line.startswith("{") and not action:
                # Try to parse JSON action
                json_lines = []
                brace_count = 0

                # Collect all lines that form a complete JSON object
                while i < len(lines):
                    current_line = lines[i]
                    json_lines.append(current_line)

                    # Count braces to find complete JSON
                    brace_count += current_line.count("{") - current_line.count("}")

                    if brace_count == 0 and len(json_lines) > 0:
                        break
                    i += 1

                try:
                    json_str = "\n".join(json_lines)
                    parsed_action = json.loads(json_str)
                    # Validate it looks like an action
                    if isinstance(parsed_action, dict) and "action_type" in parsed_action:
                        action = parsed_action
                except json.JSONDecodeError:
                    pass

            i += 1

        return tool_calls, action

    async def _gather_historical_context(self):
        """Gather historical actions and snapshots to build enhanced context."""
        start_time = time.time()
        try:
            # Use direct KB query interface like in llm_reasoner.py
            if not self.tools.get("query_knowledge_base") or not hasattr(
                self.tools["query_knowledge_base"], "kb_query"
            ):
                self.logger.warning("Knowledge base interface not available for historical context")
                self.last_historical_context = None
                duration_ms = (time.time() - start_time) * 1000
                self.perf_logger.log_metric(
                    "historical_context_error", duration_ms, {"error": "kb_interface_not_available"}
                )
                return

            kb_query = self.tools["query_knowledge_base"].kb_query

            # Query for recent snapshots using the same pattern as llm_reasoner.py
            snapshots_start_time = time.time()
            snapshots = await kb_query.query_structured(
                data_types=["observation"],
                filters={"source": "swim_snapshotter", "tags": ["snapshot"]},
                limit=3,
            )
            snapshots_duration_ms = (time.time() - snapshots_start_time) * 1000
            self.perf_logger.log_metric(
                "historical_snapshots_query",
                snapshots_duration_ms,
                {"result_count": len(snapshots) if snapshots else 0},
            )

            # Query for recent adaptation decisions/actions
            actions_start_time = time.time()
            actions = await kb_query.query_structured(
                data_types=["adaptation_decision"],
                limit=8,
            )
            actions_duration_ms = (time.time() - actions_start_time) * 1000
            self.perf_logger.log_metric(
                "historical_actions_query",
                actions_duration_ms,
                {"result_count": len(actions) if actions else 0},
            )

            # Convert action history deque to list
            action_history_list = list(self.action_history)
            reactive_logs_list = list(self.reactive_logs)

            # Build context using ContextBuilder
            if snapshots or actions:
                context_build_start_time = time.time()
                historical_context = self.context_builder.build_context_from_kb_results(
                    snapshots, action_history_list, reactive_logs_list, actions
                )
                context_build_duration_ms = (time.time() - context_build_start_time) * 1000
                self.perf_logger.log_metric(
                    "context_building",
                    context_build_duration_ms,
                    {"snapshots_count": len(snapshots or []), "actions_count": len(actions or [])},
                )

                # Extract only the historical actions summary for the prompt
                self.last_historical_context = historical_context.get("historical_actions")
                self.logger.info(
                    f"Built historical context with {len(snapshots or [])} snapshots and {len(actions or [])} KB actions"
                )
            else:
                self.last_historical_context = None
                self.logger.info("No historical data available for context building")

            total_duration_ms = (time.time() - start_time) * 1000
            self.perf_logger.log_metric(
                "historical_context_total",
                total_duration_ms,
                {
                    "success": True,
                    "snapshots_available": len(snapshots) if snapshots else 0,
                    "actions_available": len(actions) if actions else 0,
                },
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.perf_logger.log_metric("historical_context_error", duration_ms, {"error": str(e)})
            self.logger.error(f"Failed to gather historical context: {e}")
            self.last_historical_context = None

    async def validate_input(self, context: ReasoningContext) -> bool:
        """Validate input context."""
        return True  # Basic validation - could be enhanced

    def get_required_knowledge_types(self, context: ReasoningContext) -> List[str]:
        """Get required knowledge types (not used in agentic approach)."""
        return []

    def extract_search_terms(self, context: ReasoningContext) -> List[str]:
        """Extract search terms (not used in agentic approach)."""
        return []


def create_agentic_reasoner_agent(
    agent_id: str,
    config_path: str,
    llm_api_key: str,
    nats_url: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    use_improved_grpc: bool = True,
    grpc_timeout_config: Optional[Dict[str, float]] = None,
) -> "ReasonerAgent":
    """
    Create a reasoner agent with agentic LLM reasoning implementation using Gemini Flash 2.5.

    Args:
        agent_id: Unique identifier for the agent
        config_path: Path to configuration file
        llm_api_key: API key for Gemini LLM
        nats_url: NATS server URL
        logger: Logger instance
        use_improved_grpc: Whether to use the improved GRPC client
        grpc_timeout_config: Custom timeout configuration for GRPC client
    """
    from .reasoner_agent import ReasonerAgent

    # Create agent with custom GRPC client if requested
    if use_improved_grpc:
        # Load configuration to get GRPC address
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        grpc_address = (
            config.get("digital_twin", {}).get("grpc", {}).get("address", "localhost:50051")
        )

        # Default timeout configuration
        default_timeouts = {
            "query_timeout": 20.0,
            "simulation_timeout": 90.0,
            "diagnosis_timeout": 45.0,
            "connection_timeout": 15.0,
            "default_timeout": 30.0,
        }

        if grpc_timeout_config:
            default_timeouts.update(grpc_timeout_config)

        # Create improved GRPC client
        improved_dt_client = ImprovedGRPCDigitalTwinClient(
            grpc_address=grpc_address,
            logger=logger,
            **default_timeouts,
            max_retries=3,
            circuit_breaker_enabled=True,
            failure_threshold=5,
            recovery_timeout=60.0,
        )

        # Create agent with standard constructor
        agent = ReasonerAgent(
            agent_id=agent_id,
            reasoning_implementations={},
            config_path=config_path,
            nats_url=nats_url,
            logger=logger,
        )

        # Replace the default GRPC client with improved one
        if agent.dt_query:
            agent.dt_query = improved_dt_client
            if logger:
                logger.info("Replaced default GRPC client with improved version")
    else:
        # Use default agent creation
        agent = ReasonerAgent(
            agent_id=agent_id,
            reasoning_implementations={},
            config_path=config_path,
            nats_url=nats_url,
            logger=logger,
        )

    # Log available interfaces
    if logger:
        logger.info(f"Creating agentic reasoner with KB interface: {agent.kb_query is not None}")
        logger.info(f"Creating agentic reasoner with DT interface: {agent.dt_query is not None}")
        if agent.dt_query:
            grpc_addr = getattr(agent.dt_query, "grpc_address", "unknown")
            client_type = (
                "ImprovedGRPCDigitalTwinClient" if use_improved_grpc else "GRPCDigitalTwinClient"
            )
            logger.info(f"Digital Twin client: {client_type} at {grpc_addr}")

    # Create agentic reasoner for each reasoning type
    # Import the ReasoningType that the agent actually uses (local definition in reasoner_agent.py)
    from .reasoner_agent import ReasoningType as AgentReasoningType

    # Get the prompt config path for the agentic reasoner
    agentic_prompt_config_path = os.path.join(
        os.path.dirname(__file__), "agentic_reasoner_prompt.yaml"
    )

    for reasoning_type in AgentReasoningType:
        agentic_reasoner = AgenticLLMReasoner(
            api_key=llm_api_key,
            # reasoning_type=reasoning_type,
            config=config,
            kb_query_interface=agent.kb_query,
            dt_interface=agent.dt_query,
            prompt_config_path=agentic_prompt_config_path,
            logger=logger,
            system_name="switch_yolo",
        )

        agent.add_reasoning_implementation(reasoning_type, agentic_reasoner)
        if logger:
            logger.info(f"Added reasoning implementation for {reasoning_type.value}")

    return agent


def create_agentic_reasoner_with_bayesian_world_model(
    agent_id: str,
    config_path: str,
    llm_api_key: str,
    nats_url: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> "ReasonerAgent":
    """
    Create a reasoner agent that uses the Bayesian/Kalman filter world model
    instead of the Gemini LLM world model for deterministic predictions.
    """
    # Update config to use Bayesian world model
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify world model configuration
    if "world_model" not in config:
        config["world_model"] = {}

    config["world_model"]["implementation"] = "bayesian"
    config["world_model"]["config"] = {
        "prediction_horizon_minutes": 120,
        "max_history_points": 2000,
        "correlation_threshold": 0.7,
        "anomaly_threshold": 2.5,
        "process_noise": 0.01,
        "measurement_noise": 0.1,
        "learning_rate": 0.05,
    }

    # Ensure digital twin configuration exists for GRPC client
    if "digital_twin" not in config:
        config["digital_twin"] = {}
    if "grpc" not in config["digital_twin"]:
        config["digital_twin"]["grpc"] = {}

    # Set default GRPC configuration if not present
    grpc_config = config["digital_twin"]["grpc"]
    if "host" not in grpc_config:
        grpc_config["host"] = "localhost"
    if "port" not in grpc_config:
        grpc_config["port"] = 50051

    # Save modified config temporarily
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    try:
        # Create standard reasoner agent (it will use the modified config)
        agent = ReasonerAgent(
            agent_id=agent_id,
            reasoning_implementations={},
            config_path=temp_config_path,
            nats_url=nats_url,
            logger=logger,
        )

        # Replace the default GRPC client with improved one if possible
        if agent.dt_query:
            try:
                grpc_address = f"{grpc_config['host']}:{grpc_config['port']}"
                improved_client = ImprovedGRPCDigitalTwinClient(
                    grpc_address=grpc_address,
                    logger=logger,
                    query_timeout=30.0,
                    simulation_timeout=120.0,
                    diagnosis_timeout=60.0,
                    max_retries=3,
                    circuit_breaker_enabled=True,
                )
                agent.dt_query = improved_client
                if logger:
                    logger.info("Replaced default GRPC client with improved version")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not create improved GRPC client, using default: {e}")

        # Create agentic reasoner for each reasoning type
        # Import the ReasoningType that the agent actually uses (local definition in reasoner_agent.py)
        from .reasoner_agent import ReasoningType as AgentReasoningType

        # Get the prompt config path for the agentic reasoner
        agentic_prompt_config_path = os.path.join(
            os.path.dirname(__file__), "agentic_reasoner_prompt.yaml"
        )

        for reasoning_type in AgentReasoningType:
            agentic_reasoner = AgenticLLMReasoner(
                api_key=llm_api_key,
                reasoning_type=reasoning_type,
                config=config,
                kb_query_interface=agent.kb_query,
                dt_interface=agent.dt_query,
                prompt_config_path=agentic_prompt_config_path,
                logger=logger,
            )
            agent.add_reasoning_implementation(reasoning_type, agentic_reasoner)
            if logger:
                logger.info(f"Added reasoning implementation for {reasoning_type.value}")

        if logger:
            logger.info("Created agentic reasoner with Bayesian/Kalman filter world model")
            logger.info(f"Digital Twin will use Bayesian world model at {grpc_address}")

        return agent

    finally:
        # Clean up temporary config file
        try:
            os.unlink(temp_config_path)
        except Exception:
            pass
