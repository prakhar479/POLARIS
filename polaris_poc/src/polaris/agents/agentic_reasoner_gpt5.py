"""
Agentic LLM Reasoner Implementation with OpenAI GPT

An autonomous agentic reasoner that can dynamically decide which tools to use
(Knowledge Base queries, Digital Twin interactions) and make adaptation decisions
based on its analysis. Uses OpenAI GPT as the reasoning engine.
"""

import json
import asyncio
import time
from urllib import response
import uuid
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime, timezone, timedelta
import tempfile
import yaml
from pathlib import Path

from openai import AsyncOpenAI

from .reasoner_core import (
    ReasoningInterface,
    ReasoningContext,
    ReasoningResult,
    ReasoningType,
)

from collections import deque

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


class TokenUsageLogger:
    """Logs average token usage per reasoning decision."""

    def __init__(self, log_dir: str = "."):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "token_usage.jsonl"

    def log_average_tokens(
        self, avg_input_tokens: float, avg_output_tokens: float, details: Dict[str, Any]
    ):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "avg_input_tokens": round(avg_input_tokens, 2),
            "avg_output_tokens": round(avg_output_tokens, 2),
            "avg_reasoning_tokens": details.get("avg_reasoning_tokens", 0),
            "details": details,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


class PerformanceLogger:
    """Simple performance logger for overhead metrics."""

    def __init__(self, log_dir: str = "logs_gpt/overhead"):
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

                for action in raw_actions:
                    if isinstance(action, str):
                        # Convert string action to proper action dictionary
                        action_dict = {
                            "action_id": str(uuid.uuid4()),
                            "action_type": action,
                            "target": "system",
                            "params": {},
                        }
                        formatted_actions.append(action_dict)
                    elif isinstance(action, dict):
                        # Already a dictionary, use as-is
                        formatted_actions.append(action)
                    else:
                        self.logger.warning(f"Unknown action format: {action}")

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
            "goals": {"target_utilization": 0.8, "target_response_time_ms_weighted": 750.0},
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
    Agentic LLM reasoner that can dynamically decide which tools to use
    and make autonomous adaptation decisions using OpenAI GPT.
    """

    def __init__(
        self,
        api_key: str,
        reasoning_type: ReasoningType,
        kb_query_interface: Optional[KnowledgeQueryInterface] = None,
        dt_interface: Optional[DigitalTwinInterface] = None,
        model: str = "gpt-5",
        max_completion_tokens: int = 8192,
        temperature: float = 0.3,
        max_tool_calls: int = 5,
        prompt_config_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = "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"
        self.reasoning_type = reasoning_type
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.max_tool_calls = max_tool_calls
        self.max_retries = 3
        self.prompt_config_path = prompt_config_path  # Store for reloading
        self.logger = logger or logging.getLogger(f"AgenticReasoner.{reasoning_type.value}")

        # Initialize performance logger
        self.perf_logger = PerformanceLogger()
        self.token_logger = TokenUsageLogger()

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Initialize context builder and action history tracking
        self.context_builder = ContextBuilder(self.logger, {})
        self.action_history = deque(maxlen=20)  # Store the last 20 actions
        # Local short history maintained for fast context (last 3 actions)
        self.local_action_history = deque(maxlen=3)
        self.reactive_logs = deque(maxlen=3)  # Store the last 3 reactive logs
        self.last_historical_context = None  # Cache for historical actions context

        # Initialize tools
        self.tools = {}
        if kb_query_interface:
            self.tools["query_knowledge_base"] = KnowledgeBaseTool(
                kb_query_interface, self.logger, self.perf_logger
            )
            self.logger.info("Knowledge Base tool initialized")
        if dt_interface:
            self.tools["query_digital_twin"] = DigitalTwinTool(
                dt_interface, self.logger, self.perf_logger
            )
            self.logger.info("Digital Twin tool initialized")

        if not self.tools:
            self.logger.warning(
                "No tools available - agentic reasoner will have limited capabilities"
            )
        else:
            self.logger.info(
                f"Agentic reasoner initialized with {len(self.tools)} tools: {list(self.tools.keys())}"
            )

        # System prompt for the agentic reasoner
        self.prompt_config = self._load_prompt_config()
        self.system_prompt = self._build_system_prompt()

        self.logger.info(
            f"Initialized AgenticLLMReasoner with GPT and {len(self.tools)} tools available"
        )

    def _load_prompt_config(self) -> Dict[str, Any]:
        """Load prompt configuration from YAML file."""
        try:
            # Use stored prompt_config_path if available, otherwise use default
            if self.prompt_config_path:
                prompt_file_path = self.prompt_config_path
            else:
                # Get the directory where this file is located
                current_dir = os.path.dirname(__file__)
                prompt_file_path = os.path.join(current_dir, "agentic_reasoner_prompt.yaml")

            with open(prompt_file_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                return config.get("prompt_config", {})
        except Exception as e:
            self.logger.error(f"Failed to load prompt configuration: {e}")
            # Return a minimal fallback configuration
            return {
                "fixed_constraints": {},
                "thresholds": {},
                "template_parts": {},
                "template": "You are an autonomous adaptive system controller.",
            }

    def reload_prompt_config(self) -> bool:
        """Reload prompt configuration from file."""
        if not self.prompt_config_path:
            self.logger.warning("No prompt_config_path set, cannot reload config")
            return False

        try:
            self.logger.info(f"Reloading prompt config from {self.prompt_config_path}")
            old_config = self.prompt_config.copy()

            # Reload the config
            self.prompt_config = self._load_prompt_config()

            # Rebuild the system prompt with new config
            self.system_prompt = self._build_system_prompt()

            self.logger.info(f"✅ Prompt config reloaded successfully")
            self.logger.info(f"   Old config keys: {list(old_config.keys())}")
            self.logger.info(f"   New config keys: {list(self.prompt_config.keys())}")

            # Log specific changes
            if "thresholds" in old_config and "thresholds" in self.prompt_config:
                old_thresholds = old_config.get("thresholds", {})
                new_thresholds = self.prompt_config.get("thresholds", {})
                if old_thresholds != new_thresholds:
                    self.logger.info(f"   Thresholds updated: {old_thresholds} -> {new_thresholds}")

            if "template_parts" in old_config and "template_parts" in self.prompt_config:
                old_template_parts = old_config.get("template_parts", {})
                new_template_parts = self.prompt_config.get("template_parts", {})
                if old_template_parts != new_template_parts:
                    self.logger.info(
                        f"   Template parts updated: {list(new_template_parts.keys())}"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Failed to reload prompt config: {e}", exc_info=True)
            return False

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agentic reasoner using YAML configuration."""
        try:
            # Get configuration sections
            fixed_constraints = self.prompt_config.get("fixed_constraints", {})
            thresholds = self.prompt_config.get("thresholds", {})
            template_parts = self.prompt_config.get("template_parts", {})
            template = self.prompt_config.get("template", "")

            # Build tools description
            tools_description = ""
            standalone_mode = ""
            if self.tools:
                tools_description = "\n\nAvailable Tools:\n"
                for tool_name, tool in self.tools.items():
                    tools_description += f"- {tool_name}: {tool.description}\n"
                    tools_description += f"  Parameters: {json.dumps(tool.parameters, indent=2)}\n"
            else:
                tools_description = "\n\nNote: No external tools are currently available. You must make decisions based solely on the provided input data.\n"
                standalone_mode = " operating in standalone mode"

            # Prepare all template variables by combining constraints and thresholds
            template_vars = {}
            template_vars.update(fixed_constraints)
            template_vars.update(thresholds)
            template_vars["tools_description"] = tools_description
            template_vars["standalone_mode"] = standalone_mode

            # Format each template part with variables
            formatted_parts = {}
            for part_name, part_content in template_parts.items():
                try:
                    formatted_parts[part_name] = part_content.format(**template_vars)
                except KeyError as e:
                    self.logger.warning(
                        f"Missing variable {e} in template part '{part_name}', using unformatted content"
                    )
                    formatted_parts[part_name] = part_content

            # Format the main template with all parts and variables
            all_template_vars = {}
            all_template_vars.update(template_vars)
            all_template_vars.update(formatted_parts)

            try:
                return template.format(**all_template_vars)
            except KeyError as e:
                self.logger.error(f"Missing variable {e} in main template, using fallback")
                return self._build_fallback_prompt(tools_description, standalone_mode)

        except Exception as e:
            self.logger.error(f"Failed to build system prompt from YAML: {e}")
            return self._build_fallback_prompt("", " operating in standalone mode")

    def _build_fallback_prompt(self, tools_description: str, standalone_mode: str) -> str:
        """Build a fallback system prompt if YAML loading fails."""
        return f"""You are an autonomous adaptive system controller{standalone_mode} for making adaptation decisions.

Your role is to:
1. Analyze the current system situation
2. Use available tools to gather additional information as needed
3. Make informed adaptation decisions based on your analysis
4. Generate appropriate control actions

{tools_description}

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

IMPORTANT: Action Cooldowns
- ADD_SERVER and REMOVE_SERVER actions have a 2-minute cooldown period
- Check the 'cooldown_status' in the historical_actions_summary to see if cooldowns have expired
- If cooldown_expired is false, DO NOT take that action type
- The cooldown_remaining_minutes field shows exactly how much time is left

Time Calculation:
- All timestamps are in UTC format
- Use the provided cooldown_status calculations instead of manually calculating time differences
- The system automatically calculates minutes_since_last_action for you

Please analyze the system context and make adaptation decisions in JSON format:
{{
  "action_type": "ACTION_TYPE",
  "source": "agentic_reasoner", 
  "action_id": "generated-uuid",
  "params": {{}},
  "priority": "low|medium|high",
  "reasoning": "Brief explanation of the decision"
}}
"""

    async def reason(
        self, context: ReasoningContext, knowledge: Optional[List[Dict[str, Any]]] = None
    ) -> ReasoningResult:
        """Execute agentic reasoning with dynamic tool usage."""
        overall_start_time = time.time()
        reasoning_steps = ["Starting agentic LLM reasoning with GPT"]
        tool_calls_made = 0

        # Performance tracking
        tool_call_timings = []
        llm_call_timings = []
        token_usage_records = []

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

            # Start the reasoning loop with messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            final_action = None

            for iteration in range(self.max_tool_calls + 1):  # +1 for final decision
                reasoning_steps.append(f"Reasoning iteration {iteration + 1}")

                # Get LLM response
                llm_start_time = time.time()
                llm_response, inp, op, reasoning_tokens = await self._call_gpt(messages)
                token_usage_records.append((inp, op, reasoning_tokens))
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
                reasoning_steps.append(f"Received GPT response (iteration {iteration + 1})")
                self.logger.debug(f"GPT Response (iteration {iteration + 1}): {llm_response}")

                # Add assistant response to history
                messages.append({"role": "assistant", "content": llm_response})

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
                        messages.append({"role": "user", "content": tool_results_text})
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
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please provide your final decision and action in the required JSON format.",
                        }
                    )
                    llm_start_time = time.time()
                    final_response, inp, op, reasoning_tokens = await self._call_gpt(messages)
                    token_usage_records.append((inp, op, reasoning_tokens))
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

            # Append final action to local short history (last 3 actions)
            try:
                record = {
                    "action_type": final_action.get("action_type"),
                    "action_id": final_action.get("action_id"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "priority": final_action.get("priority"),
                    "reasoning": final_action.get("reasoning"),
                }
                # Keep a very small human-readable snapshot in local history
                self.local_action_history.append(record)

                # Also push a lightweight entry into the longer action_history for persistence/debugging
                try:
                    self.action_history.append(
                        {
                            "timestamp": record["timestamp"],
                            "content": {
                                "action_type": record["action_type"],
                                "action_id": record["action_id"],
                                "priority": record["priority"],
                                "reasoning": record["reasoning"],
                            },
                        }
                    )
                except Exception:
                    # Non-fatal - don't let history append failures break reasoning
                    pass
            except Exception:
                # ignore history failures
                pass

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

            # --- Average token usage logging ---
            if token_usage_records:
                avg_input = sum(i for i, _, _ in token_usage_records) / len(token_usage_records)
                avg_output = sum(o for _, o, _ in token_usage_records) / len(token_usage_records)
                avg_reasoning = sum(r for _, _, r in token_usage_records) / len(token_usage_records)
                self.token_logger.log_average_tokens(
                    avg_input,
                    avg_output,
                    {
                        "model": self.model,
                        "decision_id": context.session_id,
                        "reasoning_type": self.reasoning_type.value,
                        "llm_calls_made": len(token_usage_records),
                        "avg_reasoning_tokens": round(avg_reasoning, 2),
                    },
                )

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

        # Add local short action history (most recent 3 actions) to the prompt context
        if hasattr(self, "local_action_history") and len(self.local_action_history) > 0:
            # convert deque entries to list for serialization
            try:
                enhanced_context["local_action_history"] = list(self.local_action_history)
            except Exception:
                enhanced_context["local_action_history"] = []

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

    async def _call_gpt(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int, int]:
        """Call GPT API with message history. Returns (response_text, input_tokens, output_tokens, reasoning_tokens)."""
        for attempt in range(self.max_retries):
            call_start_time = time.time()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                    reasoning_effort="medium",
                )

                call_duration_ms = (time.time() - call_start_time) * 1000

                # Extract text from response
                if response and response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                        input_tokens = response.usage.prompt_tokens if response.usage else 0
                        output_tokens = response.usage.completion_tokens if response.usage else 0

                        # Extract reasoning tokens from completion_tokens_details if available
                        reasoning_tokens = 0
                        if (
                            response.usage
                            and hasattr(response.usage, "completion_tokens_details")
                            and response.usage.completion_tokens_details
                            and hasattr(
                                response.usage.completion_tokens_details, "reasoning_tokens"
                            )
                        ):
                            reasoning_tokens = (
                                response.usage.completion_tokens_details.reasoning_tokens
                            )

                        self.perf_logger.log_metric(
                            "gpt_api_call",
                            call_duration_ms,
                            {
                                "attempt": attempt + 1,
                                "success": True,
                                "response_length": len(content),
                                "model": self.model,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "reasoning_tokens": reasoning_tokens,
                            },
                        )
                        return content.strip(), input_tokens, output_tokens, reasoning_tokens

                self.perf_logger.log_metric(
                    "gpt_api_call",
                    call_duration_ms,
                    {
                        "attempt": attempt + 1,
                        "success": False,
                        "error": "empty_response",
                        "model": self.model,
                    },
                )
                raise ValueError("Empty response from GPT")

            except Exception as e:
                call_duration_ms = (time.time() - call_start_time) * 1000
                self.perf_logger.log_metric(
                    "gpt_api_call",
                    call_duration_ms,
                    {
                        "attempt": attempt + 1,
                        "success": False,
                        "error": str(e),
                        "model": self.model,
                    },
                )
                self.logger.warning(f"GPT call attempt {attempt + 1} failed: {e}")
                if attempt + 1 == self.max_retries:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

        raise Exception("GPT call failed after all retries")

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
    Create a reasoner agent with agentic LLM reasoning implementation using OpenAI GPT.

    Args:
        agent_id: Unique identifier for the agent
        config_path: Path to configuration file
        llm_api_key: API key for OpenAI GPT LLM
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
            reasoning_type=reasoning_type,
            kb_query_interface=agent.kb_query,
            dt_interface=agent.dt_query,
            prompt_config_path=agentic_prompt_config_path,
            logger=logger,
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
    instead of the GPT LLM world model for deterministic predictions.
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
