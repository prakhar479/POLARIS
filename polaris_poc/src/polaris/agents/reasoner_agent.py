"""
Reasoner Agent Architecture with NATS Base Class and Reasoning Interfaces.

This module provides a clean separation between NATS communication (base class)
and reasoning implementation (interfaces to be implemented later).
It also includes a gRPC client for communicating with the Digital Twin agent.
"""

import json
import asyncio
import time
import uuid
import yaml
import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
import nats
from nats.errors import TimeoutError
import grpc
from .reasoner_core import (
    ReasoningType,
    ReasoningContext,
    ReasoningResult,
    ReasoningInterface,
)

# Assuming these are your existing models
from polaris.knowledge_base.models import KBEntry, KBQuery, KBResponse

# Proto-generated files for Digital Twin gRPC communication.
from polaris.proto import digital_twin_pb2, digital_twin_pb2_grpc


class ReasoningType(Enum):
    """Types of reasoning operations."""

    INFERENCE = "inference"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    DECISION = "decision"
    PREDICTION = "prediction"


@dataclass
class ReasoningContext:
    """Context information for reasoning operations."""

    session_id: str
    timestamp: float
    input_data: Dict[str, Any]
    reasoning_type: ReasoningType
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""

    result: Any
    confidence: float
    reasoning_steps: List[str]
    context: ReasoningContext
    execution_time: float
    kb_queries_made: int = 0
    dt_queries_made: int = 0

    def to_dict(self) -> Dict[str, Any]:
        context_dict = asdict(self.context)
        if isinstance(context_dict.get("reasoning_type"), ReasoningType):
            context_dict["reasoning_type"] = context_dict["reasoning_type"].value

        return {
            "result": self.result,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "context": context_dict,
            "execution_time": self.execution_time,
            "kb_queries_made": self.kb_queries_made,
            "dt_queries_made": self.dt_queries_made,
        }


# ===============================================================
# == DIGITAL TWIN INTERFACES (Updated based on digital_twin_probe.py)
# ===============================================================


@dataclass
class DTQuery:
    """Represents a query to the Digital Twin."""

    query_content: str
    query_type: str = "natural_language"  # e.g., current_state, historical
    parameters: Optional[Dict[str, Any]] = None
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DTSimulation:
    """Represents a simulation request to the Digital Twin."""

    simulation_type: str  # e.g., forecast, what_if, scenario
    actions: List[Dict[str, Any]]
    horizon_minutes: int = 60
    parameters: Optional[Dict[str, Any]] = None
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DTDiagnosis:
    """Represents a diagnosis request to the Digital Twin."""

    anomaly_description: str
    context: Optional[Dict[str, Any]] = None
    diagnosis_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DTResponse:
    """Represents a generic response from the Digital Twin."""

    success: bool
    result: Optional[Any] = None
    confidence: float = 0.0
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # More specific fields for different response types
    future_states: Optional[List[Dict[str, Any]]] = None
    hypotheses: Optional[List[Dict[str, Any]]] = None


class DigitalTwinInterface(ABC):
    """Abstract interface for querying the Digital Twin."""

    @abstractmethod
    async def query(self, query: DTQuery) -> Optional[DTResponse]:
        """Send a query to the Digital Twin."""
        pass

    @abstractmethod
    async def simulate(self, simulation: DTSimulation) -> Optional[DTResponse]:
        """Run a simulation in the Digital Twin."""
        pass

    @abstractmethod
    async def diagnose(self, diagnosis: DTDiagnosis) -> Optional[DTResponse]:
        """Request a diagnosis from the Digital Twin."""
        pass

    @abstractmethod
    async def connect(self):
        """Connect to the Digital Twin service."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the Digital Twin service."""
        pass


# ===============================================================
# == REASONING AND KNOWLEDGE BASE INTERFACES
# ===============================================================


class ReasoningInterface(ABC):
    """Interface for reasoning implementations."""

    @abstractmethod
    async def reason(
        self, context: ReasoningContext, knowledge: Optional[List[Dict[str, Any]]] = None
    ) -> ReasoningResult:
        pass

    @abstractmethod
    async def validate_input(self, context: ReasoningContext) -> bool:
        pass

    @abstractmethod
    def get_required_knowledge_types(self, context: ReasoningContext) -> List[str]:
        pass

    @abstractmethod
    def extract_search_terms(self, context: ReasoningContext) -> List[str]:
        pass


class KnowledgeQueryInterface(ABC):
    """Interface for knowledge base query operations."""

    @abstractmethod
    async def query_structured(
        self,
        data_types: List[str],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Optional[List[Dict[str, Any]]]:
        pass

    @abstractmethod
    async def query_natural_language(
        self, query_text: str, limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        pass

    @abstractmethod
    async def query_recent_observations(self, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        pass

    @abstractmethod
    async def query_raw_telemetry(self, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        pass

    @abstractmethod
    async def store_reasoning_result(
        self, context: ReasoningContext, result: Any, confidence: float
    ) -> bool:
        pass


class GRPCDigitalTwinClient(DigitalTwinInterface):
    """gRPC implementation for querying the Digital Twin."""

    def __init__(self, grpc_address: str, logger: Optional[logging.Logger] = None):
        self.grpc_address = grpc_address
        self.logger = logger or logging.getLogger(f"DigitalTwinClient")
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[digital_twin_pb2_grpc.DigitalTwinStub] = None

    async def connect(self):
        """Establish the gRPC connection."""
        if not self.channel:
            self.logger.info(f"Connecting to Digital Twin gRPC server at {self.grpc_address}")
            try:
                self.channel = grpc.aio.insecure_channel(self.grpc_address)
                self.stub = digital_twin_pb2_grpc.DigitalTwinStub(self.channel)

                # Test the connection with a simple health check
                await self._test_connection()
                self.logger.info("Digital Twin gRPC connection established successfully")
            except Exception as e:
                self.logger.error(f"Failed to connect to Digital Twin: {e}")
                if self.channel:
                    await self.channel.close()
                    self.channel = None
                    self.stub = None
                raise

    async def disconnect(self):
        """Close the gRPC connection."""
        if self.channel:
            self.logger.info("Disconnecting from Digital Twin gRPC server.")
            await self.channel.close()
            self.channel = None
            self.stub = None

    async def query(self, query: DTQuery) -> Optional[DTResponse]:
        if not self.stub:
            self.logger.error("Digital Twin client not connected.")
            return None

        request_pb = digital_twin_pb2.QueryRequest(
            query_id=query.query_id,
            query_type=query.query_type,
            query_content=query.query_content,
            parameters={k: str(v) for k, v in (query.parameters or {}).items()},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        try:
            response_pb = await self.stub.Query(request_pb, timeout=10.0)
            # Parse result - it might be JSON string or plain string
            result = None
            if response_pb.result:
                try:
                    result = json.loads(response_pb.result)
                except json.JSONDecodeError:
                    # If it's not valid JSON, use as plain string
                    result = response_pb.result

            return DTResponse(
                success=response_pb.success,
                result=result,
                confidence=response_pb.confidence,
                explanation=response_pb.explanation,
                metadata=dict(response_pb.metadata),
            )
        except grpc.aio.AioRpcError as e:
            self.logger.error(f"gRPC Query call failed: {e.details()}", extra={"code": e.code()})
            return None
        except Exception as e:
            self.logger.error(f"Error processing gRPC Query response: {e}", exc_info=True)
            return None

    async def simulate(self, simulation: DTSimulation) -> Optional[DTResponse]:
        if not self.stub:
            self.logger.error("Digital Twin client not connected.")
            return None

        pb_actions = []
        for action in simulation.actions:
            pb_action = digital_twin_pb2.ControlAction(
                action_id=action.get("action_id", str(uuid.uuid4())),
                action_type=action.get("action_type", ""),
                target=action.get("target", ""),
                params={k: str(v) for k, v in action.get("params", {}).items()},
            )
            pb_actions.append(pb_action)

        request_pb = digital_twin_pb2.SimulationRequest(
            simulation_id=simulation.simulation_id,
            simulation_type=simulation.simulation_type,
            actions=pb_actions,
            horizon_minutes=simulation.horizon_minutes,
            parameters={k: str(v) for k, v in (simulation.parameters or {}).items()},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        try:
            response_pb = await self.stub.Simulate(request_pb, timeout=30.0)
            # Parse future states properly from the protobuf response
            future_states = []
            for state in response_pb.future_states:
                future_state = {
                    "timestamp": state.timestamp,
                    "metrics": dict(state.metrics),
                    "confidence": state.confidence,
                    "description": state.description,
                }
                future_states.append(future_state)

            return DTResponse(
                success=response_pb.success,
                confidence=response_pb.confidence,
                explanation=response_pb.explanation,
                metadata=dict(response_pb.metadata),
                future_states=future_states,
            )
        except grpc.aio.AioRpcError as e:
            self.logger.error(f"gRPC Simulate call failed: {e.details()}", extra={"code": e.code()})
            return None
        except Exception as e:
            self.logger.error(f"Error processing gRPC Simulate response: {e}", exc_info=True)
            return None

    async def diagnose(self, diagnosis: DTDiagnosis) -> Optional[DTResponse]:
        if not self.stub:
            self.logger.error("Digital Twin client not connected.")
            return None

        request_pb = digital_twin_pb2.DiagnosisRequest(
            diagnosis_id=diagnosis.diagnosis_id,
            anomaly_description=diagnosis.anomaly_description,
            context={k: str(v) for k, v in (diagnosis.context or {}).items()},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        try:
            response_pb = await self.stub.Diagnose(request_pb, timeout=20.0)
            hypotheses = [
                {
                    "hypothesis": h.hypothesis,
                    "probability": h.probability,
                    "reasoning": h.reasoning,
                    "evidence": list(h.evidence),
                }
                for h in response_pb.hypotheses
            ]
            return DTResponse(
                success=response_pb.success,
                confidence=response_pb.confidence,
                explanation=response_pb.explanation,
                metadata=dict(response_pb.metadata),
                hypotheses=hypotheses,
            )
        except grpc.aio.AioRpcError as e:
            self.logger.error(f"gRPC Diagnose call failed: {e.details()}", extra={"code": e.code()})
            return None
        except Exception as e:
            self.logger.error(f"Error processing gRPC Diagnose response: {e}", exc_info=True)
            return None

    async def _test_connection(self):
        """Test the gRPC connection with a simple management request."""
        if not self.stub:
            raise Exception("gRPC stub not initialized")

        try:
            # Send a simple health check request
            request = digital_twin_pb2.ManagementRequest(
                request_id=str(uuid.uuid4()),
                operation="health_check",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Short timeout for connection test
            response = await self.stub.Manage(request, timeout=5.0)

            if not response.success:
                self.logger.warning(f"Digital Twin health check returned: {response.result}")

        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                # Health check not implemented, but connection works
                self.logger.debug(
                    "Digital Twin health check not implemented, but connection is working"
                )
            else:
                raise Exception(f"gRPC connection test failed: {e.details()}")
        except Exception as e:
            raise Exception(f"Connection test failed: {e}")


class NATSReasonerBase(ABC):
    """Base class providing NATS communication functionality for reasoner agents."""

    def __init__(
        self,
        agent_id: str,
        config_path: str,
        nats_url: Optional[str] = None,
        kb_request_timeout: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.agent_id = agent_id
        self.kb_request_timeout = kb_request_timeout
        self.nc: Optional[nats.NATS] = None
        self.active_sessions: Dict[str, "ReasoningContext"] = {}
        self.reasoning_history: List["ReasoningResult"] = []

        # === Load configuration ===
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Apply environment variable overrides
        self._apply_env_overrides(self.config)

        # NATS URL (config > arg > default)
        self.nats_url = nats_url or self.config.get("nats", {}).get("url", "nats://localhost:4222")

        # Subjects from config
        self.telemetry_subject = self.config.get("telemetry", {}).get(
            "stream_subject", "polaris.telemetry.events.stream"
        )
        self.execution_action_subject = self.config.get("execution", {}).get(
            "action_subject", "polaris.execution.actions"
        )
        self.reasoner_kernel_subject = self.config.get("reasoner", {}).get(
            "kernel_request_subject", "polaris.reasoner.kernel.requests"
        )

        # Verification routing configuration
        reasoner_config = self.config.get("reasoner", {})
        action_routing = reasoner_config.get("action_routing", {})
        self.enable_verification = False
        self.verification_level = action_routing.get("default_verification_level", "policy")
        self.verification_timeout = action_routing.get("verification_timeout_sec", 45)
        self.verification_failure_action = action_routing.get(
            "verification_failure_action", "reject"
        )

        # Verification subjects
        verification_config = self.config.get("verification", {})
        self.verification_input_subject = verification_config.get(
            "input_subject", "polaris.verification.requests"
        )
        self.verification_output_subject = verification_config.get(
            "output_subject", "polaris.verification.results"
        )

        # Telemetry KB config
        self.telemetry_kb_enabled = (
            self.config.get("telemetry", {}).get("knowledge_base", {}).get("enabled", False)
        )
        self.telemetry_kb_buffer_size = (
            self.config.get("telemetry", {}).get("knowledge_base", {}).get("buffer_size", 100)
        )

        # Setup logger
        if logger is None:
            log_name = self.config.get("logger", {}).get("name", f"Reasoner.{agent_id}")
            self.logger = logging.getLogger(log_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                fmt = self.config.get("logger", {}).get("format", "pretty")
                if fmt == "json":
                    formatter = logging.Formatter(
                        json.dumps(
                            {
                                "time": "%(asctime)s",
                                "name": "%(name)s",
                                "level": "%(levelname)s",
                                "msg": "%(message)s",
                            }
                        )
                    )
                else:
                    formatter = logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

            log_level = self.config.get("logger", {}).get("level", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        else:
            self.logger = logger

    def _apply_env_overrides(self, config: dict, prefix: str = "") -> None:
        """Recursively override config with environment variables."""
        for key, value in config.items():
            env_key = (prefix + key).upper().replace(".", "_")
            if isinstance(value, dict):
                self._apply_env_overrides(value, prefix=env_key + "_")
            else:
                if env_key in os.environ:
                    config[key] = os.environ[env_key]

    async def connect(self) -> None:
        """Connect to NATS server and set up subscriptions."""
        self.nc = await nats.connect(self.nats_url)
        await self._setup_subscriptions()
        self.logger.info(f"Reasoner {self.agent_id} connected to NATS ({self.nats_url})")

    async def disconnect(self) -> None:
        """Disconnect from NATS server."""
        if self.nc:
            await self.nc.close()
        self.logger.info(f"Reasoner {self.agent_id} disconnected from NATS")

    async def _setup_subscriptions(self) -> None:
        """Set up default subscriptions for the reasoner."""
        await self.listen("system.notifications", self._handle_system_notification)
        await self.listen(self.reasoner_kernel_subject, self._handle_kernel_request)

    async def publish(self, topic: str, message: dict) -> None:
        """Publish a message to a specific topic."""
        if not self.nc:
            raise RuntimeError("Not connected to NATS")
        await self.nc.publish(topic, json.dumps(message).encode())

    async def listen(self, topic: str, callback: Callable) -> None:
        """Listen for messages on a specific topic."""
        if not self.nc:
            raise RuntimeError("Not connected to NATS")

        async def message_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                await callback(data, msg)
            except Exception as e:
                self.logger.error(f"Error processing message on topic {topic}: {e}", exc_info=True)

        await self.nc.subscribe(topic, cb=message_handler)

    async def _handle_kernel_request(self, data: Dict[str, Any], msg) -> None:
        """Handle telemetry reasoning requests from the kernel."""
        # self.logger.info(f"Received kernel telemetry request with data: {data}")
        try:
            context = ReasoningContext(
                session_id=data.get("session_id", str(uuid.uuid4())),
                timestamp=time.time(),
                input_data=data,
                reasoning_type=ReasoningType(data.get("reasoning_type", "inference")),
                metadata=data.get("metadata", {}),
            )

            # self.logger.info(f"Starting reasoning session {context.session_id} with input: {context.input_data}")

            result = await self.perform_reasoning(context)

            # Handle case where result might already be a dict (e.g., from meta-learner)
            if hasattr(result, "to_dict"):
                result = result.to_dict()
            elif isinstance(result, dict):
                # Result is already a dict, wrap it in the expected format
                result = {
                    "result": result,
                    "confidence": 0.5,
                    "reasoning_steps": ["Meta-learner operation"],
                    "context": context.to_dict() if hasattr(context, "to_dict") else str(context),
                    "execution_time": 0.0,
                }

            self.logger.info(
                f"Reasoning session {context.session_id} completed with result: {result['result']}"
            )
            # Placeholder for action generation logic
            action_message = result["result"]

            if self.enable_verification:
                self.logger.info(
                    f"Publishing reasoning action to verification layer",
                    extra={"action": action_message, "verification_level": self.verification_level},
                )
                await self._publish_action_for_verification(action_message)
            else:
                self.logger.info(
                    f"Publishing reasoning action directly to execution layer (verification disabled)",
                    extra={"action": action_message},
                )
                await self.publish(self.execution_action_subject, action_message)

            await msg.respond(json.dumps({"success": True, "result": result}).encode())

        except Exception as e:
            self.logger.error(f"Kernel request failed: {e}", exc_info=True)
            await msg.respond(json.dumps({"success": False, "error": str(e)}).encode())

    async def query_knowledge_base(self, query_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a query to the knowledge base and return the response."""
        try:
            response = await self.nc.request(
                "polaris.knowledge.query",
                json.dumps(query_data).encode(),
                timeout=self.kb_request_timeout,
            )
            response_data = json.loads(response.data.decode())
            return response_data if response_data.get("success") else None
        except TimeoutError:
            self.logger.warning(f"KB query timeout after {self.kb_request_timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"KB query error: {e}", exc_info=True)
            return None

    async def get_kb_stats(self) -> Optional[Dict[str, Any]]:
        """Get knowledge base statistics."""
        try:
            response = await self.nc.request(
                "polaris.knowledge.stats", b"", timeout=self.kb_request_timeout
            )
            return json.loads(response.data.decode())
        except TimeoutError:
            self.logger.warning(f"KB stats timeout after {self.kb_request_timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"KB stats error: {e}", exc_info=True)
            return None

    async def _publish_action_for_verification(self, action_message: Dict[str, Any]) -> None:
        """Publish action to verification adapter for validation before execution."""
        import uuid

        try:
            # Create verification request
            verification_request = {
                "request_id": str(uuid.uuid4()),
                "action": action_message,
                "context": {
                    "source": "reasoner_agent",
                    "agent_id": self.agent_id,
                    "timestamp": time.time(),
                },
                "verification_level": self.verification_level,
                "timeout_sec": self.verification_timeout,
                "requester": f"reasoner_agent_{self.agent_id}",
            }

            self.logger.debug(
                "Sending action for verification",
                extra={
                    "request_id": verification_request["request_id"],
                    "action_type": action_message.get("action_type"),
                    "verification_level": self.verification_level,
                },
            )

            # Publish to verification input subject
            await self.publish(self.verification_input_subject, verification_request)

        except Exception as e:
            self.logger.error(
                f"Failed to publish action for verification: {e}",
                extra={"action": action_message},
                exc_info=True,
            )

            # Handle verification failure based on configuration
            if self.verification_failure_action == "bypass":
                self.logger.warning(
                    "Bypassing verification due to failure, sending directly to execution"
                )
                await self.publish(self.execution_action_subject, action_message)
            elif self.verification_failure_action == "reject":
                self.logger.error("Rejecting action due to verification failure")
                # Could publish to an error/rejected actions subject here
            # "retry" would require more complex retry logic

    async def _handle_system_notification(self, data: Dict[str, Any], msg) -> None:
        """Handle system notifications."""
        notification_type = data.get("type")
        if notification_type == "shutdown":
            self.logger.info(f"Reasoner {self.agent_id} received shutdown notification")
            await self.disconnect()
        elif notification_type == "health_check":
            await self.publish(
                "system.health_responses",
                {
                    "agent_id": self.agent_id,
                    "status": "healthy",
                    "active_sessions": len(self.active_sessions),
                    "reasoning_history_size": len(self.reasoning_history),
                },
            )

    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        return [
            result.to_dict() if hasattr(result, "to_dict") else result
            for result in self.reasoning_history
        ]

    def clear_reasoning_history(self) -> None:
        self.reasoning_history.clear()

    @abstractmethod
    async def perform_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        pass


class DefaultKnowledgeQuery(KnowledgeQueryInterface):
    """Default implementation of knowledge base query operations using NATS."""

    def __init__(self, nats_base: NATSReasonerBase, logger: Optional[logging.Logger] = None):
        self.nats_base = nats_base
        self.logger = logger or logging.getLogger(f"KBQuery.{nats_base.agent_id}")

    async def query_structured(
        self,
        data_types: List[str],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Optional[List[Dict[str, Any]]]:
        query_data = {
            "query_type": "structured",
            "data_types": data_types,
            "filters": filters or {},
            "limit": limit,
            "min_score": min_score,
        }
        return await self._execute_query(query_data)

    async def query_natural_language(
        self, query_text: str, limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        query_data = {"query_type": "natural_language", "query_text": query_text, "limit": limit}
        return await self._execute_query(query_data)

    async def query_recent_observations(self, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        return await self.query_structured(["observation"], limit=limit)

    async def query_raw_telemetry(self, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        return await self.query_structured(["raw_telemetry_event"], limit=limit)

    async def _execute_query(self, query_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        try:
            response = await self.nats_base.query_knowledge_base(query_data)
            if response and response.get("results"):
                return response["results"]
            return None
        except Exception as e:
            self.logger.error(f"KB query failed: {e}", exc_info=True)
            return None

    async def store_reasoning_result(
        self, context: ReasoningContext, result: Any, confidence: float
    ) -> bool:
        try:
            telemetry_event = {
                "name": f"reasoning_{context.reasoning_type.value}",
                "value": confidence,
                "unit": "confidence_score",
                "source": f"reasoner_{self.nats_base.agent_id}",
                "timestamp": time.time(),
                "metadata": {
                    "session_id": context.session_id,
                    "reasoning_type": context.reasoning_type.value,
                    "result_summary": str(result)[:200],
                    "input_summary": self._summarize_input(context.input_data),
                },
            }
            # Publish reasoning result as a telemetry event for monitoring
            await self.nats_base.publish(self.nats_base.telemetry_subject, telemetry_event)
            return True
        except Exception as e:
            self.logger.error(f"Error storing reasoning result: {e}", exc_info=True)
            return False

    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "keys_count": len(input_data.keys()),
            "data_types": {k: type(v).__name__ for k, v in input_data.items()},
            "has_numerical_data": any(isinstance(v, (int, float)) for v in input_data.values()),
            "has_text_data": any(isinstance(v, str) for v in input_data.values()),
        }


class ReasonerAgent(NATSReasonerBase):
    """Main reasoner agent that combines NATS communication with pluggable reasoning implementations."""

    def __init__(
        self,
        agent_id: str,
        reasoning_implementations: Dict[ReasoningType, ReasoningInterface],
        config_path: str,
        nats_url: Optional[str] = None,
        kb_request_timeout: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(agent_id, config_path, nats_url, kb_request_timeout, logger)
        self.reasoning_implementations = reasoning_implementations
        self.kb_query = DefaultKnowledgeQuery(self, self.logger)

        # Initialize Digital Twin client from config
        # Initialize Digital Twin client from config
        dt_config = self.config.get("digital_twin", {})
        grpc_config = dt_config.get("grpc", {})
        host = grpc_config.get("host")
        port = grpc_config.get("port")

        system_name=self.config.get("system_name", "generic")

        if host and port:
            dt_grpc_address = f"{host}:{port}"
            self.dt_query: Optional[DigitalTwinInterface] = GRPCDigitalTwinClient(
                dt_grpc_address, self.logger
            )
        else:
            self.dt_query = None
            self.logger.warning(
                "Digital Twin gRPC host and/or port not configured. DT queries will be disabled."
            )

    async def connect(self) -> None:
        """Connect NATS client and Digital Twin gRPC client."""
        await super().connect()
        if self.dt_query:
            await self.dt_query.connect()

        # Subscribe to threshold update notifications from meta-learner
        await self._subscribe_to_threshold_updates()

    async def _subscribe_to_threshold_updates(self) -> None:
        """Subscribe to threshold update notifications."""
        try:
            await self.nc.subscribe("polaris.meta_learner.update", cb=self._handle_threshold_update)
            self.logger.info("Subscribed to polaris.meta_learner.update")
        except Exception as e:
            self.logger.error(f"Failed to subscribe to threshold updates: {e}")

    async def _handle_threshold_update(self, msg):
        """Handle threshold update notification from meta-learner."""
        try:
            notification = json.loads(msg.data.decode())
            self.logger.info(f"Received threshold update notification: {notification}")

            # Reload config for all LLM reasoning implementations
            for reasoning_type, impl in self.reasoning_implementations.items():
                if hasattr(impl, "reload_prompt_config"):
                    success = impl.reload_prompt_config()
                    if success:
                        self.logger.info(f"✅ Reloaded prompt config for {reasoning_type.value}")
                    else:
                        self.logger.warning(
                            f"⚠️  Failed to reload prompt config for {reasoning_type.value}"
                        )

        except Exception as e:
            self.logger.error(f"Error handling threshold update: {e}", exc_info=True)

    async def disconnect(self) -> None:
        """Disconnect NATS and Digital Twin clients."""
        if self.dt_query:
            await self.dt_query.disconnect()
        await super().disconnect()

    async def perform_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        start_time = time.time()
        reasoning_steps = []
        kb_queries = 0
        dt_queries = 0

        ##short circuit for testing
        self.dt_query = None
        # self.kb_query = None

        try:
            self.active_sessions[context.session_id] = context
            reasoning_steps.append(f"Started reasoning session: {context.session_id}")

            if context.reasoning_type not in self.reasoning_implementations:
                raise ValueError(f"No implementation for reasoning type: {context.reasoning_type}")

            reasoning_impl = self.reasoning_implementations[context.reasoning_type]
            reasoning_steps.append(f"Using {context.reasoning_type.value} reasoning implementation")

            if not await reasoning_impl.validate_input(context):
                raise ValueError("Input validation failed")
            reasoning_steps.append("Input validation passed")

            if self.dt_query:
                reasoning_steps.append("Querying Digital Twin for current system state.")
                dt_response = await self.dt_query.query(
                    DTQuery(query_type="current_state", query_content="Get system overview")
                )
                dt_queries += 1
                if dt_response and dt_response.success:
                    reasoning_steps.append(
                        f"DT response received with confidence {dt_response.confidence}"
                    )
                    context.metadata = context.metadata or {}
                    context.metadata["digital_twin_state"] = dt_response.result
                else:
                    reasoning_steps.append("Failed to get response from Digital Twin.")

            reasoning_steps.append(f"Executing {context.reasoning_type.value} reasoning logic")
            result = await reasoning_impl.reason(context, None)

            reasoning_steps.append("Storing reasoning result")
            if self.kb_query:
                await self.kb_query.store_reasoning_result(
                    context, result.result, result.confidence
                )

            result.execution_time = time.time() - start_time
            result.kb_queries_made = kb_queries
            result.dt_queries_made = dt_queries
            result.reasoning_steps = reasoning_steps + result.reasoning_steps

            self.reasoning_history.append(result)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            reasoning_steps.append(f"Error occurred: {str(e)}")
            self.logger.error(
                f"Reasoning failed in session {context.session_id}: {e}", exc_info=True
            )

            return ReasoningResult(
                result={"error": str(e)},
                confidence=0.0,
                reasoning_steps=reasoning_steps,
                context=context,
                execution_time=execution_time,
                kb_queries_made=kb_queries,
                dt_queries_made=dt_queries,
            )
        finally:
            self.active_sessions.pop(context.session_id, None)

    def add_reasoning_implementation(
        self, reasoning_type: ReasoningType, implementation: ReasoningInterface
    ):
        self.reasoning_implementations[reasoning_type] = implementation

    def remove_reasoning_implementation(self, reasoning_type: ReasoningType):
        self.reasoning_implementations.pop(reasoning_type, None)

    def get_supported_reasoning_types(self) -> List[ReasoningType]:
        return list(self.reasoning_implementations.keys())

    async def predict(self, input_data: dict) -> dict:
        context = ReasoningContext(
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
            input_data=input_data,
            reasoning_type=ReasoningType.PREDICTION,
        )
        result = await self.perform_reasoning(context)
        return result.to_dict() if hasattr(result, "to_dict") else result


class SkeletonReasoningImplementation(ReasoningInterface):
    """Skeleton implementation for reference - replace with actual implementations."""

    async def reason(
        self, context: ReasoningContext, knowledge: Optional[List[Dict[str, Any]]] = None
    ) -> ReasoningResult:

        dt_state = context.metadata.get("digital_twin_state", "not available")

        return ReasoningResult(
            result={
                "message": f"Skeleton {context.reasoning_type.value} reasoning complete.",
                "digital_twin_state": dt_state,
            },
            confidence=0.5,
            reasoning_steps=[f"Skeleton {context.reasoning_type.value} reasoning called"],
            context=context,
            execution_time=0.001,
        )

    async def validate_input(self, context: ReasoningContext) -> bool:
        return True

    def get_required_knowledge_types(self, context: ReasoningContext) -> List[str]:
        return ["observation", "adaptation_decision", "system_goal"]

    def extract_search_terms(self, context: ReasoningContext) -> List[str]:
        terms = []
        for key, value in context.input_data.items():
            if isinstance(value, str) and len(value) > 3:
                terms.extend(value.lower().split())
        return list(set(terms))[:10]


def create_reasoner_agent(
    agent_id: str,
    config_path: str,
    reasoning_implementations: Optional[Dict[ReasoningType, ReasoningInterface]] = None,
    nats_url: Optional[str] = None,
    kb_timeout: float = 30.0,
    logger: Optional[logging.Logger] = None,
    gemini_api_key: Optional[str] = None,
) -> "ReasonerAgent":
    """
    Create a reasoner agent with optional LLM-based reasoning implementations.

    Args:
        agent_id: Unique identifier for the agent
        config_path: Path to YAML configuration file
        reasoning_implementations: Optional custom reasoning implementations
        nats_url: Optional NATS server URL override
        kb_timeout: Timeout for knowledge base queries
        logger: Optional logger instance
        gemini_api_key: Gemini API key for LLM reasoning (if None, uses skeleton implementation)

    Returns:
        ReasonerAgent configured with reasoning implementations
    """
    from .reasoner_agent import ReasonerAgent, ReasoningType, SkeletonReasoningImplementation
    from .llm_reasoner import LLMReasoningImplementation

    if reasoning_implementations is None:
        reasoning_implementations = {}

        # Use LLM reasoner only
        if gemini_api_key:
            for reasoning_type in ReasoningType:
                llm_reasoner = LLMReasoningImplementation(
                    api_key=gemini_api_key, reasoning_type=reasoning_type, logger=logger
                )
                llm_reasoner.configure_basic()
                reasoning_implementations[reasoning_type] = llm_reasoner
        else:
            raise ValueError("Gemini API key must be provided to use LLM reasoning.")

    return ReasonerAgent(
        agent_id, reasoning_implementations, config_path, nats_url, kb_timeout, logger
    )
