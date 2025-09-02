"""
Reasoner Agent Architecture with NATS Base Class and Reasoning Interfaces.

This module provides a clean separation between NATS communication (base class)
and reasoning implementation (interfaces to be implemented later).
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
from dataclasses import dataclass, asdict
from enum import Enum
import nats
from nats.errors import TimeoutError

# Assuming these are your existing models
from polaris.knowledge_base.models import KBEntry, KBQuery, KBResponse


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
        }


class ReasoningInterface(ABC):
    """Interface for reasoning implementations."""
    
    @abstractmethod
    async def reason(self, 
                    context: ReasoningContext, 
                    knowledge: Optional[List[Dict[str, Any]]] = None) -> ReasoningResult:
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
    async def query_relevant_knowledge(self, 
                                     search_terms: List[str],
                                     knowledge_types: List[str],
                                     reasoning_type: ReasoningType,
                                     limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def store_reasoning_result(self, 
                                   context: ReasoningContext,
                                   result: Any,
                                   confidence: float) -> bool:
        pass


class NATSReasonerBase(ABC):
    """Base class providing NATS communication functionality for reasoner agents."""
    


    def __init__(self, 
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
        self.nats_url = (
            nats_url or
            self.config.get("nats", {}).get("url", "nats://localhost:4222")
        )

        # Subjects from config
        self.telemetry_subject = self.config.get("telemetry", {}).get("stream_subject", "polaris.telemetry.events.stream")
        self.execution_action_subject = self.config.get("execution", {}).get("action_subject", "polaris.execution.actions")
        self.reasoner_kernel_subject = self.config.get("reasoner", {}).get("kernel_request_subject", "polaris.reasoner.kernel.requests")

        # Telemetry KB config
        self.telemetry_kb_enabled = self.config.get("telemetry", {}).get("knowledge_base", {}).get("enabled", False)
        self.telemetry_kb_buffer_size = self.config.get("telemetry", {}).get("knowledge_base", {}).get("buffer_size", 100)

        # Setup logger
        if logger is None:
            log_name = self.config.get("logger", {}).get("name", f"Reasoner.{agent_id}")
            self.logger = logging.getLogger(log_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                fmt = self.config.get("logger", {}).get("format", "pretty")
                if fmt == "json":
                    formatter = logging.Formatter(json.dumps({
                        "time": "%(asctime)s",
                        "name": "%(name)s",
                        "level": "%(levelname)s",
                        "msg": "%(message)s"
                    }))
                else:
                    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        # Existing
        await self.listen("system.notifications", self._handle_system_notification)
        # New: listen for telemetry requests from kernel
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
        self.logger.info(f"Received kernel telemetry request")
        try:
            context = ReasoningContext(
                session_id=data.get("session_id", str(uuid.uuid4())),
                timestamp=time.time(),
                input_data=data.get("input_data", {}),
                reasoning_type=ReasoningType(data.get("reasoning_type", "inference")),  # default fallback
                metadata=data.get("metadata", {})
            )

            result = await self.perform_reasoning(context)

        ###############################################################replace here with actual action generation logic############################################################################
            action_message = {
                    "action_type": "SET_DIMMER",
                    "source": "fast_controller",
                    "action_id": str(uuid.uuid4()),
                    "params": {"value": 0.5},
                    "priority": "low",
                }
            self.logger.info(f"Publishing reasoning action to execution layer", extra={"action": action_message})
            await self.publish(self.execution_action_subject, action_message)
            await msg.respond(json.dumps({"success": True}).encode())

        except Exception as e:
            self.logger.error(f"Kernel request failed: {e}", exc_info=True)
            await msg.respond(json.dumps({"success": False, "error": str(e)}).encode())

    
    async def query_knowledge_base(self, query_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a query to the knowledge base and return the response."""
        try:
            response = await self.nc.request(
                "polaris.knowledge.query",
                json.dumps(query_data).encode(),
                timeout=self.kb_request_timeout
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
                "polaris.knowledge.stats",
                b"",
                timeout=self.kb_request_timeout
            )
            return json.loads(response.data.decode())
        except TimeoutError:
            self.logger.warning(f"KB stats timeout after {self.kb_request_timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"KB stats error: {e}", exc_info=True)
            return None
    

    async def _handle_system_notification(self, data: Dict[str, Any], msg) -> None:
        """Handle system notifications."""
        notification_type = data.get("type")
        if notification_type == "shutdown":
            self.logger.info(f"Reasoner {self.agent_id} received shutdown notification")
            await self.disconnect()
        elif notification_type == "health_check":
            await self.publish("system.health_responses", {
                "agent_id": self.agent_id,
                "status": "healthy",
                "active_sessions": len(self.active_sessions),
                "reasoning_history_size": len(self.reasoning_history)
            })
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        return [result.to_dict() for result in self.reasoning_history]
    
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
    
    async def query_relevant_knowledge(self, 
                                     search_terms: List[str],
                                     knowledge_types: List[str],
                                     reasoning_type: ReasoningType,
                                     limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        query_data = {
            "query_type": "structured",
            "search_term": " ".join(search_terms),
            "data_types": knowledge_types,
            "limit": limit,
            "min_score": 0.7
        }
        response = await self.nats_base.query_knowledge_base(query_data)
        if response and response.get("results"):
            return response["results"]
        return None
    
    async def store_reasoning_result(self, 
                                   context: ReasoningContext,
                                   result: Any,
                                   confidence: float) -> bool:
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
                    "input_summary": self._summarize_input(context.input_data)
                }
            }
            # Uncomment and configure when telemetry stream is ready
            # await self.nats_base.nc.publish(
            #     "polaris.telemetry.events.stream",
            #     json.dumps(telemetry_event).encode()
            # )
            return True
        except Exception as e:
            self.logger.error(f"Error storing reasoning result: {e}", exc_info=True)
            return False
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "keys_count": len(input_data.keys()),
            "data_types": {k: type(v).__name__ for k, v in input_data.items()},
            "has_numerical_data": any(isinstance(v, (int, float)) for v in input_data.values()),
            "has_text_data": any(isinstance(v, str) for v in input_data.values())
        }


class ReasonerAgent(NATSReasonerBase):
    """Main reasoner agent that combines NATS communication with pluggable reasoning implementations."""
    
    def __init__(self, 
                 agent_id: str,
                 reasoning_implementations: Dict[ReasoningType, ReasoningInterface],
                config_path: str,
                 nats_url: str = "nats://localhost:4222",
                 kb_request_timeout: float = 30.0,
                 logger: Optional[logging.Logger] = None):
        super().__init__(agent_id, config_path, nats_url, kb_request_timeout, logger)
        self.reasoning_implementations = reasoning_implementations
        self.kb_query = DefaultKnowledgeQuery(self, self.logger)
    
    async def perform_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        start_time = time.time()
        reasoning_steps = []
        kb_queries = 0
        
        try:
            self.active_sessions[context.session_id] = context
            reasoning_steps.append(f"Started reasoning session: {context.session_id}")
            
            if context.reasoning_type not in self.reasoning_implementations:
                raise ValueError(f"No implementation found for reasoning type: {context.reasoning_type}")
            
            reasoning_impl = self.reasoning_implementations[context.reasoning_type]
            reasoning_steps.append(f"Using {context.reasoning_type.value} reasoning implementation")
            
            if not await reasoning_impl.validate_input(context):
                raise ValueError("Input validation failed")
            reasoning_steps.append("Input validation passed")
            
            search_terms = reasoning_impl.extract_search_terms(context)
            knowledge_types = reasoning_impl.get_required_knowledge_types(context)
            
            reasoning_steps.append("Querying knowledge base for relevant information")
            relevant_knowledge = await self.kb_query.query_relevant_knowledge(
                search_terms, knowledge_types, context.reasoning_type
            )
            kb_queries += 1 if relevant_knowledge else 0
            
            reasoning_steps.append(f"Executing {context.reasoning_type.value} reasoning logic")
            result = await reasoning_impl.reason(context, relevant_knowledge)
            
            reasoning_steps.append("Storing reasoning result in knowledge base")
            await self.kb_query.store_reasoning_result(context, result.result, result.confidence)
            kb_queries += 1
            
            result.execution_time = time.time() - start_time
            result.kb_queries_made = kb_queries
            result.reasoning_steps = reasoning_steps
            
            self.reasoning_history.append(result)
            self.active_sessions.pop(context.session_id, None)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            reasoning_steps.append(f"Error occurred: {str(e)}")
            self.logger.error(f"Reasoning failed in session {context.session_id}: {e}", exc_info=True)
            
            error_result = ReasoningResult(
                result={"error": str(e)},
                confidence=0.0,
                reasoning_steps=reasoning_steps,
                context=context,
                execution_time=execution_time,
                kb_queries_made=kb_queries
            )
            self.active_sessions.pop(context.session_id, None)
            return error_result
    
    def add_reasoning_implementation(self, reasoning_type: ReasoningType, implementation: ReasoningInterface):
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
            reasoning_type=ReasoningType.PREDICTION
        )
        result = await self.perform_reasoning(context)
        return result.to_dict()


class SkeletonReasoningImplementation(ReasoningInterface):
    """Skeleton implementation for reference - replace with actual implementations."""
    
    async def reason(self, 
                    context: ReasoningContext, 
                    knowledge: Optional[List[Dict[str, Any]]] = None) -> ReasoningResult:
        return ReasoningResult(
            result={"message": f"Skeleton {context.reasoning_type.value} reasoning not implemented"},
            confidence=0.0,
            reasoning_steps=[f"Skeleton {context.reasoning_type.value} reasoning called"],
            context=context,
            execution_time=0.001
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


def create_reasoner_agent(agent_id: str,
                          config_path: str,
                         reasoning_implementations: Optional[Dict[ReasoningType, ReasoningInterface]] = None,
                         nats_url: str = "nats://localhost:4222",
                         kb_timeout: float = 30.0,
                         logger: Optional[logging.Logger] = None) -> ReasonerAgent:
    if reasoning_implementations is None:
        reasoning_implementations = {
            reasoning_type: SkeletonReasoningImplementation()
            for reasoning_type in ReasoningType
        }
    return ReasonerAgent(
    agent_id,
    reasoning_implementations,
    config_path,       # 3rd positional arg
    nats_url,
    kb_timeout,
    logger
)



async def main():
    """Example of how to use the reasoner agent framework with POLARIS KB."""
    reasoner = create_reasoner_agent("polaris_reasoner_001")
    
    try:
        await reasoner.connect()
        reasoner.logger.info(f"Supported reasoning types: {reasoner.get_supported_reasoning_types()}")
        
        kb_stats = await reasoner.kb_query.nats_base.get_kb_stats()
        if kb_stats:
            reasoner.logger.info(f"KB Stats: {kb_stats['service']['events_processed']} events processed")
        
        context = ReasoningContext(
            session_id="polaris_example_session",
            timestamp=time.time(),
            input_data={
                "goal": "Optimize POLARIS system performance",
                "current_metrics": {"cpu_usage": 80, "memory_usage": 60},
                "search_query": "performance optimization telemetry"
            },
            reasoning_type=ReasoningType.PLANNING
        )
        
        result = await reasoner.perform_reasoning(context)
        reasoner.logger.info(f"Reasoning result: {result.to_dict()}")
        
    finally:
        await reasoner.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
