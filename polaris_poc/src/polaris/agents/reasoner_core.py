"""
Core reasoning interfaces, dataclasses, and enums.

This module defines the shared contracts used by ReasonerAgent
and different reasoning implementations (LLM, Skeleton, etc.)
to avoid circular imports.
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum


class ReasoningType(Enum):
    """Types of reasoning operations."""
    INFERENCE = "inference"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    DECISION = "decision"
    PREDICTION = "prediction"


@dataclass
class ReasoningContext:
    session_id: str
    reasoning_type: ReasoningType
    input_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reasoning_type": self.reasoning_type.value,  # <-- FIX here
            "input_data": self.input_data,
            "metadata": self.metadata or {},
            "timestamp": self.timestamp,
        }



@dataclass
class ReasoningResult:
    result: Dict[str, Any]
    confidence: float
    reasoning_steps: List[str]
    context: "ReasoningContext"
    execution_time: float
    kb_queries_made: int = 0
    dt_queries_made: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "context": self.context.to_dict() if hasattr(self.context, "to_dict") else str(self.context),
            "execution_time": self.execution_time,
            "kb_queries_made": self.kb_queries_made,
            "dt_queries_made": self.dt_queries_made,
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
