"""Formal verification interfaces and domain models for POLARIS.

Defines the Neuro-Symbolic Verifier abstraction, verification decisions,
safety invariants, and verification contexts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from polaris.abstractions.system_contract import SystemContract
    from polaris.core.models import AdaptationAction, SystemState
    from polaris.core.topology import SystemTopology


class VerificationDecision(str, Enum):
    """Outcome of formal verification."""

    ACCEPTED = "accepted"  # Action satisfies all safety invariants without modification.
    CLAMPED = "clamped"  # Action violated soft boundaries and was projected onto safe envelope.
    REJECTED = (
        "rejected"  # Action violated hard structural or safety invariants and cannot execute.
    )


class InvariantSeverity(str, Enum):
    """Severity level of an invariant check."""

    CRITICAL = "critical"  # Hard violation; action must be rejected.
    WARNING = "warning"  # Soft violation; action can be clamped or logged.
    INFO = "info"  # Advisory notification.


@dataclass(frozen=True)
class InvariantViolation:
    """Detailed record of an invariant violation."""

    invariant_name: str
    severity: InvariantSeverity
    message: str
    violating_parameter: Optional[str] = None
    current_value: Optional[Any] = None
    admissible_bound: Optional[Any] = None


@dataclass
class VerificationResult:
    """Formal result produced by a Verifier for a candidate adaptation action."""

    decision: VerificationDecision
    original_action: AdaptationAction
    verified_action: Optional[AdaptationAction]
    violations: List[InvariantViolation] = field(default_factory=list)
    explanation: str = ""
    latency_ms: float = 0.0

    @property
    def is_executable(self) -> bool:
        """Return True if the verified action is safe to execute (accepted or clamped)."""
        return self.decision in (VerificationDecision.ACCEPTED, VerificationDecision.CLAMPED)


@dataclass
class VerificationContext:
    """Contextual evidence supplied to the Verifier during plan validation."""

    system_id: str
    system_state: SystemState
    system_contract: Optional[SystemContract] = None
    recent_actions: Sequence[AdaptationAction] = field(default_factory=list)
    topology: Optional[SystemTopology] = None
    peer_states: Dict[str, SystemState] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyInvariant(ABC):
    """Abstract base class for formal safety invariants evaluated by the Verifier."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the invariant."""
        pass

    @property
    def severity(self) -> InvariantSeverity:
        """Default severity level when this invariant fails."""
        return InvariantSeverity.CRITICAL

    @abstractmethod
    def evaluate(
        self,
        action: AdaptationAction,
        context: VerificationContext,
    ) -> Tuple[bool, Optional[InvariantViolation], Optional[Dict[str, Any]]]:
        """Evaluate action against invariant.

        Args:
            action: Candidate adaptation action.
            context: System and topological context.

        Returns:
            Tuple of:
            - is_satisfied (bool): True if invariant holds.
            - violation (Optional[InvariantViolation]): Violation details if unsatisfied.
            - clamped_parameters (Optional[Dict[str, Any]]): Clamped parameter overrides if soft violation.
        """
        pass


class Verifier(ABC):
    """Neuro-symbolic verification gatekeeper interface.

    Validates candidate adaptation actions before execution against static,
    topological, rate-of-change, and temporal logic invariants.
    """

    @abstractmethod
    async def verify(
        self,
        action: AdaptationAction,
        context: VerificationContext,
    ) -> VerificationResult:
        """Formally verify and optionally clamp a candidate action.

        Args:
            action: Candidate action proposed by Reasoner or Fast Controller.
            context: Current verification context (state, contract, topology, history).

        Returns:
            VerificationResult indicating acceptance, clamping, or rejection.
        """
        pass

    @abstractmethod
    def register_invariant(self, invariant: SafetyInvariant) -> None:
        """Register a domain or safety invariant."""
        pass

    @abstractmethod
    def get_invariants(self) -> Sequence[SafetyInvariant]:
        """Return sequence of registered invariants."""
        pass
