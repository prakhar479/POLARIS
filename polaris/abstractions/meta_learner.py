"""Meta-Learner interface for autonomous parameter tuning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from polaris.abstractions.strategy import AdaptationStrategy


class ProposalStatus(str, Enum):
    """Status of a parameter update proposal."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


@dataclass
class ParameterProposal:
    """Proposal for a parameter update."""

    proposal_id: str
    parameter_path: str
    current_value: Any
    proposed_value: Any
    rationale: str
    confidence: float
    expected_impact: str
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None


@dataclass
class PerformanceAnalysis:
    """Analysis of system performance."""

    system_id: str
    time_window_hours: float
    success_rate: float
    insights: Dict[str, Any]
    recommendations: List[str]


@dataclass
class AppliedUpdate:
    """Record of an applied parameter update."""

    proposal_id: str
    success: bool
    error_message: Optional[str] = None


class MetaLearner(ABC):
    """Interface for autonomous parameter tuning and strategy optimization.

    The Meta-Learner operates asynchronously in the background, analyzing historical
    system behavior to optimize adaptation strategies over time.
    """

    @abstractmethod
    async def analyze_performance(
        self, system_id: str, time_window_hours: float = 24.0
    ) -> PerformanceAnalysis:
        """Analyze recent system performance.

        Args:
            system_id: System to analyze
            time_window_hours: How far back to look

        Returns:
            PerformanceAnalysis with insights and recommendations
        """
        pass

    @abstractmethod
    async def propose_strategy_updates(
        self, strategy: AdaptationStrategy, analysis: PerformanceAnalysis
    ) -> List[ParameterProposal]:
        """Propose parameter updates for a strategy.

        Args:
            strategy: Current strategy instance
            analysis: Performance analysis results

        Returns:
            List of proposed parameter changes
        """
        pass

    @abstractmethod
    async def validate_proposals(
        self, proposals: List[ParameterProposal]
    ) -> List[ParameterProposal]:
        """Validate and rank proposals by safety and impact.

        Args:
            proposals: Proposed changes

        Returns:
            Filtered and ranked proposals
        """
        pass

    async def apply_proposals(
        self, strategy: AdaptationStrategy, proposals: List[ParameterProposal]
    ) -> List[AppliedUpdate]:
        """Apply approved parameter updates to strategy.

        Default implementation uses strategy's tuning interface.
        Handles exceptions gracefully to prevent entire cycle failure.
        """
        results = []
        for proposal in proposals:
            if proposal.status == ProposalStatus.APPROVED:
                try:
                    success = await strategy.update_parameter(
                        proposal.parameter_path, proposal.proposed_value
                    )
                    error_message = (
                        None
                        if success
                        else f"Parameter update returned False for {proposal.parameter_path}"
                    )
                except Exception as e:
                    success = False
                    error_message = f"Exception during parameter update: {str(e)}"

                results.append(
                    AppliedUpdate(
                        proposal_id=proposal.proposal_id,
                        success=success,
                        error_message=error_message,
                    )
                )
        return results
