"""
Statistical meta-learner with Bayesian optimization and Kalman filtering.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import uuid

from polaris.abstractions.meta_learner import (
    MetaLearner,
    ParameterProposal,
    PerformanceAnalysis,
    ProposalStatus
)
from polaris.abstractions.strategy import AdaptationStrategy
from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger
from polaris.abstractions.world_model import WorldModel


class StatisticalMetaLearner(MetaLearner):
    """
    Statistical meta-learner using heuristics and statistical analysis.

    Future: Add Bayesian optimization and Kalman filtering for advanced tuning.
    For now: Simple rule-based parameter adjustment.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        logger: Logger,
        conservative_mode: bool = True,
        world_model: Optional[WorldModel] = None,
    ):
        self.knowledge_store = knowledge_store
        self.logger = logger
        self.conservative_mode = conservative_mode
        # Optional world model for incorporating predictive uncertainty
        self.world_model = world_model

    async def analyze_performance(
        self,
        system_id: str,
        time_window_hours: float = 24.0
    ) -> PerformanceAnalysis:
        """Analyze system performance over time window."""

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=time_window_hours)

        # Query historical data
        states = await self.knowledge_store.query_states(
            system_id, start_time, end_time
        )
        actions = await self.knowledge_store.query_actions(
            system_id, start_time, end_time
        )

        # Calculate success rate
        if actions:
            successful = sum(
                1 for _, result in actions
                if hasattr(result, 'status') and result.status.value == 'success'
            )
            success_rate = successful / len(actions)
        else:
            success_rate = 0.0

        # Simple insights
        insights = {
            'total_states': len(states),
            'total_adaptations': len(actions),
            'success_rate': success_rate,
            'time_window_hours': time_window_hours
        }

        # Optionally augment with world model uncertainty information
        if self.world_model is not None:
            try:
                wm_insights = await self.world_model.get_insights()
                system_insights = wm_insights.get(system_id, {})

                # Aggregate simple uncertainty metrics (e.g., average std across metrics)
                std_values = []
                for name, info in system_insights.items():
                    if isinstance(info, dict) and 'std' in info:
                        try:
                            std_values.append(float(info['std']))
                        except (TypeError, ValueError):
                            continue

                avg_std = sum(std_values) / len(std_values) if std_values else 0.0

                regime_info = system_insights.get('regime') if isinstance(system_insights, dict) else None

                insights['world_model_uncertainty'] = {
                    'avg_metric_std': avg_std,
                    'regime': regime_info,
                }
            except Exception:  # Best-effort, do not break analysis on WM issues
                pass

        # Basic recommendations
        recommendations = []
        if success_rate < 0.7:
            recommendations.append(
                "Low success rate - consider adjusting thresholds")
        if len(actions) > 100:
            recommendations.append(
                "High adaptation frequency - consider increasing cooldown")

        return PerformanceAnalysis(
            system_id=system_id,
            time_window_hours=time_window_hours,
            success_rate=success_rate,
            insights=insights,
            recommendations=recommendations
        )

    async def propose_strategy_updates(
        self,
        strategy: AdaptationStrategy,
        analysis: PerformanceAnalysis
    ) -> List[ParameterProposal]:
        """Propose parameter updates based on analysis."""

        proposals = []
        tunable_params = strategy.get_tunable_parameters()

        # Simple heuristic: if success rate is low, suggest small adjustments
        if analysis.success_rate < 0.7:
            for param_path, spec in tunable_params.items():
                if 'threshold' in param_path.lower() and 'high' in param_path.lower():
                    # Suggest increasing high thresholds slightly
                    current = spec.current_value
                    max_val = spec.max_value or (current * 1.5)

                    if self.conservative_mode:
                        # Small 5% increase
                        proposed = min(current * 1.05, max_val)
                    else:
                        # Larger 10% increase
                        proposed = min(current * 1.10, max_val)

                    if proposed != current:
                        base_confidence = 0.6
                        wm_unc = analysis.insights.get('world_model_uncertainty', {})
                        avg_std = wm_unc.get('avg_metric_std', 0.0) if isinstance(wm_unc, dict) else 0.0
                        # If world model suggests high variability, be slightly more cautious
                        if avg_std > 10.0:
                            confidence = max(0.4, base_confidence - 0.1)
                        else:
                            confidence = base_confidence

                        proposals.append(ParameterProposal(
                            proposal_id=str(uuid.uuid4()),
                            parameter_path=param_path,
                            current_value=current,
                            proposed_value=proposed,
                            rationale=(
                                f"Low success rate ({analysis.success_rate:.2%}). "
                                f"Increasing threshold to reduce adaptation frequency."
                            ),
                            confidence=confidence,
                            expected_impact="May reduce false positives",
                            status=ProposalStatus.PENDING
                        ))

        # Suggest increase cooldown if too many adaptations
        if analysis.insights.get('total_adaptations', 0) > 100:
            for param_path, spec in tunable_params.items():
                if 'cooldown' in param_path.lower():
                    current = spec.current_value
                    max_val = spec.max_value or 300

                    proposed = min(current * 1.2, max_val)  # 20% increase

                    if proposed != current:
                        proposals.append(ParameterProposal(
                            proposal_id=str(uuid.uuid4()),
                            parameter_path=param_path,
                            current_value=current,
                            proposed_value=proposed,
                            rationale="High adaptation frequency. Increasing cooldown.",
                            confidence=0.7,
                            expected_impact="Reduce adaptation churn",
                            status=ProposalStatus.PENDING
                        ))

        return proposals

    async def validate_proposals(
        self,
        proposals: List[ParameterProposal]
    ) -> List[ParameterProposal]:
        """Validate and rank proposals."""

        validated = []
        for proposal in proposals:
            # Simple validation: approve if confidence > 0.5
            if proposal.confidence >= 0.5:
                proposal.status = ProposalStatus.APPROVED
                validated.append(proposal)
            else:
                proposal.status = ProposalStatus.REJECTED

        # Sort by confidence
        validated.sort(key=lambda p: p.confidence, reverse=True)

        return validated
