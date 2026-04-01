"""Statistical meta-learner with Bayesian optimization and Kalman filtering."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.meta_learner import (
    MetaLearner,
    ParameterProposal,
    PerformanceAnalysis,
    ProposalStatus,
)
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationStrategy, ParameterSpec
from polaris.core.models import AdaptationAction, ExecutionResult
from polaris.infrastructure.constants import (
    DEFAULT_EXECUTION_TIME_NORMALIZATION,
    DEFAULT_STD_DEVIATION_THRESHOLD,
)
from polaris.meta_learner.bayesian_optimizer import (
    AcquisitionFunction,
    GaussianProcessOptimizer,
    ParameterConfiguration,
    ParameterSpace,
    ParameterType,
)


class StatisticalMetaLearner(MetaLearner):
    """Statistical meta-learner using Bayesian optimization for intelligent parameter tuning.

    Combines Gaussian Process-based optimization with world model uncertainty
    integration for sophisticated parameter adaptation decisions.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        enable_bayesian_optimization: bool = True,
        enable_learning: bool = True,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
        conservative_mode: bool = False,
        world_model: Optional[Any] = None,
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        exploration_weight: float = 0.1,
        min_samples_for_optimization: int = 10,
    ):
        """Initialize statistical meta-learner with learning capabilities."""
        self.knowledge_store = knowledge_store
        self.logger = logger
        self.metrics = metrics
        self.conservative_mode = conservative_mode
        self.world_model = world_model
        self.enable_bayesian_optimization = enable_bayesian_optimization
        self.enable_learning = enable_learning
        self.acquisition_function = acquisition_function
        self.exploration_weight = exploration_weight
        self.min_samples_for_optimization = min_samples_for_optimization

        # Bayesian optimizer cache (per system)
        self._optimizers: Dict[str, GaussianProcessOptimizer] = {}

    async def analyze_performance(
        self, system_id: str, time_window_hours: float = 24.0
    ) -> PerformanceAnalysis:
        """Analyze system performance over time window."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=time_window_hours)

        # Query historical data
        states = await self.knowledge_store.query_states(system_id, start_time, end_time)
        actions = await self.knowledge_store.query_actions(system_id, start_time, end_time)

        # Calculate success rate
        if actions:
            successful = sum(
                1
                for _, result in actions
                if hasattr(result, "status") and result.status.value == "success"
            )
            success_rate = successful / len(actions)
        else:
            success_rate = 0.0

        # Simple insights
        insights: Dict[str, Any] = {
            "total_states": len(states),
            "total_adaptations": len(actions),
            "success_rate": success_rate,
            "time_window_hours": time_window_hours,
        }

        # Optionally augment with world model uncertainty information
        if self.world_model is not None:
            try:
                wm_insights = await self.world_model.get_insights()
                system_insights = wm_insights.get(system_id, {})

                # Aggregate simple uncertainty metrics (e.g., average std across metrics)
                std_values = []
                for _name, info in system_insights.items():
                    if isinstance(info, dict) and "std" in info:
                        try:
                            std_values.append(float(info["std"]))
                        except (TypeError, ValueError):
                            continue

                avg_std = sum(std_values) / len(std_values) if std_values else 0.0

                regime_info = (
                    system_insights.get("regime") if isinstance(system_insights, dict) else None
                )

                insights["world_model_uncertainty"] = {
                    "avg_metric_std": avg_std,
                    "regime": regime_info,
                }
            except Exception:
                # Best-effort world model insights - do not break analysis on WM issues
                pass

        # Basic recommendations
        recommendations = []
        if success_rate < 0.7:
            recommendations.append("Low success rate - consider adjusting thresholds")
        if len(actions) > 100:
            recommendations.append("High adaptation frequency - consider increasing cooldown")

        return PerformanceAnalysis(
            system_id=system_id,
            time_window_hours=time_window_hours,
            success_rate=success_rate,
            insights=insights,
            recommendations=recommendations,
        )

    async def propose_strategy_updates(
        self, strategy: AdaptationStrategy, analysis: PerformanceAnalysis
    ) -> List[ParameterProposal]:
        """Propose parameter updates using Bayesian optimization or heuristics."""
        tunable_params = strategy.get_tunable_parameters()
        if not tunable_params:
            return []

        # Try Bayesian optimization first if enabled
        if self.enable_bayesian_optimization:
            bayesian_proposals = await self._propose_bayesian_updates(
                strategy, analysis, tunable_params
            )
            if bayesian_proposals:
                return bayesian_proposals

        # Fallback to heuristic-based approach
        return await self._propose_heuristic_updates(strategy, analysis, tunable_params)

    async def _propose_bayesian_updates(
        self,
        strategy: AdaptationStrategy,
        analysis: PerformanceAnalysis,
        tunable_params: Dict[str, Any],
    ) -> List[ParameterProposal]:
        """Propose parameter updates using Bayesian optimization."""
        try:
            # Get or create optimizer for this system
            optimizer = self._get_or_create_optimizer(analysis.system_id, tunable_params)

            # Collect historical configuration-performance data
            historical_data: List[ParameterConfiguration] = (
                await self._collect_historical_configurations(analysis.system_id, tunable_params)
            )

            # Train optimizer on historical data
            if not optimizer.fit(historical_data):
                # Not enough data for Bayesian optimization
                return []

            # Get optimization confidence
            optimization_confidence = optimizer.get_optimization_confidence()

            # Get suggested parameters
            suggested_params = optimizer.suggest_next_parameters(n_suggestions=3)

            # Create proposals from suggestions
            proposals = []
            for params in suggested_params:
                for param_path, suggested_value in params.items():
                    if param_path not in tunable_params:
                        continue

                    spec = tunable_params[param_path]
                    current_value = spec.current_value

                    # Skip if suggestion is too similar to current value
                    if self._values_are_similar(current_value, suggested_value, spec):
                        continue

                    # Apply conservative constraints
                    if self.conservative_mode:
                        suggested_value = self._apply_conservative_constraints(
                            current_value, suggested_value, spec
                        )

                    # Calculate confidence based on optimization and world model uncertainty
                    base_confidence = 0.5 + 0.4 * optimization_confidence  # 0.5-0.9 range

                    # Adjust confidence based on world model uncertainty
                    wm_unc = analysis.insights.get("world_model_uncertainty", {})
                    avg_std = wm_unc.get("avg_metric_std", 0.0) if isinstance(wm_unc, dict) else 0.0
                    if avg_std > DEFAULT_STD_DEVIATION_THRESHOLD:
                        confidence = max(
                            0.3, base_confidence - 0.2
                        )  # More cautious with high uncertainty
                    else:
                        confidence = base_confidence

                    proposals.append(
                        ParameterProposal(
                            proposal_id=str(uuid.uuid4()),
                            parameter_path=param_path,
                            current_value=current_value,
                            proposed_value=suggested_value,
                            rationale=(
                                f"Bayesian optimization suggests {param_path} change from {current_value} "
                                f"to {suggested_value} to improve performance. "
                                f"optimization confidence: {optimization_confidence: .2f}"
                            ),
                            confidence=confidence,
                            expected_impact="Data-driven performance optimization",
                            status=ProposalStatus.PENDING,
                        )
                    )

            return proposals

        except Exception as e:
            if self.logger:
                self.logger.error(f"Bayesian optimization failed: {e}")
            return []

    async def _propose_heuristic_updates(
        self,
        strategy: AdaptationStrategy,
        analysis: PerformanceAnalysis,
        tunable_params: Dict[str, Any],
    ) -> List[ParameterProposal]:
        """Propose parameter updates using original heuristic approach."""
        proposals = []

        # Simple heuristic: if success rate is low, suggest small adjustments
        if analysis.success_rate < 0.7:
            for param_path, spec in tunable_params.items():
                kind = getattr(spec, "kind", None)
                is_high_threshold = kind == "threshold_high"
                if not is_high_threshold:
                    # Fallback to legacy name-based heuristic
                    is_high_threshold = (
                        "threshold" in param_path.lower() and "high" in param_path.lower()
                    )
                if not is_high_threshold:
                    continue

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
                    wm_unc = analysis.insights.get("world_model_uncertainty", {})
                    avg_std = wm_unc.get("avg_metric_std", 0.0) if isinstance(wm_unc, dict) else 0.0
                    # If world model suggests high variability, be slightly more cautious
                    if avg_std > DEFAULT_STD_DEVIATION_THRESHOLD:
                        confidence = max(0.4, base_confidence - 0.1)
                    else:
                        confidence = base_confidence

                    proposals.append(
                        ParameterProposal(
                            proposal_id=str(uuid.uuid4()),
                            parameter_path=param_path,
                            current_value=current,
                            proposed_value=proposed,
                            rationale=(
                                f"Low success rate ({analysis.success_rate: .2%}). "
                                f"Increasing threshold to reduce adaptation frequency."
                            ),
                            confidence=confidence,
                            expected_impact="May reduce false positives",
                            status=ProposalStatus.PENDING,
                        )
                    )

        # Suggest increase cooldown if too many adaptations
        if analysis.insights.get("total_adaptations", 0) > 100:
            for param_path, spec in tunable_params.items():
                kind = getattr(spec, "kind", None)
                is_cooldown = kind == "cooldown"
                if not is_cooldown:
                    # Fallback to legacy name-based heuristic
                    is_cooldown = "cooldown" in param_path.lower()
                if not is_cooldown:
                    continue

                current = spec.current_value
                max_val = spec.max_value or 300

                proposed = min(current * 1.2, max_val)  # 20% increase

                if proposed != current:
                    proposals.append(
                        ParameterProposal(
                            proposal_id=str(uuid.uuid4()),
                            parameter_path=param_path,
                            current_value=current,
                            proposed_value=proposed,
                            rationale="High adaptation frequency. Increasing cooldown.",
                            confidence=0.7,
                            expected_impact="Reduce adaptation churn",
                            status=ProposalStatus.PENDING,
                        )
                    )

        return proposals

    def _get_or_create_optimizer(
        self, system_id: str, tunable_params: Dict[str, Any]
    ) -> GaussianProcessOptimizer:
        """Get or create Bayesian optimizer for a system."""
        if system_id not in self._optimizers:
            # Create parameter spaces from tunable parameters
            parameter_spaces = []

            for param_path, spec in tunable_params.items():
                param_type = self._determine_parameter_type(spec)

                # Skip unsupported parameter types
                if param_type not in [
                    ParameterType.CONTINUOUS,
                    ParameterType.DISCRETE,
                    ParameterType.CATEGORICAL,
                ]:
                    continue

                if param_type == ParameterType.CONTINUOUS:
                    param_space = ParameterSpace(
                        name=param_path,
                        param_type=param_type,
                        min_value=spec.min_value or (spec.current_value * 0.5),
                        max_value=spec.max_value or (spec.current_value * 2.0),
                        current_value=spec.current_value,
                    )
                elif param_type == ParameterType.DISCRETE:
                    param_space = ParameterSpace(
                        name=param_path,
                        param_type=param_type,
                        min_value=spec.min_value or int(spec.current_value * 0.5),
                        max_value=spec.max_value or int(spec.current_value * 2),
                        current_value=spec.current_value,
                    )
                else:  # CATEGORICAL
                    param_space = ParameterSpace(
                        name=param_path,
                        param_type=param_type,
                        allowed_values=spec.allowed_values or [spec.current_value],
                        current_value=spec.current_value,
                    )

                parameter_spaces.append(param_space)

            # Create optimizer
            self._optimizers[system_id] = GaussianProcessOptimizer(
                parameter_spaces=parameter_spaces,
                acquisition_function=self.acquisition_function,
                exploration_weight=self.exploration_weight,
                min_samples_for_optimization=self.min_samples_for_optimization,
            )

        return self._optimizers[system_id]

    def _determine_parameter_type(self, spec: ParameterSpec) -> ParameterType:
        """Determine parameter type from specification."""
        # Check explicit type first
        if hasattr(spec, "type"):
            if spec.type in (int, "int"):
                return ParameterType.DISCRETE
            elif spec.type in (float, "float"):
                return ParameterType.CONTINUOUS
            elif spec.type in (str, "str"):
                return ParameterType.CATEGORICAL

        # Use allowed_values to determine categorical
        if hasattr(spec, "allowed_values") and spec.allowed_values:
            return ParameterType.CATEGORICAL

        # Use current_value type as fallback
        if hasattr(spec, "current_value"):
            current_type = type(spec.current_value)
            if current_type in (int, "int"):
                return ParameterType.DISCRETE
            if current_type in (float, "float"):
                return ParameterType.CONTINUOUS
            if current_type in (str, "str"):
                return ParameterType.CATEGORICAL

        # Default to continuous
        return ParameterType.CONTINUOUS

    async def _collect_historical_configurations(
        self, system_id: str, tunable_params: Dict[str, Any]
    ) -> List[ParameterConfiguration]:
        """Collect historical parameter configurations and performance data."""
        # Query historical actions and their outcomes
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=168)  # Last 7 days

        try:
            actions = await self.knowledge_store.query_actions(system_id, start_time, end_time)

            configurations = []
            for action, result in actions:
                # Extract parameter configuration from action
                if hasattr(action, "parameters") and action.parameters:
                    # Calculate performance metric for this configuration
                    performance = self._calculate_performance_metric(action, result)

                    config = ParameterConfiguration(
                        parameters=action.parameters,
                        performance=performance,
                        metadata={
                            "action_type": action.action_type,
                            "timestamp": getattr(action, "timestamp", datetime.now(timezone.utc)),
                            "result_status": getattr(result, "status", "unknown"),
                        },
                    )
                    configurations.append(config)

            return configurations

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to collect historical configurations: {e}")
            return []

    def _calculate_performance_metric(
        self, action: AdaptationAction, result: ExecutionResult
    ) -> float:
        """Calculate performance metric from action result."""
        # Base performance from action result
        base_performance = 0.5  # Default neutral performance

        if hasattr(result, "status"):
            if result.status.value == "success":
                base_performance = 0.8
            elif result.status.value == "failed":
                base_performance = 0.2

        # Adjust based on action context if available
        if hasattr(action, "context") and action.context:
            # Consider execution time, resource usage, etc.
            context = action.context
            if isinstance(context, dict):
                # Faster execution = better performance
                execution_time = context.get("execution_time")
                if execution_time and isinstance(execution_time, (int, float)):
                    time_bonus = max(
                        0, 1.0 - execution_time / DEFAULT_EXECUTION_TIME_NORMALIZATION
                    )  # Normalize to 0-1
                    base_performance += 0.1 * time_bonus

        return float(np.clip(base_performance, 0.0, 1.0))

    def _values_are_similar(self, current: Any, suggested: Any, spec: ParameterSpec) -> bool:
        """Check if suggested value is too similar to current value."""
        if isinstance(current, (int, float)) and isinstance(suggested, (int, float)):
            # For numeric values, check relative difference
            if current == 0:
                return abs(suggested) < 0.01
            relative_diff = abs(suggested - current) / abs(current)
            return relative_diff < 0.05  # Less than 5% change
        else:
            # For categorical/discrete values, check equality
            return bool(current == suggested)

    def _apply_conservative_constraints(
        self, current: Any, suggested: Any, spec: ParameterSpec
    ) -> Any:
        """Apply conservative constraints to suggested parameter changes."""
        if isinstance(current, (int, float)) and isinstance(suggested, (int, float)):
            # Limit change magnitude to 15% for conservative mode
            max_change = abs(current) * 0.15
            if abs(suggested - current) > max_change:
                # Apply maximum allowed change in the same direction
                if suggested > current:
                    suggested = current + max_change
                else:
                    suggested = current - max_change

            # Respect bounds
            if spec.min_value is not None:
                suggested = max(spec.min_value, suggested)
            if spec.max_value is not None:
                suggested = min(spec.max_value, suggested)

        return suggested

    async def validate_proposals(
        self, proposals: List[ParameterProposal]
    ) -> List[ParameterProposal]:
        """Validate proposed parameter updates."""
        validated_proposals = []
        for proposal in proposals:
            # Simple validation: approve if confidence > 0.5
            if proposal.confidence >= 0.5:
                proposal.status = ProposalStatus.APPROVED
                validated_proposals.append(proposal)
            else:
                proposal.status = ProposalStatus.REJECTED

        # Sort by confidence
        validated_proposals.sort(key=lambda p: p.confidence, reverse=True)

        return validated_proposals
