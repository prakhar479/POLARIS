"""LLM-based meta-learner using AI for intelligent parameter tuning."""

import json
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.meta_learner import (
    AppliedUpdate,
    MetaLearner,
    ParameterProposal,
    PerformanceAnalysis,
    ProposalStatus,
)
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationStrategy, ParameterSpec
from polaris.infrastructure.constants import DEFAULT_MAX_TOKENS
from polaris.infrastructure.llm import LLMClient, LLMMessage
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector


class LLMMetaLearner(MetaLearner):
    """LLM-powered meta-learner for intelligent strategy optimization.

    Uses an LLM to analyze system performance and propose parameter updates using
    natural language reasoning and deep analysis.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_store: KnowledgeStore,
        logger: Logger,
        auto_apply: bool = False,
        temperature: float = 0.1,
        analysis_system_prompt: Optional[str] = None,
        optimization_system_prompt: Optional[str] = None,
        per_system_prompts: Optional[Dict[str, Dict[str, str]]] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize LLM meta-learner.

        Args:
            llm_client: LLM client for reasoning
            knowledge_store: Knowledge store for historical data
            logger: Logger instance
            auto_apply: Whether to auto-apply approved changes
            temperature: LLM temperature (lower = more conservative)
            analysis_system_prompt: Optional custom system prompt for analysis
            optimization_system_prompt: Optional custom system prompt for optimization
            per_system_prompts: Optional system-specific prompt overrides
            metrics: Optional metrics collector for tracking performance
        """
        self.llm = llm_client
        self.knowledge_store = knowledge_store
        self.logger = logger
        self.auto_apply = auto_apply
        self.temperature = temperature
        self.metrics = metrics or NullMetricsCollector()
        self.analysis_system_prompt = analysis_system_prompt
        self.optimization_system_prompt = optimization_system_prompt
        self._per_system_prompts = per_system_prompts or {}

    async def analyze_performance(
        self, system_id: str, time_window_hours: float = 24.0
    ) -> PerformanceAnalysis:
        """Analyze system performance using LLM."""
        self.metrics.increment(
            "polaris.meta_learning.llm.analysis_requests",
            tags={"system_id": system_id},
        )

        # Handle null knowledge store gracefully
        if not self.knowledge_store:
            self.logger.warning(
                "Knowledge store not available for system %s, using fallback analysis",
                system_id=system_id,
            )
            self.metrics.increment(
                "polaris.meta_learning.llm.analysis_knowledge_store_unavailable",
                tags={"system_id": system_id},
            )
            return PerformanceAnalysis(
                system_id=system_id,
                time_window_hours=time_window_hours,
                success_rate=0.0,
                insights={
                    "error": "knowledge_store_unavailable",
                    "message": "Knowledge store not configured",
                },
                recommendations=["Configure knowledge store for better analysis"],
            )

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=time_window_hours)

        # Query historical data
        states = await self.knowledge_store.query_states(system_id, start_time, end_time)
        actions = await self.knowledge_store.query_actions(system_id, start_time, end_time)

        # Calculate basic statistics
        if actions:
            successful = sum(
                1
                for _, result in actions
                if hasattr(result, "status") and result.status.value == "success"
            )
            success_rate = successful / len(actions)
        else:
            success_rate = 1.0  # No actions = no failures

        # Prepare data for LLM analysis
        analysis_prompt = self._build_analysis_prompt(
            system_id, states, actions, success_rate, time_window_hours
        )

        try:
            # Call LLM for analysis
            messages = [
                LLMMessage(role="system", content=self._get_system_prompt(system_id)),
                LLMMessage(role="user", content=analysis_prompt),
            ]

            llm_start = datetime.now(timezone.utc)
            response = await self.llm.generate(
                messages, temperature=self.temperature, max_tokens=DEFAULT_MAX_TOKENS
            )

            duration = (datetime.now(timezone.utc) - llm_start).total_seconds()
            self.metrics.histogram(
                "polaris.meta_learning.llm.analysis_llm_duration_seconds",
                duration,
                tags={"system_id": system_id},
            )

            # Parse LLM response
            analysis_data = self._parse_analysis_response(response.content)

            insights = {
                "total_states": len(states),
                "total_adaptations": len(actions),
                "success_rate": success_rate,
                "llm_analysis": analysis_data.get("analysis", ""),
                "identified_issues": analysis_data.get("issues", []),
            }

            self.metrics.increment(
                "polaris.meta_learning.llm.analysis_success",
                tags={"system_id": system_id},
            )

            return PerformanceAnalysis(
                system_id=system_id,
                time_window_hours=time_window_hours,
                success_rate=success_rate,
                insights=insights,
                recommendations=analysis_data.get("recommendations", []),
            )

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            self.metrics.increment(
                "polaris.meta_learning.llm.analysis_errors",
                tags={"system_id": system_id},
            )
            # Fallback to basic analysis
            return PerformanceAnalysis(
                system_id=system_id,
                time_window_hours=time_window_hours,
                success_rate=success_rate,
                insights={"total_adaptations": len(actions)},
                recommendations=[],
            )

    async def propose_strategy_updates(
        self, strategy: AdaptationStrategy, analysis: PerformanceAnalysis
    ) -> List[ParameterProposal]:
        """Use LLM to propose parameter updates."""
        self.metrics.increment(
            "polaris.meta_learning.llm.proposals_requests",
            tags={"system_id": analysis.system_id},
        )

        # Get tunable parameters
        tunable_params = strategy.get_tunable_parameters()

        if not tunable_params:
            self.metrics.increment(
                "polaris.meta_learning.llm.proposals_no_tunable_parameters",
                tags={"system_id": analysis.system_id},
            )
            return []

        try:
            strategy_metrics = await strategy.get_performance_metrics()
            if isinstance(strategy_metrics, dict):
                analysis.insights.setdefault("strategy_metrics", strategy_metrics)
        except Exception:
            self.logger.debug("Unable to collect strategy performance metrics")

        # Build prompt for parameter optimization
        prompt = self._build_optimization_prompt(analysis, tunable_params)

        try:
            messages = [
                LLMMessage(
                    role="system", content=self._get_optimization_system_prompt(analysis.system_id)
                ),
                LLMMessage(role="user", content=prompt),
            ]

            response = await self.llm.generate(
                messages, temperature=self.temperature, max_tokens=DEFAULT_MAX_TOKENS
            )

            # Parse proposals
            proposals_data = self._parse_proposals_response(response.content)

            proposals = []
            for prop_data in proposals_data:
                param_path = prop_data.get("parameter")
                if param_path not in tunable_params:
                    continue

                spec = tunable_params[param_path]
                proposed_value = prop_data.get("proposed_value")

                # Validate proposed value
                if not self._validate_proposal(spec, proposed_value):
                    continue

                proposals.append(
                    ParameterProposal(
                        proposal_id=str(uuid.uuid4()),
                        parameter_path=param_path,
                        current_value=spec.current_value,
                        proposed_value=proposed_value,
                        rationale=prop_data.get("rationale", ""),
                        confidence=prop_data.get("confidence", 0.7),
                        expected_impact=prop_data.get("expected_impact", ""),
                        status=ProposalStatus.PENDING,
                        created_at=datetime.now(timezone.utc),
                    )
                )

            self.metrics.gauge(
                "polaris.meta_learning.llm.proposals_generated",
                len(proposals),
                tags={"system_id": analysis.system_id},
            )

            return proposals

        except Exception as e:
            self.logger.error(f"LLM proposal generation failed: {e}")
            self.metrics.increment(
                "polaris.meta_learning.llm.proposals_errors",
                tags={"system_id": analysis.system_id},
            )
            return []

    async def validate_proposals(
        self, proposals: List[ParameterProposal], system_state: Optional[Dict[str, Any]] = None
    ) -> List[ParameterProposal]:
        """Enhanced validation with multi-factor scoring."""
        validated = []
        approved = 0
        rejected = 0

        for index, proposal in enumerate(proposals):
            # Calculate comprehensive validation score
            validation_score = self._calculate_validation_score(proposal, system_state)

            # Update proposal confidence with validation score
            proposal = replace(proposal, confidence=validation_score)
            proposals[index] = proposal

            # Apply configurable threshold
            approval_threshold = 0.7  # Could be configurable

            if validation_score >= approval_threshold:
                proposal.status = ProposalStatus.APPROVED
                validated.append(proposal)
                approved += 1
            else:
                proposal.status = ProposalStatus.REJECTED
                rejected += 1

        self.metrics.gauge(
            "polaris.meta_learning.llm.proposals_approved",
            approved,
        )
        self.metrics.gauge(
            "polaris.meta_learning.llm.proposals_rejected",
            rejected,
        )
        self.metrics.gauge(
            "polaris.meta_learning.llm.validation_avg_score",
            sum(p.confidence for p in proposals) / len(proposals) if proposals else 0,
        )

        avg_score = sum(p.confidence for p in proposals) / len(proposals) if proposals else 0.0
        self.logger.info(
            "Proposal validation completed",
            approved=approved,
            rejected=rejected,
            avg_score=avg_score,
        )

        return validated

    async def apply_proposals(
        self, strategy: AdaptationStrategy, proposals: List[ParameterProposal]
    ) -> List[AppliedUpdate]:
        """Apply approved proposals, gated by the ``auto_apply`` flag.

        When ``auto_apply=False`` (the default) no updates are written to the live
        strategy — proposals are recorded for transparency only. When
        ``auto_apply=True`` each approved proposal is forwarded to
        ``strategy.update_parameter``.
        """
        if not self.auto_apply:
            self.logger.info(
                "Meta-learner auto_apply is disabled — skipping parameter application",
                proposals=len(proposals),
            )
            return []

        results: List[AppliedUpdate] = []
        for proposal in proposals:
            if proposal.status != ProposalStatus.APPROVED:
                continue
            try:
                success = await strategy.update_parameter(
                    proposal.parameter_path, proposal.proposed_value
                )
                results.append(AppliedUpdate(proposal_id=proposal.proposal_id, success=success))
                self.logger.info(
                    "Meta-learner applied parameter update",
                    parameter=proposal.parameter_path,
                    proposed_value=proposal.proposed_value,
                    success=success,
                )
            except Exception as e:
                self.logger.error(
                    "Meta-learner failed to apply parameter update",
                    parameter=proposal.parameter_path,
                    error=str(e),
                )
                results.append(
                    AppliedUpdate(
                        proposal_id=proposal.proposal_id,
                        success=False,
                        error_message=str(e),
                    )
                )
        return results

    def _get_system_prompt(self, system_id: Optional[str] = None) -> str:
        """System prompt for performance analysis, with optional system-specific overrides."""
        # Per-system override if provided
        if system_id and self._per_system_prompts:
            per_system = self._per_system_prompts.get(system_id, {})
            override = per_system.get("analysis_system_prompt")
            if override:
                return override

        # Global analysis prompt override
        if self.analysis_system_prompt:
            try:
                return self.analysis_system_prompt.format(system_id=system_id or "")
            except Exception:
                return self.analysis_system_prompt

        return """You are an expert system analyst for self-adaptive systems.

Analyze the historical performance data and identify:
1. Performance trends and patterns
2. Potential issues or inefficiencies
3. Opportunities for optimization
4. Specific recommendations for improvement

Focus on:
- Response time patterns and anomalies
- Resource utilization trends
- Error rates and reliability issues
- Adaptation effectiveness
- System stability indicators

Respond in JSON format:
{
    "analysis": "detailed analysis of system behavior with specific metrics",
    "issues": ["identified issue 1 with impact assessment", "issue 2", ...],
    "recommendations": ["specific recommendation 1 with expected benefit", "recommendation 2", ...]
}

Be thorough, data-driven, and provide actionable insights. Quantify improvements where possible."""

    def _get_optimization_system_prompt(self, system_id: Optional[str] = None) -> str:
        """System prompt for parameter optimization, with optional system-specific overrides."""
        # Per-system override if provided
        if system_id and self._per_system_prompts:
            per_system = self._per_system_prompts.get(system_id, {})
            override = per_system.get("optimization_system_prompt")
            if override:
                return override

        # Global optimization prompt override
        if self.optimization_system_prompt:
            try:
                return self.optimization_system_prompt.format(system_id=system_id or "")
            except Exception:
                return self.optimization_system_prompt

        return """You are an expert parameter optimizer for self-adaptive systems.

Given performance analysis and tunable parameters, propose specific parameter changes to improve system performance.

Consider:
- Current system performance and bottlenecks
- Parameter sensitivity and interaction effects
- Risk vs reward for each change
- Gradual improvement approach
- System stability and safety

For each parameter change, provide:
- parameter: the parameter path (e.g., "strategy.threshold")
- proposed_value: new value to set (must be within bounds)
- rationale: detailed explanation of why this change will help
- confidence: 0.0-1.0 confidence score (higher for safer, well-justified changes)
- expected_impact: expected improvement with metrics if possible

Respond in JSON format:
{
    "proposals": [
        {
            "parameter": "param.path",
            "proposed_value": value,
            "rationale": "detailed explanation with expected benefits",
            "confidence": 0.8,
            "expected_impact": "expected improvement description with metrics"
        }
    ]
}

Be conservative - prioritize system stability. Only propose changes you're confident will improve performance.
Start with small adjustments."""

    def _build_analysis_prompt(
        self, system_id: str, states: list, actions: list, success_rate: float, time_window: float
    ) -> str:
        """Build prompt for performance analysis."""
        return f"""Analyze the performance of system '{system_id}' over the last {time_window} hours.

**Statistics:**
- Total state observations: {len(states)}
- Total adaptation actions: {len(actions)}
- Adaptation success rate: {success_rate: .1%}

**Recent Adaptations:**
{self._format_recent_actions(actions[-10:])}

**Rolling Window Metrics (Last {min(5, len(states))} intervals):**
{self._analyze_rolling_metrics(states)}

**Overall Trends:**
{self._analyze_metric_trends(states)}

Provide a comprehensive analysis and recommendations for optimization.
"""

    def _analyze_rolling_metrics(self, states: list, num_windows: int = 5) -> str:
        """Analyze metrics over discrete rolling windows to provide rich, compact context."""
        if not states:
            return "No state data available."

        # Group states into temporal windows (simplified by dividing the array)
        window_size = max(1, len(states) // num_windows)
        windows = [states[i : i + window_size] for i in range(0, len(states), window_size)][
            -num_windows:
        ]

        summary = []
        for i, window in enumerate(windows):
            window_metrics: dict[str, list[float]] = {}
            for state in window:
                try:
                    metrics = (
                        getattr(state, "metrics", state) if not isinstance(state, dict) else state
                    )
                    for k, v in (metrics.items() if isinstance(metrics, dict) else {}):
                        if isinstance(v, (int, float)):
                            window_metrics.setdefault(k, []).append(v)
                except Exception:
                    continue

            if window_metrics:
                avgs = {k: sum(v) / len(v) for k, v in window_metrics.items()}
                metrics_str = ", ".join(f"{k}: {v:.2f}" for k, v in avgs.items() if k)
                summary.append(f"Window {i+1}: {metrics_str}")

        return "\n".join(summary) if summary else "Insufficient numeric data for rolling metrics."

    def _build_optimization_prompt(
        self, analysis: PerformanceAnalysis, tunable_params: Dict
    ) -> str:
        """Build prompt for parameter optimization."""
        params_desc = "\n".join(
            [
                f"- {path}: current={spec.current_value}, min={spec.min_value}, max={spec.max_value}"
                for path, spec in tunable_params.items()
            ]
        )

        compact_metrics = self._compact_strategy_metrics(
            analysis.insights.get("strategy_metrics", {})
        )

        return f"""Analysis:
**Success Rate:** {analysis.success_rate: .1%}

**Recommendations:** {', '.join(analysis.recommendations) if analysis.recommendations else 'None'}
**Insights:** {analysis.insights.get('llm_analysis', 'None')}
**Strategy Metrics:** {compact_metrics}

**Tunable Parameters:**
{params_desc}

Propose optimization changes based on the above.
"""

    def _compact_strategy_metrics(self, metrics: dict) -> str:
        """Provide a concise summary of strategy metrics to prevent prompt bloating."""
        if not metrics:
            return "{}"
        summary = []
        for k, v in metrics.items():
            if isinstance(v, list) and v:
                nums = [x for x in v if isinstance(x, (int, float))]
                if nums:
                    summary.append(
                        f"{k}: avg={sum(nums)/len(nums):.2f}, latest={nums[-1]:.2f} (n={len(v)})"
                    )
                else:
                    summary.append(f"{k}: {len(v)} items")
            elif isinstance(v, dict):
                summary.append(f"{k}: {list(v.keys())}")
            else:
                summary.append(f"{k}: {v}")
        return " | ".join(summary)

    def _format_recent_actions(self, actions: list) -> str:
        """Format recent actions for prompt."""
        if not actions:
            return "No recent adaptations"

        lines = []
        for action, result in actions[-5:]:
            status = result.status.value if hasattr(result, "status") else "unknown"
            lines.append(f"  - {action.action_type}: {status}")

        return "\n".join(lines)

    def _analyze_metric_trends(self, states: list) -> str:
        """Analyze metric trends from states with actual statistical analysis."""
        if len(states) < 2:
            return "Insufficient data for trend analysis (need at least 2 data points)"

        try:
            # Extract common metrics from states
            metric_trends = {}

            # Gather all observed metric keys
            all_metric_keys = set()
            for state in states:
                try:
                    if hasattr(state, "metrics"):
                        all_metric_keys.update(state.metrics.keys())
                    elif isinstance(state, dict):
                        all_metric_keys.update(state.keys())
                except (AttributeError, TypeError, KeyError):
                    pass

            if not all_metric_keys:
                return "Unable to extract metrics from state data"

            # Analyze each metric trend
            for metric_key in all_metric_keys:
                values = []
                for state in states:
                    try:
                        if hasattr(state, "metrics") and metric_key in state.metrics:
                            val = state.metrics[metric_key]
                            if isinstance(val, (int, float)):
                                values.append(val)
                        elif isinstance(state, dict) and metric_key in state:
                            val = state[metric_key]
                            if isinstance(val, (int, float)):
                                values.append(val)
                    except (AttributeError, TypeError, KeyError):
                        continue

                if len(values) >= 2:
                    trend = self._calculate_trend_direction(values)
                    change_pct = self._calculate_percentage_change(values)
                    metric_trends[metric_key] = {
                        "direction": trend,
                        "change_pct": change_pct,
                        "latest": values[-1],
                        "average": sum(values) / len(values),
                    }

            if not metric_trends:
                return "No numeric metrics found for trend analysis"

            # Format trend analysis for LLM
            trend_summary = []
            for metric, data in metric_trends.items():
                if data["direction"] == "up":
                    trend_summary.append(
                        f"{metric}: INCREASING (+{data['change_pct']: .1f}% from "
                        f"{data['average']: .2f} to {data['latest']: .2f})"
                    )
                elif data["direction"] == "down":
                    trend_summary.append(
                        f"{metric}: DECREASING ({data['change_pct']: .1f}% from "
                        f"{data['average']: .2f} to {data['latest']: .2f})"
                    )
                else:
                    trend_summary.append(
                        f"{metric}: STABLE (avg: {data['average']: .2f}, current: {data['latest']: .2f})"
                    )

            return "\n".join(trend_summary)

        except Exception as e:
            self.logger.warning(f"Error in metric trend analysis: {e}")
            return f"Trend analysis failed: {str(e)}"

    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse LLM analysis response with improved robustness."""
        try:
            # Extract JSON with better error handling
            response = response.strip()
            if not response:
                return {"analysis": "", "issues": [], "recommendations": []}

            json_content = response

            # Handle ```json blocks
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0].strip()
                    if json_part:
                        json_content = json_part

            # Handle generic ``` blocks
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 3:
                    json_part = parts[1].strip()
                    if json_part:
                        json_content = json_part

            data = json.loads(json_content)

            # Validate structure and provide defaults
            if not isinstance(data, dict):
                return {"analysis": response, "issues": [], "recommendations": []}

            return {
                "analysis": data.get("analysis", ""),
                "issues": data.get("issues", []) if isinstance(data.get("issues"), list) else [],
                "recommendations": (
                    data.get("recommendations", [])
                    if isinstance(data.get("recommendations"), list)
                    else []
                ),
            }

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return {"analysis": response, "issues": [], "recommendations": []}

    def _parse_proposals_response(self, response: str) -> List[Dict]:
        """Parse LLM proposals response with improved robustness."""
        try:
            response = response.strip()
            if not response:
                return []

            json_content = response

            # Handle ```json blocks
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0].strip()
                    if json_part:
                        json_content = json_part

            # Handle generic ``` blocks
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 3:
                    json_part = parts[1].strip()
                    if json_part:
                        json_content = json_part

            data = json.loads(json_content)

            # Validate structure
            if not isinstance(data, dict):
                return []

            proposals = data.get("proposals", [])
            if not isinstance(proposals, list):
                return []

            return proposals

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return []

    def _validate_proposal(self, spec: ParameterSpec, proposed_value: Any) -> bool:
        """Validate proposed value against parameter spec."""
        try:
            # Handle categorical values first so string comparisons do not fall through
            if spec.allowed_values:
                return proposed_value in spec.allowed_values

            # Check type
            if spec.type == float:
                proposed_value = float(proposed_value)
            elif spec.type == int:
                proposed_value = int(proposed_value)
            elif spec.type == str:
                proposed_value = str(proposed_value)

            # Check bounds
            if spec.min_value is not None and proposed_value < spec.min_value:
                return False
            if spec.max_value is not None and proposed_value > spec.max_value:
                return False

            return True
        except Exception:
            return False

    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return "unknown"

        # Simple linear regression to determine trend
        n = len(values)
        x = list(range(n))

        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Determine direction based on slope relative to average value
        avg_value = y_mean
        if avg_value == 0:
            return "stable" if abs(slope) < 0.01 else ("up" if slope > 0 else "down")

        relative_change = abs(slope) / avg_value

        if relative_change < 0.05:  # Less than 5% change per time unit
            return "stable"
        elif slope > 0:
            return "up"
        else:
            return "down"

    def _calculate_percentage_change(self, values: List[float]) -> float:
        """Calculate percentage change from first to last value."""
        if len(values) < 2:
            return 0.0

        first_val = values[0]
        last_val = values[-1]

        if first_val == 0:
            return 0.0 if last_val == 0 else 100.0

        return ((last_val - first_val) / first_val) * 100

    def _calculate_validation_score(
        self, proposal: ParameterProposal, system_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate comprehensive validation score for a proposal."""
        score = proposal.confidence  # Start with LLM confidence

        # Factor 1: Parameter sensitivity (conservative for critical params)
        critical_params = ["threshold", "timeout", "max_retries", "rate_limit"]
        if any(critical in proposal.parameter_path.lower() for critical in critical_params):
            score *= 0.8  # Reduce score for critical parameters

        # Factor 2: Change magnitude (smaller changes are safer)
        try:
            current_val = (
                float(proposal.current_value)
                if isinstance(proposal.current_value, (int, float, str))
                else 0
            )
            proposed_val = (
                float(proposal.proposed_value)
                if isinstance(proposal.proposed_value, (int, float, str))
                else 0
            )

            if current_val != 0:
                change_pct = abs((proposed_val - current_val) / current_val)
                if change_pct > 0.5:  # More than 50% change
                    score *= 0.7
                elif change_pct > 0.2:  # More than 20% change
                    score *= 0.85
        except (ValueError, TypeError):
            pass  # Skip if values aren't numeric

        # Factor 3: System stability (reduce score if system is unstable)
        if system_state:
            try:
                # Look for stability indicators
                error_rate = system_state.get("error_rate", 0)
                cpu_usage = system_state.get("cpu_usage", 0)

                if error_rate > 0.1:  # High error rate
                    score *= 0.8
                if cpu_usage > 0.9:  # High CPU usage
                    score *= 0.85
            except (AttributeError, TypeError):
                pass

        # Factor 4: Rationale quality (longer, more detailed rationales are better)
        if len(proposal.rationale) < 20:
            score *= 0.9  # Penalize short rationales

        # Ensure score stays within bounds
        return max(0.0, min(1.0, score))
