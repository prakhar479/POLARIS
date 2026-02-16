"""
LLM-based meta-learner using AI for intelligent parameter tuning.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import uuid
import json

from polaris.abstractions.meta_learner import (
    MetaLearner,
    ParameterProposal,
    PerformanceAnalysis,
    ProposalStatus,
    AppliedUpdate
)
from polaris.abstractions.strategy import AdaptationStrategy
from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.infrastructure.llm import LLMClient, LLMMessage


class LLMMetaLearner(MetaLearner):
    """
    LLM-powered meta-learner for intelligent strategy optimization.

    Uses an LLM to analyze system performance and propose parameter
    updates using natural language reasoning and deep analysis.
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
        """
        Initialize LLM meta-learner.

        Args:
            llm_client: LLM client for reasoning
            knowledge_store: Knowledge store for historical data
            logger: Logger instance
            auto_apply: Whether to auto-apply approved changes
            temperature: LLM temperature (lower = more conservative)
        """
        self.llm = llm_client
        self.knowledge_store = knowledge_store
        self.logger = logger
        self.auto_apply = auto_apply
        self.temperature = temperature
        self.metrics = metrics
        self.analysis_system_prompt = analysis_system_prompt
        self.optimization_system_prompt = optimization_system_prompt
        self._per_system_prompts = per_system_prompts or {}

    async def analyze_performance(
        self,
        system_id: str,
        time_window_hours: float = 24.0
    ) -> PerformanceAnalysis:
        """Analyze system performance using LLM."""

        if self.metrics:
            self.metrics.increment(
                "polaris.meta_learning.llm.analysis_requests",
                tags={"system_id": system_id},
            )

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=time_window_hours)

        # Query historical data
        states = await self.knowledge_store.query_states(
            system_id, start_time, end_time
        )
        actions = await self.knowledge_store.query_actions(
            system_id, start_time, end_time
        )

        # Calculate basic statistics
        if actions:
            successful = sum(
                1 for _, result in actions
                if hasattr(result, 'status') and result.status.value == 'success'
            )
            success_rate = successful / len(actions)
        else:
            success_rate = 1.0  # No actions = no failures

        # Prepare data for LLM analysis
        analysis_prompt = self._build_analysis_prompt(
            system_id,
            states,
            actions,
            success_rate,
            time_window_hours
        )

        try:
            # Call LLM for analysis
            messages = [
                LLMMessage(role="system", content=self._get_system_prompt(system_id)),
                LLMMessage(role="user", content=analysis_prompt)
            ]

            llm_start = datetime.now(timezone.utc)
            response = await self.llm.generate(
                messages,
                temperature=self.temperature,
                max_tokens=1024
            )

            if self.metrics:
                duration = (datetime.now(timezone.utc) - llm_start).total_seconds()
                self.metrics.histogram(
                    "polaris.meta_learning.llm.analysis_llm_duration_seconds",
                    duration,
                    tags={"system_id": system_id},
                )

            # Parse LLM response
            analysis_data = self._parse_analysis_response(response.content)

            insights = {
                'total_states': len(states),
                'total_adaptations': len(actions),
                'success_rate': success_rate,
                'llm_analysis': analysis_data.get('analysis', ''),
                'identified_issues': analysis_data.get('issues', [])
            }

            if self.metrics:
                self.metrics.increment(
                    "polaris.meta_learning.llm.analysis_success",
                    tags={"system_id": system_id},
                )

            return PerformanceAnalysis(
                system_id=system_id,
                time_window_hours=time_window_hours,
                success_rate=success_rate,
                insights=insights,
                recommendations=analysis_data.get('recommendations', [])
            )

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            if self.metrics:
                self.metrics.increment(
                    "polaris.meta_learning.llm.analysis_errors",
                    tags={"system_id": system_id},
                )
            # Fallback to basic analysis
            return PerformanceAnalysis(
                system_id=system_id,
                time_window_hours=time_window_hours,
                success_rate=success_rate,
                insights={'total_adaptations': len(actions)},
                recommendations=[]
            )

    async def propose_strategy_updates(
        self,
        strategy: AdaptationStrategy,
        analysis: PerformanceAnalysis
    ) -> List[ParameterProposal]:
        """Use LLM to propose parameter updates."""

        if self.metrics:
            self.metrics.increment(
                "polaris.meta_learning.llm.proposals_requests",
                tags={"system_id": analysis.system_id},
            )

        # Get tunable parameters
        tunable_params = strategy.get_tunable_parameters()

        if not tunable_params:
            if self.metrics:
                self.metrics.increment(
                    "polaris.meta_learning.llm.proposals_no_tunable_parameters",
                    tags={"system_id": analysis.system_id},
                )
            return []

        # Build prompt for parameter optimization
        prompt = self._build_optimization_prompt(
            analysis,
            tunable_params
        )

        try:
            messages = [
                LLMMessage(role="system",
                           content=self._get_optimization_system_prompt(analysis.system_id)),
                LLMMessage(role="user", content=prompt)
            ]

            response = await self.llm.generate(
                messages,
                temperature=self.temperature,
                max_tokens=1024
            )

            # Parse proposals
            proposals_data = self._parse_proposals_response(response.content)

            proposals = []
            for prop_data in proposals_data:
                param_path = prop_data.get('parameter')
                if param_path not in tunable_params:
                    continue

                spec = tunable_params[param_path]
                proposed_value = prop_data.get('proposed_value')

                # Validate proposed value and coerce type
                proposed_value = self._validate_proposal(spec, proposed_value)
                if proposed_value is None:
                    continue

                proposals.append(ParameterProposal(
                    proposal_id=str(uuid.uuid4()),
                    parameter_path=param_path,
                    current_value=spec.current_value,
                    proposed_value=proposed_value,
                    rationale=prop_data.get('rationale', ''),
                    confidence=prop_data.get('confidence', 0.7),
                    expected_impact=prop_data.get('expected_impact', ''),
                    status=ProposalStatus.PENDING,
                    created_at=datetime.now(timezone.utc)
                ))

            if self.metrics:
                self.metrics.gauge(
                    "polaris.meta_learning.llm.proposals_generated",
                    len(proposals),
                    tags={"system_id": analysis.system_id},
                )

            return proposals

        except Exception as e:
            self.logger.error(f"LLM proposal generation failed: {e}")
            if self.metrics:
                self.metrics.increment(
                    "polaris.meta_learning.llm.proposals_errors",
                    tags={"system_id": analysis.system_id},
                )
            return []

    async def validate_proposals(
        self,
        proposals: List[ParameterProposal]
    ) -> List[ParameterProposal]:
        """Validate proposals (approve high-confidence ones)."""

        approved = 0
        rejected = 0

        validated = []
        for proposal in proposals:
            if proposal.confidence >= 0.7:
                proposal.status = ProposalStatus.APPROVED
                validated.append(proposal)
                approved += 1
            else:
                proposal.status = ProposalStatus.REJECTED
                rejected += 1

        if self.metrics:
            self.metrics.gauge(
                "polaris.meta_learning.llm.proposals_approved",
                approved,
            )
            self.metrics.gauge(
                "polaris.meta_learning.llm.proposals_rejected",
                rejected,
            )

        return validated

    async def apply_proposals(
        self,
        strategy: AdaptationStrategy,
        proposals: List[ParameterProposal]
    ) -> List[AppliedUpdate]:
        """Apply approved proposals only when auto_apply is enabled."""
        if not self.auto_apply:
            return []
        return await super().apply_proposals(strategy, proposals)

    def _get_system_prompt(self, system_id: Optional[str] = None) -> str:
        """System prompt for performance analysis, with optional system-specific overrides."""

        # Per-system override if provided
        if system_id and self._per_system_prompts:
            per_system = self._per_system_prompts.get(system_id, {})
            override = per_system.get('analysis_system_prompt')
            if override:
                return override

        # Global analysis prompt override
        if self.analysis_system_prompt:
            try:
                return self.analysis_system_prompt.format(system_id=system_id or "")
            except Exception as e:
                self.logger.warning(f"analysis_system_prompt formatting failed: {e}")
                return self.analysis_system_prompt

        return """You are an expert system analyst for self-adaptive systems.

Analyze the historical performance data and identify:
1. Performance trends and patterns
2. Potential issues or inefficiencies
3. Opportunities for optimization
4. Specific recommendations for improvement

Respond in JSON format:
{
    "analysis": "detailed analysis of system behavior",
    "issues": ["identified issue 1", "issue 2", ...],
    "recommendations": ["recommendation 1", "recommendation 2", ...]
}

Be thorough and data-driven in your analysis."""

    def _get_optimization_system_prompt(self, system_id: Optional[str] = None) -> str:
        """System prompt for parameter optimization, with optional system-specific overrides."""

        # Per-system override if provided
        if system_id and self._per_system_prompts:
            per_system = self._per_system_prompts.get(system_id, {})
            override = per_system.get('optimization_system_prompt')
            if override:
                return override

        # Global optimization prompt override
        if self.optimization_system_prompt:
            try:
                return self.optimization_system_prompt.format(system_id=system_id or "")
            except Exception as e:
                self.logger.warning(f"optimization_system_prompt formatting failed: {e}")
                return self.optimization_system_prompt

        return """You are an expert parameter optimizer for self-adaptive systems.

Given performance analysis and tunable parameters, propose specific parameter changes to improve system performance.

For each parameter change, provide:
- parameter: the parameter path
- proposed_value: new value to set
- rationale: why this change will help
- confidence: 0.0-1.0 confidence score
- expected_impact: expected improvement

Respond in JSON format:
{
    "proposals": [
        {
            "parameter": "param.path",
            "proposed_value": value,
            "rationale": "explanation",
            "confidence": 0.8,
            "expected_impact": "description"
        }
    ]
}

Be conservative - only propose changes you're confident will improve performance."""

    def _build_analysis_prompt(
        self,
        system_id: str,
        states: list,
        actions: list,
        success_rate: float,
        time_window: float
    ) -> str:
        """Build prompt for performance analysis."""

        return f"""Analyze the performance of system '{system_id}' over the last {time_window} hours.

**Statistics:**
- Total state observations: {len(states)}
- Total adaptation actions: {len(actions)}
- Adaptation success rate: {success_rate:.1%}

**Recent Adaptations:**
{self._format_recent_actions(actions[-10:])}

**Performance Trends:**
{self._analyze_metric_trends(states)}

Provide a comprehensive analysis and recommendations for optimization.
"""

    def _build_optimization_prompt(
        self,
        analysis: PerformanceAnalysis,
        tunable_params: Dict
    ) -> str:
        """Build prompt for parameter optimization."""

        params_desc = "\n".join([
            f"- {path}: current={spec.current_value}, min={spec.min_value}, max={spec.max_value}"
            for path, spec in tunable_params.items()
        ])

        return f"""Based on this performance analysis:

**Success Rate:** {analysis.success_rate:.1%}
**Recommendations:** {', '.join(analysis.recommendations)}
**Insights:** {analysis.insights.get('llm_analysis', 'No analysis available')}

**Tunable Parameters:**
{params_desc}

Propose specific parameter changes to improve system performance.
"""

    def _format_recent_actions(self, actions: list) -> str:
        """Format recent actions for prompt."""
        if not actions:
            return "No recent adaptations"

        lines = []
        for action, result in actions[-5:]:
            status = result.status.value if hasattr(
                result, 'status') else 'unknown'
            lines.append(f"  - {action.action_type}: {status}")

        return "\n".join(lines)

    def _analyze_metric_trends(self, states: list) -> str:
        """Analyze metric trends from states."""
        if len(states) < 2:
            return "Insufficient data for trend analysis"

        # Simple trend analysis
        return f"Analyzed {len(states)} state snapshots"

    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse LLM analysis response with improved robustness."""
        try:
            # Extract JSON with better error handling
            response = response.strip()
            if not response:
                return {'analysis': '', 'issues': [], 'recommendations': []}
            
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
                return {'analysis': response, 'issues': [], 'recommendations': []}
            
            return {
                'analysis': data.get('analysis', ''),
                'issues': data.get('issues', []) if isinstance(data.get('issues'), list) else [],
                'recommendations': data.get('recommendations', []) if isinstance(data.get('recommendations'), list) else []
            }
            
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse analysis response JSON: {e}")
            return {'analysis': response, 'issues': [], 'recommendations': []}

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
            
            proposals = data.get('proposals', [])
            if not isinstance(proposals, list):
                return []
            
            return proposals
            
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse proposals response JSON: {e}")
            return []

    def _validate_proposal(self, spec, proposed_value) -> Optional[Any]:
        """Validate proposed value against parameter spec and return coerced value."""
        try:
            # Check type
            if spec.type == float:
                proposed_value = float(proposed_value)
            elif spec.type == int:
                proposed_value = int(proposed_value)

            # Check bounds
            if spec.min_value is not None and proposed_value < spec.min_value:
                return False
            if spec.max_value is not None and proposed_value > spec.max_value:
                return False

            # Check allowed values
            if spec.allowed_values and proposed_value not in spec.allowed_values:
                return None

            return proposed_value
        except Exception:
            return None
