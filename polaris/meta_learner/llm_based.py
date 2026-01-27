"""
LLM-based meta-learner using AI for intelligent parameter tuning.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone
import uuid
import json

from polaris.abstractions.meta_learner import (
    MetaLearner,
    ParameterProposal,
    PerformanceAnalysis,
    ProposalStatus
)
from polaris.abstractions.strategy import AdaptationStrategy
from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger
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
        temperature: float = 0.1
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

    async def analyze_performance(
        self,
        system_id: str,
        time_window_hours: float = 24.0
    ) -> PerformanceAnalysis:
        """Analyze system performance using LLM."""

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
                LLMMessage(role="system", content=self._get_system_prompt()),
                LLMMessage(role="user", content=analysis_prompt)
            ]

            response = await self.llm.generate(
                messages,
                temperature=self.temperature,
                max_tokens=1024
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

            return PerformanceAnalysis(
                system_id=system_id,
                time_window_hours=time_window_hours,
                success_rate=success_rate,
                insights=insights,
                recommendations=analysis_data.get('recommendations', [])
            )

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
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

        # Get tunable parameters
        tunable_params = strategy.get_tunable_parameters()

        if not tunable_params:
            return []

        # Build prompt for parameter optimization
        prompt = self._build_optimization_prompt(
            analysis,
            tunable_params
        )

        try:
            messages = [
                LLMMessage(role="system",
                           content=self._get_optimization_system_prompt()),
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

                # Validate proposed value
                if not self._validate_proposal(spec, proposed_value):
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

            return proposals

        except Exception as e:
            self.logger.error(f"LLM proposal generation failed: {e}")
            return []

    async def validate_proposals(
        self,
        proposals: List[ParameterProposal]
    ) -> List[ParameterProposal]:
        """Validate proposals (approve high-confidence ones)."""

        validated = []
        for proposal in proposals:
            if proposal.confidence >= 0.7:
                proposal.status = ProposalStatus.APPROVED
                validated.append(proposal)
            else:
                proposal.status = ProposalStatus.REJECTED

        return validated

    def _get_system_prompt(self) -> str:
        """System prompt for performance analysis."""
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

    def _get_optimization_system_prompt(self) -> str:
        """System prompt for parameter optimization."""
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
            
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
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
            
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return []

    def _validate_proposal(self, spec, proposed_value) -> bool:
        """Validate proposed value against parameter spec."""
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
                return False

            return True
        except:
            return False
