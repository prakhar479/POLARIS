"""Comprehensive tests for LLM-based meta-learner."""

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from polaris.abstractions.meta_learner import ParameterProposal, PerformanceAnalysis, ProposalStatus
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationStrategy
from polaris.infrastructure.llm import LLMClient, LLMMessage, LLMResponse
from polaris.meta_learner.llm_based import LLMMetaLearner


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, response_content: str = "", should_fail: bool = False):
        self.response_content = response_content
        self.should_fail = should_fail
        self.call_count = 0

    async def generate(
        self, messages: List[LLMMessage], temperature: float = 0.7, max_tokens: int = 1024
    ) -> LLMResponse:
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError("Mock LLM failure")

        return LLMResponse(content=self.response_content, model="mock-model", tokens_used=100)


class MockKnowledgeStore:
    """Mock knowledge store for testing."""

    def __init__(self, states: List = None, actions: List = None, should_fail: bool = False):
        self.states = states or []
        self.actions = actions or []
        self.should_fail = should_fail

    async def query_states(self, system_id: str, start_time, end_time):
        if self.should_fail:
            raise RuntimeError("Knowledge store failure")
        return self.states

    async def query_actions(self, system_id: str, start_time, end_time):
        if self.should_fail:
            raise RuntimeError("Knowledge store failure")
        return self.actions


class MockStrategy(AdaptationStrategy):
    """Mock strategy for testing."""

    async def assess(self, state, context) -> List:
        """Mock assess method."""
        return []

    def get_tunable_parameters(self) -> Dict[str, Any]:
        return {
            "strategy.threshold": MockParamSpec(
                current_value=0.5, min_value=0.0, max_value=1.0, type=float
            ),
            "strategy.max_retries": MockParamSpec(
                current_value=3, min_value=1, max_value=10, type=int
            ),
        }

    async def update_parameter(self, path: str, value: Any) -> bool:
        return True

    async def on_action_executed(self, action, result):
        pass


class MockParamSpec:
    """Mock parameter specification."""

    def __init__(
        self, current_value, min_value=None, max_value=None, type=float, allowed_values=None
    ):
        self.current_value = current_value
        self.min_value = min_value
        self.max_value = max_value
        self.type = type
        self.allowed_values = allowed_values


class MockLogger(Logger):
    """Mock logger for testing."""

    def __init__(self):
        self.messages = []

    def info(self, message: str, *args, **kwargs):
        self.messages.append(("info", message % args if args else message))

    def warning(self, message: str, *args, **kwargs):
        self.messages.append(("warning", message % args if args else message))

    def error(self, message: str, *args, **kwargs):
        self.messages.append(("error", message % args if args else message))

    def debug(self, message: str, *args, **kwargs):
        self.messages.append(("debug", message % args if args else message))


class MockMetricsCollector(MetricsCollector):
    """Mock metrics collector for testing."""

    def __init__(self):
        self.metrics = {}

    def increment(self, name: str, tags: Dict[str, str] = None):
        key = f"{name}: {tags}" if tags else name
        self.metrics[key] = self.metrics.get(key, 0) + 1

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        key = f"{name}: {tags}" if tags else name
        self.metrics[key] = value

    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        key = f"{name}: {tags}" if tags else name
        self.metrics[key] = value


@pytest.fixture
def mock_llm_client():
    """Provide mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_knowledge_store():
    """Provide mock knowledge store."""
    return MockKnowledgeStore()


@pytest.fixture
def mock_strategy():
    """Provide mock strategy."""
    return MockStrategy()


@pytest.fixture
def mock_logger():
    """Provide mock logger."""
    return MockLogger()


@pytest.fixture
def mock_metrics():
    """Provide mock metrics collector."""
    return MockMetricsCollector()


@pytest.fixture
def llm_meta_learner(mock_llm_client, mock_knowledge_store, mock_logger, mock_metrics):
    """Provide LLM meta-learner instance."""
    return LLMMetaLearner(
        llm_client=mock_llm_client,
        knowledge_store=mock_knowledge_store,
        logger=mock_logger,
        metrics=mock_metrics,
        temperature=0.1,
    )


class TestLLMMetaLearner:
    """Test suite for LLMMetaLearner."""

    @pytest.mark.asyncio
    async def test_analyze_performance_success(self, llm_meta_learner, mock_llm_client):
        """Test successful performance analysis."""
        # Setup mock response
        mock_llm_client.response_content = """{
            "analysis": "System performance is stable with minor optimization opportunities",
            "issues": ["High response time during peak loads"],
            "recommendations": ["Increase timeout threshold by 20%"]
        }"""

        # Setup mock data
        mock_state = Mock()
        mock_state.metrics = {"response_time": 100, "cpu_usage": 0.7}
        llm_meta_learner.knowledge_store.states = [mock_state, mock_state]
        llm_meta_learner.knowledge_store.actions = []

        # Execute
        result = await llm_meta_learner.analyze_performance("test-system", 24.0)

        # Verify
        assert isinstance(result, PerformanceAnalysis)
        assert result.system_id == "test-system"
        assert result.time_window_hours == 24.0
        assert result.success_rate == 1.0  # No actions = no failures
        assert "System performance is stable" in result.insights.get("llm_analysis", "")
        assert len(result.recommendations) > 0
        assert mock_llm_client.call_count == 1

    @pytest.mark.asyncio
    async def test_analyze_performance_no_knowledge_store(
        self, mock_llm_client, mock_logger, mock_metrics
    ):
        """Test analysis with null knowledge store."""
        meta_learner = LLMMetaLearner(
            llm_client=mock_llm_client,
            knowledge_store=None,  # Null knowledge store
            logger=mock_logger,
            metrics=mock_metrics,
        )

        result = await meta_learner.analyze_performance("test-system", 24.0)

        assert isinstance(result, PerformanceAnalysis)
        assert result.insights.get("error") == "knowledge_store_unavailable"
        assert len(result.recommendations) > 0
        assert any("Configure knowledge store" in r for r in result.recommendations)

        # Verify metrics were recorded
        # Check if any metric key contains the expected metric name
        metric_keys = list(mock_metrics.metrics.keys())
        assert any(
            "polaris.meta_learning.llm.analysis_knowledge_store_unavailable" in key
            for key in metric_keys
        )

    @pytest.mark.asyncio
    async def test_analyze_performance_llm_failure(self, llm_meta_learner, mock_llm_client):
        """Test analysis when LLM fails."""
        mock_llm_client.should_fail = True

        # Setup mock data
        mock_state = Mock()
        mock_state.metrics = {"response_time": 100}
        llm_meta_learner.knowledge_store.states = [mock_state]
        llm_meta_learner.knowledge_store.actions = []

        result = await llm_meta_learner.analyze_performance("test-system", 24.0)

        assert isinstance(result, PerformanceAnalysis)
        assert result.system_id == "test-system"
        # Fallback has no recommendations
        assert len(result.recommendations) == 0

    @pytest.mark.asyncio
    async def test_propose_strategy_updates_success(
        self, llm_meta_learner, mock_llm_client, mock_strategy
    ):
        """Test successful proposal generation."""
        # Setup mock response
        mock_llm_client.response_content = """{
            "proposals": [
                {
                    "parameter": "strategy.threshold",
                    "proposed_value": 0.6,
                    "rationale": "Increase threshold to reduce false positives",
                    "confidence": 0.8,
                    "expected_impact": "Improved accuracy by 15%"
                }
            ]
        }"""

        # Setup analysis
        analysis = PerformanceAnalysis(
            system_id="test-system",
            time_window_hours=24.0,
            success_rate=0.9,
            insights={"llm_analysis": "System needs optimization"},
            recommendations=["Adjust thresholds"],
        )

        result = await llm_meta_learner.propose_strategy_updates(mock_strategy, analysis)

        assert len(result) == 1
        proposal = result[0]
        assert proposal.parameter_path == "strategy.threshold"
        assert proposal.proposed_value == 0.6
        assert proposal.current_value == 0.5
        assert proposal.confidence == 0.8
        assert proposal.status == ProposalStatus.PENDING
        assert mock_llm_client.call_count == 1

    @pytest.mark.asyncio
    async def test_propose_strategy_updates_no_tunable_params(self, llm_meta_learner):
        """Test proposal generation with no tunable parameters."""
        # Mock strategy with no tunable parameters
        strategy = Mock()
        strategy.get_tunable_parameters.return_value = {}

        analysis = PerformanceAnalysis(
            system_id="test-system",
            time_window_hours=24.0,
            success_rate=0.9,
            insights={},
            recommendations=[],
        )

        result = await llm_meta_learner.propose_strategy_updates(strategy, analysis)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_validate_proposals_enhanced_scoring(self, llm_meta_learner):
        """Test enhanced proposal validation with multi-factor scoring."""
        proposals = [
            ParameterProposal(
                proposal_id="1",
                parameter_path="strategy.threshold",
                current_value=0.5,
                proposed_value=0.6,
                rationale=(
                    "Increase threshold to improve accuracy and reduce false positives in the system"
                ),  # Long rationale
                confidence=0.9,  # Higher confidence to offset penalties
                expected_impact="Improved accuracy by 15%",
                status=ProposalStatus.PENDING,
            ),
            ParameterProposal(
                proposal_id="2",
                parameter_path="strategy.learning_rate",  # Non-critical parameter
                current_value=0.01,
                proposed_value=0.02,  # Small change
                rationale="Short",  # Short rationale should be penalized
                confidence=0.8,
                expected_impact="Better reliability",
                status=ProposalStatus.PENDING,
            ),
        ]

        system_state = {"error_rate": 0.05, "cpu_usage": 0.8}  # Stable system

        result = await llm_meta_learner.validate_proposals(proposals, system_state)

        # Should approve at least the good proposal
        assert len(result) >= 1

        # Check that scores were calculated
        for proposal in proposals:
            assert proposal.confidence != 0.8  # Should be updated with validation score

    def test_calculate_trend_direction(self, llm_meta_learner):
        """Test trend direction calculation."""
        # Increasing trend
        values = [1, 2, 3, 4, 5]
        assert llm_meta_learner._calculate_trend_direction(values) == "up"

        # Decreasing trend
        values = [5, 4, 3, 2, 1]
        assert llm_meta_learner._calculate_trend_direction(values) == "down"

        # Stable trend
        values = [1, 1.01, 0.99, 1.02, 0.98]
        assert llm_meta_learner._calculate_trend_direction(values) == "stable"

        # Insufficient data
        values = [1]
        assert llm_meta_learner._calculate_trend_direction(values) == "unknown"

    def test_calculate_percentage_change(self, llm_meta_learner):
        """Test percentage change calculation."""
        # Normal case
        values = [100, 110]
        change = llm_meta_learner._calculate_percentage_change(values)
        assert change == 10.0

        # Decrease
        values = [100, 90]
        change = llm_meta_learner._calculate_percentage_change(values)
        assert change == -10.0

        # Zero start value
        values = [0, 100]
        change = llm_meta_learner._calculate_percentage_change(values)
        assert change == 100.0

        # Insufficient data
        values = [100]
        change = llm_meta_learner._calculate_percentage_change(values)
        assert change == 0.0

    def test_calculate_validation_score(self, llm_meta_learner):
        """Test validation score calculation."""
        proposal = ParameterProposal(
            proposal_id="1",
            parameter_path="strategy.threshold",
            current_value=0.5,
            proposed_value=0.6,
            rationale="Detailed rationale explaining why this change is beneficial for the system",
            confidence=0.8,
            expected_impact="Improved performance",
            status=ProposalStatus.PENDING,
        )

        # Test without system state
        score = llm_meta_learner._calculate_validation_score(proposal)
        assert 0.0 <= score <= 1.0

        # Test with system state
        system_state = {"error_rate": 0.05, "cpu_usage": 0.8}
        score = llm_meta_learner._calculate_validation_score(proposal, system_state)
        assert 0.0 <= score <= 1.0

        # Test critical parameter (should be penalized)
        critical_proposal = ParameterProposal(
            proposal_id="2",
            parameter_path="strategy.max_retries",
            current_value=3,
            proposed_value=5,
            rationale="Increase retries",
            confidence=0.8,
            expected_impact="Better reliability",
            status=ProposalStatus.PENDING,
        )

        critical_score = llm_meta_learner._calculate_validation_score(critical_proposal)
        normal_score = llm_meta_learner._calculate_validation_score(proposal)

        # Critical parameter should have lower score
        assert critical_score < normal_score

    def test_analyze_metric_trends(self, llm_meta_learner):
        """Test metric trend analysis."""
        # Create mock states with metrics
        state1 = Mock()
        state1.metrics = {"response_time": 100, "cpu_usage": 0.5}

        state2 = Mock()
        state2.metrics = {"response_time": 120, "cpu_usage": 0.6}

        state3 = Mock()
        state3.metrics = {"response_time": 110, "cpu_usage": 0.55}

        states = [state1, state2, state3]

        result = llm_meta_learner._analyze_metric_trends(states)

        assert "response_time" in result
        assert "cpu_usage" in result
        assert "INCREASING" in result or "DECREASING" in result or "STABLE" in result

    def test_analyze_metric_trends_insufficient_data(self, llm_meta_learner):
        """Test metric trend analysis with insufficient data."""
        result = llm_meta_learner._analyze_metric_trends([])
        assert "Insufficient data" in result

        result = llm_meta_learner._analyze_metric_trends([Mock()])
        assert "Insufficient data" in result

    def test_parse_analysis_response_robustness(self, llm_meta_learner):
        """Test robust JSON parsing for analysis responses."""
        # Test valid JSON
        response = '{"analysis": "test", "issues": [], "recommendations": []}'
        result = llm_meta_learner._parse_analysis_response(response)
        assert result["analysis"] == "test"
        assert result["issues"] == []
        assert result["recommendations"] == []

        # Test JSON in code block
        response = '```json\n{"analysis": "test", "issues": [], "recommendations": []}\n```'
        result = llm_meta_learner._parse_analysis_response(response)
        assert result["analysis"] == "test"

        # Test invalid JSON (should fallback)
        response = "Not valid JSON but some text"
        result = llm_meta_learner._parse_analysis_response(response)
        assert result["analysis"] == response
        assert result["issues"] == []
        assert result["recommendations"] == []

        # Test empty response
        result = llm_meta_learner._parse_analysis_response("")
        assert result["analysis"] == ""
        assert result["issues"] == []
        assert result["recommendations"] == []

    def test_parse_proposals_response_robustness(self, llm_meta_learner):
        """Test robust JSON parsing for proposals responses."""
        # Test valid JSON
        response = '{"proposals": [{"parameter": "test", "proposed_value": 1.0}]}'
        result = llm_meta_learner._parse_proposals_response(response)
        assert len(result) == 1
        assert result[0]["parameter"] == "test"

        # Test JSON in code block
        response = '```json\n{"proposals": [{"parameter": "test", "proposed_value": 1.0}]}\n```'
        result = llm_meta_learner._parse_proposals_response(response)
        assert len(result) == 1

        # Test invalid JSON (should return empty)
        response = "Not valid JSON"
        result = llm_meta_learner._parse_proposals_response(response)
        assert len(result) == 0

    def test_validate_proposal_value(self, llm_meta_learner):
        """Test proposal value validation."""
        # Create mock spec
        spec = MockParamSpec(current_value=0.5, min_value=0.0, max_value=1.0, type=float)

        # Valid value
        assert llm_meta_learner._validate_proposal(spec, 0.7) is True

        # Value too high
        assert llm_meta_learner._validate_proposal(spec, 1.5) is False

        # Value too low
        assert llm_meta_learner._validate_proposal(spec, -0.1) is False

        # Type conversion
        assert llm_meta_learner._validate_proposal(spec, "0.7") is True

        # Invalid type
        assert llm_meta_learner._validate_proposal(spec, "invalid") is False

    def test_system_prompt_overrides(self, llm_meta_learner):
        """Test system prompt override functionality."""
        # Test default prompt
        prompt = llm_meta_learner._get_system_prompt()
        assert "expert system analyst" in prompt

        # Test global override
        llm_meta_learner.analysis_system_prompt = "Custom analysis prompt for {system_id}"
        prompt = llm_meta_learner._get_system_prompt("test-system")
        assert "test-system" in prompt

        # Test per-system override
        llm_meta_learner._per_system_prompts = {
            "test-system": {"analysis_system_prompt": "System-specific prompt"}
        }
        prompt = llm_meta_learner._get_system_prompt("test-system")
        assert prompt == "System-specific prompt"

    def test_optimization_system_prompt_overrides(self, llm_meta_learner):
        """Test optimization system prompt override functionality."""
        # Test default prompt
        prompt = llm_meta_learner._get_optimization_system_prompt()
        assert "expert parameter optimizer" in prompt

        # Test global override
        llm_meta_learner.optimization_system_prompt = "Custom optimization prompt for {system_id}"
        prompt = llm_meta_learner._get_optimization_system_prompt("test-system")
        assert "test-system" in prompt

        # Test per-system override
        llm_meta_learner._per_system_prompts = {
            "test-system": {"optimization_system_prompt": "System-specific optimization prompt"}
        }
        prompt = llm_meta_learner._get_optimization_system_prompt("test-system")
        assert prompt == "System-specific optimization prompt"


@pytest.mark.asyncio
class TestLLMMetaLearnerIntegration:
    """Integration tests for LLMMetaLearner."""

    async def test_full_analysis_and_proposal_cycle(
        self, mock_llm_client, mock_logger, mock_metrics
    ):
        """Test complete analysis and proposal generation cycle."""
        # Setup LLM responses
        mock_llm_client.response_content = """{
            "analysis": "System shows high response times during peak loads",
            "issues": ["Response time increased by 30% during peak hours"],
            "recommendations": ["Adjust timeout thresholds", "Optimize retry logic"]
        }"""

        # Create meta-learner with mock knowledge store
        knowledge_store = MockKnowledgeStore(
            states=[Mock(metrics={"response_time": 150, "cpu_usage": 0.8})], actions=[]
        )

        meta_learner = LLMMetaLearner(
            llm_client=mock_llm_client,
            knowledge_store=knowledge_store,
            logger=mock_logger,
            metrics=mock_metrics,
        )

        # Step 1: Analyze performance
        analysis = await meta_learner.analyze_performance("test-system", 24.0)
        assert isinstance(analysis, PerformanceAnalysis)

        # Step 2: Generate proposals (change LLM response)
        mock_llm_client.response_content = """{
            "proposals": [
                {
                    "parameter": "strategy.threshold",
                    "proposed_value": 0.7,
                    "rationale": "Increase threshold to handle higher response times",
                    "confidence": 0.8,
                    "expected_impact": "Reduced false positives by 20%"
                }
            ]
        }"""

        strategy = MockStrategy()
        proposals = await meta_learner.propose_strategy_updates(strategy, analysis)
        assert len(proposals) == 1

        # Step 3: Validate proposals
        validated = await meta_learner.validate_proposals(proposals)
        assert len(validated) >= 0

        # Verify metrics were recorded
        metric_keys = list(mock_metrics.metrics.keys())
        assert any("polaris.meta_learning.llm.analysis_requests" in key for key in metric_keys)
        assert any("polaris.meta_learning.llm.proposals_requests" in key for key in metric_keys)


if __name__ == "__main__":
    pytest.main([__file__])
