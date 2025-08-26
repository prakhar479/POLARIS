"""
Test suite for Meta-Learner Agent interface and example implementation.

This module provides comprehensive tests for the meta-learner agent interface,
validating the abstract contract and testing the example implementation.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from polaris.agents.meta_learner_agent import (
    BaseMetaLearnerAgent,
    TriggerType,
    ParameterType,
    MetaLearningContext,
    ParameterUpdate,
    CalibrationRequest,
    CalibrationResult,
    MetaLearningInsights,
    MetaLearningError,
)
from polaris.agents.example_meta_learner import ExampleMetaLearnerAgent


class TestMetaLearnerInterface:
    """Test the abstract Meta-Learner Agent interface."""

    def test_abstract_interface_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetaLearnerAgent("test-agent")

    def test_trigger_type_enum(self):
        """Test TriggerType enum values."""
        assert TriggerType.PERIODIC == "periodic"
        assert TriggerType.EVENT_DRIVEN == "event_driven"
        assert TriggerType.PERFORMANCE_DRIVEN == "performance_driven"
        assert TriggerType.THRESHOLD_VIOLATION == "threshold_violation"
        assert TriggerType.MANUAL == "manual"

    def test_parameter_type_enum(self):
        """Test ParameterType enum values."""
        assert ParameterType.UTILITY_WEIGHTS == "utility_weights"
        assert ParameterType.POLICY_PARAMETERS == "policy_parameters"
        assert ParameterType.COORDINATION_STRATEGIES == "coordination_strategies"
        assert ParameterType.THRESHOLD_VALUES == "threshold_values"
        assert ParameterType.CONTROL_GAINS == "control_gains"
        assert ParameterType.LEARNING_RATES == "learning_rates"


class TestMetaLearningDataModels:
    """Test the data models used by meta-learning agents."""

    def test_meta_learning_context_creation(self):
        """Test MetaLearningContext model creation."""
        context = MetaLearningContext(
            trigger_type=TriggerType.PERIODIC, trigger_source="scheduler"
        )

        assert context.trigger_type == TriggerType.PERIODIC
        assert context.trigger_source == "scheduler"
        assert context.time_window_hours == 24.0  # default
        assert context.focus_areas == []  # default
        assert context.constraints == {}  # default

    def test_parameter_update_creation(self):
        """Test ParameterUpdate model creation."""
        update = ParameterUpdate(
            parameter_type=ParameterType.UTILITY_WEIGHTS,
            parameter_path="coordinator.utility.performance",
            old_value=0.5,
            new_value=0.6,
            confidence=0.85,
            reasoning="Performance degradation detected",
            expected_impact="Faster adaptation responses",
            risk_assessment="Low risk",
        )

        assert update.parameter_type == ParameterType.UTILITY_WEIGHTS
        assert update.parameter_path == "coordinator.utility.performance"
        assert update.old_value == 0.5
        assert update.new_value == 0.6
        assert update.confidence == 0.85
        assert update.update_id is not None  # auto-generated
        assert isinstance(update.timestamp, datetime)

    def test_calibration_request_creation(self):
        """Test CalibrationRequest model creation."""
        request = CalibrationRequest(
            target_metrics=["cpu_usage", "memory_usage"],
            calibration_data={"historical": "data"},
            validation_window_hours=2.0,
        )

        assert request.target_metrics == ["cpu_usage", "memory_usage"]
        assert request.calibration_data == {"historical": "data"}
        assert request.validation_window_hours == 2.0
        assert request.request_id is not None  # auto-generated

    def test_meta_learning_insights_creation(self):
        """Test MetaLearningInsights model creation."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        end_time = datetime.now(timezone.utc)

        insights = MetaLearningInsights(
            analysis_window={"start": start_time, "end": end_time},
            adaptation_patterns=[{"pattern": "scale_out", "frequency": 5}],
            performance_trends={"response_time": 1.2},
            coordination_effectiveness={"auction_success": 0.9},
            recommendations=["Increase performance weight"],
            confidence_overall=0.85,
        )

        assert insights.analysis_window["start"] == start_time
        assert insights.analysis_window["end"] == end_time
        assert len(insights.adaptation_patterns) == 1
        assert insights.performance_trends["response_time"] == 1.2
        assert insights.confidence_overall == 0.85
        assert insights.insights_id is not None  # auto-generated


class TestExampleMetaLearnerAgent:
    """Test the example meta-learner implementation."""

    @pytest.fixture
    def mock_kb_client(self):
        """Mock knowledge base client."""
        return MagicMock()

    @pytest.fixture
    def mock_world_model_client(self):
        """Mock world model client."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_kb_client, mock_world_model_client):
        """Create test agent instance."""
        return ExampleMetaLearnerAgent(
            agent_id="test-agent",
            knowledge_base_client=mock_kb_client,
            world_model_client=mock_world_model_client,
            config={"min_confidence_threshold": 0.7, "analysis_window_hours": 24.0},
        )

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "test-agent"
        assert agent.min_confidence_threshold == 0.7
        assert agent.analysis_window_hours == 24.0
        assert agent.applied_updates_count == 0
        assert not agent.running

    @pytest.mark.asyncio
    async def test_analyze_adaptation_patterns(self, agent):
        """Test adaptation pattern analysis."""
        context = MetaLearningContext(
            trigger_type=TriggerType.PERIODIC, trigger_source="test"
        )

        insights = await agent.analyze_adaptation_patterns(context)

        assert isinstance(insights, MetaLearningInsights)
        assert insights.confidence_overall > 0.0
        assert len(insights.adaptation_patterns) >= 0
        assert agent.last_analysis_time is not None

    @pytest.mark.asyncio
    async def test_calibrate_world_model(self, agent):
        """Test world model calibration."""
        request = CalibrationRequest(
            target_metrics=["cpu_usage", "response_time"],
            calibration_data={"validation_data": {}},
            validation_window_hours=1.0,
        )

        result = await agent.calibrate_world_model(request)

        assert isinstance(result, CalibrationResult)
        assert result.request_id == request.request_id
        assert result.success
        assert 0.0 <= result.improvement_score <= 1.0
        assert agent.last_calibration_time is not None

    @pytest.mark.asyncio
    async def test_propose_parameter_updates(self, agent):
        """Test parameter update proposal."""
        insights = MetaLearningInsights(
            analysis_window={
                "start": datetime.now(timezone.utc),
                "end": datetime.now(timezone.utc),
            },
            performance_trends={"response_time": 1.2},  # Degraded performance
            coordination_effectiveness={"auction_success_rate": 0.75},  # Low success
            confidence_overall=0.9,
        )

        context = MetaLearningContext(
            trigger_type=TriggerType.PERFORMANCE_DRIVEN, trigger_source="test"
        )

        updates = await agent.propose_parameter_updates(insights, context)

        assert isinstance(updates, list)
        for update in updates:
            assert isinstance(update, ParameterUpdate)
            assert update.confidence >= agent.min_confidence_threshold
            assert update.reasoning is not None
            assert update.expected_impact is not None

    @pytest.mark.asyncio
    async def test_validate_updates(self, agent):
        """Test parameter update validation."""
        updates = [
            ParameterUpdate(
                parameter_type=ParameterType.UTILITY_WEIGHTS,
                parameter_path="test.parameter",
                old_value=0.5,
                new_value=0.6,
                confidence=0.8,
                reasoning="Test update",
                expected_impact="Test impact",
                risk_assessment="Low risk",
            )
        ]

        validated = await agent.validate_updates(updates)

        assert isinstance(validated, list)
        assert len(validated) <= len(updates)  # May filter some updates
        for update in validated:
            assert update.confidence >= agent.min_confidence_threshold

    @pytest.mark.asyncio
    async def test_apply_updates(self, agent):
        """Test parameter update application."""
        updates = [
            ParameterUpdate(
                parameter_type=ParameterType.UTILITY_WEIGHTS,
                parameter_path="test.parameter",
                old_value=0.5,
                new_value=0.6,
                confidence=0.8,
                reasoning="Test update",
                expected_impact="Test impact",
                risk_assessment="Low risk",
            )
        ]

        results = await agent.apply_updates(updates)

        assert isinstance(results, dict)
        assert len(results) == len(updates)
        for update_id, success in results.items():
            assert isinstance(update_id, str)
            assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_handle_trigger_periodic(self, agent):
        """Test handling periodic trigger."""
        trigger_data = {"source": "scheduler", "time_window_hours": 12.0}

        result = await agent.handle_trigger(TriggerType.PERIODIC, trigger_data)

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_handle_trigger_performance_driven(self, agent):
        """Test handling performance-driven trigger."""
        trigger_data = {
            "source": "monitoring",
            "metrics": {"response_time": 1.5, "error_rate": 0.05},
            "focus_areas": ["performance"],
        }

        result = await agent.handle_trigger(
            TriggerType.PERFORMANCE_DRIVEN, trigger_data
        )

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_get_learning_status(self, agent):
        """Test getting learning status."""
        status = await agent.get_learning_status()

        assert isinstance(status, dict)
        assert "agent_id" in status
        assert "running" in status
        assert "total_updates_applied" in status
        assert status["agent_id"] == agent.agent_id

    @pytest.mark.asyncio
    async def test_configure_triggers(self, agent):
        """Test trigger configuration."""
        trigger_config = {
            TriggerType.PERIODIC: {"interval_hours": 12.0},
            TriggerType.PERFORMANCE_DRIVEN: {"threshold": 1.2},
        }

        result = await agent.configure_triggers(trigger_config)

        assert isinstance(result, bool)
        assert result  # Should succeed

    @pytest.mark.asyncio
    async def test_export_learning_history(self, agent):
        """Test learning history export."""
        history = await agent.export_learning_history(time_window_hours=24.0)

        assert isinstance(history, dict)
        assert "agent_id" in history
        assert "export_timestamp" in history
        assert "time_window_hours" in history
        assert "history" in history
        assert history["agent_id"] == agent.agent_id


class TestErrorHandling:
    """Test error handling in meta-learning operations."""

    @pytest.fixture
    def failing_agent(self):
        """Create an agent that fails operations for testing."""

        class FailingMetaLearnerAgent(BaseMetaLearnerAgent):
            async def analyze_adaptation_patterns(self, context):
                raise MetaLearningError("Analysis failed")

            async def calibrate_world_model(self, request):
                raise Exception("Calibration failed")

            async def propose_parameter_updates(self, insights, context):
                raise Exception("Update proposal failed")

            async def validate_updates(self, updates, context=None):
                raise Exception("Validation failed")

            async def apply_updates(self, updates):
                raise Exception("Application failed")

            async def handle_trigger(self, trigger_type, trigger_data):
                raise Exception("Trigger handling failed")

        return FailingMetaLearnerAgent("failing-agent")

    @pytest.mark.asyncio
    async def test_analysis_error_handling(self, failing_agent):
        """Test error handling in pattern analysis."""
        context = MetaLearningContext(
            trigger_type=TriggerType.PERIODIC, trigger_source="test"
        )

        with pytest.raises(MetaLearningError):
            await failing_agent.analyze_adaptation_patterns(context)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
