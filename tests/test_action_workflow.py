"""Tests for ActionWorkflow lifecycle, verification observation windows, and automated rollback."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.system_contract import ActionSchema, SLOContract, SystemContract
from polaris.core.adaptation_pipeline import AdaptationPipeline
from polaris.core.events import InMemoryEventBus, WorkflowEvent
from polaris.core.models import (
    ActionWorkflow,
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
    WorkflowStatus,
)
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector
from polaris.knowledge.memory import InMemoryKnowledgeStore
from tests.conftest import MockLogger


@pytest.fixture
def mock_logger():
    return MockLogger()


@pytest.fixture
def mock_event_bus():
    return InMemoryEventBus()


@pytest.fixture
def knowledge_store():
    return InMemoryKnowledgeStore()


class TestActionWorkflowModel:
    """Test ActionWorkflow domain model."""

    def test_workflow_initialization_defaults(self):
        action = AdaptationAction(
            action_id="act-1",
            action_type="scale_up",
            target_system="sys-1",
        )
        workflow = ActionWorkflow(action=action)

        assert workflow.workflow_id != ""
        assert workflow.action == action
        assert workflow.status == WorkflowStatus.PENDING
        assert workflow.rollback_action is None
        assert workflow.verification_window_seconds == 0.0
        assert workflow.execution_result is None
        assert workflow.rollback_result is None
        assert workflow.created_at is not None
        assert workflow.updated_at is not None

    def test_workflow_inherits_action_fields(self):
        rollback = AdaptationAction(
            action_id="act-rb",
            action_type="scale_down",
            target_system="sys-1",
        )
        action = AdaptationAction(
            action_id="act-1",
            action_type="scale_up",
            target_system="sys-1",
            rollback_action=rollback,
            verification_window_seconds=2.5,
        )
        workflow = ActionWorkflow(action=action)

        assert workflow.rollback_action == rollback
        assert workflow.verification_window_seconds == 2.5

    def test_workflow_transition(self):
        action = AdaptationAction(action_id="1", action_type="scale", target_system="sys-1")
        workflow = ActionWorkflow(action=action)

        t0 = workflow.updated_at
        workflow.transition_to(WorkflowStatus.EXECUTING)
        assert workflow.status == WorkflowStatus.EXECUTING
        assert workflow.updated_at >= t0

        workflow.transition_to(WorkflowStatus.FAILED, error="Connection reset")
        assert workflow.status == WorkflowStatus.FAILED
        assert workflow.error_message == "Connection reset"


class TestAdaptationPipelineWorkflow:
    """Test AdaptationPipeline with ActionWorkflow execution, verification, and rollback."""

    @pytest.mark.asyncio
    async def test_successful_action_with_verification_window(
        self, mock_logger, mock_event_bus, knowledge_store
    ):
        """Action executes and succeeds post-verification."""
        await mock_event_bus.start()
        workflow_events = []
        mock_event_bus.subscribe(WorkflowEvent, lambda ev: workflow_events.append(ev))

        action = AdaptationAction(
            action_id="act-1",
            action_type="scale_up",
            target_system="sys-1",
            verification_window_seconds=0.01,
        )

        strategy = Mock()
        strategy.requires_system_contract = False
        strategy.assess = AsyncMock(return_value=[action])
        strategy.on_action_executed = AsyncMock()

        connector = Mock()
        connector.validate_action = AsyncMock(return_value=True)
        connector.execute_action = AsyncMock(
            return_value=ExecutionResult(
                action_id="act-1",
                status=ExecutionStatus.SUCCESS,
                result_data={"scaled": True},
            )
        )
        # Post-action telemetry is healthy
        post_state = SystemState(
            system_id="sys-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"response_time": MetricValue("response_time", 0.25)},
            health_status=HealthStatus.HEALTHY,
        )
        connector.collect_telemetry = AsyncMock(return_value=post_state)

        pipeline = AdaptationPipeline(
            strategy=strategy,
            knowledge_store=knowledge_store,
            world_model=None,
            event_bus=mock_event_bus,
            logger=mock_logger,
            metrics=NullMetricsCollector(),
            config=Mock(systems=[]),
        )

        initial_state = SystemState(
            system_id="sys-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"response_time": MetricValue("response_time", 0.9)},
            health_status=HealthStatus.WARNING,
        )

        result = await pipeline.run(initial_state, connector)
        assert result is True

        # Check stored workflows
        stored_workflows = await knowledge_store.query_workflows("sys-1")
        assert len(stored_workflows) == 1
        wf = stored_workflows[0]
        assert wf.status == WorkflowStatus.COMPLETED
        assert wf.execution_result is not None
        assert wf.execution_result.status == ExecutionStatus.SUCCESS

        # Check workflow events
        statuses = [ev.current_status for ev in workflow_events]
        assert WorkflowStatus.VALIDATING in statuses
        assert WorkflowStatus.COMPLETED in statuses

    @pytest.mark.asyncio
    async def test_action_verification_degradation_triggers_rollback(
        self, mock_logger, mock_event_bus, knowledge_store
    ):
        """Action executes, but post-verification health degrades to CRITICAL, triggering rollback."""
        await mock_event_bus.start()
        workflow_events = []
        mock_event_bus.subscribe(WorkflowEvent, lambda ev: workflow_events.append(ev))

        rollback = AdaptationAction(
            action_id="act-rb",
            action_type="scale_down",
            target_system="sys-1",
        )
        action = AdaptationAction(
            action_id="act-1",
            action_type="scale_up",
            target_system="sys-1",
            rollback_action=rollback,
            verification_window_seconds=0.01,
        )

        strategy = Mock()
        strategy.requires_system_contract = False
        strategy.assess = AsyncMock(return_value=[action])
        strategy.on_action_executed = AsyncMock()

        connector = Mock()
        connector.validate_action = AsyncMock(return_value=True)
        connector.execute_action = AsyncMock(
            side_effect=[
                # 1. Main action succeeds
                ExecutionResult(
                    action_id="act-1",
                    status=ExecutionStatus.SUCCESS,
                    result_data={"scaled": True},
                ),
                # 2. Rollback action succeeds
                ExecutionResult(
                    action_id="act-rb",
                    status=ExecutionStatus.SUCCESS,
                    result_data={"scaled_down": True},
                ),
            ]
        )
        # Verification telemetry shows system degraded to CRITICAL
        degraded_state = SystemState(
            system_id="sys-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"response_time": MetricValue("response_time", 5.0)},
            health_status=HealthStatus.CRITICAL,
        )
        connector.collect_telemetry = AsyncMock(return_value=degraded_state)

        pipeline = AdaptationPipeline(
            strategy=strategy,
            knowledge_store=knowledge_store,
            world_model=None,
            event_bus=mock_event_bus,
            logger=mock_logger,
            metrics=NullMetricsCollector(),
            config=Mock(systems=[]),
        )

        initial_state = SystemState(
            system_id="sys-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"response_time": MetricValue("response_time", 1.0)},
            health_status=HealthStatus.WARNING,
        )

        result = await pipeline.run(initial_state, connector)
        assert result is True

        # Verify rollback was executed
        assert connector.execute_action.call_count == 2
        rollback_call_action = connector.execute_action.call_args_list[1].args[0]
        assert rollback_call_action.action_type == "scale_down"

        # Check stored workflow
        stored_workflows = await knowledge_store.query_workflows("sys-1")
        assert len(stored_workflows) == 1
        wf = stored_workflows[0]
        assert wf.status == WorkflowStatus.ROLLED_BACK
        assert wf.rollback_result is not None
        assert wf.rollback_result.status == ExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_action_verification_slo_breach_triggers_rollback(
        self, mock_logger, mock_event_bus, knowledge_store
    ):
        """Action executes, but post-verification violates SLOContract, triggering rollback."""
        await mock_event_bus.start()

        rollback = AdaptationAction(
            action_id="act-rb",
            action_type="scale_down",
            target_system="sys-1",
        )
        action = AdaptationAction(
            action_id="act-1",
            action_type="scale_up",
            target_system="sys-1",
            rollback_action=rollback,
            verification_window_seconds=0.01,
        )

        strategy = Mock()
        strategy.assess = AsyncMock(return_value=[action])
        strategy.on_action_executed = AsyncMock()

        connector = Mock()
        connector.validate_action = AsyncMock(return_value=True)
        connector.execute_action = AsyncMock(
            side_effect=[
                ExecutionResult(
                    action_id="act-1",
                    status=ExecutionStatus.SUCCESS,
                    result_data={},
                ),
                ExecutionResult(
                    action_id="act-rb",
                    status=ExecutionStatus.SUCCESS,
                    result_data={},
                ),
            ]
        )

        # Health is HEALTHY, but response_time violates SLO: response_time <= 1.0
        post_state = SystemState(
            system_id="sys-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"response_time": MetricValue("response_time", 2.5)},
            health_status=HealthStatus.HEALTHY,
        )
        connector.collect_telemetry = AsyncMock(return_value=post_state)

        contract = SystemContract(
            system_id="sys-1",
            supported_action_types=("scale_up", "scale_down"),
            slos=(SLOContract(metric_name="response_time", operator="<=", target_value=1.0),),
        )

        pipeline = AdaptationPipeline(
            strategy=strategy,
            knowledge_store=knowledge_store,
            world_model=None,
            event_bus=mock_event_bus,
            logger=mock_logger,
            metrics=NullMetricsCollector(),
            config=Mock(systems=[]),
        )

        initial_state = SystemState(
            system_id="sys-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"response_time": MetricValue("response_time", 1.2)},
            health_status=HealthStatus.HEALTHY,
        )

        result = await pipeline.run(initial_state, connector, system_contract=contract)
        assert result is True

        # Rollback was executed because SLO violated
        assert connector.execute_action.call_count == 2
        wf = (await knowledge_store.query_workflows("sys-1"))[0]
        assert wf.status == WorkflowStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_contract_driven_rollback_and_verification_window(
        self, mock_logger, mock_event_bus, knowledge_store
    ):
        """Action has no explicit rollback, but ActionSchema in contract specifies defaults."""
        await mock_event_bus.start()

        # Action does not specify rollback or verification window
        action = AdaptationAction(
            action_id="act-contract-test",
            action_type="scale_up",
            target_system="sys-1",
        )

        strategy = Mock()
        strategy.assess = AsyncMock(return_value=[action])
        strategy.on_action_executed = AsyncMock()

        connector = Mock()
        connector.validate_action = AsyncMock(return_value=True)
        connector.execute_action = AsyncMock(
            side_effect=[
                ExecutionResult(
                    action_id="act-contract-test",
                    status=ExecutionStatus.SUCCESS,
                    result_data={},
                ),
                ExecutionResult(
                    action_id="auto-rb",
                    status=ExecutionStatus.SUCCESS,
                    result_data={},
                ),
            ]
        )
        # Degraded state
        connector.collect_telemetry = AsyncMock(
            return_value=SystemState(
                system_id="sys-1",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
            )
        )

        # Contract provides ActionSchema with default verification and rollback
        contract = SystemContract(
            system_id="sys-1",
            supported_action_types=("scale_up", "scale_down"),
            actions={
                "scale_up": ActionSchema(
                    action_type="scale_up",
                    rollback_action="scale_down",
                    default_verification_window_seconds=0.01,
                ),
                "scale_down": ActionSchema(action_type="scale_down"),
            },
        )

        pipeline = AdaptationPipeline(
            strategy=strategy,
            knowledge_store=knowledge_store,
            world_model=None,
            event_bus=mock_event_bus,
            logger=mock_logger,
            metrics=NullMetricsCollector(),
            config=Mock(systems=[]),
        )

        initial_state = SystemState(
            system_id="sys-1",
            timestamp=datetime.now(timezone.utc),
            metrics={},
            health_status=HealthStatus.HEALTHY,
        )

        result = await pipeline.run(initial_state, connector, system_contract=contract)
        assert result is True

        # Auto rollback triggered using contract schema!
        assert connector.execute_action.call_count == 2
        assert connector.execute_action.call_args_list[1].args[0].action_type == "scale_down"
        wf = (await knowledge_store.query_workflows("sys-1"))[0]
        assert wf.status == WorkflowStatus.ROLLED_BACK
