"""Tests for Formal Neuro-Symbolic Verifier Agent in POLARIS.

Validates contract schema checking, continuous parameter safety envelopes,
rate-of-change (delta) clamping, metric temporal logic anti-flapping,
and topological blast-radius backpressure.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.system_contract import ActionSchema, SystemContract
from polaris.abstractions.verifier import (
    InvariantSeverity,
    VerificationContext,
    VerificationDecision,
)
from polaris.core.adaptation_pipeline import AdaptationPipeline
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.core.safety import SafetyConfig, SafetyPolicyEngine
from polaris.core.topology import SystemTopology
from polaris.core.verifier import (
    ContractSchemaInvariant,
    NeuroSymbolicVerifier,
    ParameterBoundsClampingInvariant,
    RateOfChangeClampingInvariant,
    TemporalDwellInvariant,
    TopologicalSafetyInvariant,
)
from tests.conftest import MockLogger, MockMetricsCollector


def make_test_contract() -> SystemContract:
    """Create a sample SystemContract with schema and bounds."""
    dimmer_schema = ActionSchema(
        action_type="set_dimmer",
        description="Adjust optional content dimmer factor",
        parameters_schema={
            "dimmer": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            }
        },
        required_parameters=("dimmer",),
    )
    scale_schema = ActionSchema(
        action_type="scale_up",
        description="Add a server replica",
        parameters_schema={"step": {"type": "integer", "minimum": 1, "maximum": 5}},
    )
    scale_down_schema = ActionSchema(
        action_type="scale_down",
        description="Remove a server replica",
        parameters_schema={"step": {"type": "integer", "minimum": 1, "maximum": 5}},
    )

    return SystemContract(
        system_id="web-server-1",
        supported_action_types=("set_dimmer", "scale_up", "scale_down"),
        actions={
            "set_dimmer": dimmer_schema,
            "scale_up": scale_schema,
            "scale_down": scale_down_schema,
        },
    )


def make_system_state(
    system_id: str = "web-server-1",
    health: HealthStatus = HealthStatus.HEALTHY,
    dimmer: float = 0.8,
) -> SystemState:
    """Helper to create SystemState with dimmer metric."""
    return SystemState(
        system_id=system_id,
        timestamp=datetime.now(timezone.utc),
        metrics={
            "dimmer": MetricValue(name="dimmer", value=dimmer),
            "response_time": MetricValue(name="response_time", value=250.0),
        },
        health_status=health,
    )


@pytest.mark.asyncio
async def test_verifier_accepts_valid_action():
    verifier = NeuroSymbolicVerifier()
    contract = make_test_contract()
    state = make_system_state()

    action = AdaptationAction(
        action_id="act-1",
        action_type="set_dimmer",
        target_system="web-server-1",
        parameters={"dimmer": 0.75},
    )
    context = VerificationContext(
        system_id="web-server-1",
        system_state=state,
        system_contract=contract,
    )

    result = await verifier.verify(action, context)
    assert result.decision == VerificationDecision.ACCEPTED
    assert result.is_executable is True
    assert result.verified_action == action
    assert "ACCEPTED" in result.explanation
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_verifier_rejects_unsupported_action():
    verifier = NeuroSymbolicVerifier()
    contract = make_test_contract()
    state = make_system_state()

    action = AdaptationAction(
        action_id="act-bad",
        action_type="reboot_cluster",  # Not in contract
        target_system="web-server-1",
    )
    context = VerificationContext(
        system_id="web-server-1",
        system_state=state,
        system_contract=contract,
    )

    result = await verifier.verify(action, context)
    assert result.decision == VerificationDecision.REJECTED
    assert result.is_executable is False
    assert result.verified_action is None
    assert "not supported by contract" in result.explanation


@pytest.mark.asyncio
async def test_verifier_rejects_missing_required_parameter():
    verifier = NeuroSymbolicVerifier()
    contract = make_test_contract()
    state = make_system_state()

    action = AdaptationAction(
        action_id="act-missing",
        action_type="set_dimmer",
        target_system="web-server-1",
        parameters={},  # 'dimmer' is required
    )
    context = VerificationContext(
        system_id="web-server-1",
        system_state=state,
        system_contract=contract,
    )

    result = await verifier.verify(action, context)
    assert result.decision == VerificationDecision.REJECTED
    assert "Missing required parameter 'dimmer'" in result.explanation


@pytest.mark.asyncio
async def test_verifier_clamps_parameter_bounds_minimum_and_maximum():
    verifier = NeuroSymbolicVerifier(allow_clamping=True)
    contract = make_test_contract()
    state_high = make_system_state(dimmer=0.9)

    # Test parameter exceeding maximum (1.25 -> clamped to 1.0)
    action_high = AdaptationAction(
        action_id="act-high",
        action_type="set_dimmer",
        target_system="web-server-1",
        parameters={"dimmer": 1.25},
    )
    context_high = VerificationContext(
        system_id="web-server-1",
        system_state=state_high,
        system_contract=contract,
    )
    res_high = await verifier.verify(action_high, context_high)
    assert res_high.decision == VerificationDecision.CLAMPED
    assert res_high.is_executable is True
    assert res_high.verified_action is not None
    assert res_high.verified_action.parameters["dimmer"] == 1.0

    # Test parameter below minimum (-0.3 -> clamped to 0.0)
    state_low = make_system_state(dimmer=0.1)
    action_low = AdaptationAction(
        action_id="act-low",
        action_type="set_dimmer",
        target_system="web-server-1",
        parameters={"dimmer": -0.3},
    )
    context_low = VerificationContext(
        system_id="web-server-1",
        system_state=state_low,
        system_contract=contract,
    )
    res_low = await verifier.verify(action_low, context_low)
    assert res_low.decision == VerificationDecision.CLAMPED
    assert res_low.verified_action is not None
    assert res_low.verified_action.parameters["dimmer"] == 0.0


@pytest.mark.asyncio
async def test_verifier_rate_of_change_delta_clamping():
    # max delta is 0.25 per cycle
    verifier = NeuroSymbolicVerifier(allow_clamping=True)
    contract = make_test_contract()
    state = make_system_state(dimmer=0.8)

    # Proposed jump from 0.8 to 0.2 (delta = -0.6, exceeds max_delta of 0.25)
    action_steep = AdaptationAction(
        action_id="act-steep",
        action_type="set_dimmer",
        target_system="web-server-1",
        parameters={"dimmer": 0.2},
    )
    context = VerificationContext(
        system_id="web-server-1",
        system_state=state,
        system_contract=contract,
    )

    res = await verifier.verify(action_steep, context)
    assert res.decision == VerificationDecision.CLAMPED
    assert res.verified_action is not None
    # 0.8 - 0.25 = 0.55
    assert res.verified_action.parameters["dimmer"] == 0.55
    assert "rate-of-change envelope" in res.explanation


@pytest.mark.asyncio
async def test_verifier_temporal_dwell_anti_flapping():
    verifier = NeuroSymbolicVerifier()
    contract = make_test_contract()
    state = make_system_state()

    # Previous action 5 seconds ago was scale_up
    now = datetime.now(timezone.utc)
    recent_act = AdaptationAction(
        action_id="act-prev",
        action_type="scale_up",
        target_system="web-server-1",
        created_at=now,
    )

    # Candidate action right now is opposing action scale_down (dwell window is 30s)
    candidate = AdaptationAction(
        action_id="act-next",
        action_type="scale_down",
        target_system="web-server-1",
        created_at=now,
    )

    context = VerificationContext(
        system_id="web-server-1",
        system_state=state,
        system_contract=contract,
        recent_actions=[recent_act],
    )

    result = await verifier.verify(candidate, context)
    assert result.decision == VerificationDecision.REJECTED
    assert "opposes recent 'scale_up' within dwell window" in result.explanation


@pytest.mark.asyncio
async def test_verifier_topological_safety_backpressure():
    verifier = NeuroSymbolicVerifier()
    contract = make_test_contract()
    gw_state = make_system_state(system_id="gateway")

    # Downstream database is in CRITICAL state
    db_state = SystemState(
        system_id="database",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue(name="cpu", value=99.0)},
        health_status=HealthStatus.CRITICAL,
    )

    topo = SystemTopology()
    topo.add_dependency("gateway", "database")

    # Upstream gateway proposes scale_up (amplifies downstream load)
    action = AdaptationAction(
        action_id="act-gw",
        action_type="scale_up",
        target_system="gateway",
    )

    context = VerificationContext(
        system_id="gateway",
        system_state=gw_state,
        system_contract=contract,
        topology=topo,
        peer_states={"database": db_state},
    )

    result = await verifier.verify(action, context)
    assert result.decision == VerificationDecision.REJECTED
    assert "cascade backpressure requires load-shedding" in result.explanation


@pytest.mark.asyncio
async def test_verifier_integrates_with_safety_policy_engine():
    # Setup safety engine with system rate limit of 1 adaptation per 60s
    safety_cfg = SafetyConfig(
        enabled=True,
        max_system_adaptations_per_window=1,
        system_window_seconds=60.0,
    )
    safety_engine = SafetyPolicyEngine(config=safety_cfg)
    verifier = NeuroSymbolicVerifier(safety_engine=safety_engine)
    contract = make_test_contract()
    state = make_system_state()

    act1 = AdaptationAction(action_id="1", action_type="scale_up", target_system="web-server-1")
    act2 = AdaptationAction(action_id="2", action_type="scale_up", target_system="web-server-1")

    context = VerificationContext(
        system_id="web-server-1",
        system_state=state,
        system_contract=contract,
    )

    # First action passes safety engine
    res1 = await verifier.verify(act1, context)
    assert res1.decision == VerificationDecision.ACCEPTED
    safety_engine.record_action_start(act1)
    safety_engine.record_action_end(act1, success=True)

    # Second action immediately trips system rate limit
    res2 = await verifier.verify(act2, context)
    assert res2.decision == VerificationDecision.REJECTED
    assert "rate limit reached" in res2.explanation


@pytest.mark.asyncio
async def test_adaptation_pipeline_applies_verifier_clamping():
    mock_strat = MagicMock()
    mock_logger = MockLogger()
    mock_metrics = MockMetricsCollector()
    mock_connector = MagicMock()
    mock_connector.validate_action = AsyncMock(return_value=True)
    mock_connector.execute_action = AsyncMock(
        return_value=ExecutionResult(
            action_id="1",
            status=ExecutionStatus.SUCCESS,
            result_data={},
        )
    )

    config = MagicMock()
    config.get.return_value = {"enabled": True}

    pipeline = AdaptationPipeline(
        strategy=mock_strat,
        knowledge_store=AsyncMock(),
        world_model=AsyncMock(),
        event_bus=AsyncMock(),
        logger=mock_logger,
        metrics=mock_metrics,
        config=config,
    )

    contract = make_test_contract()
    state = make_system_state(dimmer=0.5)

    # Strategy proposes an action with dimmer = 1.4 (exceeds schema maximum 1.0)
    out_of_bounds_action = AdaptationAction(
        action_id="act-clamp-test",
        action_type="set_dimmer",
        target_system="web-server-1",
        parameters={"dimmer": 1.4},
    )
    mock_strat.assess = AsyncMock(return_value=[out_of_bounds_action])
    mock_strat.on_action_executed = AsyncMock()

    executed = await pipeline.run(state, mock_connector, system_contract=contract)

    assert executed is True
    # Verify that the connector was called with clamped parameters!
    executed_action = mock_connector.execute_action.call_args[0][0]
    assert executed_action.parameters["dimmer"] <= 1.0
    assert any(m[1] == "polaris.verifier.clamped" for m in mock_metrics.metrics)


@pytest.mark.asyncio
async def test_adaptation_pipeline_verifier_rejection_skips_execution():
    mock_strat = MagicMock()
    mock_logger = MockLogger()
    mock_metrics = MockMetricsCollector()
    mock_connector = MagicMock()
    mock_connector.validate_action = AsyncMock(return_value=True)
    mock_connector.execute_action = AsyncMock()

    config = MagicMock()
    config.get.return_value = {"enabled": True}

    pipeline = AdaptationPipeline(
        strategy=mock_strat,
        knowledge_store=AsyncMock(),
        world_model=AsyncMock(),
        event_bus=AsyncMock(),
        logger=mock_logger,
        metrics=mock_metrics,
        config=config,
    )

    contract = make_test_contract()
    state = make_system_state()

    # Strategy proposes an invalid unsupported action
    invalid_action = AdaptationAction(
        action_id="act-inv",
        action_type="delete_database",  # Not in contract!
        target_system="web-server-1",
    )
    mock_strat.assess = AsyncMock(return_value=[invalid_action])

    executed = await pipeline.run(state, mock_connector, system_contract=contract)

    assert executed is False
    # Connector execution must NOT be called for rejected action
    assert mock_connector.execute_action.call_count == 0
    assert any(m[1] == "polaris.verifier.rejected" for m in mock_metrics.metrics)


@pytest.mark.asyncio
async def test_verifier_clamping_disabled_rejects_violating_action():
    """Verify that when allow_clamping=False, actions requiring clamping are formally rejected."""
    verifier = NeuroSymbolicVerifier(allow_clamping=False)
    contract = make_test_contract()
    state = make_system_state()

    # Action that exceeds rate-of-change envelope (dimmer jump from 1.0 to 0.1 > max_step 0.25)
    action = AdaptationAction(
        action_id="act-clamp-disabled",
        action_type="set_dimmer",
        target_system="web-server-1",
        parameters={"dimmer": 0.1},
    )

    context = VerificationContext(
        system_id="web-server-1",
        system_state=state,
        system_contract=contract,
    )

    result = await verifier.verify(action, context)
    assert result.decision == VerificationDecision.REJECTED
    assert result.verified_action is None
    assert "clamping is disabled" in result.explanation
