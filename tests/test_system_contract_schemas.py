"""Tests for ActionSchema, MetricSchema, SLOContract, and contract-driven Pareto arbitration."""

from datetime import datetime, timezone

import pytest

from polaris.abstractions.connector_capabilities import ConnectorCapabilities
from polaris.abstractions.strategy import AdaptationContext
from polaris.abstractions.system_contract import (
    ActionSchema,
    MetricDirection,
    MetricSchema,
    MetricType,
    SLOContract,
    SystemContract,
)
from polaris.abstractions.world_model import WorldModel
from polaris.core.models import AdaptationAction, HealthStatus, MetricValue, SystemState
from polaris.strategies.hybrid import HybridStrategy
from polaris.world_model.statistical import StatisticalWorldModel


def test_action_schema_validation_required_and_bounds():
    """Verify ActionSchema validates required parameters and bounds."""
    schema = ActionSchema(
        action_type="scale_replicas",
        required_parameters=("replicas",),
        parameters_schema={
            "replicas": {"type": "integer", "minimum": 1, "maximum": 50},
            "step_factor": {"type": "number", "minimum": 0.1, "maximum": 2.0},
        },
        performance_impact="positive",
        cost_impact="negative",
    )

    # Missing required parameter
    valid, err = schema.validate_parameters({})
    assert valid is False
    assert "Missing required parameter 'replicas'" in (err or "")

    # Invalid type
    valid, err = schema.validate_parameters({"replicas": "five"})
    assert valid is False
    assert "must be an integer" in (err or "")

    # Below minimum
    valid, err = schema.validate_parameters({"replicas": 0})
    assert valid is False
    assert "must be >= 1" in (err or "")

    # Above maximum
    valid, err = schema.validate_parameters({"replicas": 55})
    assert valid is False
    assert "must be <= 50" in (err or "")

    # Boolean rejected as integer
    valid, err = schema.validate_parameters({"replicas": True})
    assert valid is False
    assert "must be an integer" in (err or "")

    # Valid
    valid, err = schema.validate_parameters({"replicas": 5, "step_factor": 1.5})
    assert valid is True
    assert err is None


def test_metric_schema_and_slo_contract_violations():
    """Verify MetricSchema directionality and SLOContract violation checks."""
    metric = MetricSchema(
        name="p99_latency_ms",
        metric_type=MetricType.GAUGE,
        direction=MetricDirection.MINIMIZE,
        unit="ms",
        target_value=200.0,
    )
    assert metric.direction == MetricDirection.MINIMIZE

    slo_latency = SLOContract(metric_name="p99_latency_ms", operator="<=", target_value=250.0)
    assert slo_latency.is_violated(200.0) is False
    assert slo_latency.is_violated(250.0) is False
    assert slo_latency.is_violated(250.1) is True

    slo_availability = SLOContract(metric_name="uptime_ratio", operator=">=", target_value=0.999)
    assert slo_availability.is_violated(1.0) is False
    assert slo_availability.is_violated(0.998) is True

    slo_exact = SLOContract(metric_name="cluster_mode", operator="!=", target_value=0.0)
    assert slo_exact.is_violated(0.0) is True
    assert slo_exact.is_violated(1.0) is False


def test_system_contract_alias_resolution_and_validation():
    """Verify SystemContract handles canonical names, aliases, and schemas."""
    schema = ActionSchema(
        action_type="scale_up",
        required_parameters=("count",),
        parameters_schema={"count": {"type": "integer", "minimum": 1}},
    )
    caps = ConnectorCapabilities.from_supported_action_types(
        action_types=["scale_up", "scale_down"],
        action_aliases={"add_worker": "scale_up"},
        actions={"scale_up": schema},
    )

    contract = SystemContract.from_capabilities("worker-pool", "CustomConnector", caps)

    # Resolution via alias
    found_schema = contract.get_action_schema("add_worker")
    assert found_schema is not None
    assert found_schema.action_type == "scale_up"

    # Action validation
    valid_act = AdaptationAction(
        action_id="1",
        action_type="add_worker",
        target_system="worker-pool",
        parameters={"count": 3},
    )
    valid, err = contract.validate_action(valid_act)
    assert valid is True
    assert err is None

    # Invalid action parameters
    invalid_act = AdaptationAction(
        action_id="2", action_type="scale_up", target_system="worker-pool", parameters={"count": -1}
    )
    valid, err = contract.validate_action(invalid_act)
    assert valid is False
    assert "must be >= 1" in (err or "")

    # Unsupported action
    unsupported = AdaptationAction(
        action_id="3", action_type="reboot_node", target_system="worker-pool"
    )
    valid, err = contract.validate_action(unsupported)
    assert valid is False
    assert "is not supported" in (err or "")


def test_world_model_is_stressed_interface():
    """Verify WorldModel.is_stressed default and statistical implementation."""

    # Base ABC default
    class MinimalModel(WorldModel):
        async def update(self, state):
            pass

        async def predict(self, action, current_state):
            from polaris.abstractions.world_model import PredictionResult

            return PredictionResult({}, 1.0)

        async def get_insights(self):
            return {}

    from unittest.mock import MagicMock

    base_model = MinimalModel()
    assert base_model.is_stressed("sys-1") is False

    # Statistical world model
    mock_ks = MagicMock()
    stat_model = StatisticalWorldModel(knowledge_store=mock_ks)
    assert stat_model.is_stressed("sys-1") is False

    # Set high regime probability
    stat_model._regime_probs["sys-1"] = {"low": 0.1, "normal": 0.2, "high": 0.7}
    assert stat_model.is_stressed("sys-1") is True


@pytest.mark.asyncio
async def test_hybrid_strategy_contract_driven_pareto_arbitration():
    """Verify HybridStrategy uses ActionSchema and SLOContract for arbitrary software domains."""
    from unittest.mock import AsyncMock

    # Custom database exemplar: actions have no "scale_up" or "dimmer" in names
    schema_flush = ActionSchema(
        action_type="flush_memtable",
        performance_impact="positive",  # Relieves write stall
        cost_impact="negative",  # Consumes disk I/O
        qos_impact="positive",
    )
    schema_throttle = ActionSchema(
        action_type="throttle_writes",
        performance_impact="positive",  # Protects latency
        cost_impact="positive",  # Saves I/O
        qos_impact="negative",  # Degrades write SLA
    )

    slo = SLOContract(metric_name="write_latency_ms", operator="<=", target_value=10.0)

    contract = SystemContract(
        system_id="db-cluster",
        connector_type="DBConnector",
        supported_action_types=("flush_memtable", "throttle_writes"),
        actions={"flush_memtable": schema_flush, "throttle_writes": schema_throttle},
        slos=(slo,),
    )

    # State where SLO is violated (write latency 25ms > 10ms target)
    stressed_state = SystemState(
        system_id="db-cluster",
        timestamp=datetime.now(timezone.utc),
        metrics={"write_latency_ms": MetricValue("write_latency_ms", 25.0, "ms")},
        health_status=HealthStatus.HEALTHY,  # health_status is healthy, but SLO is violated!
    )

    context = AdaptationContext(
        system_id="db-cluster",
        historical_states=[],
        system_contract=contract,
    )

    act_flush = AdaptationAction(
        action_id="1", action_type="flush_memtable", target_system="db-cluster"
    )
    act_throttle = AdaptationAction(
        action_id="2", action_type="throttle_writes", target_system="db-cluster"
    )

    class MockStrat:
        def __init__(self, actions):
            self.actions = actions

        async def assess(self, s, c):
            return self.actions

        async def get_performance_metrics(self):
            return {"success_rate": 0.9}

        def get_tunable_parameters(self):
            return {}

    strat_flush = MockStrat([act_flush])
    strat_throttle = MockStrat([act_throttle])

    # Favor QoS preservation (w_qos=0.6, w_perf=0.3, w_cost=0.1) -> flush_memtable should win
    hybrid_qos = HybridStrategy(
        strategies=[(strat_throttle, 10.0), (strat_flush, 5.0)],
        selection_mode="pareto",
        objective_weights={"performance": 0.3, "cost": 0.1, "qos": 0.6},
    )
    selected_qos = await hybrid_qos.assess(stressed_state, context)
    assert len(selected_qos) == 1
    assert selected_qos[0].action_type == "flush_memtable"

    # Favor Cost/I-O saving (w_cost=0.7, w_perf=0.2, w_qos=0.1) -> throttle_writes should win
    hybrid_cost = HybridStrategy(
        strategies=[(strat_flush, 10.0), (strat_throttle, 5.0)],
        selection_mode="pareto",
        objective_weights={"performance": 0.2, "cost": 0.7, "qos": 0.1},
    )
    selected_cost = await hybrid_cost.assess(stressed_state, context)
    assert len(selected_cost) == 1
    assert selected_cost[0].action_type == "throttle_writes"
