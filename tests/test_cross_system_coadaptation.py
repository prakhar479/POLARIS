"""Tests for cross-system co-adaptation and cascade backpressure."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.abstractions.system_contract import ActionSchema, SystemContract
from polaris.core.adaptation_pipeline import AdaptationPipeline
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.core.topology import SystemTopology
from polaris.infrastructure.config import PolarisConfig
from polaris.knowledge.memory import InMemoryKnowledgeStore
from polaris.strategies.hybrid import HybridStrategy


def make_state(
    system_id: str,
    health: HealthStatus = HealthStatus.HEALTHY,
    metrics: dict | None = None,
) -> SystemState:
    metric_values = {}
    if metrics:
        for k, v in metrics.items():
            metric_values[k] = MetricValue(name=k, value=v, timestamp=datetime.now(timezone.utc))
    return SystemState(
        system_id=system_id,
        health_status=health,
        metrics=metric_values,
        timestamp=datetime.now(timezone.utc),
    )


def test_pareto_downstream_healthy_prefers_scale_up():
    mock_strat = MagicMock()
    strategy = HybridStrategy(strategies=[(mock_strat, 1)], selection_mode="pareto")

    # Upstream system (gateway) under high load (high response time)
    gw_state = make_state(
        "gateway",
        health=HealthStatus.WARNING,
        metrics={"average_response_time": 800.0, "average_utilization": 85.0},
    )

    # Downstream dependency (database) is healthy
    db_state = make_state(
        "database",
        health=HealthStatus.HEALTHY,
        metrics={"average_utilization": 30.0},
    )

    topo = SystemTopology()
    topo.add_dependency("gateway", "database")

    context = AdaptationContext(
        system_id="gateway",
        historical_states=[],
        topology=topo,
        peer_states={"database": db_state},
        upstream_systems=[],
        downstream_systems=["database"],
    )

    scale_action = AdaptationAction(action_id="1", target_system="gateway", action_type="scale_up")
    dimmer_action = AdaptationAction(
        action_id="2",
        target_system="gateway",
        action_type="dimmer",
        parameters={"dimmer": 0.5},
    )

    u_scale = strategy._calculate_pareto_utility(scale_action, 0.9, gw_state, context)
    u_dimmer = strategy._calculate_pareto_utility(dimmer_action, 0.9, gw_state, context)

    # When downstream is healthy, scaling up should have higher utility to maintain QoS
    assert u_scale > u_dimmer


def test_pareto_downstream_stressed_activates_cascade_backpressure():
    mock_strat = MagicMock()
    strategy = HybridStrategy(strategies=[(mock_strat, 1)], selection_mode="pareto")

    # Upstream system (gateway) under high load
    gw_state = make_state(
        "gateway",
        health=HealthStatus.WARNING,
        metrics={"average_response_time": 800.0, "average_utilization": 85.0},
    )

    # Downstream dependency (database) is in CRITICAL state / overloaded
    db_state = make_state(
        "database",
        health=HealthStatus.CRITICAL,
        metrics={"average_utilization": 98.0},
    )

    topo = SystemTopology()
    topo.add_dependency("gateway", "database")

    context = AdaptationContext(
        system_id="gateway",
        historical_states=[],
        topology=topo,
        peer_states={"database": db_state},
        upstream_systems=[],
        downstream_systems=["database"],
    )

    scale_action = AdaptationAction(action_id="1", target_system="gateway", action_type="scale_up")
    dimmer_action = AdaptationAction(
        action_id="2",
        target_system="gateway",
        action_type="dimmer",
        parameters={"dimmer": 0.5},
    )

    u_scale = strategy._calculate_pareto_utility(scale_action, 0.9, gw_state, context)
    u_dimmer = strategy._calculate_pareto_utility(dimmer_action, 0.9, gw_state, context)

    # With downstream stressed, cascade backpressure penalizes scale_up and boosts dimmer
    assert u_dimmer > u_scale


def test_pareto_downstream_stress_with_contract_schemas():
    mock_strat = MagicMock()
    strategy = HybridStrategy(strategies=[(mock_strat, 1)], selection_mode="pareto")

    gw_contract = SystemContract(
        system_id="gateway",
        actions={
            "expand_capacity": ActionSchema(
                action_type="expand_capacity",
                performance_impact="positive",
                cost_impact="negative",
            ),
            "throttle_traffic": ActionSchema(
                action_type="throttle_traffic",
                performance_impact="negative",
                cost_impact="positive",
            ),
        },
    )

    gw_state = make_state(
        "gateway",
        health=HealthStatus.WARNING,
        metrics={"latency": 900.0},
    )

    db_state = make_state(
        "database",
        health=HealthStatus.CRITICAL,
        metrics={"cpu_usage": 95.0},
    )

    context = AdaptationContext(
        system_id="gateway",
        historical_states=[],
        system_contract=gw_contract,
        peer_states={"database": db_state},
        downstream_systems=["database"],
    )

    expand_action = AdaptationAction(
        action_id="1", target_system="gateway", action_type="expand_capacity"
    )
    throttle_action = AdaptationAction(
        action_id="2", target_system="gateway", action_type="throttle_traffic"
    )

    u_expand = strategy._calculate_pareto_utility(expand_action, 0.9, gw_state, context)
    u_throttle = strategy._calculate_pareto_utility(throttle_action, 0.9, gw_state, context)

    # Throttle (load shedding) should beat capacity expansion when downstream is collapsing
    assert u_throttle > u_expand


@pytest.mark.asyncio
async def test_adaptation_pipeline_propagates_topology_and_peer_states():
    store = InMemoryKnowledgeStore()
    topo = SystemTopology()
    topo.add_dependency("frontend", "backend")
    await store.store_topology(topo)

    # Store latest backend state in knowledge store
    backend_state = make_state(
        "backend",
        health=HealthStatus.CRITICAL,
        metrics={"utilization": 99.0},
    )
    await store.store_state(backend_state)

    captured_context = None

    class CaptureStrategy:
        async def assess(self, state, context):
            nonlocal captured_context
            captured_context = context
            return []

        async def get_performance_metrics(self):
            return {}

        def get_tunable_parameters(self):
            return {}

        def update_parameter(self, name, val):
            return True

        async def on_action_executed(self, action, result):
            pass

    pipeline = AdaptationPipeline(
        strategy=CaptureStrategy(),
        knowledge_store=store,
        world_model=None,
        event_bus=AsyncMock(),
        logger=MagicMock(),
        metrics=MagicMock(),
        config=PolarisConfig(),
    )

    connector = AsyncMock()
    connector.get_system_id.return_value = "frontend"
    connector.execute_action.return_value = ExecutionResult(
        action_id="1", status=ExecutionStatus.SUCCESS, result_data={}
    )

    frontend_state = make_state("frontend", health=HealthStatus.HEALTHY)
    await pipeline.run(frontend_state, connector)

    assert captured_context is not None
    assert captured_context.topology is not None
    assert captured_context.downstream_systems == ["backend"]
    assert "backend" in captured_context.peer_states
    assert captured_context.peer_states["backend"].system_id == "backend"
    assert captured_context.peer_states["backend"].health_status == HealthStatus.CRITICAL
