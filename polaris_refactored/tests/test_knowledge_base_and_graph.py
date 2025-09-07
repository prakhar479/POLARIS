import asyncio
from datetime import datetime, timedelta

import pytest

from polaris_refactored.src.infrastructure.data_storage import (
    InMemoryGraphStorageBackend,
    PolarisDataStore,
)
from polaris_refactored.src.digital_twin.knowledge_base import PolarisKnowledgeBase
from polaris_refactored.src.domain.models import (
    SystemState,
    MetricValue,
    HealthStatus,
    LearnedPattern,
)
from polaris_refactored.src.framework.events import TelemetryEvent


@pytest.mark.asyncio
async def test_store_and_query_state():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })
    await ds.start()
    kb = PolarisKnowledgeBase(ds)

    system_id = "sys-A"
    state = SystemState(
        system_id=system_id,
        timestamp=datetime.utcnow(),
        metrics={
            "cpu": MetricValue(name="cpu", value=0.5, unit="ratio"),
            "latency_ms": MetricValue(name="latency_ms", value=120.0, unit="ms"),
        },
        health_status=HealthStatus.HEALTHY,
        metadata={"zone": "us-east-1"},
    )
    await kb.store_telemetry(TelemetryEvent(system_state=state))

    current = await kb.get_current_state(system_id)
    assert current is not None
    assert current.system_id == system_id

    start = datetime.utcnow() - timedelta(hours=1)
    end = datetime.utcnow() + timedelta(hours=1)
    history = await kb.get_historical_states(system_id, start, end)
    assert len(history) >= 1

    await ds.stop()


@pytest.mark.asyncio
async def test_system_relationships_and_chain():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })
    await ds.start()
    kb = PolarisKnowledgeBase(ds)

    await kb.add_system_relationship("A", "B", "depends_on", strength=0.8)
    await kb.add_system_relationship("B", "C", "depends_on", strength=0.9)

    deps = await kb.query_system_dependencies("A")
    assert any(d.target_system == "B" for d in deps)

    dependents = await kb.get_dependent_systems("B")
    assert "A" in dependents

    chain = await kb.get_dependency_chain("A", max_depth=2)
    assert chain["root"] == "A"
    assert "A" in chain["graph"]
    assert "B" in chain["graph"]["A"]

    await ds.stop()


@pytest.mark.asyncio
async def test_learned_patterns_store_and_query():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })
    await ds.start()
    kb = PolarisKnowledgeBase(ds)

    pattern = LearnedPattern(
        pattern_id="p1",
        pattern_type="reinforcement",
        conditions={"system_id": "S1", "env": "prod"},
        outcomes={"estimated_reward": 1.0, "recommended_actions": [{"action_type": "scale_out", "parameters": {"scale_factor": 2}}]},
        confidence=0.9,
        learned_at=datetime.utcnow(),
        usage_count=0,
    )
    await kb.store_learned_pattern(pattern)

    results = await kb.query_patterns(pattern_type="reinforcement", conditions={"system_id": "S1"})
    assert any(p.pattern_id == "p1" for p in results)

    similar = await kb.get_similar_patterns(current_conditions={"system_id": "S1", "env": "prod"}, similarity_threshold=0.5)
    assert any(p.pattern_id == "p1" for p in similar)

    await ds.stop()
