import asyncio
from datetime import datetime
import pytest

from polaris_refactored.src.infrastructure.data_storage import (
    PolarisDataStore,
    InMemoryGraphStorageBackend,
)
from polaris_refactored.src.infrastructure.data_storage import (
    SystemDependencyRepository,
    LearnedPatternRepository,
)
from polaris_refactored.src.digital_twin.knowledge_base import PolarisKnowledgeBase
from polaris_refactored.src.digital_twin.telemetry_subscriber import (
    TelemetryToKnowledgeBaseHandler,
    subscribe_telemetry_persistence,
)
from polaris_refactored.src.framework.events import TelemetryEvent, PolarisEventBus
from polaris_refactored.src.domain.models import (
    SystemState,
    MetricValue,
    HealthStatus,
    LearnedPattern,
)
from polaris_refactored.src.infrastructure.exceptions import DataStoreError


@pytest.mark.asyncio
async def test_data_store_start_stop_and_errors():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })

    # Not started yet -> get_repository should error
    with pytest.raises(DataStoreError):
        ds.get_repository("system_states")

    await ds.start()
    # After start it should provide repositories
    assert isinstance(ds.get_repository("system_states").collection_name, str)

    await ds.stop()
    # After stop, get_repository should error
    with pytest.raises(DataStoreError):
        ds.get_repository("system_states")


@pytest.mark.asyncio
async def test_dependency_repo_neighbors_with_type_filter():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({"graph": backend, "document": backend, "time_series": backend})
    await ds.start()
    repo: SystemDependencyRepository = ds.get_repository("system_dependencies")  # type: ignore[assignment]

    # Create mixed relationship types
    from polaris_refactored.src.domain.models import SystemDependency
    await repo.save(SystemDependency("A", "B", "depends_on", 1.0, {}))
    await repo.save(SystemDependency("A", "C", "replicates", 0.5, {}))

    out_all = await repo.get_neighbors("A", direction="out")
    assert {d.target_system for d in out_all} == {"B", "C"}

    out_depends = await repo.get_neighbors("A", direction="out", relationship_type="depends_on")
    assert {d.target_system for d in out_depends} == {"B"}

    # Incoming for B should include A
    in_b = await repo.get_neighbors("B", direction="in")
    assert any(d.source_system == "A" for d in in_b)

    await ds.stop()


@pytest.mark.asyncio
async def test_learned_pattern_repo_save_delete_and_list():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({"graph": backend, "document": backend, "time_series": backend})
    await ds.start()

    patterns: LearnedPatternRepository = ds.get_repository("learned_patterns")  # type: ignore[assignment]
    p = LearnedPattern(
        pattern_id="pp",
        pattern_type="pattern",
        conditions={"a": 1},
        outcomes={},
        confidence=0.4,
        learned_at=datetime.utcnow(),
    )
    await patterns.save(p)

    all1 = await patterns.list_all()
    assert any(x.pattern_id == "pp" for x in all1)

    assert await patterns.delete("pp")
    all2 = await patterns.list_all()
    assert not any(x.pattern_id == "pp" for x in all2)

    await ds.stop()


@pytest.mark.asyncio
async def test_kb_similarity_edge_cases_sorting_and_threshold():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({"graph": backend, "document": backend, "time_series": backend})
    await ds.start()
    kb = PolarisKnowledgeBase(ds)

    # Add patterns with different confidences
    p1 = LearnedPattern("p1", "pattern", {"k": 1}, {}, 0.9, datetime.utcnow())
    p2 = LearnedPattern("p2", "pattern", {"k": 2}, {}, 0.5, datetime.utcnow())
    p3 = LearnedPattern("p3", "pattern", {"k": 1, "x": 3}, {}, 0.7, datetime.utcnow())
    await kb.store_learned_pattern(p1)
    await kb.store_learned_pattern(p2)
    await kb.store_learned_pattern(p3)

    # Empty conditions -> sorted by confidence desc
    sims = await kb.get_similar_patterns({}, similarity_threshold=0.0)
    ids = [p.pattern_id for p in sims]
    assert ids == ["p1", "p3", "p2"]

    # With conditions -> ensure threshold filters correctly
    sims2 = await kb.get_similar_patterns({"k": 1}, similarity_threshold=0.5)
    ids2 = {p.pattern_id for p in sims2}
    assert "p1" in ids2 and "p3" in ids2 and "p2" not in ids2

    await ds.stop()


@pytest.mark.asyncio
async def test_telemetry_handler_can_handle_and_filter():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({"graph": backend, "document": backend, "time_series": backend})
    await ds.start()
    kb = PolarisKnowledgeBase(ds)
    handler = TelemetryToKnowledgeBaseHandler(kb)

    # can_handle type check
    state = SystemState("sysZ", datetime.utcnow(), {"cpu": MetricValue("cpu", 1.0)}, HealthStatus.HEALTHY)
    evt = TelemetryEvent(system_state=state)
    assert handler.can_handle(evt)

    # filter in subscription
    bus = PolarisEventBus(worker_count=1)
    await bus.start()
    sub_id = await subscribe_telemetry_persistence(bus, kb, system_id_filter="sysZ")
    await bus.publish_telemetry(evt)

    # A different system should not be persisted
    state2 = SystemState("sysQ", datetime.utcnow(), {"cpu": MetricValue("cpu", 2.0)}, HealthStatus.HEALTHY)
    evt2 = TelemetryEvent(system_state=state2)
    await bus.publish_telemetry(evt2)

    # Wait for worker
    for _ in range(10):
        current = await kb.get_current_state("sysZ")
        if current is not None:
            break
        await asyncio.sleep(0.05)

    assert (await kb.get_current_state("sysZ")) is not None
    assert (await kb.get_current_state("sysQ")) is None

    await bus.stop()
    await ds.stop()
