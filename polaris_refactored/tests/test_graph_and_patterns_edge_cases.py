import pytest
from datetime import datetime

from polaris_refactored.src.infrastructure.data_storage import InMemoryGraphStorageBackend, PolarisDataStore
from polaris_refactored.src.digital_twin.knowledge_base import PolarisKnowledgeBase
from polaris_refactored.src.domain.models import LearnedPattern


@pytest.mark.asyncio
async def test_graph_remove_edge_via_repository_delete():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })
    await ds.start()
    kb = PolarisKnowledgeBase(ds)

    # Add A -> B edge
    await kb.add_system_relationship("A", "B", "depends_on", strength=1.0)
    out = await kb.query_system_dependencies("A")
    assert any(d.target_system == "B" for d in out)

    # Now remove via repository delete by key
    repo = ds.get_repository("system_dependencies")
    key = "A:depends_on:B"
    assert await repo.delete(key)  # type: ignore[attr-defined]

    out2 = await kb.query_system_dependencies("A")
    assert not any(d.target_system == "B" for d in out2)

    await ds.stop()


@pytest.mark.asyncio
async def test_learned_patterns_complex_query_and_similarity():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })
    await ds.start()
    kb = PolarisKnowledgeBase(ds)

    p1 = LearnedPattern(
        pattern_id="p1",
        pattern_type="pattern",
        conditions={"system_id": "S1", "env": "prod", "region": "us"},
        outcomes={"recommended_actions": [{"action_type": "scale_out"}]},
        confidence=0.7,
        learned_at=datetime.utcnow(),
    )
    p2 = LearnedPattern(
        pattern_id="p2",
        pattern_type="pattern",
        conditions={"system_id": "S1", "env": "staging", "region": "eu"},
        outcomes={"recommended_actions": [{"action_type": "scale_in"}]},
        confidence=0.8,
        learned_at=datetime.utcnow(),
    )
    await kb.store_learned_pattern(p1)
    await kb.store_learned_pattern(p2)

    # Query by type and subset conditions
    results = await kb.query_patterns("pattern", {"system_id": "S1", "env": "prod"})
    ids = {r.pattern_id for r in results}
    assert "p1" in ids and "p2" not in ids

    # Similarity should include p1 when conditions overlap sufficiently
    sims = await kb.get_similar_patterns({"system_id": "S1", "env": "prod"}, similarity_threshold=0.4)
    ids2 = {r.pattern_id for r in sims}
    assert "p1" in ids2

    await ds.stop()
