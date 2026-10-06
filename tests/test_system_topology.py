"""Tests for multi-system topology graph."""

import pytest

from polaris.core.topology import SystemTopology
from polaris.knowledge.memory import InMemoryKnowledgeStore


def test_topology_empty():
    topo = SystemTopology()
    assert topo.get_dependencies("svc_a") == []
    assert topo.get_dependents("svc_a") == []
    assert topo.get_impact_radius("svc_a") == {"svc_a"}
    assert topo.topological_sort() == []


def test_topology_single_dependency():
    # api_gateway depends on backend
    topo = SystemTopology()
    topo.add_dependency("api_gateway", "backend")

    assert topo.get_dependencies("api_gateway") == ["backend"]
    assert topo.get_dependents("backend") == ["api_gateway"]
    assert topo.get_dependencies("backend") == []
    assert topo.get_dependents("api_gateway") == []


def test_topology_multi_tier_pipeline():
    # client -> gateway -> auth -> db
    topo = SystemTopology()
    topo.add_dependency("gateway", "auth")
    topo.add_dependency("auth", "db")

    assert topo.get_dependencies("gateway") == ["auth"]
    assert topo.get_dependencies("auth") == ["db"]
    assert topo.get_dependents("db") == ["auth"]
    assert topo.get_dependents("auth") == ["gateway"]

    # Impact radius includes all transitively connected systems
    radius = topo.get_impact_radius("auth")
    assert radius == {"gateway", "auth", "db"}

    # Topological sort: db (0 deps), auth (1 dep), gateway (1 dep on auth)
    order = topo.topological_sort()
    assert order.index("db") < order.index("auth")
    assert order.index("auth") < order.index("gateway")


def test_topology_diamond_dependency():
    # web depends on cache and db
    # cache depends on db
    topo = SystemTopology()
    topo.add_dependency("web", "cache")
    topo.add_dependency("web", "db")
    topo.add_dependency("cache", "db")

    order = topo.topological_sort()
    assert order.index("db") < order.index("cache")
    assert order.index("cache") < order.index("web")
    assert topo.get_impact_radius("web") == {"web", "cache", "db"}


def test_topology_cyclic_resilience():
    # svc_a depends on svc_b, svc_b depends on svc_a (cyclic)
    topo = SystemTopology()
    topo.add_dependency("svc_a", "svc_b")
    topo.add_dependency("svc_b", "svc_a")

    # Should not infinite loop or fail
    order = topo.topological_sort()
    assert set(order) == {"svc_a", "svc_b"}
    assert topo.get_impact_radius("svc_a") == {"svc_a", "svc_b"}


def test_topology_serialization():
    topo = SystemTopology()
    topo.add_dependency("gateway", "service_a")
    topo.add_dependency("gateway", "service_b")

    data = topo.to_dict()
    assert "gateway" in data
    assert set(data["gateway"]) == {"service_a", "service_b"}

    restored = SystemTopology.from_dict(data)
    assert set(restored.get_dependencies("gateway")) == {"service_a", "service_b"}


@pytest.mark.asyncio
async def test_knowledge_store_topology_persistence():
    store = InMemoryKnowledgeStore()
    assert await store.get_topology() is None

    topo = SystemTopology()
    topo.add_dependency("frontend", "backend")
    await store.store_topology(topo)

    retrieved = await store.get_topology()
    assert retrieved is not None
    assert retrieved.get_dependencies("frontend") == ["backend"]
