import asyncio
from datetime import datetime
import pytest

from polaris_refactored.src.infrastructure.data_storage import InMemoryGraphStorageBackend, PolarisDataStore
from polaris_refactored.src.digital_twin.knowledge_base import PolarisKnowledgeBase
from polaris_refactored.src.digital_twin.telemetry_subscriber import subscribe_telemetry_persistence
from polaris_refactored.src.framework.events import PolarisEventBus, TelemetryEvent
from polaris_refactored.src.domain.models import SystemState, MetricValue, HealthStatus


@pytest.mark.asyncio
async def test_telemetry_persistence_subscription_end_to_end():
    # Set up in-memory backends and data store
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })
    await ds.start()
    kb = PolarisKnowledgeBase(ds)

    # Start event bus and subscribe the telemetry persistence handler
    bus = PolarisEventBus(worker_count=1)
    await bus.start()
    sub_id = await subscribe_telemetry_persistence(bus, kb)
    assert isinstance(sub_id, str) and len(sub_id) > 0

    # Publish a telemetry event
    system_id = "sys-sub"
    state = SystemState(
        system_id=system_id,
        timestamp=datetime.utcnow(),
        metrics={"cpu": MetricValue(name="cpu", value=0.75)},
        health_status=HealthStatus.HEALTHY,
    )
    await bus.publish_telemetry(TelemetryEvent(system_state=state))

    # Wait briefly for async worker to process
    for _ in range(10):
        current = await kb.get_current_state(system_id)
        if current is not None:
            break
        await asyncio.sleep(0.05)
    current = await kb.get_current_state(system_id)
    assert current is not None
    assert "cpu" in current.metrics
    assert current.metrics["cpu"].value == 0.75

    # Cleanup
    await bus.stop()
    await ds.stop()
