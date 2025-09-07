import pytest
from datetime import datetime
from polaris_refactored.src.infrastructure.data_storage.data_store import PolarisDataStore
from polaris_refactored.src.infrastructure.exceptions import DataStoreError
from polaris_refactored.src.infrastructure.data_storage import InMemoryGraphStorageBackend
from polaris_refactored.src.domain.models import SystemState, MetricValue, HealthStatus, AdaptationAction


@pytest.mark.asyncio
async def test_unit_of_work_commit_rollback():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })
    await ds.start()

    # Get repos
    state_repo = ds.get_repository("system_states")
    action_repo = ds.get_repository("adaptation_actions")

    # Create test data
    state = SystemState(
        system_id="test-sys",
        timestamp=datetime.utcnow(),
        metrics={"cpu": MetricValue("cpu", 0.5)},
        health_status=HealthStatus.HEALTHY,
    )
    action = AdaptationAction(
        action_id="act-1",
        action_type="scale_out",
        target_system="test-sys",
        parameters={"count": 1},
        priority=1,
        timeout_seconds=60,
    )

    # Test commit
    async with ds.unit_of_work() as uow:
        uow.add_save("system_states", state)
        uow.add_save("adaptation_actions", action)
        # commit happens on exit

    # Verify committed
    saved_state = await state_repo.get_by_id(f"{state.system_id}_{state.timestamp.isoformat()}")
    assert saved_state is not None
    saved_action = await action_repo.get_by_id(action.action_id)
    assert saved_action is not None
    assert saved_action.action_type == "scale_out"

    # Test rollback on error
    try:
        async with ds.unit_of_work() as uow:
            uow.add_delete("adaptation_actions", action.action_id)
            raise RuntimeError("oops")
    except RuntimeError:
        pass  # expected

    # Verify rollback - action should still exist
    assert await action_repo.get_by_id(action.action_id) is not None

    # Test explicit rollback
    async with ds.unit_of_work() as uow:
        uow.add_delete("adaptation_actions", action.action_id)
        await uow.rollback()

    assert await action_repo.get_by_id(action.action_id) is not None

    await ds.stop()


@pytest.mark.asyncio
async def test_unit_of_work_error_handling():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({"graph": backend, "document": backend, "time_series": backend})
    await ds.start()

    # Test invalid repo name
    with pytest.raises(DataStoreError, match="not found"):
        async with ds.unit_of_work() as uow:
            uow.add_save("nonexistent_repo", object())

    # Test commit without start
    async with ds.unit_of_work() as uow:
        # This should work inside the context manager
        await uow.commit()
        
    # Test accessing uow after context manager exits
    with pytest.raises(DataStoreError, match="No active transaction to commit"):
        await uow.commit()

    await ds.stop()
