import pytest
from datetime import datetime

from polaris_refactored.src.digital_twin.world_model import (
    CompositeWorldModel,
    StatisticalWorldModel,
    MLWorldModel,
)
from polaris_refactored.src.framework.events import TelemetryEvent
from polaris_refactored.src.domain.models import SystemState, MetricValue, HealthStatus


@pytest.mark.asyncio
async def test_composite_model_tolerates_internal_errors():
    class BadModel(MLWorldModel):
        async def update_system_state(self, telemetry):
            raise RuntimeError("boom")
        async def predict_system_behavior(self, system_id: str, time_horizon: int):
            raise RuntimeError("boom")
        async def simulate_adaptation_impact(self, system_id: str, action):
            raise RuntimeError("boom")

    class GoodModel(MLWorldModel):
        async def predict_system_behavior(self, system_id: str, time_horizon: int):
            from polaris_refactored.src.digital_twin.world_model import PredictionResult
            return PredictionResult({"cpu": 0.9}, 0.8)
        async def simulate_adaptation_impact(self, system_id: str, action):
            from polaris_refactored.src.digital_twin.world_model import SimulationResult
            return SimulationResult({"latency": 120.0}, 0.7)

    comp = CompositeWorldModel([BadModel(), GoodModel()])

    # update should not raise
    state = SystemState("sys", datetime.utcnow(), {"cpu": MetricValue("cpu", 0.5)}, HealthStatus.HEALTHY)
    await comp.update_system_state(TelemetryEvent(system_state=state))

    pred = await comp.predict_system_behavior("sys", 60)
    assert pred.outcomes.get("cpu") == pytest.approx(0.9, rel=1e-6)
    assert pred.probability > 0.0

    sim = await comp.simulate_adaptation_impact("sys", {})
    assert sim.outcomes.get("latency") == pytest.approx(120.0, rel=1e-6)
    assert sim.probability > 0.0


@pytest.mark.asyncio
async def test_statistical_simulation_no_scale_change_and_non_numeric_ignored():
    wm = StatisticalWorldModel(window=2)
    sys_id = "s"
    # metric includes a non-numeric value which should be ignored in windows
    state = SystemState(
        sys_id,
        datetime.utcnow(),
        {"cpu": MetricValue("cpu", 0.8), "status": MetricValue("status", "ok")},
        HealthStatus.HEALTHY,
    )
    await wm.update_system_state(TelemetryEvent(system_state=state))

    # scale_factor = 1 -> no change
    sim = await wm.simulate_adaptation_impact(sys_id, {"parameters": {"scale_factor": 1}})
    # cpu outcome equals base value
    assert sim.outcomes.get("cpu") == pytest.approx(0.8, rel=1e-6)
