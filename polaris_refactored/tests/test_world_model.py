import pytest
from datetime import datetime

from polaris_refactored.src.framework.events import TelemetryEvent
from polaris_refactored.src.domain.models import SystemState, MetricValue, HealthStatus
from polaris_refactored.src.digital_twin.world_model import (
    StatisticalWorldModel,
    CompositeWorldModel,
    MLWorldModel,
)


@pytest.mark.asyncio
async def test_statistical_world_model_update_and_predict():
    wm = StatisticalWorldModel(window=3)
    sys_id = "sys-1"

    # feed three telemetry updates
    for v in [0.3, 0.6, 0.9]:
        state = SystemState(
            system_id=sys_id,
            timestamp=datetime.utcnow(),
            metrics={"cpu": MetricValue(name="cpu", value=v)},
            health_status=HealthStatus.HEALTHY,
        )
        evt = TelemetryEvent(system_state=state)
        await wm.update_system_state(evt)

    pred = await wm.predict_system_behavior(sys_id, time_horizon=60)
    # Expect mean approx (0.3+0.6+0.9)/3=0.6
    assert "cpu" in pred.outcomes
    assert abs(pred.outcomes["cpu"] - 0.6) < 1e-6
    assert pred.probability > 0.0

    # simulate scale out impact should reduce cpu
    sim = await wm.simulate_adaptation_impact(sys_id, {"parameters": {"scale_factor": 2}})
    assert "cpu" in sim.outcomes
    assert sim.outcomes["cpu"] <= pred.outcomes["cpu"]


@pytest.mark.asyncio
async def test_composite_world_model_weighted_merge():
    # Model A predicts cpu=1.0 with high confidence
    class MockModelA(MLWorldModel):
        async def predict_system_behavior(self, system_id: str, time_horizon: int):
            from polaris_refactored.src.digital_twin.world_model import PredictionResult
            return PredictionResult({"cpu": 1.0}, 0.9)
        async def simulate_adaptation_impact(self, system_id: str, action):
            from polaris_refactored.src.digital_twin.world_model import SimulationResult
            return SimulationResult({"latency": 100.0}, 0.8)

    # Model B predicts cpu=0.0 with low confidence
    class MockModelB(MLWorldModel):
        async def predict_system_behavior(self, system_id: str, time_horizon: int):
            from polaris_refactored.src.digital_twin.world_model import PredictionResult
            return PredictionResult({"cpu": 0.0}, 0.1)
        async def simulate_adaptation_impact(self, system_id: str, action):
            from polaris_refactored.src.digital_twin.world_model import SimulationResult
            return SimulationResult({"latency": 200.0}, 0.2)

    comp = CompositeWorldModel([MockModelA(), MockModelB()])
    pred = await comp.predict_system_behavior("sys", 60)
    # Weighted average should be closer to 1.0
    assert pred.outcomes["cpu"] > 0.5
    assert pred.probability > 0.0

    sim = await comp.simulate_adaptation_impact("sys", {})
    # Weighted latency closer to 100 than 200
    assert sim.outcomes["latency"] < 150.0
    assert sim.probability > 0.0


@pytest.mark.asyncio
async def test_ml_world_model_stub_defaults():
    ml = MLWorldModel()
    pred = await ml.predict_system_behavior("sys", 60)
    assert isinstance(pred.outcomes, dict)
    assert 0.0 < pred.probability <= 0.5
    sim = await ml.simulate_adaptation_impact("sys", {})
    assert isinstance(sim.outcomes, dict)
    assert 0.0 < sim.probability <= 0.5
