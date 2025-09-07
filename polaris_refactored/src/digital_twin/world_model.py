"""
World Model Implementation

Placeholder for the world model that will be implemented in task 6.1.
This provides the interface definitions for now.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..domain.models import SystemState
from ..framework.events import TelemetryEvent
from ..infrastructure.di import Injectable
from .knowledge_base import PolarisKnowledgeBase


class PredictionResult:
    """Result of a world model prediction."""
    
    def __init__(self, outcomes: Dict[str, Any], probability: float):
        self.outcomes = outcomes
        self.probability = probability


class SimulationResult:
    """Result of a world model simulation."""
    
    def __init__(self, outcomes: Dict[str, Any], probability: float):
        self.outcomes = outcomes
        self.probability = probability


class PolarisWorldModel(Injectable, ABC):
    """
    Abstract base class for POLARIS world models.
    
    This will be fully implemented in task 6.1.
    """
    
    @abstractmethod
    async def update_system_state(self, telemetry: TelemetryEvent) -> None:
        """Update the world model with new telemetry data."""
        pass
    
    @abstractmethod
    async def predict_system_behavior(self, system_id: str, time_horizon: int) -> PredictionResult:
        """Predict future system behavior."""
        pass
    
    @abstractmethod
    async def simulate_adaptation_impact(self, system_id: str, action: Any) -> SimulationResult:
        """Simulate the impact of an adaptation action."""
        pass


class CompositeWorldModel(PolarisWorldModel):
    """
    Composite world model that combines multiple model strategies.
    
    This will be fully implemented in task 6.1.
    """
    
    def __init__(self, models: List[PolarisWorldModel], weights: Optional[Dict[str, float]] = None):
        self.models = models
        # Optional per-model weighting by class name
        self.weights = weights or {}
    
    async def update_system_state(self, telemetry: TelemetryEvent) -> None:
        """Update all constituent models."""
        for m in self.models:
            try:
                await m.update_system_state(telemetry)
            except Exception:
                # Best-effort update; individual model errors should not break the ensemble
                continue
    
    async def predict_system_behavior(self, system_id: str, time_horizon: int) -> PredictionResult:
        """Combine predictions from all models with confidence-weighted aggregation."""
        combined: Dict[str, Any] = {}
        total_weight = 0.0
        # Aggregate by summing weighted numeric predictions when possible; last-write-wins for others
        for m in self.models:
            try:
                pred = await m.predict_system_behavior(system_id, time_horizon)
            except Exception:
                # Skip failing model
                continue
            w = self.weights.get(m.__class__.__name__, 1.0) * max(pred.probability, 0.0)
            if w <= 0:
                continue
            total_weight += w
            for k, v in pred.outcomes.items():
                if isinstance(v, (int, float)):
                    combined[k] = combined.get(k, 0.0) + w * float(v)
                else:
                    # Non-numeric: prefer higher-weight prediction
                    if k not in combined:
                        combined[k] = v
        # Normalize numeric aggregates
        if total_weight > 0:
            for k, v in list(combined.items()):
                if isinstance(v, (int, float)):
                    combined[k] = float(v) / total_weight
        confidence = min(1.0, total_weight / max(1.0, len(self.models))) if self.models else 0.0
        return PredictionResult(combined, confidence)
    
    async def simulate_adaptation_impact(self, system_id: str, action: Any) -> SimulationResult:
        """Combine simulation results from all models using confidence-weighted merge."""
        outcomes: Dict[str, Any] = {}
        total_weight = 0.0
        for m in self.models:
            try:
                sim = await m.simulate_adaptation_impact(system_id, action)
            except Exception:
                # Skip failing model
                continue
            w = self.weights.get(m.__class__.__name__, 1.0) * max(sim.probability, 0.0)
            if w <= 0:
                continue
            total_weight += w
            for k, v in sim.outcomes.items():
                if isinstance(v, (int, float)):
                    outcomes[k] = outcomes.get(k, 0.0) + w * float(v)
                else:
                    if k not in outcomes:
                        outcomes[k] = v
        if total_weight > 0:
            for k, v in list(outcomes.items()):
                if isinstance(v, (int, float)):
                    outcomes[k] = float(v) / total_weight
        probability = min(1.0, total_weight / max(1.0, len(self.models))) if self.models else 0.0
        return SimulationResult(outcomes, probability)


class StatisticalWorldModel(PolarisWorldModel):
    """Statistical world model implementation.

    Maintains simple rolling means per metric per system to provide naive forecasts
    and uses heuristics to simulate action impact.
    """

    def __init__(self, knowledge_base: Optional[PolarisKnowledgeBase] = None, window: int = 5):
        self._kb = knowledge_base
        self._window = max(1, window)
        # In-memory rolling windows: system_id -> metric_name -> list[float]
        self._windows: Dict[str, Dict[str, List[float]]] = {}

    async def update_system_state(self, telemetry: TelemetryEvent) -> None:
        state = telemetry.system_state
        sys_map = self._windows.setdefault(state.system_id, {})
        for mname, mval in state.metrics.items():
            try:
                val = float(mval.value)  # only track numeric
            except (TypeError, ValueError):
                continue
            arr = sys_map.setdefault(mname, [])
            arr.append(val)
            if len(arr) > self._window:
                arr.pop(0)

    async def predict_system_behavior(self, system_id: str, time_horizon: int) -> PredictionResult:
        sys_map = self._windows.get(system_id, {})
        preds: Dict[str, Any] = {}
        count_vals = 0
        for mname, arr in sys_map.items():
            if arr:
                mean = sum(arr) / len(arr)
                preds[mname] = mean  # naive constant prediction
                count_vals += 1
        confidence = min(1.0, (count_vals / max(1, len(sys_map)))) if sys_map else 0.0
        return PredictionResult(preds, confidence)

    async def simulate_adaptation_impact(self, system_id: str, action: Any) -> SimulationResult:
        """Very simple heuristic: if action includes 'scale_out' with factor f, reduce load metrics.

        Expected action structure (flexible): {"action_type": str, "parameters": {"scale_factor": float}}
        """
        params = {}
        if isinstance(action, dict):
            params = action.get("parameters", {}) if isinstance(action.get("parameters"), dict) else {}
        factor = float(params.get("scale_factor", 1.0))
        outcomes: Dict[str, Any] = {}
        sys_map = self._windows.get(system_id, {})
        for mname, arr in sys_map.items():
            if not arr:
                continue
            base = sum(arr) / len(arr)
            if "cpu" in mname.lower() or "latency" in mname.lower():
                outcomes[mname] = base / max(1.0, factor)
            else:
                outcomes[mname] = base
        probability = 0.5 if outcomes else 0.0
        return SimulationResult(outcomes, probability)


class MLWorldModel(PolarisWorldModel):
    """Machine learning world model implementation (minimal stub).

    Provides DI hooks to consume knowledge base context. The default implementation
    returns low-confidence defaults until a real model provider is integrated.
    """

    def __init__(self, knowledge_base: Optional[PolarisKnowledgeBase] = None):
        self._kb = knowledge_base

    async def update_system_state(self, telemetry: TelemetryEvent) -> None:
        # In a future implementation, extract features and update an online model.
        return None
    
    async def predict_system_behavior(self, system_id: str, time_horizon: int) -> PredictionResult:
        # Minimal stub prediction: no-op with low confidence
        return PredictionResult({}, 0.1)
    
    async def simulate_adaptation_impact(self, system_id: str, action: Any) -> SimulationResult:
        # Minimal stub simulation: no-op with low probability
        return SimulationResult({}, 0.1)