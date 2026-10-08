"""Pluggable domain surrogates for counterfactual world model predictions.

Provides analytical, physics-based, and queuing theory surrogates:
- QueuingDomainSurrogate: M/M/m queuing capacity and load shedding models (web servers / microservices)
- ModelSwitchingDomainSurrogate: Multi-model accuracy/latency profiles (ML inference / computer vision)
"""

from __future__ import annotations

from typing import Dict, Optional

from polaris.abstractions.world_model import DomainSurrogate
from polaris.core.models import AdaptationAction, SystemState

DEFAULT_VISION_MODELS: Dict[str, Dict[str, float]] = {
    "yolov5n": {"confidence": 0.52, "latency_s": 0.040, "cpu_percent": 28.0, "rate": 280.0},
    "yolov5s": {"confidence": 0.62, "latency_s": 0.065, "cpu_percent": 40.0, "rate": 260.0},
    "yolov5m": {"confidence": 0.69, "latency_s": 0.096, "cpu_percent": 48.0, "rate": 244.0},
    "yolov5l": {"confidence": 0.76, "latency_s": 0.145, "cpu_percent": 64.0, "rate": 210.0},
    "yolov5x": {"confidence": 0.82, "latency_s": 0.210, "cpu_percent": 78.0, "rate": 180.0},
}


class QueuingDomainSurrogate(DomainSurrogate):
    """M/M/m queuing capacity and load-shedding domain surrogate.

    Simulates expected counterfactual metric shifts when scaling horizontal worker
    instances or adjusting admission control / dimmer ratios.
    """

    def can_handle(self, system_id: str, action_type: str) -> bool:
        """Check if action is a horizontal scaling or load-shedding action."""
        token = action_type.strip().lower()
        return token in (
            "scale_up",
            "scale-up",
            "scale_down",
            "scale-down",
            "set_dimmer",
            "set-dimmer",
        )

    def predict_deltas(self, action: AdaptationAction, state: SystemState) -> Dict[str, float]:
        """Predict queuing metric deltas for scaling and dimmer actions."""
        deltas: Dict[str, float] = {}
        act_type = action.action_type.strip().lower()
        params = action.parameters or {}

        s_count_mv = state.metrics.get("server_count")
        s_val = float(s_count_mv.value) if s_count_mv and s_count_mv.value is not None else 2.0

        if act_type in ("scale_up", "scale-up"):
            deltas["server_count"] = 1.0
            if "active_servers" in state.metrics:
                deltas["active_servers"] = 1.0

            ratio = s_val / (s_val + 1.0)
            if "average_utilization" in state.metrics:
                old_u = float(state.metrics["average_utilization"].value)
                new_u = round(old_u * ratio, 4)
                deltas["average_utilization"] = round(new_u - old_u, 4)

            if "average_response_time" in state.metrics:
                old_r = float(state.metrics["average_response_time"].value)
                r_delta = -round(old_r * 0.25, 2)
                deltas["average_response_time"] = r_delta

        elif act_type in ("scale_down", "scale-down"):
            new_s = max(1.0, s_val - 1.0)
            deltas["server_count"] = new_s - s_val
            if "active_servers" in state.metrics:
                deltas["active_servers"] = new_s - s_val

            ratio = s_val / new_s if new_s > 0 else 1.33
            if "average_utilization" in state.metrics:
                old_u = float(state.metrics["average_utilization"].value)
                new_u = min(1.0, round(old_u * ratio, 4))
                deltas["average_utilization"] = round(new_u - old_u, 4)

            if "average_response_time" in state.metrics:
                old_r = float(state.metrics["average_response_time"].value)
                r_delta = round(old_r * 0.30, 2)
                deltas["average_response_time"] = r_delta

        elif act_type in ("set_dimmer", "set-dimmer"):
            dim_val = params.get("value", params.get("dimmer"))
            if dim_val is not None:
                try:
                    d_target = float(dim_val)
                    curr_dim = (
                        float(state.metrics["dimmer"].value) if "dimmer" in state.metrics else 1.0
                    )
                    deltas["dimmer"] = d_target - curr_dim
                    if "average_response_time" in state.metrics:
                        d_diff = d_target - curr_dim
                        r_delta = round(d_diff * 120.0, 2)
                        deltas["average_response_time"] = r_delta
                except (ValueError, TypeError):
                    pass

        return deltas


class ModelSwitchingDomainSurrogate(DomainSurrogate):
    """Machine learning model switching domain surrogate.

    Simulates expected QoS deltas (latency, confidence, CPU, throughput)
    when switching between machine learning model variants.
    """

    def __init__(self, model_profiles: Optional[Dict[str, Dict[str, float]]] = None):
        """Initialize with model profiles."""
        self._profiles = dict(model_profiles or DEFAULT_VISION_MODELS)

    def can_handle(self, system_id: str, action_type: str) -> bool:
        """Check if action is a model switching adaptation."""
        token = action_type.strip().lower()
        return token in ("switch_model", "model_switch", "switch")

    def predict_deltas(self, action: AdaptationAction, state: SystemState) -> Dict[str, float]:
        """Predict QoS deltas for model switching."""
        deltas: Dict[str, float] = {}
        params = action.parameters or {}
        m_name = params.get("model_name") or params.get("model")

        if m_name and str(m_name) in self._profiles:
            prof = self._profiles[str(m_name)]
            mapping = {
                "confidence_mean": prof["confidence"],
                "response_time": prof["latency_s"],
                "cpu_usage": prof["cpu_percent"],
                "inference_rate": prof["rate"],
            }
            for m_k, prof_v in mapping.items():
                if m_k in state.metrics:
                    old_v = float(state.metrics[m_k].value)
                    deltas[m_k] = round(prof_v - old_v, 4)

        return deltas
