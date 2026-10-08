"""SWITCH system connector for POLARIS framework.

Connects to the SWITCH self-adaptive Machine Learning-Enabled System (MLS) exemplar.
Supports dynamic vision model switching (e.g. YOLOv5/COCO object detection)
balancing latency, accuracy, and CPU utilization.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

from polaris.abstractions.connector import Connector
from polaris.abstractions.connector_capabilities import ConnectorCapabilities
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.system_contract import (
    ActionSchema,
    MetricDirection,
    MetricSchema,
    MetricType,
    SLOContract,
)
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)

if TYPE_CHECKING:
    import httpx

DEFAULT_SWITCH_PORT = 5001
DEFAULT_SWITCH_MODELS: Dict[str, Dict[str, float]] = {
    "yolov5n": {"confidence": 0.52, "latency_s": 0.040, "cpu_percent": 28.0, "rate": 280.0},
    "yolov5s": {"confidence": 0.62, "latency_s": 0.065, "cpu_percent": 40.0, "rate": 260.0},
    "yolov5m": {"confidence": 0.69, "latency_s": 0.096, "cpu_percent": 48.0, "rate": 244.0},
    "yolov5l": {"confidence": 0.76, "latency_s": 0.145, "cpu_percent": 64.0, "rate": 210.0},
    "yolov5x": {"confidence": 0.82, "latency_s": 0.210, "cpu_percent": 78.0, "rate": 180.0},
}


class SWITCHConnector(Connector):
    """Connector for SWITCH ML-model switching adaptive system.

    Provides telemetry for confidence, median response time, CPU usage,
    and inference rate, with actions to dynamically switch models.
    Supports live REST endpoints and deterministic trace-driven simulation.
    """

    def __init__(
        self,
        base_url: str = f"http://localhost:{DEFAULT_SWITCH_PORT}",
        system_id: str = "switch",
        timeout: float = 10.0,
        synthetic_mode: bool = False,
        initial_model: str = "yolov5m",
        models: Optional[Dict[str, Dict[str, float]]] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize SWITCH connector."""
        self.base_url = base_url.rstrip("/")
        self.system_id = system_id
        self.timeout = timeout
        self.synthetic_mode = synthetic_mode
        self._models = dict(models or DEFAULT_SWITCH_MODELS)
        self._active_model = initial_model if initial_model in self._models else "yolov5m"
        self._switch_count = 0
        self._step_counter = 0
        self._logger = logger
        self._metrics = metrics
        self._connected = False
        self._client: Optional["httpx.AsyncClient"] = None

    async def _ensure_client(self) -> "httpx.AsyncClient":
        """Ensure async HTTP client is initialized."""
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "SWITCHConnector requires 'httpx' for REST mode. Install with: pip install httpx"
            ) from exc

        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def connect(self) -> bool:
        """Connect to SWITCH managed system or initialize synthetic simulation."""
        if self.synthetic_mode:
            self._connected = True
            if self._logger:
                self._logger.info(
                    "SWITCHConnector initialized in synthetic simulation mode",
                    system_id=self.system_id,
                    active_model=self._active_model,
                )
            if self._metrics:
                self._metrics.increment("polaris.connector.switch.connected")
            return True

        try:
            client = await self._ensure_client()
            resp = await client.get("/health")
            resp.raise_for_status()
            self._connected = True
            if self._logger:
                self._logger.info(
                    "SWITCHConnector connected to HTTP endpoint",
                    base_url=self.base_url,
                    system_id=self.system_id,
                )
            if self._metrics:
                self._metrics.increment("polaris.connector.switch.connected")
            return True
        except Exception as exc:
            if self._logger:
                self._logger.warning(
                    f"SWITCH HTTP connection failed ({exc}); falling back to synthetic simulation mode",
                    system_id=self.system_id,
                )
            # Automatic fallback to high-fidelity synthetic mode for reproducibility
            self.synthetic_mode = True
            self._connected = True
            if self._metrics:
                self._metrics.increment("polaris.connector.switch.synthetic_fallback")
            return True

    async def disconnect(self) -> bool:
        """Disconnect and cleanup resources."""
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._logger:
            self._logger.info("SWITCHConnector disconnected", system_id=self.system_id)
        return True

    async def get_system_id(self) -> str:
        """Return target system ID."""
        return self.system_id

    async def collect_telemetry(self) -> SystemState:
        """Collect QoS telemetry metrics from SWITCH."""
        now = datetime.now(timezone.utc)
        self._step_counter += 1

        if self.synthetic_mode:
            model_info = self._models.get(self._active_model, self._models["yolov5m"])
            # Simulate slight natural variance across frames
            noise_factor = 1.0 + 0.04 * ((self._step_counter % 5) - 2)
            conf = min(0.99, max(0.1, model_info["confidence"] * noise_factor))
            latency = max(0.01, model_info["latency_s"] * noise_factor)
            cpu = min(100.0, max(5.0, model_info["cpu_percent"] * noise_factor))
            rate = max(10.0, model_info["rate"] * noise_factor)

            # Determine health status based on latency SLA (target: <= 0.15s)
            health = HealthStatus.HEALTHY
            if latency > 0.15:
                health = HealthStatus.WARNING
            if latency > 0.20:
                health = HealthStatus.CRITICAL

            metrics = {
                "confidence_mean": MetricValue("confidence_mean", round(conf, 4), timestamp=now),
                "response_time": MetricValue(
                    "response_time", round(latency, 4), unit="s", timestamp=now
                ),
                "cpu_usage": MetricValue("cpu_usage", round(cpu, 2), unit="percent", timestamp=now),
                "inference_rate": MetricValue(
                    "inference_rate", round(rate, 2), unit="inf/min", timestamp=now
                ),
                "switch_count": MetricValue("switch_count", self._switch_count, timestamp=now),
            }

            return SystemState(
                system_id=self.system_id,
                timestamp=now,
                metrics=metrics,
                health_status=health,
                metadata={
                    "active_model": self._active_model,
                    "available_models": list(self._models.keys()),
                    "synthetic_mode": True,
                },
            )

        # HTTP REST telemetry collection
        client = await self._ensure_client()
        resp = await client.get("/telemetry")
        resp.raise_for_status()
        data = resp.json()

        conf = float(data.get("confidence_mean", 0.65))
        latency = float(data.get("response_time", 0.10))
        cpu = float(data.get("cpu_usage", 50.0))
        rate = float(data.get("inference_rate", 240.0))
        self._active_model = str(data.get("active_model", self._active_model))
        self._switch_count = int(data.get("switch_count", self._switch_count))

        health = HealthStatus.HEALTHY
        if latency > 0.15:
            health = HealthStatus.WARNING
        if latency > 0.20:
            health = HealthStatus.CRITICAL

        metrics = {
            "confidence_mean": MetricValue("confidence_mean", conf, timestamp=now),
            "response_time": MetricValue("response_time", latency, unit="s", timestamp=now),
            "cpu_usage": MetricValue("cpu_usage", cpu, unit="percent", timestamp=now),
            "inference_rate": MetricValue("inference_rate", rate, unit="inf/min", timestamp=now),
            "switch_count": MetricValue("switch_count", self._switch_count, timestamp=now),
        }

        return SystemState(
            system_id=self.system_id,
            timestamp=now,
            metrics=metrics,
            health_status=health,
            metadata={"active_model": self._active_model, "synthetic_mode": False},
        )

    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute model switching adaptation action."""
        t_start = time.perf_counter()
        action_type = action.action_type.strip().lower()

        if action_type not in ("switch_model", "model_switch", "switch"):
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=f"Unsupported action type: {action.action_type}",
            )

        params = action.parameters or {}
        target_model = params.get("model_name") or params.get("model")
        if not target_model or str(target_model) not in self._models:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=f"Invalid or missing model_name: {target_model}; choose from {list(self._models.keys())}",
            )

        target_model_str = str(target_model)
        prev_model = self._active_model

        if self.synthetic_mode:
            if target_model_str != prev_model:
                self._active_model = target_model_str
                self._switch_count += 1

            latency_ms = int((time.perf_counter() - t_start) * 1000)
            if self._logger:
                self._logger.info(
                    f"SWITCH model transitioned: {prev_model} -> {target_model_str}",
                    action_id=action.action_id,
                    total_switches=self._switch_count,
                )
            if self._metrics:
                self._metrics.increment(
                    "polaris.connector.switch.action_executed",
                    tags={"from_model": prev_model, "to_model": target_model_str},
                )

            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.SUCCESS,
                result_data={
                    "previous_model": prev_model,
                    "active_model": target_model_str,
                    "total_switches": self._switch_count,
                },
                execution_time_ms=latency_ms,
            )

        # HTTP REST action execution
        try:
            client = await self._ensure_client()
            resp = await client.post("/switch", json={"model_name": target_model_str})
            resp.raise_for_status()
            data = resp.json()
            if target_model_str != prev_model:
                self._switch_count += 1
            self._active_model = target_model_str

            latency_ms = int((time.perf_counter() - t_start) * 1000)
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.SUCCESS,
                result_data=data,
                execution_time_ms=latency_ms,
            )
        except Exception as exc:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=str(exc),
            )

    async def get_supported_actions(self) -> List[AdaptationAction]:
        """Return supported adaptation actions."""
        return [
            AdaptationAction(
                action_id="",
                action_type="switch_model",
                target_system=self.system_id,
                parameters={"model_name": self._active_model},
            )
        ]

    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate candidate adaptation action."""
        act_type = action.action_type.strip().lower()
        if act_type not in ("switch_model", "model_switch", "switch"):
            return False
        params = action.parameters or {}
        target_model = params.get("model_name") or params.get("model")
        return bool(target_model and str(target_model) in self._models)

    async def get_capabilities(self) -> ConnectorCapabilities:
        """Expose formal contract capabilities for SWITCH."""
        switch_schema = ActionSchema(
            action_type="switch_model",
            description="Dynamically switch vision detection model to trade off accuracy, latency, and CPU.",
            parameters_schema={
                "model_name": {
                    "type": "string",
                    "enum": list(self._models.keys()),
                    "description": "Target vision model identifier",
                }
            },
            required_parameters=("model_name",),
            rollback_action="switch_model",
            expected_duration_seconds=0.05,
            default_verification_window_seconds=1.0,
            performance_impact="positive",
            qos_impact="positive",
        )

        metrics = {
            "confidence_mean": MetricSchema(
                name="confidence_mean",
                metric_type=MetricType.GAUGE,
                direction=MetricDirection.MAXIMIZE,
                description="Average bounding box detection confidence (accuracy surrogate)",
                target_value=0.70,
                warning_threshold=0.60,
                critical_threshold=0.50,
            ),
            "response_time": MetricSchema(
                name="response_time",
                metric_type=MetricType.GAUGE,
                direction=MetricDirection.MINIMIZE,
                unit="s",
                description="Per-frame inference latency (median)",
                target_value=0.10,
                warning_threshold=0.15,
                critical_threshold=0.20,
            ),
            "cpu_usage": MetricSchema(
                name="cpu_usage",
                metric_type=MetricType.GAUGE,
                direction=MetricDirection.MINIMIZE,
                unit="percent",
                description="CPU utilization percentage",
                target_value=50.0,
                warning_threshold=70.0,
                critical_threshold=85.0,
            ),
            "inference_rate": MetricSchema(
                name="inference_rate",
                metric_type=MetricType.RATE,
                direction=MetricDirection.MAXIMIZE,
                unit="inf/min",
                description="Inference throughput per minute",
                target_value=240.0,
            ),
        }

        slos = (
            SLOContract(
                metric_name="response_time",
                operator="<=",
                target_value=0.15,
                description="Median response time must not exceed 150ms",
                priority=1.0,
            ),
            SLOContract(
                metric_name="confidence_mean",
                operator=">=",
                target_value=0.65,
                description="Detection confidence must remain at or above 65%",
                priority=0.8,
            ),
        )

        return ConnectorCapabilities(
            supported_action_types=("switch_model",),
            action_aliases={"model_switch": "switch_model", "switch": "switch_model"},
            metadata={
                "exemplar": "SWITCH",
                "domain": "ml_enabled_systems",
                "dataset": "COCO_2017",
                "models": self._models,
            },
            actions={"switch_model": switch_schema},
            metrics=metrics,
            slos=slos,
        )
