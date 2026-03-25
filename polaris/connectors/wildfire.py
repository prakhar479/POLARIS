"""Wildfire system connector for POLARIS framework."""

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from polaris.abstractions.connector import Connector
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.infrastructure.constants import (
    DEFAULT_CONNECTOR_TIMEOUT,
    DEFAULT_WILDFIRE_PORT,
    HTTP_STATUS_MAX_SUCCESS,
    HTTP_STATUS_MIN_SUCCESS,
    MILLISECONDS_PER_SECOND,
)

if TYPE_CHECKING:
    import httpx


# Use string-based import to avoid circular imports
def _get_connector_class() -> type:
    from polaris.abstractions.connector import Connector

    return Connector


class WildfireConnector(Connector):
    """Connector for Wildfire fire spread simulation system."""

    def __init__(
        self,
        base_url: str = f"http://localhost:{DEFAULT_WILDFIRE_PORT}",
        system_id: str = "wildfire",
        timeout: float = DEFAULT_CONNECTOR_TIMEOUT,
        session_id: Optional[str] = None,
        auto_create_session: bool = True,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        """Initialize Wildfire connector with API endpoint and configuration."""
        self.base_url = base_url.rstrip("/")
        self.system_id = system_id
        self.timeout = timeout
        self.session_id = session_id
        self.auto_create_session = auto_create_session
        self._logger = logger
        self._metrics = metrics
        self._client: Optional["httpx.AsyncClient"] = None
        self._connected = False

    async def _ensure_client(self) -> "httpx.AsyncClient":
        """Ensure HTTP client is initialized."""
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "WildfireConnector requires 'httpx'. Install with: pip install httpx"
            ) from exc

        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def connect(self) -> bool:
        """Connect to Wildfire API and verify connectivity."""
        try:
            client = await self._ensure_client()

            if self.session_id:
                await client.put(f"/api/v1/sessions/{self.session_id}/current")
            elif self.auto_create_session:
                resp = await client.post("/api/v1/sessions")
                resp.raise_for_status()
                data = resp.json()
                self.session_id = data.get("session_id", self.session_id)

            health_resp = await client.get("/health")
            health_resp.raise_for_status()
            self._connected = True

            if self._logger:
                self._logger.info(
                    "WildfireConnector connected",
                    base_url=self.base_url,
                    system_id=self.system_id,
                )
            if self._metrics:
                self._metrics.increment("polaris.connector.wildfire.connected")

            return True
        except Exception as exc:
            self._connected = False
            if self._logger:
                self._logger.error(
                    "WildfireConnector connection failed",
                    error=str(exc),
                    base_url=self.base_url,
                )
            if self._metrics:
                self._metrics.increment("polaris.connector.wildfire.connection_errors")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Wildfire API and clean up resources."""
        self._connected = False
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._logger:
            self._logger.info("WildfireConnector disconnected", base_url=self.base_url)
        if self._metrics:
            self._metrics.increment("polaris.connector.wildfire.disconnected")
        return True

    async def get_system_id(self) -> str:
        """Return the system identifier for this connector."""
        return self.system_id

    async def collect_telemetry(self) -> SystemState:
        """Collect current system state and metrics from Wildfire API."""
        if not self._connected:
            connected = await self.connect()
            if not connected:
                return SystemState(
                    system_id=self.system_id,
                    timestamp=datetime.now(timezone.utc),
                    metrics={},
                    health_status=HealthStatus.UNHEALTHY,
                    metadata={"error": "Unable to connect to Wildfire adapter"},
                )

        start = time.monotonic()
        try:
            client = await self._ensure_client()
            resp = await client.get("/api/v1/sim/metrics")
            if resp.status_code != 200:
                raise RuntimeError(f"Unexpected status code {resp.status_code}")

            payload: Dict[str, Any] = resp.json()
            metrics_payload: Dict[str, Any] = payload.get("metrics", {})
            now = datetime.now(timezone.utc)

            metrics: Dict[str, MetricValue] = {}

            timestep_value = metrics_payload.get("timestep", payload.get("timestep", 0))
            metrics["timestep"] = MetricValue(
                name="timestep",
                value=int(timestep_value),
                unit="step",
                timestamp=now,
            )

            num_agents = metrics_payload.get("num_agents")
            if isinstance(num_agents, int):
                metrics["num_agents"] = MetricValue(
                    name="num_agents",
                    value=num_agents,
                    unit="count",
                    timestamp=now,
                )

            mr1_values = metrics_payload.get("mr1_values") or []
            if isinstance(mr1_values, list) and mr1_values:
                try:
                    mr1_floats = [float(v) for v in mr1_values]
                    mr1_avg = (
                        sum(mr1_floats) / num_agents
                        if isinstance(num_agents, int) and num_agents > 0
                        else 0.0
                    )
                    metrics["mr1_avg"] = MetricValue(
                        name="mr1_avg",
                        value=mr1_avg,
                        unit="score",
                        timestamp=now,
                    )
                    mr1_total = sum(mr1_floats)
                    metrics["mr1_total"] = MetricValue(
                        name="mr1_total",
                        value=mr1_total,
                        unit="score",
                        timestamp=now,
                    )
                except (TypeError, ValueError):
                    pass

            mr2_value = metrics_payload.get("mr2_value")
            if mr2_value is not None:
                try:
                    metrics["mr2_value"] = MetricValue(
                        name="mr2_value",
                        value=float(mr2_value),
                        unit="count",
                        timestamp=now,
                    )
                except (TypeError, ValueError):
                    pass

            burning = metrics_payload.get("fire_cells_burning")
            total = metrics_payload.get("fire_cells_total")
            if isinstance(burning, int):
                metrics["fire_cells_burning"] = MetricValue(
                    name="fire_cells_burning",
                    value=burning,
                    unit="cells",
                    timestamp=now,
                )
            if isinstance(total, int) and total > 0 and isinstance(burning, int):
                ratio = 100.0 * float(burning) / float(total)
                metrics["fire_cells_burning_ratio"] = MetricValue(
                    name="fire_cells_burning_ratio",
                    value=ratio,
                    unit="percent",
                    timestamp=now,
                )

            if self._metrics:
                duration = time.monotonic() - start
                self._metrics.histogram(
                    "polaris.connector.wildfire.telemetry_duration_seconds",
                    duration,
                )

            return SystemState(
                system_id=self.system_id,
                timestamp=now,
                metrics=metrics,
                health_status=HealthStatus.HEALTHY,
                metadata={"raw_metrics": metrics_payload},
            )
        except Exception as exc:
            if self._logger:
                self._logger.error(
                    "WildfireConnector telemetry collection failed",
                    error=str(exc),
                )
            if self._metrics:
                self._metrics.increment("polaris.connector.wildfire.telemetry_errors")
            return SystemState(
                system_id=self.system_id,
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
                metadata={"error": str(exc)},
            )

    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute adaptation action on Wildfire system."""
        if not self._connected:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message="Not connected to Wildfire adapter",
            )

        start = time.monotonic()
        try:
            client = await self._ensure_client()
            action_type = action.action_type.lower()

            # Action handlers mapping
            action_handlers = {
                "wildfire_reset": lambda: client.post("/api/v1/sim/reset"),
                "wildfire_pause": lambda: client.post("/api/v1/sim/pause"),
                "wildfire_resume": lambda: client.post("/api/v1/sim/resume"),
                "wildfire_step": lambda: client.post("/api/v1/sim/step"),
                "wildfire_move": lambda: client.post(
                    "/api/v1/sim/action", json=(action.parameters or {}).get("actions", [])
                ),
                "wildfire_batch_actions": lambda: client.post(
                    "/api/v1/sim/batch-actions",
                    json={"actions": (action.parameters or {}).get("actions", [])},
                ),
            }

            handler = action_handlers.get(action_type)
            if handler:
                resp = await handler()
            else:
                if self._metrics:
                    self._metrics.increment("polaris.connector.wildfire.actions_unsupported")
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    result_data={},
                    error_message=f"Unsupported action type: {action.action_type}",
                )

            status_success = HTTP_STATUS_MIN_SUCCESS <= resp.status_code < HTTP_STATUS_MAX_SUCCESS
            response_data: Dict[str, Any] = {}
            try:
                response_data = resp.json()
            except Exception:
                response_data = {"raw_text": resp.text}

            exec_status = ExecutionStatus.SUCCESS if status_success else ExecutionStatus.FAILED
            error_message = (
                None if status_success else response_data.get("error") or f"HTTP {resp.status_code}"
            )

            duration_ms = int((time.monotonic() - start) * MILLISECONDS_PER_SECOND)
            if self._metrics:
                self._metrics.histogram(
                    "polaris.connector.wildfire.action_execution_duration_ms",
                    duration_ms,
                    tags={"action_type": action.action_type},
                )
                self._metrics.increment(
                    "polaris.connector.wildfire.actions_executed",
                    tags={"action_type": action.action_type, "status": exec_status.value},
                )

            if self._logger:
                self._logger.info(
                    "WildfireConnector action executed",
                    action_type=action.action_type,
                    status=exec_status.value,
                )

            return ExecutionResult(
                action_id=action.action_id,
                status=exec_status,
                result_data={"response": response_data},
                error_message=error_message,
                execution_time_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * MILLISECONDS_PER_SECOND)
            if self._metrics:
                self._metrics.histogram(
                    "polaris.connector.wildfire.action_execution_duration_ms",
                    duration_ms,
                    tags={"action_type": action.action_type},
                )
                self._metrics.increment(
                    "polaris.connector.wildfire.actions_executed",
                    tags={
                        "action_type": action.action_type,
                        "status": ExecutionStatus.FAILED.value,
                    },
                )
            if self._logger:
                self._logger.error(
                    "WildfireConnector action execution failed",
                    action_type=action.action_type,
                    error=str(exc),
                )
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=str(exc),
                execution_time_ms=duration_ms,
            )

    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if action type is supported by Wildfire system."""
        allowed_types = {
            "wildfire_reset",
            "wildfire_pause",
            "wildfire_resume",
            "wildfire_step",
            "wildfire_move",
            "wildfire_batch_actions",
        }
        if action.target_system.lower() != self.system_id.lower():
            return False
        return action.action_type.lower() in allowed_types

    async def get_supported_actions(self) -> List[AdaptationAction]:
        """Return list of supported adaptation actions for Wildfire system."""
        return [
            AdaptationAction(
                action_id="",
                action_type="wildfire_reset",
                target_system=self.system_id,
                parameters={},
            ),
            AdaptationAction(
                action_id="",
                action_type="wildfire_pause",
                target_system=self.system_id,
                parameters={},
            ),
            AdaptationAction(
                action_id="",
                action_type="wildfire_resume",
                target_system=self.system_id,
                parameters={},
            ),
            AdaptationAction(
                action_id="",
                action_type="wildfire_step",
                target_system=self.system_id,
                parameters={},
            ),
            AdaptationAction(
                action_id="",
                action_type="wildfire_move",
                target_system=self.system_id,
                parameters={"actions": []},
            ),
            AdaptationAction(
                action_id="",
                action_type="wildfire_batch_actions",
                target_system=self.system_id,
                parameters={"actions": []},
            ),
        ]
