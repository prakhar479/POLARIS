"""Universal HTTP / REST system connector for POLARIS framework.

Provides domain-agnostic integration with arbitrary web services, microservices,
Prometheus metric endpoints, and REST APIs via declarative configuration.
"""

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from polaris.abstractions.connector import Connector
from polaris.abstractions.connector_capabilities import ConnectorCapabilities
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
    HTTP_STATUS_MAX_SUCCESS,
    HTTP_STATUS_MIN_SUCCESS,
    MILLISECONDS_PER_SECOND,
)


class HttpConnector(Connector):
    """Domain-agnostic HTTP/REST connector for arbitrary software services."""

    def __init__(
        self,
        base_url: str,
        system_id: str = "http_service",
        telemetry_endpoint: str = "/metrics",
        actions_endpoint: str = "/actions",
        health_endpoint: Optional[str] = "/health",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_CONNECTOR_TIMEOUT,
        supported_actions: Optional[List[str]] = None,
        action_schemas: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize the generic HTTP connector.

        Args:
            base_url: Base URL of the target service (e.g. 'http://localhost:8080')
            system_id: Unique system identifier
            telemetry_endpoint: Path or URL to retrieve telemetry snapshot
            actions_endpoint: Path or URL to submit adaptation actions
            health_endpoint: Optional health probe path (e.g. '/health' or '/live')
            headers: Optional HTTP headers (authentication, tokens, etc.)
            timeout: Request timeout in seconds
            supported_actions: List of supported action type strings
            action_schemas: Optional mapping of action names to ActionSchemas
            dependencies: Optional list of downstream dependency system IDs
            logger: Optional structured logger
            metrics: Optional metrics collector
        """
        self.base_url = base_url.rstrip("/")
        self.system_id = system_id
        self.telemetry_endpoint = telemetry_endpoint
        self.actions_endpoint = actions_endpoint
        self.health_endpoint = health_endpoint
        self.headers = headers or {}
        self.timeout = float(timeout)
        self.supported_actions = list(supported_actions or [])
        self._action_schemas = dict(action_schemas or {})
        self._dependencies = list(dependencies or [])
        self.logger = logger
        self.metrics = metrics

        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
            )
        return self._client

    async def connect(self) -> bool:
        """Probe the target service health or base endpoint."""
        client = self._get_client()
        probe_path = self.health_endpoint or self.telemetry_endpoint or "/"

        try:
            response = await client.get(probe_path)
            # Accept any responsive status code (2xx, 3xx, 401, 403, 404 all indicate host is reachable)
            self._connected = response.status_code < 500
            if self.logger:
                self.logger.info(
                    "HttpConnector connected",
                    system_id=self.system_id,
                    base_url=self.base_url,
                    probe_status=response.status_code,
                )
            return self._connected
        except Exception as exc:
            self._connected = False
            if self.logger:
                self.logger.warning(
                    "HttpConnector connection failed",
                    system_id=self.system_id,
                    base_url=self.base_url,
                    error=str(exc),
                )
            return False

    async def disconnect(self) -> bool:
        """Close HTTP client connection."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._connected = False
        return True

    async def get_system_id(self) -> str:
        """Return system identifier."""
        return self.system_id

    def _flatten_metrics(
        self, data: Dict[str, Any], prefix: str = "", now: Optional[datetime] = None
    ) -> Dict[str, MetricValue]:
        """Recursively flatten nested dictionary into dot-separated MetricValues."""
        now_dt = now or datetime.now(timezone.utc)
        result: Dict[str, MetricValue] = {}

        for k, v in data.items():
            metric_key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                result[metric_key] = MetricValue(
                    name=metric_key,
                    value=float(v),
                    timestamp=now_dt,
                )
            elif isinstance(v, dict):
                result.update(self._flatten_metrics(v, prefix=metric_key, now=now_dt))

        return result

    def _parse_prometheus_text(self, text: str, now: datetime) -> Dict[str, MetricValue]:
        """Parse Prometheus exposition text format into MetricValue objects."""
        metrics: Dict[str, MetricValue] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Match: metric_name{labels} value or metric_name value
            match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)", line)
            if match:
                name, _labels, val_str = match.groups()
                try:
                    val = float(val_str)
                    metrics[name] = MetricValue(
                        name=name,
                        value=val,
                        timestamp=now,
                    )
                except ValueError:
                    continue
        return metrics

    async def collect_telemetry(self) -> SystemState:
        """Collect telemetry from configured endpoint (JSON or Prometheus format)."""
        client = self._get_client()
        now = datetime.now(timezone.utc)
        endpoint = self.telemetry_endpoint

        try:
            response = await client.get(endpoint)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP error {response.status_code}: {response.text}")

            content_type = response.headers.get("content-type", "")
            metrics: Dict[str, MetricValue] = {}
            health = HealthStatus.HEALTHY

            if "json" in content_type:
                payload = response.json()
                if isinstance(payload, dict):
                    # Check for explicit health status field
                    raw_health = str(payload.get("health") or payload.get("status") or "").lower()
                    if raw_health in ("critical", "unhealthy", "down", "error"):
                        health = HealthStatus.CRITICAL
                    elif raw_health in ("warning", "degraded", "warn"):
                        health = HealthStatus.WARNING
                    elif raw_health in ("healthy", "ok", "up"):
                        health = HealthStatus.HEALTHY

                    # Flatten metrics
                    raw_metrics = payload.get("metrics")
                    metrics_dict: Dict[str, Any] = (
                        raw_metrics if isinstance(raw_metrics, dict) else payload
                    )
                    metrics = self._flatten_metrics(metrics_dict, now=now)
            else:
                # Fallback to Prometheus line format
                metrics = self._parse_prometheus_text(response.text, now=now)

            return SystemState(
                system_id=self.system_id,
                health_status=health,
                metrics=metrics,
                timestamp=now,
            )

        except Exception as exc:
            if self.logger:
                self.logger.error(
                    "HttpConnector telemetry collection failed",
                    system_id=self.system_id,
                    endpoint=endpoint,
                    error=str(exc),
                )
            return SystemState(
                system_id=self.system_id,
                health_status=HealthStatus.CRITICAL,
                metrics={},
                timestamp=now,
            )

    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute action via HTTP POST mutation."""
        client = self._get_client()
        start_time = time.time()
        endpoint = self.actions_endpoint

        payload = {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "parameters": action.parameters or {},
            "target_system": self.system_id,
        }

        try:
            response = await client.post(endpoint, json=payload)
            elapsed_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND

            if HTTP_STATUS_MIN_SUCCESS <= response.status_code <= HTTP_STATUS_MAX_SUCCESS:
                res_data = {}
                try:
                    res_data = response.json()
                except Exception:
                    res_data = {"text": response.text}

                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.SUCCESS,
                    result_data=res_data,
                    execution_time_ms=int(elapsed_ms),
                    completed_at=datetime.now(timezone.utc),
                )
            else:
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    result_data={},
                    error_message=f"HTTP {response.status_code}: {response.text}",
                    execution_time_ms=int(elapsed_ms),
                    completed_at=datetime.now(timezone.utc),
                )

        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=str(exc),
                execution_time_ms=int(elapsed_ms),
                completed_at=datetime.now(timezone.utc),
            )

    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate whether action type is supported."""
        if not self.supported_actions:
            return True
        return action.action_type in self.supported_actions

    async def get_supported_actions(self) -> List[AdaptationAction]:
        """Return list of supported action templates."""
        return [
            AdaptationAction(
                action_id="",
                target_system=self.system_id,
                action_type=act,
            )
            for act in self.supported_actions
        ]

    async def get_capabilities(self) -> ConnectorCapabilities:
        """Return ConnectorCapabilities including schemas and dependencies."""
        return ConnectorCapabilities.from_supported_action_types(
            action_types=self.supported_actions,
            actions=self._action_schemas,
            dependencies=self._dependencies,
        )
