"""OpenTelemetry (OTel) metrics ingestion receiver for POLARIS.

Provides automated ingestion of OTLP JSON / dictionary telemetry payloads exported
by OpenTelemetry Collectors, SDKs, and agents, converting them into native POLARIS
SystemState snapshots and streaming them directly into the monitoring loop.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.core.models import HealthStatus, MetricValue, SystemState


@dataclass
class OtelReceiverConfig:
    """Configuration for OpenTelemetry telemetry receiver."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 4318
    endpoint: str = "/v1/metrics"
    default_system_id: str = "otel_service"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OtelReceiverConfig":
        """Build OtelReceiverConfig from dictionary with validation."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            host=str(data.get("host", "127.0.0.1")),
            port=int(data.get("port", 4318)),
            endpoint=str(data.get("endpoint", "/v1/metrics")),
            default_system_id=str(data.get("default_system_id", "otel_service")),
        )


class OtelMetricParser:
    """Parser transforming OpenTelemetry OTLP metric structures into POLARIS models."""

    @staticmethod
    def _extract_any_value(val_dict: Any) -> Any:
        """Extract Python primitive from OTLP AnyValue representation."""
        if not isinstance(val_dict, dict):
            return val_dict

        if "stringValue" in val_dict:
            return val_dict["stringValue"]
        if "intValue" in val_dict:
            return int(val_dict["intValue"])
        if "doubleValue" in val_dict:
            return float(val_dict["doubleValue"])
        if "boolValue" in val_dict:
            return bool(val_dict["boolValue"])
        if "arrayValue" in val_dict:
            vals = val_dict["arrayValue"].get("values", [])
            return [OtelMetricParser._extract_any_value(v) for v in vals]
        if "kvlistValue" in val_dict:
            kvs = val_dict["kvlistValue"].get("values", [])
            return {
                kv.get("key", ""): OtelMetricParser._extract_any_value(kv.get("value", {}))
                for kv in kvs
            }
        # Fallback to first dict value if present
        if val_dict:
            return next(iter(val_dict.values()))
        return None

    @classmethod
    def _extract_attributes(cls, attr_list: Any) -> Dict[str, str]:
        """Convert OTLP attribute array into string key-value dictionary."""
        attributes: Dict[str, str] = {}
        if isinstance(attr_list, list):
            for item in attr_list:
                if isinstance(item, dict) and "key" in item:
                    k = str(item["key"])
                    raw_v = cls._extract_any_value(item.get("value", ""))
                    attributes[k] = str(raw_v)
        elif isinstance(attr_list, dict):
            for k, v in attr_list.items():
                attributes[str(k)] = str(cls._extract_any_value(v))
        return attributes

    @staticmethod
    def _timestamp_from_nanos(nanos: Any) -> datetime:
        """Convert Unix nanosecond timestamp to UTC datetime."""
        if nanos is not None:
            try:
                ns = int(nanos)
                if ns > 0:
                    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
            except (ValueError, TypeError):
                pass
        return datetime.now(timezone.utc)

    @classmethod
    def parse_otlp_payload(
        cls, payload: Dict[str, Any], default_system_id: str = "otel_service"
    ) -> List[SystemState]:
        """Parse an OTLP ExportMetricsServiceRequest dictionary into SystemStates."""
        states: List[SystemState] = []
        resource_metrics = payload.get("resourceMetrics", [])
        if not isinstance(resource_metrics, list):
            return states

        now = datetime.now(timezone.utc)

        for res_metric in resource_metrics:
            if not isinstance(res_metric, dict):
                continue

            # Extract resource attributes
            resource = res_metric.get("resource", {})
            res_attrs = cls._extract_attributes(resource.get("attributes", []))

            # Determine target system ID: prefer service.name, k8s.pod.name, host.name
            system_id = (
                res_attrs.get("service.name")
                or res_attrs.get("k8s.pod.name")
                or res_attrs.get("host.name")
                or default_system_id
            )

            metrics_by_name: Dict[str, MetricValue] = {}
            latest_ts = now
            health = HealthStatus.HEALTHY

            # Scope metrics (OpenAPI/OTLP v0.19+) or instrumentationLibraryMetrics (older)
            scope_metrics = res_metric.get("scopeMetrics") or res_metric.get(
                "instrumentationLibraryMetrics", []
            )
            if not isinstance(scope_metrics, list):
                continue

            for s_metric in scope_metrics:
                if not isinstance(s_metric, dict):
                    continue

                metric_list = s_metric.get("metrics", [])
                if not isinstance(metric_list, list):
                    continue

                for metric in metric_list:
                    if not isinstance(metric, dict):
                        continue

                    metric_name = metric.get("name")
                    if not metric_name:
                        continue

                    unit = metric.get("unit")

                    # 1. Gauge
                    if "gauge" in metric:
                        data_points = metric["gauge"].get("dataPoints", [])
                        for dp in data_points:
                            val = dp.get("asDouble")
                            if val is None:
                                val = dp.get("asInt")
                            if val is not None:
                                ts = cls._timestamp_from_nanos(
                                    dp.get("timeUnixNano") or dp.get("startTimeUnixNano")
                                )
                                latest_ts = max(latest_ts, ts)
                                tags = {
                                    **res_attrs,
                                    **cls._extract_attributes(dp.get("attributes")),
                                }
                                metrics_by_name[metric_name] = MetricValue(
                                    name=metric_name,
                                    value=float(val),
                                    unit=unit,
                                    tags=tags,
                                    timestamp=ts,
                                )

                    # 2. Sum / Counter
                    elif "sum" in metric:
                        data_points = metric["sum"].get("dataPoints", [])
                        for dp in data_points:
                            val = dp.get("asDouble")
                            if val is None:
                                val = dp.get("asInt")
                            if val is not None:
                                ts = cls._timestamp_from_nanos(
                                    dp.get("timeUnixNano") or dp.get("startTimeUnixNano")
                                )
                                latest_ts = max(latest_ts, ts)
                                tags = {
                                    **res_attrs,
                                    **cls._extract_attributes(dp.get("attributes")),
                                }
                                metrics_by_name[metric_name] = MetricValue(
                                    name=metric_name,
                                    value=float(val),
                                    unit=unit,
                                    tags=tags,
                                    timestamp=ts,
                                )

                    # 3. Histogram
                    elif "histogram" in metric:
                        data_points = metric["histogram"].get("dataPoints", [])
                        for dp in data_points:
                            ts = cls._timestamp_from_nanos(
                                dp.get("timeUnixNano") or dp.get("startTimeUnixNano")
                            )
                            latest_ts = max(latest_ts, ts)
                            tags = {**res_attrs, **cls._extract_attributes(dp.get("attributes"))}

                            count = dp.get("count")
                            h_sum = dp.get("sum")

                            if count is not None:
                                c_name = f"{metric_name}.count"
                                metrics_by_name[c_name] = MetricValue(
                                    name=c_name,
                                    value=float(count),
                                    unit="count",
                                    tags=tags,
                                    timestamp=ts,
                                )
                            if h_sum is not None:
                                s_name = f"{metric_name}.sum"
                                metrics_by_name[s_name] = MetricValue(
                                    name=s_name,
                                    value=float(h_sum),
                                    unit=unit,
                                    tags=tags,
                                    timestamp=ts,
                                )
                            if count and h_sum is not None and float(count) > 0:
                                avg_name = f"{metric_name}.avg"
                                metrics_by_name[avg_name] = MetricValue(
                                    name=avg_name,
                                    value=float(h_sum) / float(count),
                                    unit=unit,
                                    tags=tags,
                                    timestamp=ts,
                                )
                            if "min" in dp and dp["min"] is not None:
                                m_name = f"{metric_name}.min"
                                metrics_by_name[m_name] = MetricValue(
                                    name=m_name,
                                    value=float(dp["min"]),
                                    unit=unit,
                                    tags=tags,
                                    timestamp=ts,
                                )
                            if "max" in dp and dp["max"] is not None:
                                mx_name = f"{metric_name}.max"
                                metrics_by_name[mx_name] = MetricValue(
                                    name=mx_name,
                                    value=float(dp["max"]),
                                    unit=unit,
                                    tags=tags,
                                    timestamp=ts,
                                )

                    # 4. Summary
                    elif "summary" in metric:
                        data_points = metric["summary"].get("dataPoints", [])
                        for dp in data_points:
                            ts = cls._timestamp_from_nanos(
                                dp.get("timeUnixNano") or dp.get("startTimeUnixNano")
                            )
                            latest_ts = max(latest_ts, ts)
                            tags = {**res_attrs, **cls._extract_attributes(dp.get("attributes"))}
                            count = dp.get("count")
                            s_sum = dp.get("sum")
                            if count is not None:
                                metrics_by_name[f"{metric_name}.count"] = MetricValue(
                                    name=f"{metric_name}.count",
                                    value=float(count),
                                    unit="count",
                                    tags=tags,
                                    timestamp=ts,
                                )
                            if s_sum is not None:
                                metrics_by_name[f"{metric_name}.sum"] = MetricValue(
                                    name=f"{metric_name}.sum",
                                    value=float(s_sum),
                                    unit=unit,
                                    tags=tags,
                                    timestamp=ts,
                                )
                            if count and s_sum is not None and float(count) > 0:
                                avg_name = f"{metric_name}.avg"
                                metrics_by_name[avg_name] = MetricValue(
                                    name=avg_name,
                                    value=float(s_sum) / float(count),
                                    unit=unit,
                                    tags=tags,
                                    timestamp=ts,
                                )
                            for qv in dp.get("quantileValues", []):
                                q_val = qv.get("quantile")
                                q_num = qv.get("value")
                                if q_val is not None and q_num is not None:
                                    q_pct = int(round(float(q_val) * 100))
                                    q_name = f"{metric_name}.p{q_pct}"
                                    metrics_by_name[q_name] = MetricValue(
                                        name=q_name,
                                        value=float(q_num),
                                        unit=unit,
                                        tags=tags,
                                        timestamp=ts,
                                    )

            # Check health status indicator metrics if present
            if "system.health" in metrics_by_name:
                h_val = metrics_by_name["system.health"].value
                if isinstance(h_val, (int, float)) and not isinstance(h_val, bool):
                    if h_val >= 2.0:
                        health = HealthStatus.CRITICAL
                    elif h_val >= 1.0:
                        health = HealthStatus.WARNING

            if metrics_by_name:
                states.append(
                    SystemState(
                        system_id=system_id,
                        health_status=health,
                        metrics=metrics_by_name,
                        timestamp=latest_ts,
                    )
                )

        return states


class OtelTelemetryReceiver:
    """Asynchronous HTTP/OTLP telemetry receiver for ingestion into POLARIS."""

    def __init__(
        self,
        config: Optional[OtelReceiverConfig] = None,
        on_state_received: Optional[Callable[[SystemState], Awaitable[bool]]] = None,
        parser: Optional[OtelMetricParser] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize the OpenTelemetry receiver.

        Args:
            config: Receiver configuration (host, port, endpoint)
            on_state_received: Async callback when a SystemState is parsed
            parser: OTel parser instance
            logger: Optional structured logger
            metrics: Optional metrics collector
        """
        self.config = config or OtelReceiverConfig()
        self.on_state_received = on_state_received
        self.parser = parser or OtelMetricParser()
        self.logger = logger
        self.metrics = metrics

        self._server: Optional[asyncio.Server] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Return True if the receiver server is running."""
        return self._is_running

    async def ingest_payload(self, payload: Dict[str, Any]) -> int:
        """Parse OTLP payload and forward each SystemState to the registered callback."""
        states = self.parser.parse_otlp_payload(
            payload, default_system_id=self.config.default_system_id
        )
        ingested = 0
        for state in states:
            if self.on_state_received:
                success = await self.on_state_received(state)
                if success:
                    ingested += 1
            else:
                ingested += 1

        if self.metrics:
            self.metrics.increment(
                "polaris.otel.ingested_states_total",
                value=float(ingested),
                tags={"endpoint": self.config.endpoint},
            )

        return ingested

    async def ingest_json(self, json_str: str) -> int:
        """Parse raw JSON OTLP payload string and ingest."""
        data = json.loads(json_str)
        return await self.ingest_payload(data)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming HTTP connection for OTLP push requests."""
        try:
            # Read HTTP request line
            line = await reader.readline()
            if not line:
                writer.close()
                await writer.wait_closed()
                return

            req_line = line.decode("utf-8", errors="replace").strip()
            parts = req_line.split(" ")
            if len(parts) < 2:
                writer.close()
                await writer.wait_closed()
                return

            method, path = parts[0].upper(), parts[1]

            # Read headers
            headers: Dict[str, str] = {}
            while True:
                hdr_line = await reader.readline()
                if not hdr_line or hdr_line == b"\r\n" or hdr_line == b"\n":
                    break
                hdr_str = hdr_line.decode("utf-8", errors="replace").strip()
                if ":" in hdr_str:
                    k, v = hdr_str.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            content_length = int(headers.get("content-length", 0))

            if method == "POST" and (path == self.config.endpoint or path == "/v1/metrics"):
                body_bytes = await reader.readexactly(content_length) if content_length > 0 else b""
                try:
                    payload = json.loads(body_bytes.decode("utf-8"))
                    await self.ingest_payload(payload)
                    res_body = b'{"status":"ok"}'
                    resp = (
                        f"HTTP/1.1 200 OK\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: {len(res_body)}\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode("utf-8") + res_body
                except Exception as exc:
                    err_body = json.dumps({"error": str(exc)}).encode("utf-8")
                    resp = (
                        f"HTTP/1.1 400 Bad Request\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: {len(err_body)}\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode("utf-8") + err_body

            elif method == "GET" and path in ("/health", "/live", "/"):
                health_body = b'{"status":"healthy"}'
                resp = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(health_body)}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode("utf-8") + health_body
            else:
                resp = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"

            writer.write(resp)
            await writer.drain()

        except Exception as exc:
            if self.logger:
                self.logger.warning("Error processing OTel client request", error=str(exc))
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self) -> None:
        """Start the asynchronous HTTP server for incoming OTel metric pushes."""
        if self._is_running:
            return

        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.config.host,
            port=self.config.port,
        )
        self._is_running = True
        if self.logger:
            self.logger.info(
                "OtelTelemetryReceiver listening",
                host=self.config.host,
                port=self.config.port,
                endpoint=self.config.endpoint,
            )

    async def stop(self) -> None:
        """Stop the receiver HTTP server and clean up connections."""
        if not self._is_running or self._server is None:
            return

        self._server.close()
        await self._server.wait_closed()
        self._server = None
        self._is_running = False
        if self.logger:
            self.logger.info("OtelTelemetryReceiver stopped")
