"""Tests for OpenTelemetry (OTel) Metric Parser and Ingestion Receiver."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from polaris.core.models import HealthStatus, SystemState
from polaris.core.polaris import Polaris, PolarisConfig
from polaris.infrastructure.otel_receiver import (
    OtelMetricParser,
    OtelReceiverConfig,
    OtelTelemetryReceiver,
)

SAMPLE_OTLP_PAYLOAD = {
    "resourceMetrics": [
        {
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "cart_service"}},
                    {"key": "deployment.environment", "value": {"stringValue": "production"}},
                ]
            },
            "scopeMetrics": [
                {
                    "metrics": [
                        {
                            "name": "cart.active_sessions",
                            "unit": "sessions",
                            "gauge": {
                                "dataPoints": [
                                    {
                                        "asInt": 128,
                                        "timeUnixNano": "1710000000000000000",
                                        "attributes": [
                                            {"key": "tenant", "value": {"stringValue": "acme"}}
                                        ],
                                    }
                                ]
                            },
                        },
                        {
                            "name": "cart.checkout_latency",
                            "unit": "ms",
                            "histogram": {
                                "dataPoints": [
                                    {
                                        "count": 50,
                                        "sum": 2500.0,
                                        "min": 10.0,
                                        "max": 150.0,
                                        "timeUnixNano": "1710000000000000000",
                                    }
                                ]
                            },
                        },
                        {
                            "name": "cart.orders_completed",
                            "unit": "orders",
                            "sum": {
                                "dataPoints": [
                                    {
                                        "asDouble": 340.0,
                                        "timeUnixNano": "1710000000000000000",
                                    }
                                ]
                            },
                        },
                    ]
                }
            ],
        }
    ]
}


def test_otel_metric_parser_extracts_gauge_histogram_sum():
    """Verify OTLP parser extracts gauge, sum, and histogram metric models."""
    states = OtelMetricParser.parse_otlp_payload(SAMPLE_OTLP_PAYLOAD)
    assert len(states) == 1

    state = states[0]
    assert state.system_id == "cart_service"
    assert state.health_status == HealthStatus.HEALTHY

    # Gauge check
    assert "cart.active_sessions" in state.metrics
    gauge_metric = state.metrics["cart.active_sessions"]
    assert gauge_metric.value == 128.0
    assert gauge_metric.unit == "sessions"
    assert gauge_metric.tags["service.name"] == "cart_service"
    assert gauge_metric.tags["tenant"] == "acme"

    # Histogram checks: count, sum, avg, min, max
    assert "cart.checkout_latency.count" in state.metrics
    assert state.metrics["cart.checkout_latency.count"].value == 50.0
    assert "cart.checkout_latency.sum" in state.metrics
    assert state.metrics["cart.checkout_latency.sum"].value == 2500.0
    assert "cart.checkout_latency.avg" in state.metrics
    assert state.metrics["cart.checkout_latency.avg"].value == 50.0
    assert "cart.checkout_latency.min" in state.metrics
    assert state.metrics["cart.checkout_latency.min"].value == 10.0
    assert "cart.checkout_latency.max" in state.metrics
    assert state.metrics["cart.checkout_latency.max"].value == 150.0

    # Sum check
    assert "cart.orders_completed" in state.metrics
    assert state.metrics["cart.orders_completed"].value == 340.0


def test_otel_metric_parser_health_status_detection():
    """Verify system.health indicator adjusts health status to CRITICAL."""
    payload = {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "db_service"}}]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": "system.health",
                                "gauge": {
                                    "dataPoints": [{"asDouble": 2.0}]  # >= 2.0 indicates CRITICAL
                                },
                            }
                        ]
                    }
                ],
            }
        ]
    }
    states = OtelMetricParser.parse_otlp_payload(payload)
    assert len(states) == 1
    assert states[0].health_status == HealthStatus.CRITICAL


@pytest.mark.asyncio
async def test_otel_receiver_ingest_payload_callback():
    """Verify OtelTelemetryReceiver invokes on_state_received callback."""
    received = []

    async def on_state(state: SystemState) -> bool:
        received.append(state)
        return True

    receiver = OtelTelemetryReceiver(on_state_received=on_state)
    count = await receiver.ingest_payload(SAMPLE_OTLP_PAYLOAD)

    assert count == 1
    assert len(received) == 1
    assert received[0].system_id == "cart_service"


@pytest.mark.asyncio
async def test_otel_receiver_live_http_server():
    """Verify live HTTP server receives OTLP POST and GET health probes."""
    received = []

    async def on_state(state: SystemState) -> bool:
        received.append(state)
        return True

    # Use port 49182 for isolated test
    config = OtelReceiverConfig(
        enabled=True,
        host="127.0.0.1",
        port=49182,
        endpoint="/v1/metrics",
    )
    receiver = OtelTelemetryReceiver(config=config, on_state_received=on_state)

    await receiver.start()
    assert receiver.is_running is True

    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:49182") as client:
            # 1. Health probe
            health_resp = await client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json() == {"status": "healthy"}

            # 2. Post OTLP payload
            metrics_resp = await client.post("/v1/metrics", json=SAMPLE_OTLP_PAYLOAD)
            assert metrics_resp.status_code == 200
            assert metrics_resp.json() == {"status": "ok"}
            assert len(received) == 1
            assert received[0].system_id == "cart_service"

    finally:
        await receiver.stop()
        assert receiver.is_running is False


def test_polaris_config_instantiates_otel_receiver():
    """Verify Polaris configuration properly instantiates OtelTelemetryReceiver."""
    cfg = PolarisConfig(
        otel={
            "enabled": True,
            "host": "127.0.0.1",
            "port": 4318,
            "endpoint": "/v1/metrics",
            "default_system_id": "k8s_cluster",
        }
    )
    polaris = Polaris(config=cfg)
    assert polaris.otel_receiver is not None
    assert polaris.otel_receiver.config.enabled is True
    assert polaris.otel_receiver.config.port == 4318
    assert polaris.otel_receiver.config.default_system_id == "k8s_cluster"
