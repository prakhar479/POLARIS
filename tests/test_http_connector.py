"""Tests for generic HttpConnector and Prometheus scraping."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from polaris.abstractions.connector_capabilities import ConnectorCapabilities
from polaris.connectors.http_connector import HttpConnector
from polaris.core.models import (
    AdaptationAction,
    ExecutionStatus,
    HealthStatus,
)


@pytest.mark.asyncio
async def test_http_connector_initialization():
    connector = HttpConnector(
        base_url="https://api.example.com",
        system_id="payment_svc",
        telemetry_endpoint="/v1/metrics",
        actions_endpoint="/v1/actions",
        supported_actions=["throttle", "scale"],
        dependencies=["db_svc"],
    )

    assert await connector.get_system_id() == "payment_svc"
    assert connector.base_url == "https://api.example.com"
    caps = await connector.get_capabilities()
    assert isinstance(caps, ConnectorCapabilities)
    assert set(caps.supported_action_types) == {"throttle", "scale"}
    assert caps.dependencies == ("db_svc",)


@pytest.mark.asyncio
async def test_http_connector_connect_success():
    connector = HttpConnector(base_url="http://testserver")
    mock_resp = httpx.Response(200, json={"status": "healthy"})

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.is_closed = False
    connector._client = mock_client

    connected = await connector.connect()
    assert connected is True


@pytest.mark.asyncio
async def test_http_connector_connect_failure():
    connector = HttpConnector(base_url="http://testserver")

    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    mock_client.is_closed = False
    connector._client = mock_client

    connected = await connector.connect()
    assert connected is False


@pytest.mark.asyncio
async def test_http_connector_collect_telemetry_json():
    connector = HttpConnector(base_url="http://testserver", system_id="web")

    json_data = {
        "health": "warning",
        "cpu_usage": 88.5,
        "memory": {
            "used_mb": 2048.0,
            "pct": 75.0,
        },
        "requests_per_sec": 450,
    }

    mock_resp = httpx.Response(
        200,
        json=json_data,
        headers={"content-type": "application/json"},
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.is_closed = False
    connector._client = mock_client

    state = await connector.collect_telemetry()

    assert state.system_id == "web"
    assert state.health_status == HealthStatus.WARNING
    assert state.metrics["cpu_usage"].value == 88.5
    assert state.metrics["memory.used_mb"].value == 2048.0
    assert state.metrics["memory.pct"].value == 75.0
    assert state.metrics["requests_per_sec"].value == 450.0


@pytest.mark.asyncio
async def test_http_connector_collect_telemetry_prometheus_format():
    connector = HttpConnector(base_url="http://testserver", system_id="prom_svc")

    prom_text = """
    # HELP http_requests_total Total number of HTTP requests
    # TYPE http_requests_total counter
    http_requests_total 12345.0
    # HELP node_cpu_seconds_total Seconds spent in cpu modes
    node_cpu_utilization{cpu="0"} 85.5
    """

    mock_resp = httpx.Response(
        200,
        text=prom_text,
        headers={"content-type": "text/plain; version=0.0.4"},
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.is_closed = False
    connector._client = mock_client

    state = await connector.collect_telemetry()

    assert state.system_id == "prom_svc"
    assert "http_requests_total" in state.metrics
    assert state.metrics["http_requests_total"].value == 12345.0
    assert "node_cpu_utilization" in state.metrics
    assert state.metrics["node_cpu_utilization"].value == 85.5


@pytest.mark.asyncio
async def test_http_connector_execute_action_success():
    connector = HttpConnector(
        base_url="http://testserver",
        actions_endpoint="/api/v1/adapt",
        supported_actions=["scale_up"],
    )

    action = AdaptationAction(
        action_id="act_123",
        target_system="http_service",
        action_type="scale_up",
        parameters={"replicas": 5},
    )

    mock_resp = httpx.Response(200, json={"result": "scaled", "new_replicas": 5})

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.is_closed = False
    connector._client = mock_client

    result = await connector.execute_action(action)

    assert result.action_id == "act_123"
    assert result.status == ExecutionStatus.SUCCESS
    assert result.result_data == {"result": "scaled", "new_replicas": 5}
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["action_type"] == "scale_up"
    assert call_kwargs["json"]["parameters"] == {"replicas": 5}


@pytest.mark.asyncio
async def test_http_connector_execute_action_failure():
    connector = HttpConnector(
        base_url="http://testserver",
        actions_endpoint="/api/v1/adapt",
    )

    action = AdaptationAction(
        action_id="act_999",
        target_system="http_service",
        action_type="invalid_action",
    )

    mock_resp = httpx.Response(500, text="Internal Server Error")

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.is_closed = False
    connector._client = mock_client

    result = await connector.execute_action(action)

    assert result.action_id == "act_999"
    assert result.status == ExecutionStatus.FAILED
    assert "500" in (result.error_message or "")


@pytest.mark.asyncio
async def test_component_builder_builds_http_connector():
    from unittest.mock import MagicMock

    from polaris.core.component_builder import ComponentBuilder
    from polaris.infrastructure.config import SystemConfig

    sys_cfg = SystemConfig(
        id="order_api",
        connector_type="http",
        connection={
            "base_url": "http://127.0.0.1:8000",
            "telemetry_endpoint": "/status",
            "actions_endpoint": "/adapt",
            "headers": {"X-Token": "secret"},
        },
    )

    mock_cfg = MagicMock()
    mock_cfg.observability.metrics.enabled = False
    connectors = ComponentBuilder.build_connectors(
        systems_config=[sys_cfg],
        logger=MagicMock(),
        metrics=None,
        config=mock_cfg,
    )

    assert len(connectors) == 1
    assert isinstance(connectors[0], HttpConnector)
    assert await connectors[0].get_system_id() == "order_api"
    assert connectors[0].base_url == "http://127.0.0.1:8000"
    assert connectors[0].telemetry_endpoint == "/status"
