"""Tests for OpenAPI / Swagger Action Synthesizer and dynamic HTTP connector routing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from polaris.abstractions.system_contract import SystemContract
from polaris.connectors.http_connector import HttpConnector
from polaris.core.models import AdaptationAction, ExecutionStatus
from polaris.infrastructure.openapi_synthesizer import (
    HttpActionEndpoint,
    OpenApiSynthesizer,
    SynthesizedApi,
)

SAMPLE_OPENAPI_V3 = {
    "openapi": "3.0.0",
    "info": {
        "title": "Payment Gateway API",
        "version": "2.4.0",
    },
    "servers": [{"url": "http://payment.internal.net:8080"}],
    "paths": {
        "/services/payment/scale": {
            "post": {
                "operationId": "scalePaymentService",
                "summary": "Scale the payment service worker pool",
                "x-polaris-performance-impact": "positive",
                "x-polaris-cost-impact": "negative",
                "x-polaris-rollback-action": "scale_down",
                "x-polaris-verification-window-seconds": 45.0,
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["replicas"],
                                "properties": {
                                    "replicas": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 20,
                                        "description": "Target number of active worker pods",
                                    }
                                },
                            }
                        }
                    },
                },
            }
        },
        "/rate-limit/{tier}": {
            "put": {
                "operationId": "update_rate_limit",
                "summary": "Update rate limit tier quota",
                "parameters": [
                    {
                        "name": "tier",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "requests_per_minute",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "integer", "minimum": 10},
                    },
                ],
            }
        },
        "/cache/flush": {
            "post": {
                "summary": "Flush in-memory payment cache",
                "description": "Invalidates all active cache entries",
            }
        },
        "/metrics": {
            "get": {
                "summary": "Prometheus metrics",
                "responses": {"200": {"description": "Prometheus text"}},
            }
        },
    },
}

SAMPLE_SWAGGER_V2 = {
    "swagger": "2.0",
    "info": {"title": "Legacy Gateway", "version": "1.0.0"},
    "host": "legacy.api.com",
    "basePath": "/v1",
    "schemes": ["https"],
    "paths": {
        "/throttle": {
            "post": {
                "operationId": "throttle_traffic",
                "summary": "Throttle incoming client requests",
                "parameters": [
                    {
                        "name": "body",
                        "in": "body",
                        "required": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "rate": {"type": "number", "minimum": 0.1, "maximum": 1.0}
                            },
                            "required": ["rate"],
                        },
                    }
                ],
            }
        }
    },
}

SAMPLE_REF_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Ref API", "version": "1.0"},
    "paths": {
        "/database/pool": {
            "post": {
                "operationId": "resize_db_pool",
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/PoolConfig"}}
                    }
                },
            }
        }
    },
    "components": {
        "schemas": {
            "PoolConfig": {
                "type": "object",
                "required": ["max_connections"],
                "properties": {
                    "max_connections": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 500,
                        "description": "Max pooled DB connections",
                    }
                },
            }
        }
    },
}


def test_openapi_v3_synthesis_and_validation():
    """Verify OpenAPI v3 parsing, action schemas, and contract generation."""
    synth = OpenApiSynthesizer.synthesize_from_dict(
        SAMPLE_OPENAPI_V3, system_id="payment_svc", mutation_only=True
    )
    assert synth.system_id == "payment_svc"
    assert synth.title == "Payment Gateway API"
    assert synth.version == "2.4.0"
    assert synth.base_url == "http://payment.internal.net:8080"

    # Only mutation endpoints synthesized (POST, PUT, not GET /metrics)
    assert "scale_payment_service" in synth.action_schemas
    assert "update_rate_limit" in synth.action_schemas
    assert "post_cache_flush" in synth.action_schemas
    assert "get_metrics" not in synth.action_schemas

    # Verify action schema properties and bounds
    scale_schema = synth.action_schemas["scale_payment_service"]
    assert scale_schema.performance_impact == "positive"
    assert scale_schema.cost_impact == "negative"
    assert scale_schema.rollback_action == "scale_down"
    assert scale_schema.default_verification_window_seconds == 45.0
    assert "replicas" in scale_schema.parameters_schema
    assert scale_schema.parameters_schema["replicas"]["type"] == "integer"
    assert scale_schema.parameters_schema["replicas"]["minimum"] == 1
    assert scale_schema.parameters_schema["replicas"]["maximum"] == 20

    # Validation succeeds on valid params
    is_valid, err = scale_schema.validate_parameters({"replicas": 10})
    assert is_valid is True
    assert err is None

    # Validation fails on out of bounds
    is_valid, err = scale_schema.validate_parameters({"replicas": 50})
    assert is_valid is False
    assert "must be <= 20" in err

    # Validation fails on missing required param
    is_valid, err = scale_schema.validate_parameters({})
    assert is_valid is False
    assert "Missing required parameter" in err

    # Contract conversion
    contract = synth.to_system_contract()
    assert isinstance(contract, SystemContract)
    assert contract.system_id == "payment_svc"
    assert "scale_payment_service" in contract.supported_action_types


def test_swagger_v2_synthesis():
    """Verify Swagger 2.0 parsing and base_url derivation."""
    synth = OpenApiSynthesizer.synthesize_from_dict(
        SAMPLE_SWAGGER_V2, system_id="legacy_gw", mutation_only=True
    )
    assert synth.base_url == "https://legacy.api.com/v1"
    assert "throttle_traffic" in synth.action_schemas
    schema = synth.action_schemas["throttle_traffic"]
    assert "rate" in schema.parameters_schema
    assert schema.parameters_schema["rate"]["type"] == "number"
    assert schema.parameters_schema["rate"]["minimum"] == 0.1

    # Naming heuristic gives negative QoS impact for throttle
    assert schema.qos_impact == "negative"


def test_ref_schema_resolution():
    """Verify local $ref resolution for components/schemas."""
    synth = OpenApiSynthesizer.synthesize_from_dict(SAMPLE_REF_SPEC, system_id="db_sys")
    assert "resize_db_pool" in synth.action_schemas
    schema = synth.action_schemas["resize_db_pool"]
    assert "max_connections" in schema.parameters_schema
    assert schema.parameters_schema["max_connections"]["minimum"] == 5
    assert schema.parameters_schema["max_connections"]["maximum"] == 500
    assert "max_connections" in schema.required_parameters


def test_synthesize_from_json_and_yaml():
    """Verify JSON and YAML serialization input parsers."""
    json_str = json.dumps(SAMPLE_OPENAPI_V3)
    synth_json = OpenApiSynthesizer.synthesize_from_json(json_str, system_id="json_sys")
    assert "scale_payment_service" in synth_json.action_schemas

    yaml_str = """
openapi: 3.0.0
info:
  title: Microservice
  version: 1.0.0
paths:
  /workers/add:
    post:
      operationId: add_worker
      summary: Add a worker
"""
    synth_yaml = OpenApiSynthesizer.synthesize_from_yaml(yaml_str, system_id="yaml_sys")
    assert "add_worker" in synth_yaml.action_schemas
    # Heuristic for add_worker
    assert synth_yaml.action_schemas["add_worker"].performance_impact == "positive"


@pytest.mark.asyncio
async def test_http_connector_with_openapi_spec_dispatch():
    """Verify HttpConnector executes actions using OpenAPI endpoint bindings."""
    connector = HttpConnector(
        base_url="http://localhost:8080",
        system_id="payment_svc",
        openapi_spec=SAMPLE_OPENAPI_V3,
    )

    # Connector should have synthesized actions automatically
    assert "scale_payment_service" in connector.supported_actions
    assert "update_rate_limit" in connector.supported_actions

    # Mock client and test POST body routing
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_response = MagicMock(status_code=200, text='{"status":"scaled"}')
    mock_response.json.return_value = {"status": "scaled"}
    mock_client.request = AsyncMock(return_value=mock_response)
    connector._client = mock_client

    action = AdaptationAction(
        action_id="act-1",
        action_type="scale_payment_service",
        target_system="payment_svc",
        parameters={"replicas": 4},
    )

    res = await connector.execute_action(action)
    assert res.status == ExecutionStatus.SUCCESS
    assert res.result_data == {"status": "scaled"}
    # Verifies mapped endpoint /services/payment/scale was hit with POST and JSON body
    mock_client.request.assert_called_once_with(
        "POST", "/services/payment/scale", params=None, json={"replicas": 4}
    )


@pytest.mark.asyncio
async def test_http_connector_with_path_and_query_parameters():
    """Verify HttpConnector handles path substitution and query parameters."""
    connector = HttpConnector(
        base_url="http://localhost:8080",
        system_id="payment_svc",
        openapi_spec=SAMPLE_OPENAPI_V3,
    )

    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_response = MagicMock(status_code=200, text='{"rate": 50}')
    mock_response.json.return_value = {"rate": 50}
    mock_client.request = AsyncMock(return_value=mock_response)
    connector._client = mock_client

    action = AdaptationAction(
        action_id="act-2",
        action_type="update_rate_limit",
        target_system="payment_svc",
        parameters={"tier": "enterprise", "requests_per_minute": 5000},
    )

    res = await connector.execute_action(action)
    assert res.status == ExecutionStatus.SUCCESS
    # Verifies path was substituted (/rate-limit/enterprise) and query param passed
    mock_client.request.assert_called_once_with(
        "PUT",
        "/rate-limit/enterprise",
        params={"requests_per_minute": 5000},
        json={},
    )


@pytest.mark.asyncio
async def test_http_connector_fallback_default_post():
    """Verify HttpConnector falls back to actions_endpoint when action is unmapped."""
    connector = HttpConnector(
        base_url="http://localhost:8080",
        system_id="custom_sys",
        actions_endpoint="/v1/execute",
    )

    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_response = MagicMock(status_code=200, text='{"status":"ok"}')
    mock_response.json.return_value = {"status": "ok"}
    mock_client.post = AsyncMock(return_value=mock_response)
    connector._client = mock_client

    action = AdaptationAction(
        action_id="act-3",
        action_type="custom_action",
        target_system="custom_sys",
        parameters={"key": "val"},
    )

    res = await connector.execute_action(action)
    assert res.status == ExecutionStatus.SUCCESS
    mock_client.post.assert_called_once_with(
        "/v1/execute",
        json={
            "action_id": "act-3",
            "action_type": "custom_action",
            "parameters": {"key": "val"},
            "target_system": "custom_sys",
        },
    )


@pytest.mark.asyncio
async def test_openapi_synthesize_from_url():
    """Verify asynchronous OpenAPI fetching and synthesis from a remote URL."""
    fake_json = json.dumps(SAMPLE_OPENAPI_V3)
    mock_resp = MagicMock(status_code=200, text=fake_json)
    mock_resp.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__.return_value = mock_http_client
    mock_http_client.__aexit__.return_value = None
    mock_http_client.get = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_http_client):
        synth = await OpenApiSynthesizer.synthesize_from_url(
            "http://example.com/openapi.json", system_id="remote_sys"
        )
        assert synth.system_id == "remote_sys"
        assert "scale_payment_service" in synth.action_schemas
