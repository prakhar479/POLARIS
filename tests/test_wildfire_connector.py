import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from polaris.connectors.wildfire import WildfireConnector
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    SystemState,
)


@pytest.fixture
def mock_metrics():
    metrics = MagicMock()
    metrics.increment = MagicMock()
    metrics.histogram = MagicMock()
    return metrics


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    return logger


@pytest.fixture
def connector(mock_metrics, mock_logger):
    return WildfireConnector(
        base_url="http://localhost:5000",
        system_id="wildfire-test",
        timeout=5.0,
        logger=mock_logger,
        metrics=mock_metrics,
    )


@pytest.mark.asyncio
async def test_connect_success(connector, mock_metrics, mock_logger):
    mock_client = AsyncMock()
    # Auto-create session first
    mock_client.post.return_value = httpx.Response(
        201, json={"session_id": "abc123", "num_agents": 2}, request=httpx.Request("POST", "/api/v1/sessions")
    )
    mock_client.get.return_value = httpx.Response(
        200, json={"status": "healthy", "active_sessions": 1}, request=httpx.Request("GET", "/health")
    )

    with patch.object(connector, "_ensure_client", return_value=mock_client):
        result = await connector.connect()

    assert result is True
    assert connector._connected is True
    assert connector.session_id == "abc123"
    mock_metrics.increment.assert_any_call("polaris.connector.wildfire.connected")
    mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_connect_failure(connector, mock_metrics, mock_logger):
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.RequestError("Connection refused")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await connector.connect()

    assert result is False
    assert connector._connected is False
    mock_metrics.increment.assert_any_call("polaris.connector.wildfire.connection_errors")
    mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_disconnect(connector, mock_metrics, mock_logger):
    connector._connected = True
    mock_client = AsyncMock()
    connector._client = mock_client

    result = await connector.disconnect()

    assert result is True
    assert connector._connected is False
    assert connector._client is None
    mock_client.aclose.assert_called_once()
    mock_metrics.increment.assert_any_call("polaris.connector.wildfire.disconnected")
    mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_get_system_id(connector):
    assert await connector.get_system_id() == "wildfire-test"

@pytest.mark.asyncio
async def test_collect_telemetry_success(connector, mock_metrics):
    connector._connected = True
    mock_client = AsyncMock()
    mock_client.get.return_value = httpx.Response(
        200,
        json={
            "timestep": 5,
            "metrics": {
                "timestep": 5,
                "num_agents": 2,
                "mr1_values": [0.5, 0.6],
                "mr2_value": 2,
                "fire_cells_burning": 15,
                "fire_cells_total": 2500,
                "agent_positions": [{"id": 0, "position": [25, 30]}],
            },
        },
    )
    connector._client = mock_client

    state = await connector.collect_telemetry()

    assert isinstance(state, SystemState)
    assert state.system_id == "wildfire-test"
    assert state.health_status == HealthStatus.HEALTHY
    assert "timestep" in state.metrics
    assert state.metrics["timestep"].value == 5
    assert "num_agents" in state.metrics
    assert state.metrics["num_agents"].value == 2
    assert "mr1_avg" in state.metrics
    assert state.metrics["mr1_avg"].value == pytest.approx(0.55)
    assert "mr2_value" in state.metrics
    assert state.metrics["mr2_value"].value == 2
    assert "fire_cells_burning" in state.metrics
    assert state.metrics["fire_cells_burning"].value == 15
    assert "fire_cells_burning_ratio" in state.metrics
    assert state.metrics["fire_cells_burning_ratio"].value == pytest.approx(0.6)
    mock_metrics.histogram.assert_called()

@pytest.mark.asyncio
async def test_collect_telemetry_not_connected(connector, mock_metrics):
    connector._connected = False
    connector.auto_create_session = False

    state = await connector.collect_telemetry()

    assert state.health_status == HealthStatus.UNHEALTHY
    assert "Unable to connect" in state.metadata["error"]


@pytest.mark.asyncio
async def test_collect_telemetry_api_error(connector, mock_metrics, mock_logger):
    connector._connected = True
    mock_client = AsyncMock()
    mock_client.get.return_value = httpx.Response(500, text="Internal Server Error")
    connector._client = mock_client

    state = await connector.collect_telemetry()

    assert state.health_status == HealthStatus.UNHEALTHY
    assert "error" in state.metadata
    mock_metrics.increment.assert_any_call("polaris.connector.wildfire.telemetry_errors")
    mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_execute_action_wildfire_reset(connector, mock_metrics, mock_logger):
    connector._connected = True
    mock_client = AsyncMock()
    mock_client.post.return_value = httpx.Response(
        200, json={"message": "Simulation reset", "timestep": 0}
    )
    connector._client = mock_client

    action = AdaptationAction(
        action_id="test-123",
        action_type="wildfire_reset",
        target_system="wildfire-test",
        parameters={},
    )

    result = await connector.execute_action(action)

    assert result.action_id == "test-123"
    assert result.status == ExecutionStatus.SUCCESS
    assert "response" in result.result_data
    mock_metrics.histogram.assert_called()
    mock_metrics.increment.assert_any_call(
        "polaris.connector.wildfire.actions_executed",
        tags={"action_type": "wildfire_reset", "status": "success"},
    )
    mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_execute_action_wildfire_move(connector, mock_metrics, mock_logger):
    connector._connected = True
    mock_client = AsyncMock()
    mock_client.post.return_value = httpx.Response(
        200, json={"message": "Action executed successfully", "timestep": 6, "applied": 2}
    )
    connector._client = mock_client

    action = AdaptationAction(
        action_id="move-456",
        action_type="wildfire_move",
        target_system="wildfire-test",
        parameters={
            "actions": [{"uav": 0, "move": "north"}, {"uav": 1, "move": "hold"}]
        },
    )

    result = await connector.execute_action(action)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.action_id == "move-456"
    mock_client.post.assert_called_with(
        "/api/v1/sim/action",
        json=[{"uav": 0, "move": "north"}, {"uav": 1, "move": "hold"}],
    )


@pytest.mark.asyncio
async def test_execute_action_unsupported_type(connector, mock_metrics):
    connector._connected = True

    action = AdaptationAction(
        action_id="bad-789",
        action_type="unknown_action",
        target_system="wildfire-test",
        parameters={},
    )

    result = await connector.execute_action(action)

    assert result.status == ExecutionStatus.FAILED
    assert "Unsupported action type" in result.error_message
    mock_metrics.increment.assert_any_call(
        "polaris.connector.wildfire.actions_unsupported"
    )


@pytest.mark.asyncio
async def test_execute_action_not_connected(connector, mock_metrics):
    connector._connected = False

    action = AdaptationAction(
        action_id="not-connected",
        action_type="wildfire_reset",
        target_system="wildfire-test",
        parameters={},
    )

    result = await connector.execute_action(action)

    assert result.status == ExecutionStatus.FAILED
    assert "Not connected" in result.error_message


@pytest.mark.asyncio
async def test_validate_action(connector):
    valid = AdaptationAction(
        action_id="valid",
        action_type="wildfire_reset",
        target_system="wildfire-test",
        parameters={},
    )
    invalid_type = AdaptationAction(
        action_id="invalid-type",
        action_type="unknown",
        target_system="wildfire-test",
        parameters={},
    )
    wrong_target = AdaptationAction(
        action_id="wrong-target",
        action_type="wildfire_reset",
        target_system="other-system",
        parameters={},
    )

    assert await connector.validate_action(valid) is True
    assert await connector.validate_action(invalid_type) is False
    assert await connector.validate_action(wrong_target) is False


@pytest.mark.asyncio
async def test_get_supported_actions(connector):
    actions = await connector.get_supported_actions()
    assert len(actions) == 6
    action_types = {a.action_type for a in actions}
    expected = {
        "wildfire_reset",
        "wildfire_pause",
        "wildfire_resume",
        "wildfire_step",
        "wildfire_move",
        "wildfire_batch_actions",
    }
    assert action_types == expected
    for a in actions:
        assert a.target_system == "wildfire-test"
