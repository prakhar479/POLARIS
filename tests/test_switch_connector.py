"""Tests for SWITCH connector."""

import pytest

from polaris.connectors.switch import SWITCHConnector
from polaris.core.models import AdaptationAction, ExecutionStatus, HealthStatus


@pytest.mark.asyncio
async def test_switch_connector_synthetic_connection():
    connector = SWITCHConnector(
        system_id="switch_test", synthetic_mode=True, initial_model="yolov5m"
    )
    connected = await connector.connect()
    assert connected is True

    sys_id = await connector.get_system_id()
    assert sys_id == "switch_test"

    actions = await connector.get_supported_actions()
    assert any(a.action_type == "switch_model" for a in actions)

    disconnected = await connector.disconnect()
    assert disconnected is True


@pytest.mark.asyncio
async def test_switch_connector_telemetry_synthetic():
    connector = SWITCHConnector(
        system_id="switch_test", synthetic_mode=True, initial_model="yolov5m"
    )
    await connector.connect()

    state = await connector.collect_telemetry()
    assert state.system_id == "switch_test"
    assert "confidence_mean" in state.metrics
    assert "response_time" in state.metrics
    assert "cpu_usage" in state.metrics
    assert "inference_rate" in state.metrics
    assert "switch_count" in state.metrics

    # yolov5m baseline latency is ~0.096s, so health should be healthy
    assert state.health_status == HealthStatus.HEALTHY
    assert state.metadata["active_model"] == "yolov5m"
    assert state.metadata["synthetic_mode"] is True

    await connector.disconnect()


@pytest.mark.asyncio
async def test_switch_connector_execute_action():
    connector = SWITCHConnector(
        system_id="switch_test", synthetic_mode=True, initial_model="yolov5m"
    )
    await connector.connect()

    action = AdaptationAction(
        action_id="act-test-1",
        action_type="switch_model",
        target_system="switch_test",
        parameters={"model_name": "yolov5s"},
    )
    is_valid = await connector.validate_action(action)
    assert is_valid is True

    result = await connector.execute_action(action)
    assert result.status == ExecutionStatus.SUCCESS
    assert result.result_data["previous_model"] == "yolov5m"
    assert result.result_data["active_model"] == "yolov5s"
    assert result.result_data["total_switches"] == 1

    # Verify telemetry now reflects new model yolov5s
    state = await connector.collect_telemetry()
    assert state.metadata["active_model"] == "yolov5s"
    assert state.metrics["switch_count"].value == 1

    await connector.disconnect()


@pytest.mark.asyncio
async def test_switch_connector_invalid_actions():
    connector = SWITCHConnector(
        system_id="switch_test", synthetic_mode=True, initial_model="yolov5m"
    )
    await connector.connect()

    # Invalid action type
    bad_type = AdaptationAction(
        action_id="act-bad-1",
        action_type="scale_up",
        target_system="switch_test",
    )
    assert await connector.validate_action(bad_type) is False
    res_bad_type = await connector.execute_action(bad_type)
    assert res_bad_type.status == ExecutionStatus.FAILED

    # Invalid model name
    bad_model = AdaptationAction(
        action_id="act-bad-2",
        action_type="switch_model",
        target_system="switch_test",
        parameters={"model_name": "resnet9000"},
    )
    assert await connector.validate_action(bad_model) is False
    res_bad_model = await connector.execute_action(bad_model)
    assert res_bad_model.status == ExecutionStatus.FAILED

    await connector.disconnect()


@pytest.mark.asyncio
async def test_switch_connector_capabilities():
    connector = SWITCHConnector(system_id="switch_test", synthetic_mode=True)
    caps = await connector.get_capabilities()
    assert "switch_model" in caps.actions
    assert caps.actions["switch_model"].action_type == "switch_model"
    assert "confidence_mean" in caps.metrics
    assert "response_time" in caps.metrics
    assert len(caps.slos) == 2
