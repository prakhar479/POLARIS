"""Tests for the KubernetesConnector."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from polaris.connectors.kubernetes_connector import KubernetesConnector
from polaris.core.models import AdaptationAction, ExecutionStatus, HealthStatus


@pytest.fixture
def k8s_connector():
    return KubernetesConnector(kubeconfig_path="/dummy/path", namespace="test-ns")


@pytest.mark.asyncio
@patch("kubernetes.config.load_kube_config")
@patch("kubernetes.client.CoreV1Api")
@patch("kubernetes.client.AppsV1Api")
async def test_kubernetes_connect_success(
    mock_apps_api, mock_core_api, mock_load_config, k8s_connector
):
    core_mock = MagicMock()
    mock_core_api.return_value = core_mock

    success = await k8s_connector.connect()

    assert success is True
    assert k8s_connector._connected is True
    mock_load_config.assert_called_once_with(config_file="/dummy/path")
    core_mock.list_namespace.assert_called_once_with(limit=1)


@pytest.mark.asyncio
async def test_kubernetes_connect_import_error(k8s_connector):
    with patch.dict("sys.modules", {"kubernetes": None}):
        success = await k8s_connector.connect()
        assert success is False
        assert k8s_connector._connected is False


@pytest.mark.asyncio
@patch("kubernetes.client.CoreV1Api")
@patch("kubernetes.client.AppsV1Api")
async def test_kubernetes_collect_telemetry(mock_apps_api, mock_core_api, k8s_connector):
    # Set up mocks manually to avoid full connect method call dependencies for test isolation
    k8s_connector._connected = True
    k8s_connector.core_v1_api = mock_core_api.return_value
    k8s_connector.apps_v1_api = mock_apps_api.return_value

    # Mock Pod response
    pod_mock = MagicMock()
    pod_mock.status.phase = "Running"
    pod_mock_failed = MagicMock()
    pod_mock_failed.status.phase = "Failed"

    pod_list_mock = MagicMock()
    pod_list_mock.items = [pod_mock, pod_mock, pod_mock_failed]
    k8s_connector.core_v1_api.list_namespaced_pod.return_value = pod_list_mock

    # Mock Deployment response
    dep_mock = MagicMock()
    dep_mock.status.unavailable_replicas = 0
    dep_list_mock = MagicMock()
    dep_list_mock.items = [dep_mock]
    k8s_connector.apps_v1_api.list_namespaced_deployment.return_value = dep_list_mock

    state = await k8s_connector.collect_telemetry()

    assert state.system_id == "kubernetes-test-ns"
    assert "pods_total" in state.metrics
    assert state.metrics["pods_total"].value == 3.0
    assert state.metrics["pods_running"].value == 2.0
    assert state.metrics["pods_failed"].value == 1.0

    # 1 failed out of 3 -> > 0 failed -> Warning
    assert state.health_status == HealthStatus.WARNING


@pytest.mark.asyncio
@patch("kubernetes.client.AppsV1Api")
async def test_kubernetes_execute_scale(mock_apps_api, k8s_connector):
    k8s_connector._connected = True
    k8s_connector.apps_v1_api = mock_apps_api.return_value

    action = AdaptationAction(
        action_id="test-action",
        action_type="scale_deployment",
        target_system="kubernetes-test-ns",
        parameters={"deployment_name": "my-web", "replicas": 3},
    )

    result = await k8s_connector.execute_action(action)

    assert result.status == ExecutionStatus.SUCCESS
    k8s_connector.apps_v1_api.patch_namespaced_deployment_scale.assert_called_once_with(
        name="my-web", namespace="test-ns", body={"spec": {"replicas": 3}}
    )


@pytest.mark.asyncio
async def test_kubernetes_validate_action(k8s_connector):
    action1 = AdaptationAction(
        action_id="1",
        action_type="scale_deployment",
        target_system="kubernetes-test-ns",
        parameters={},
    )
    action2 = AdaptationAction(
        action_id="2",
        action_type="restart_deployment",
        target_system="kubernetes-test-ns",
        parameters={},
    )
    action3 = AdaptationAction(
        action_id="3",
        action_type="unknown_action",
        target_system="kubernetes-test-ns",
        parameters={},
    )

    assert await k8s_connector.validate_action(action1) is True
    assert await k8s_connector.validate_action(action2) is True
    assert await k8s_connector.validate_action(action3) is False
