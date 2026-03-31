"""Kubernetes system connector.

Connects Polaris to a Kubernetes cluster for self-adaptation of cloud-native workloads.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

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


class KubernetesConnector(Connector):
    """Connector for Kubernetes clusters.

    Monitors pod/node states and executes actions like scaling deployments. Requires the
    `kubernetes` python package.
    """

    def __init__(
        self,
        kubeconfig_path: Optional[str] = None,
        in_cluster: bool = False,
        namespace: str = "default",
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        """Initialize Kubernetes connector.

        Args:
            kubeconfig_path: Path to kubeconfig file (if None, relies on default env
                vars).
            in_cluster: Whether Polaris is running inside the K8s cluster.
            namespace: Default namespace to scope monitoring and actions.
            logger: Optional logger for logging events.
            metrics: Optional metrics collector for tracking performance.
        """
        self.kubeconfig_path = kubeconfig_path
        self.in_cluster = in_cluster
        self.namespace = namespace
        self._connected = False
        self._logger = logger
        self._metrics = metrics

        # Will be initialized in connect()
        self.core_v1_api: Optional[Any] = None
        self.apps_v1_api: Optional[Any] = None

    async def connect(self) -> bool:
        """Connect to Kubernetes cluster."""
        try:
            import kubernetes.client as k8s_client
            import kubernetes.config as k8s_config

            loop = asyncio.get_running_loop()

            def init_client() -> None:
                if self.in_cluster:
                    k8s_config.load_incluster_config()
                elif self.kubeconfig_path:
                    k8s_config.load_kube_config(config_file=self.kubeconfig_path)
                else:
                    k8s_config.load_kube_config()

            await loop.run_in_executor(None, init_client)

            self.core_v1_api = k8s_client.CoreV1Api()
            self.apps_v1_api = k8s_client.AppsV1Api()

            # Test connection by listing namespaces
            await loop.run_in_executor(
                None, lambda: cast(Any, self.core_v1_api).list_namespace(limit=1)
            )

            self._connected = True
            if self._logger:
                self._logger.info("KubernetesConnector connected", namespace=self.namespace)
            if self._metrics:
                self._metrics.increment("polaris.connector.kubernetes.connected")
            return True
        except ImportError:
            if self._logger:
                self._logger.error("The 'kubernetes' package is not installed.")
            return False
        except Exception as exc:
            self._connected = False
            if self._logger:
                self._logger.error("KubernetesConnector connection failed", error=str(exc))
            if self._metrics:
                self._metrics.increment("polaris.connector.kubernetes.connection_errors")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Kubernetes."""
        self._connected = False
        self.core_v1_api = None
        self.apps_v1_api = None
        if self._logger:
            self._logger.info("KubernetesConnector disconnected")
        if self._metrics:
            self._metrics.increment("polaris.connector.kubernetes.disconnected")
        return True

    async def get_system_id(self) -> str:
        """Get Kubernetes system identifier."""
        return f"kubernetes-{self.namespace}"

    async def collect_telemetry(self) -> SystemState:
        """Collect current state from Kubernetes."""
        if self._metrics:
            self._metrics.increment("polaris.connector.kubernetes.telemetry_calls")
        if not self._connected or not self.core_v1_api:
            return SystemState(
                system_id=await self.get_system_id(),
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
                metadata={"error": "Not connected"},
            )

        loop = asyncio.get_running_loop()
        try:
            metrics: Dict[str, MetricValue] = {}

            # Collect pod telemetry
            def get_pods() -> Any:
                return cast(Any, self.core_v1_api).list_namespaced_pod(self.namespace)

            pods = await loop.run_in_executor(None, get_pods)

            total_pods = len(pods.items)
            running_pods = 0
            pending_pods = 0
            failed_pods = 0

            for pod in pods.items:
                phase = pod.status.phase
                if phase == "Running":
                    running_pods += 1
                elif phase == "Pending":
                    pending_pods += 1
                elif phase == "Failed":
                    failed_pods += 1

            metrics["pods_total"] = MetricValue(
                name="pods_total",
                value=float(total_pods),
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )
            metrics["pods_running"] = MetricValue(
                name="pods_running",
                value=float(running_pods),
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )
            metrics["pods_pending"] = MetricValue(
                name="pods_pending",
                value=float(pending_pods),
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )
            metrics["pods_failed"] = MetricValue(
                name="pods_failed",
                value=float(failed_pods),
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )

            # Collect deployment telemetry
            def get_deployments() -> Any:
                return cast(Any, self.apps_v1_api).list_namespaced_deployment(self.namespace)

            deps = await loop.run_in_executor(None, get_deployments)

            total_deps = len(deps.items)
            unavailable_deps = 0

            for dep in deps.items:
                if dep.status.unavailable_replicas:
                    unavailable_deps += 1

            metrics["deployments_total"] = MetricValue(
                name="deployments_total",
                value=float(total_deps),
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )
            metrics["deployments_unavailable"] = MetricValue(
                name="deployments_unavailable",
                value=float(unavailable_deps),
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )

            # Determine health
            health = HealthStatus.HEALTHY
            if failed_pods > 0 or unavailable_deps > 0:
                health = HealthStatus.WARNING
            if (failed_pods / max(1, total_pods)) > 0.5:
                health = HealthStatus.CRITICAL

            return SystemState(
                system_id=await self.get_system_id(),
                timestamp=datetime.now(timezone.utc),
                metrics=metrics,
                health_status=health,
            )

        except Exception as exc:
            if self._logger:
                self._logger.error("Kubernetes telemetry collection failed", error=str(exc))
            if self._metrics:
                self._metrics.increment("polaris.connector.kubernetes.telemetry_errors")
            return SystemState(
                system_id=await self.get_system_id(),
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
                metadata={"error": str(exc)},
            )

    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute adaptation action on Kubernetes."""
        if not self._connected or not self.apps_v1_api:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message="Not connected to Kubernetes",
            )

        loop = asyncio.get_running_loop()
        try:
            action_type = action.action_type.lower()
            params = action.parameters or {}
            target_ns = params.get("namespace", self.namespace)

            if action_type == "scale_deployment":
                deployment_name = params.get("deployment_name")
                replicas = params.get("replicas")

                if not deployment_name or replicas is None:
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        result_data={},
                        error_message="scale_deployment requires 'deployment_name' and 'replicas' parameters",
                    )

                replicas = int(replicas)
                if replicas < 0:
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        result_data={},
                        error_message="replicas cannot be negative",
                    )

                def patch_deployment() -> None:
                    body = {"spec": {"replicas": replicas}}
                    cast(Any, self.apps_v1_api).patch_namespaced_deployment_scale(
                        name=deployment_name, namespace=target_ns, body=body
                    )

                await loop.run_in_executor(None, patch_deployment)

                if self._logger:
                    self._logger.info(
                        "Kubernetes scale deployed", deployment=deployment_name, replicas=replicas
                    )

                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.SUCCESS,
                    result_data={
                        "action_type": action_type,
                        "deployment_name": deployment_name,
                        "replicas_target": replicas,
                    },
                )

            elif action_type == "restart_deployment":
                deployment_name = params.get("deployment_name")
                if not deployment_name:
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        result_data={},
                        error_message="restart_deployment requires 'deployment_name' parameter",
                    )

                def patch_deployment_restart() -> None:
                    # Modify an annotation to trigger a rolling restart
                    now_str = datetime.now(timezone.utc).isoformat()
                    body = {
                        "spec": {
                            "template": {
                                "metadata": {"annotations": {"polaris-restarted-at": now_str}}
                            }
                        }
                    }
                    cast(Any, self.apps_v1_api).patch_namespaced_deployment(
                        name=deployment_name, namespace=target_ns, body=body
                    )

                await loop.run_in_executor(None, patch_deployment_restart)

                if self._logger:
                    self._logger.info("Kubernetes Deployment restarted", deployment=deployment_name)

                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.SUCCESS,
                    result_data={
                        "action_type": action_type,
                        "deployment_name": deployment_name,
                    },
                )

            else:
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    result_data={},
                    error_message=f"Unsupported action type: {action.action_type}",
                )

        except Exception as exc:
            if self._logger:
                self._logger.error(
                    "Kubernetes action execution failed",
                    action_type=action.action_type,
                    error=str(exc),
                )
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=str(exc),
            )

    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if action can be executed on Kubernetes."""
        valid_types = ["scale_deployment", "restart_deployment"]
        return action.action_type.lower() in valid_types

    async def get_supported_actions(self) -> List[AdaptationAction]:
        """Get list of actions supported by Kubernetes connector."""
        system_id = await self.get_system_id()
        return [
            AdaptationAction(
                action_id="",
                action_type="scale_deployment",
                target_system=system_id,
                parameters={"deployment_name": "string", "replicas": "integer"},
            ),
            AdaptationAction(
                action_id="",
                action_type="restart_deployment",
                target_system=system_id,
                parameters={"deployment_name": "string"},
            ),
        ]

    async def get_capabilities(self) -> ConnectorCapabilities:
        """Expose normalized Kubernetes capability metadata."""
        actions = await self.get_supported_actions()
        action_types = [
            action.action_type
            for action in actions
            if isinstance(action.action_type, str) and action.action_type.strip()
        ]
        return ConnectorCapabilities.from_supported_action_types(
            action_types,
            action_aliases={
                "scale deployment": "scale_deployment",
                "restart deployment": "restart_deployment",
            },
            metadata={"system_family": "kubernetes", "namespace": self.namespace},
        )
