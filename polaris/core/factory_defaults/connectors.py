"""Built-in connector factory registrations."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from polaris.infrastructure.constants import (
    DEFAULT_CONNECTOR_TIMEOUT,
    DEFAULT_WILDFIRE_PORT,
    MAX_PORT,
    MIN_PORT,
)

if TYPE_CHECKING:
    from polaris.abstractions import Connector, Logger, MetricsCollector

ConnectorFactory = Callable[[Any, "Logger", Optional["MetricsCollector"]], "Connector"]
ConnectorConfigValidator = Callable[[Dict[str, Any]], None]
RegisterConnectorFactory = Callable[[str, ConnectorFactory], None]
RegisterConnectorConfigValidator = Callable[[str, ConnectorConfigValidator], None]


def register_default_connector_factories(
    register_connector_factory: RegisterConnectorFactory,
    register_connector_config_validator: RegisterConnectorConfigValidator,
) -> None:
    """Register factories and validators for built-in connector types."""
    from polaris.connectors import (
        KubernetesConnector,
        SUAVEConnector,
        SWIMConnector,
        WildfireConnector,
    )

    def _validate_port(port: Any, connector_name: str) -> None:
        if not isinstance(port, int):
            raise ValueError(f"{connector_name} connection port must be an integer")
        if not (MIN_PORT <= port <= MAX_PORT):
            raise ValueError(
                f"{connector_name} connection port must be between {MIN_PORT} and {MAX_PORT}"
            )

    def _validate_swim_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("SWIM connection config must be a dictionary")

        host = connection.get("host")
        if host is not None and not isinstance(host, str):
            raise ValueError("SWIM connection host must be a string")

        port = connection.get("port")
        if port is not None:
            _validate_port(port, "SWIM")

    def _validate_wildfire_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("Wildfire connection config must be a dictionary")

        base_url = connection.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("Wildfire base_url must be a string")

        host = connection.get("host")
        if host is not None and not isinstance(host, str):
            raise ValueError("Wildfire host must be a string")

        port = connection.get("port")
        if port is not None:
            _validate_port(port, "Wildfire")

    def _validate_kubernetes_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("Kubernetes connection config must be a dictionary")

        kubeconfig_path = connection.get("kubeconfig_path")
        if kubeconfig_path is not None and not isinstance(kubeconfig_path, str):
            raise ValueError("Kubernetes kubeconfig_path must be a string")

        in_cluster = connection.get("in_cluster")
        if in_cluster is not None and not isinstance(in_cluster, bool):
            raise ValueError("Kubernetes in_cluster must be a boolean")

        namespace = connection.get("namespace")
        if namespace is not None and not isinstance(namespace, str):
            raise ValueError("Kubernetes namespace must be a string")

    def _swim_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
        host = system_cfg.connection.get("host", "localhost")
        port = system_cfg.connection.get("port", 4242)
        return SWIMConnector(host=host, port=port, logger=logger, metrics=metrics)

    register_connector_factory("swim", _swim_factory)
    register_connector_config_validator("swim", _validate_swim_connection)

    def _wildfire_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
        base_url = system_cfg.connection.get("base_url")
        if not base_url:
            host = system_cfg.connection.get("host", "localhost")
            port = system_cfg.connection.get("port", DEFAULT_WILDFIRE_PORT)
            base_url = f"http://{host}:{port}"

        return WildfireConnector(
            base_url=base_url,
            system_id=system_cfg.id,
            timeout=system_cfg.connection.get("timeout", DEFAULT_CONNECTOR_TIMEOUT),
            session_id=system_cfg.connection.get("session_id"),
            logger=logger,
            metrics=metrics,
        )

    register_connector_factory("wildfire", _wildfire_factory)
    register_connector_config_validator("wildfire", _validate_wildfire_connection)

    def _suave_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
        host = system_cfg.connection.get("host", "localhost")
        port = system_cfg.connection.get("port", 9090)
        connect_timeout = system_cfg.connection.get("connect_timeout", 10.0)
        service_timeout = system_cfg.connection.get("service_timeout", 5.0)

        return SUAVEConnector(
            host=host,
            port=port,
            connect_timeout=connect_timeout,
            service_timeout=service_timeout,
            logger=logger,
            metrics=metrics,
        )

    register_connector_factory("suave", _suave_factory)

    def _kubernetes_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
        kubeconfig = system_cfg.connection.get("kubeconfig_path")
        in_cluster = system_cfg.connection.get("in_cluster", False)
        namespace = system_cfg.connection.get("namespace", "default")
        return KubernetesConnector(
            kubeconfig_path=kubeconfig,
            in_cluster=in_cluster,
            namespace=namespace,
            logger=logger,
            metrics=metrics,
        )

    register_connector_factory("kubernetes", _kubernetes_factory)
    register_connector_config_validator("kubernetes", _validate_kubernetes_connection)
