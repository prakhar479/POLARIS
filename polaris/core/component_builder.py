"""Factory helpers for building Polaris components from configuration.

All methods are static so they can be tested independently of the Polaris
orchestrator and used by third-party code that wants to construct individual
components without instantiating the full framework.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from polaris.abstractions import (
        AdaptationStrategy,
        Connector,
        KnowledgeStore,
        Logger,
        MetaLearner,
        MetricsCollector,
        WorldModel,
    )
    from polaris.core.events import EventBus
    from polaris.core.registry import ConnectorRegistry
    from polaris.infrastructure.config import PolarisConfig


class ComponentBuilder:
    """Static factory methods for constructing Polaris components from config.

    Each method accepts the relevant subset of configuration and CLI overrides
    and returns a fully-initialised component ready for use.  No instance
    state is maintained — all methods are ``@staticmethod``.
    """

    # Observability

    @staticmethod
    def build_logger(
        config: "PolarisConfig",
        cli_overrides: Dict[str, Any],
    ) -> "Logger":
        """Create a logger from configuration with CLI overrides applied."""
        from polaris.core.builders.observability import build_logger as build_logger_impl

        return build_logger_impl(config, cli_overrides)

    @staticmethod
    def build_metrics(
        config: "PolarisConfig",
        cli_overrides: Dict[str, Any],
    ) -> "MetricsCollector":
        """Create a metrics collector from configuration with CLI overrides applied.

        Returns a :class:`NullMetricsCollector` when metrics are disabled so
        components can call ``self.metrics.increment(...)`` unconditionally
        without ``if self.metrics:`` guards.
        """
        from polaris.core.builders.observability import build_metrics as build_metrics_impl

        return build_metrics_impl(config, cli_overrides)

    @staticmethod
    def build_event_bus(
        config: "PolarisConfig",
        metrics: Optional["MetricsCollector"],
        logger: "Logger",
    ) -> "EventBus":
        """Create an EventBus, optionally wired to the metrics collector."""
        event_bus_metrics = (
            metrics if ComponentBuilder.should_collect(config, "event_bus", metrics) else None
        )
        from polaris.core.builders.observability import build_event_bus as build_event_bus_impl

        return build_event_bus_impl(config, event_bus_metrics, logger)

    # Core domain components

    @staticmethod
    def build_knowledge_store(
        config: "PolarisConfig",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
    ) -> "KnowledgeStore":
        """Create the knowledge store from canonical ``knowledge_store`` config."""
        from polaris.core.builders.domain import build_knowledge_store as build_knowledge_store_impl

        ks_metrics = (
            metrics if ComponentBuilder.should_collect(config, "knowledge_store", metrics) else None
        )
        return build_knowledge_store_impl(config, logger, ks_metrics)

    @staticmethod
    def build_world_model(
        config: "PolarisConfig",
        knowledge_store: "KnowledgeStore",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
    ) -> "WorldModel":
        """Create the default statistical world model."""
        from polaris.core.builders.domain import build_world_model as build_world_model_impl

        wm_metrics = (
            metrics if ComponentBuilder.should_collect(config, "world_model", metrics) else None
        )
        return build_world_model_impl(config, knowledge_store, logger, wm_metrics)

    @staticmethod
    def build_strategy(
        strategy_config: Any,
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        registry: "ConnectorRegistry",
        config: "PolarisConfig",
    ) -> "AdaptationStrategy":
        """Create a strategy from configuration."""
        from polaris.core.builders.domain import build_strategy as build_strategy_impl

        strategy_metrics = (
            metrics if ComponentBuilder.should_collect(config, "strategy", metrics) else None
        )
        return build_strategy_impl(
            strategy_config,
            logger,
            strategy_metrics,
            knowledge_store,
            world_model,
            registry,
        )

    @staticmethod
    def build_meta_learner(
        meta_config: Optional[Dict[str, Any]],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        config: "PolarisConfig",
    ) -> Optional["MetaLearner"]:
        """Create a meta-learner from configuration.

        Returns ``None`` if meta-learning is not configured or if initialisation
        fails.  Also returns the resolved meta-learning loop interval (seconds)
        as a side-effect stored on the returned object's ``_interval_seconds``
        attribute so the caller can read it.
        """
        from polaris.core.builders.domain import build_meta_learner as build_meta_learner_impl

        meta_metrics = (
            metrics if ComponentBuilder.should_collect(config, "meta_learner", metrics) else None
        )
        return build_meta_learner_impl(
            meta_config,
            knowledge_store,
            world_model,
            logger,
            meta_metrics,
        )

    @staticmethod
    def build_connectors(
        systems_config: Any,
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        config: "PolarisConfig",
    ) -> List["Connector"]:
        """Create connectors for all enabled systems in the configuration."""
        from polaris.core.builders.domain import build_connectors as build_connectors_impl

        connector_metrics = (
            metrics if ComponentBuilder.should_collect(config, "connectors", metrics) else None
        )
        return build_connectors_impl(systems_config, logger, connector_metrics)

    # Metrics export configuration

    @staticmethod
    def build_metrics_export_config(
        config: "PolarisConfig",
        cli_overrides: Dict[str, Any],
        metrics: Optional["MetricsCollector"],
    ) -> Dict[str, Any]:
        """Build the metrics auto-export configuration dict."""
        from polaris.core.builders.runtime import (
            build_metrics_export_config as build_metrics_export_config_impl,
        )

        return build_metrics_export_config_impl(config, cli_overrides, metrics)

    # Helpers

    @staticmethod
    def should_collect(
        config: "PolarisConfig",
        component_name: str,
        metrics: Optional["MetricsCollector"],
    ) -> bool:
        """Return True if per-component metrics collection is enabled."""
        if not metrics:
            return False

        metrics_config: Dict[str, Any] = {}
        if hasattr(config, "observability") and config.observability:
            metrics_config = config.observability.get("metrics", {})

        components_config = metrics_config.get("components", {})
        return bool(components_config.get(component_name, True))

    @staticmethod
    def resolve_meta_learning_interval(meta_config: Optional[Dict[str, Any]]) -> float:
        """Return the meta-learning loop interval in seconds from config."""
        from polaris.core.builders.runtime import (
            resolve_meta_learning_interval as resolve_meta_learning_interval_impl,
        )

        return resolve_meta_learning_interval_impl(meta_config)

    @staticmethod
    def resolve_meta_learning_transparency_config(
        meta_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return normalized transparency config for meta-learning records."""
        from polaris.core.builders.runtime import (
            resolve_meta_learning_transparency_config as resolve_meta_learning_transparency_config_impl,
        )

        return resolve_meta_learning_transparency_config_impl(meta_config)

    @staticmethod
    def resolve_monitoring_interval(
        config: "PolarisConfig",
        cli_overrides: Dict[str, Any],
        logger: "Logger",
    ) -> float:
        """Return the validated monitoring interval in seconds."""
        _ = logger  # kept for backwards-compatible method signature
        from polaris.core.builders.runtime import (
            resolve_monitoring_interval as resolve_monitoring_interval_impl,
        )

        return resolve_monitoring_interval_impl(config, cli_overrides)
