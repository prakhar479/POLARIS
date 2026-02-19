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

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @staticmethod
    def build_logger(
        config: "PolarisConfig",
        cli_overrides: Dict[str, Any],
    ) -> "Logger":
        """Create a logger from configuration with CLI overrides applied."""
        from polaris.infrastructure.observability.logger import create_logger

        logger_type = "structured"
        level = "INFO"
        console = True
        log_file = None
        use_colors = True

        if hasattr(config, "observability") and config.observability:
            logging_config = config.observability.get("logging", {})
            logger_type = logging_config.get("type", logger_type)
            level = logging_config.get("level", level)
            console = logging_config.get("console", console)
            use_colors = logging_config.get("use_colors", use_colors)
            if logging_config.get("file", False):
                log_file = logging_config.get("file_path", "./logs/polaris.log")

        if "log_format" in cli_overrides:
            logger_type = cli_overrides["log_format"]
        if "log_level" in cli_overrides:
            level = cli_overrides["log_level"]
        if "console_logging" in cli_overrides:
            console = bool(cli_overrides["console_logging"])
        if "log_file" in cli_overrides:
            log_file = cli_overrides["log_file"]

        return create_logger(
            logger_type=logger_type,
            name="polaris",
            level=level,
            log_file=log_file,
            console=console,
            use_colors=use_colors,
        )

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
        from polaris.infrastructure.observability.null_metrics import NullMetricsCollector

        if cli_overrides.get("metrics_enabled", True) is False:
            return NullMetricsCollector()

        metrics_config: Dict[str, Any] = {}
        if hasattr(config, "observability") and config.observability:
            metrics_config = config.observability.get("metrics", {})

        if not metrics_config.get("enabled", True):
            return NullMetricsCollector()

        collector_type = metrics_config.get("collector_type", "simple")

        if collector_type == "simple":
            from polaris.infrastructure.observability.metrics import SimpleMetricsCollector

            return SimpleMetricsCollector()

        # Fallback for unknown collector types
        from polaris.infrastructure.observability.metrics import SimpleMetricsCollector

        return SimpleMetricsCollector()

    @staticmethod
    def build_event_bus(
        config: "PolarisConfig",
        metrics: Optional["MetricsCollector"],
        logger: "Logger",
    ) -> "EventBus":
        """Create an EventBus, optionally wired to the metrics collector."""
        from polaris.core.events import EventBus

        event_bus_metrics = (
            metrics if ComponentBuilder.should_collect(config, "event_bus", metrics) else None
        )
        return EventBus(metrics=event_bus_metrics, logger=logger)

    # ------------------------------------------------------------------
    # Core domain components
    # ------------------------------------------------------------------

    @staticmethod
    def build_knowledge_store(
        config: "PolarisConfig",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
    ) -> "KnowledgeStore":
        """Create the knowledge store from config.

        If ``config.knowledge_store.db_path`` is set the SQLite-backed store is
        used (data survives restarts).  Otherwise the default in-memory store is
        used.
        """
        ks_metrics = (
            metrics if ComponentBuilder.should_collect(config, "knowledge_store", metrics) else None
        )

        db_path: Optional[str] = None
        max_states: int = 5000
        if hasattr(config, "knowledge_store") and config.knowledge_store:
            ks_cfg = config.knowledge_store
            if isinstance(ks_cfg, dict):
                db_path = ks_cfg.get("db_path")
                max_states = int(ks_cfg.get("max_states_per_system", max_states))
            elif hasattr(ks_cfg, "db_path"):
                db_path = ks_cfg.db_path
                if hasattr(ks_cfg, "max_states_per_system"):
                    max_states = int(ks_cfg.max_states_per_system)

        if db_path:
            from polaris.knowledge.sqlite_store import SQLiteKnowledgeStore

            return SQLiteKnowledgeStore(
                db_path=db_path,
                max_states_per_system=max_states,
                logger=logger,
                metrics=ks_metrics,
            )

        from polaris.knowledge import InMemoryKnowledgeStore

        return InMemoryKnowledgeStore(logger=logger, metrics=ks_metrics)

    @staticmethod
    def build_world_model(
        config: "PolarisConfig",
        knowledge_store: "KnowledgeStore",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
    ) -> "WorldModel":
        """Create the default statistical world model."""
        from polaris.world_model import StatisticalWorldModel

        wm_metrics = (
            metrics if ComponentBuilder.should_collect(config, "world_model", metrics) else None
        )
        return StatisticalWorldModel(knowledge_store, logger=logger, metrics=wm_metrics)

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
        """Create a strategy from configuration, falling back to ThresholdReactiveStrategy."""
        from polaris.core.factories import get_strategy_factory
        from polaris.strategies import ThresholdReactiveStrategy

        strategy_metrics = (
            metrics if ComponentBuilder.should_collect(config, "strategy", metrics) else None
        )

        factory = get_strategy_factory(strategy_config.type)
        if not factory:
            logger.warning(
                f"No strategy factory registered for type '{strategy_config.type}', "
                "using threshold strategy instead"
            )
            return ThresholdReactiveStrategy(logger=logger, metrics=strategy_metrics)

        try:
            return factory(
                strategy_config,
                logger,
                strategy_metrics,
                knowledge_store,
                world_model,
                registry,
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize strategy of type '{strategy_config.type}' "
                f"from config: {e}. Falling back to threshold."
            )
            return ThresholdReactiveStrategy(logger=logger, metrics=strategy_metrics)

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
        if not isinstance(meta_config, dict):
            return None

        meta_metrics = (
            metrics if ComponentBuilder.should_collect(config, "meta_learner", metrics) else None
        )
        meta_type = meta_config.get("type", "statistical")

        if meta_type == "statistical":
            from polaris.meta_learner import StatisticalMetaLearner
            from polaris.meta_learner.bayesian_optimizer import AcquisitionFunction

            stat_cfg = meta_config.get("statistical", {}) or {}
            conservative_mode = bool(stat_cfg.get("conservative_mode", True))
            enable_bayesian = bool(stat_cfg.get("enable_bayesian_optimization", True))
            min_samples = int(stat_cfg.get("min_samples_for_optimization", 10))

            acq_func_str = stat_cfg.get("acquisition_function", "expected_improvement")
            try:
                acquisition_function = AcquisitionFunction(acq_func_str)
            except ValueError:
                logger.warning(f"Unknown acquisition function '{acq_func_str}', using default")
                acquisition_function = AcquisitionFunction.EXPECTED_IMPROVEMENT

            exploration_weight = float(stat_cfg.get("exploration_weight", 0.1))

            return StatisticalMetaLearner(
                knowledge_store=knowledge_store,
                logger=logger,
                conservative_mode=conservative_mode,
                world_model=world_model,
                enable_bayesian_optimization=enable_bayesian,
                acquisition_function=acquisition_function,
                exploration_weight=exploration_weight,
                min_samples_for_optimization=min_samples,
            )

        if meta_type == "llm":
            try:
                from polaris.infrastructure.llm import create_llm_client
                from polaris.meta_learner import LLMMetaLearner

                llm_cfg = meta_config.get("llm", {}) or {}
                provider = llm_cfg.get("provider", "google")
                resilience_cfg = llm_cfg.get("resilience")
                llm_client = create_llm_client(provider, resilience=resilience_cfg)

                return LLMMetaLearner(
                    llm_client=llm_client,
                    knowledge_store=knowledge_store,
                    logger=logger,
                    auto_apply=bool(llm_cfg.get("auto_apply", False)),
                    temperature=float(llm_cfg.get("temperature", 0.1)),
                    analysis_system_prompt=llm_cfg.get("analysis_system_prompt"),
                    optimization_system_prompt=llm_cfg.get("optimization_system_prompt"),
                    per_system_prompts=llm_cfg.get("per_system_prompts"),
                    metrics=meta_metrics,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM meta-learner from config: {e}")
                return None

        logger.warning(f"Unknown meta_learner type '{meta_type}'. Meta-learning will be disabled.")
        return None

    @staticmethod
    def build_connectors(
        systems_config: Any,
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        config: "PolarisConfig",
    ) -> List["Connector"]:
        """Create connectors for all enabled systems in the configuration."""
        from polaris.core.factories import get_connector_factory

        connectors: List[Any] = []
        connector_metrics = (
            metrics if ComponentBuilder.should_collect(config, "connectors", metrics) else None
        )

        for system in systems_config:
            if not system.enabled:
                continue

            factory = get_connector_factory(system.connector_type)
            if not factory:
                logger.error(
                    f"No connector factory registered for type '{system.connector_type}', "
                    f"skipping system '{system.id}'"
                )
                continue

            try:
                connector = factory(system, logger, connector_metrics)
            except Exception as e:
                logger.error(
                    f"Failed to create connector for system '{system.id}' "
                    f"(type '{system.connector_type}'): {e}"
                )
                continue

            connectors.append(connector)

        return connectors

    # ------------------------------------------------------------------
    # Metrics export configuration
    # ------------------------------------------------------------------

    @staticmethod
    def build_metrics_export_config(
        config: "PolarisConfig",
        cli_overrides: Dict[str, Any],
        metrics: Optional["MetricsCollector"],
    ) -> Dict[str, Any]:
        """Build the metrics auto-export configuration dict."""
        if not metrics or not hasattr(metrics, "export_to_file"):
            return {"enabled": False}

        export_config: Dict[str, Any] = {}
        if hasattr(config, "observability") and config.observability:
            metrics_config = config.observability.get("metrics", {})
            export_config = metrics_config.get("export", {})

        export_enabled = export_config.get("enabled", False)
        export_dir = cli_overrides.get("metrics_export_dir") or export_config.get(
            "output_dir", "./metrics"
        )
        auto_interval = cli_overrides.get("metrics_auto_export_interval")
        if auto_interval is None:
            auto_interval = export_config.get("auto_export_interval_minutes", 0)

        if export_enabled and auto_interval is not None and auto_interval > 0:
            return {
                "enabled": True,
                "interval_minutes": auto_interval,
                "output_dir": export_dir,
                "formats": cli_overrides.get("metrics_export_formats")
                or export_config.get("formats", ["json"]),
                "experiment_name": cli_overrides.get("metrics_experiment_name")
                or export_config.get("experiment_name"),
                "include_timestamp": export_config.get("include_timestamp", True),
            }

        return {"enabled": False}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        if not isinstance(meta_config, dict):
            return 3600.0
        try:
            interval_hours = float(meta_config.get("analysis_interval_hours", 1.0))
            if interval_hours > 0:
                return interval_hours * 3600.0
        except Exception:
            pass
        return 3600.0

    @staticmethod
    def resolve_monitoring_interval(
        config: "PolarisConfig",
        cli_overrides: Dict[str, Any],
        logger: "Logger",
    ) -> float:
        """Return the validated monitoring interval in seconds."""
        interval: float = 30.0

        if hasattr(config, "monitoring") and config.monitoring:
            interval = config.monitoring.get("interval_seconds", interval)
        if "monitoring_interval" in cli_overrides:
            interval = cli_overrides["monitoring_interval"]

        try:
            interval = float(interval)
        except Exception:
            logger.warning(f"Invalid monitoring interval {interval}, falling back to 30 seconds")
            interval = 30.0

        if interval <= 0:
            logger.warning(
                f"Non-positive monitoring interval {interval}, falling back to 30 seconds"
            )
            interval = 30.0

        return interval
