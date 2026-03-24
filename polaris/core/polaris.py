"""Main Polaris framework orchestrator.

The ``Polaris`` class is a thin orchestrator that wires together focused
sub-modules:

- :mod:`polaris.core.component_builder` — factory helpers for all components
- :mod:`polaris.core.monitoring_loop` — telemetry collection + adaptation cycle
- :mod:`polaris.core.adaptation_pipeline` — assess → validate → execute pipeline
- :mod:`polaris.core.config_reloader` — hot-reload config watching
- :mod:`polaris.core.meta_learning_loop` — autonomous strategy tuning
- :mod:`polaris.core.metrics_export_loop` — periodic metrics file export
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from polaris.core.component_builder import ComponentBuilder
from polaris.core.events import EventBus
from polaris.core.registry import ConnectorRegistry
from polaris.infrastructure.config import PolarisConfig

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


class Polaris:
    """Main Polaris framework class — modular and extensible.

    Simple usage (all defaults)::

        polaris = Polaris(config_path="config.yaml")
        await polaris.run()

    Custom components::

        polaris = Polaris(
            strategy=MyStrategy(),
            world_model=MyWorldModel(),
            meta_learner=MyMetaLearner(),
        )
        await polaris.run()
    """

    def __init__(
        self,
        # Configuration
        config_path: Optional[str] = None,
        config: Optional[PolarisConfig] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        # Core components (swappable)
        strategy: Optional["AdaptationStrategy"] = None,
        world_model: Optional["WorldModel"] = None,
        knowledge_store: Optional["KnowledgeStore"] = None,
        connectors: Optional[List["Connector"]] = None,
        # Meta-learning (optional, off by default)
        meta_learner: Optional["MetaLearner"] = None,
        enable_meta_learning: bool = False,
        # Infrastructure (swappable)
        logger: Optional["Logger"] = None,
        metrics: Optional["MetricsCollector"] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialise Polaris with custom or default components."""
        self.cli_overrides: Dict[str, Any] = cli_overrides or {}

        # ── Configuration ────────────────────────────────────────────────
        if config_path:
            from polaris.infrastructure.config import load_config

            self.config: PolarisConfig = load_config(config_path)
            self._config_path: Optional[str] = config_path
        else:
            self.config = config or PolarisConfig()
            self._config_path = None

        # ── Infrastructure ───────────────────────────────────────────────
        self.logger: "Logger" = logger or ComponentBuilder.build_logger(
            self.config, self.cli_overrides
        )
        self.metrics: "MetricsCollector" = metrics or ComponentBuilder.build_metrics(
            self.config, self.cli_overrides
        )
        self.event_bus: EventBus = event_bus or ComponentBuilder.build_event_bus(
            self.config, self.metrics, self.logger
        )

        # ── Core domain components ───────────────────────────────────────
        self.knowledge_store: "KnowledgeStore" = knowledge_store or (
            ComponentBuilder.build_knowledge_store(self.config, self.logger, self.metrics)
        )
        self.world_model: "WorldModel" = world_model or ComponentBuilder.build_world_model(
            self.config, self.knowledge_store, self.logger, self.metrics
        )

        registry_metrics = (
            self.metrics
            if ComponentBuilder.should_collect(self.config, "registry", self.metrics)
            else None
        )
        self.registry: ConnectorRegistry = ConnectorRegistry(metrics=registry_metrics)
        self._connectors: List["Connector"] = connectors or []

        # ── Strategy ─────────────────────────────────────────────────────
        self.strategy: Optional["AdaptationStrategy"] = strategy
        if not self.strategy and hasattr(self.config, "strategy") and self.config.strategy:
            self.strategy = ComponentBuilder.build_strategy(
                self.config.strategy,
                self.logger,
                self.metrics,
                self.knowledge_store,
                self.world_model,
                self.registry,
                self.config,
            )
        if not self.strategy:
            from polaris.strategies import ThresholdReactiveStrategy

            self.strategy = ThresholdReactiveStrategy()

        # ── Meta-learner ─────────────────────────────────────────────────
        self.meta_learner: Optional["MetaLearner"] = meta_learner
        self._meta_learning_interval_seconds: float = 3600.0
        meta_cfg = getattr(self.config, "meta_learner", None)
        self._meta_learning_transparency_config: Dict[str, Any] = (
            ComponentBuilder.resolve_meta_learning_transparency_config(
                meta_cfg if isinstance(meta_cfg, dict) else None
            )
        )

        if self.meta_learner is None:
            meta_enabled = isinstance(meta_cfg, dict) and bool(meta_cfg.get("enabled", False))

            if enable_meta_learning or meta_enabled:
                self.meta_learner = ComponentBuilder.build_meta_learner(
                    meta_cfg if isinstance(meta_cfg, dict) else None,
                    self.knowledge_store,
                    self.world_model,
                    self.logger,
                    self.metrics,
                    self.config,
                )
                self._meta_learning_interval_seconds = (
                    ComponentBuilder.resolve_meta_learning_interval(
                        meta_cfg if isinstance(meta_cfg, dict) else None
                    )
                )

        # ── Connectors from config ───────────────────────────────────────
        if hasattr(self.config, "systems") and self.config.systems and not connectors:
            self._connectors = ComponentBuilder.build_connectors(
                self.config.systems, self.logger, self.metrics, self.config
            )

        # ── Monitoring interval ──────────────────────────────────────────
        self._monitoring_interval: float = ComponentBuilder.resolve_monitoring_interval(
            self.config, self.cli_overrides, self.logger
        )

        # ── Metrics export config ────────────────────────────────────────
        self._metrics_export_config: Dict[str, Any] = ComponentBuilder.build_metrics_export_config(
            self.config, self.cli_overrides, self.metrics
        )

        # ── Internal state ───────────────────────────────────────────────
        self._running: bool = False
        self._tasks: List[asyncio.Task[Any]] = []

        # ── Log summary ──────────────────────────────────────────────────
        self.logger.info(
            "Polaris components initialized",
            has_strategy=self.strategy is not None,
            has_world_model=self.world_model is not None,
            has_meta_learner=self.meta_learner is not None,
            metrics_enabled=self.metrics is not None,
            monitoring_interval_seconds=self._monitoring_interval,
        )
        if ComponentBuilder.should_collect(self.config, "core_framework", self.metrics):
            self.metrics.increment("polaris.core.initialized")
            self.metrics.gauge(
                "polaris.core.monitoring_interval_seconds", self._monitoring_interval
            )

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the framework and run until stopped."""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting Polaris framework")

        await self.event_bus.start()

        # Connect all configured connectors
        for connector in self._connectors:
            system_id = await connector.get_system_id()
            await self.registry.register(connector)
            connected = await connector.connect()
            if connected:
                self.logger.info(f"Connected to system: {system_id}")
            else:
                self.logger.error(f"Failed to connect to system: {system_id}")

        # Build sub-modules
        from polaris.core.adaptation_pipeline import AdaptationPipeline
        from polaris.core.config_reloader import ConfigReloader
        from polaris.core.meta_learning_loop import MetaLearningLoop
        from polaris.core.metrics_export_loop import MetricsExportLoop
        from polaris.core.monitoring_loop import MonitoringLoop

        config_reloader = ConfigReloader(
            config_path=self._config_path,
            strategy=self.strategy,
            logger=self.logger,
            metrics=self.metrics,
            config=self.config,
        )

        # Allow hot-reload to update meta-learner settings (e.g., auto_apply).
        config_reloader.update_meta_learner(self.meta_learner)

        if self.strategy is not None:
            pipeline = AdaptationPipeline(
                strategy=self.strategy,
                knowledge_store=self.knowledge_store,
                world_model=self.world_model,
                event_bus=self.event_bus,
                logger=self.logger,
                metrics=self.metrics,
                config=self.config,
                dry_run=bool(self.cli_overrides.get("dry_run", False)),
            )

        monitoring = MonitoringLoop(
            registry=self.registry,
            adaptation_pipeline=pipeline,
            config_reloader=config_reloader,
            knowledge_store=self.knowledge_store,
            world_model=self.world_model,
            event_bus=self.event_bus,
            logger=self.logger,
            metrics=self.metrics,
            interval_seconds=self._monitoring_interval,
            config=self.config,
        )

        self._tasks.append(asyncio.create_task(monitoring.run()))

        if self._metrics_export_config.get("enabled", False):
            export_loop = MetricsExportLoop(
                metrics=self.metrics,
                export_config=self._metrics_export_config,
                logger=self.logger,
                config=self.config,
            )
            self._tasks.append(asyncio.create_task(export_loop.run()))

        if self.meta_learner and self.strategy:
            meta_loop = MetaLearningLoop(
                meta_learner=self.meta_learner,
                strategy=self.strategy,
                registry=self.registry,
                logger=self.logger,
                metrics=self.metrics,
                interval_seconds=self._meta_learning_interval_seconds,
                config=self.config,
                transparency_config=self._meta_learning_transparency_config,
            )
            self._tasks.append(asyncio.create_task(meta_loop.run()))

        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop the framework gracefully."""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping Polaris framework")

        if ComponentBuilder.should_collect(self.config, "core_framework", self.metrics):
            self.metrics.increment("polaris.core.stop_called")
            self.metrics.gauge(
                "polaris.core.connectors_at_shutdown", len(self.registry.system_ids())
            )

        # Export final metrics if configured
        if (
            self.metrics
            and hasattr(self.metrics, "export_to_file")
            and self.cli_overrides.get("metrics_export_dir")
        ):
            try:
                from polaris.infrastructure.observability.export import export_polaris_metrics

                exported_files = export_polaris_metrics(
                    metrics_collector=self.metrics,  # type: ignore[arg-type]
                    output_dir=self.cli_overrides["metrics_export_dir"],
                    experiment_name=self.cli_overrides.get("metrics_experiment_name"),
                    formats=self.cli_overrides.get("metrics_export_formats", ["json"]),
                )
                self.logger.info(f"Final metrics exported to {len(exported_files)} files")
            except Exception as e:
                self.logger.error(f"Failed to export final metrics: {e}")

        for connector in self.registry.all():
            await connector.disconnect()

        await self.event_bus.stop()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def register_connector(self, connector: "Connector") -> None:
        """Register a new managed system connector."""
        self._connectors.append(connector)

    def get_knowledge_store(self) -> Optional["KnowledgeStore"]:
        """Access the knowledge store for querying."""
        return self.knowledge_store

    def get_world_model(self) -> Optional["WorldModel"]:
        """Access the world model for insights."""
        return self.world_model

    def is_running(self) -> bool:
        """Return True if the framework is currently running."""
        return self._running

    def export_metrics(self, file_path: str, format: str = "json") -> None:
        """Export collected metrics to a file.

        Args:
            file_path: Destination file path.
            format: ``'json'`` or ``'csv'``.
        """
        if hasattr(self.metrics, "export_to_file"):
            self.metrics.export_to_file(file_path, format)
        else:
            raise NotImplementedError("Metrics collector does not support export")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return the current metrics summary dict."""
        return self.metrics.get_summary()

    # ──────────────────────────────────────────────────────────────────────
    # Async context manager
    # ──────────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "Polaris":
        """Async context manager entry (does not start the run loop)."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit — stops the framework."""
        await self.stop()
