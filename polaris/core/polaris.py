"""Main Polaris framework orchestrator.

The ``Polaris`` class is a thin orchestrator that wires together focused sub-modules:

- :mod:`polaris.core.component_builder` — factory helpers for all components
- :mod:`polaris.core.monitoring_loop` — telemetry collection + adaptation cycle
- :mod:`polaris.core.adaptation_pipeline` — assess → validate → execute pipeline
- :mod:`polaris.core.config_reloader` — hot-reload config watching
- :mod:`polaris.core.meta_learning_loop` — autonomous strategy tuning
- :mod:`polaris.core.metrics_export_loop` — periodic metrics file export
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

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
        SystemContract,
        Verifier,
        WorldModel,
    )


class Polaris:
    """Main Polaris framework class — modular and extensible.

    Simple usage (all defaults)::

    polaris = Polaris(config_path="config.yaml") await polaris.run()

    Custom components::

    polaris = Polaris( strategy=MyStrategy(), world_model=MyWorldModel(),
    meta_learner=MyMetaLearner(), ) await polaris.run()
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
        verifier: Optional["Verifier"] = None,
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

        # Configuration
        if config_path:
            from polaris.infrastructure.config import load_config

            self.config: PolarisConfig = load_config(config_path)
            self._config_path: Optional[str] = config_path
        else:
            self.config = config or PolarisConfig()
            self._config_path = None

        # Infrastructure
        self.logger: "Logger" = logger or ComponentBuilder.build_logger(
            self.config, self.cli_overrides
        )
        self.metrics: "MetricsCollector" = metrics or ComponentBuilder.build_metrics(
            self.config, self.cli_overrides
        )
        self.event_bus: EventBus = event_bus or ComponentBuilder.build_event_bus(
            self.config, self.metrics, self.logger
        )

        # Core domain components
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

        # Multi-system topology graph
        from polaris.core.topology import SystemTopology

        self._topology: SystemTopology = SystemTopology()
        for sys_cfg in getattr(self.config, "systems", []):
            for dep_id in getattr(sys_cfg, "dependencies", []):
                self._topology.add_dependency(sys_cfg.id, dep_id)

        # Cluster safety guardrails
        from polaris.core.safety import SafetyPolicyEngine

        self.safety_engine: Optional[SafetyPolicyEngine] = None
        safety_cfg = getattr(self.config, "safety", None)
        if safety_cfg is not None:
            self.safety_engine = SafetyPolicyEngine(
                config=safety_cfg,
                logger=self.logger,
                metrics=self.metrics,
            )

        # Neuro-symbolic verifier gatekeeper
        from polaris.core.verifier import NeuroSymbolicVerifier

        self.verifier: "Verifier" = verifier or NeuroSymbolicVerifier(
            safety_engine=self.safety_engine,
            logger=self.logger,
            metrics=self.metrics,
        )

        # OpenTelemetry receiver
        self._otel_receiver: Optional[Any] = None
        otel_cfg_data = getattr(self.config, "otel", None)
        if otel_cfg_data:
            from polaris.infrastructure.otel_receiver import (
                OtelReceiverConfig,
                OtelTelemetryReceiver,
            )

            otel_cfg = OtelReceiverConfig.from_dict(otel_cfg_data)
            if otel_cfg.enabled:
                self._otel_receiver = OtelTelemetryReceiver(
                    config=otel_cfg,
                    on_state_received=self.ingest_telemetry,
                    logger=self.logger,
                    metrics=self.metrics,
                )

        # Strategy
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

        # Meta-learner
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

        # Connectors from config
        if hasattr(self.config, "systems") and self.config.systems and not connectors:
            self._connectors = ComponentBuilder.build_connectors(
                self.config.systems, self.logger, self.metrics, self.config
            )

        # Monitoring interval
        self._monitoring_interval: float = ComponentBuilder.resolve_monitoring_interval(
            self.config, self.cli_overrides, self.logger
        )

        # Metrics export config
        self._metrics_export_config: Dict[str, Any] = ComponentBuilder.build_metrics_export_config(
            self.config, self.cli_overrides, self.metrics
        )

        # Internal state
        self._running: bool = False
        self._tasks: List[asyncio.Task[Any]] = []
        self._monitoring_loop: Optional[Any] = None

        # Log summary
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

        # Store system topology in knowledge store
        if self.knowledge_store and hasattr(self.knowledge_store, "store_topology"):
            await self.knowledge_store.store_topology(self._topology)

        # Connect all configured connectors
        from polaris.infrastructure.contract_builder import build_system_contract

        sys_cfg_map = {sc.id: sc for sc in getattr(self.config, "systems", [])}

        for connector in self._connectors:
            system_id = await connector.get_system_id()
            connector_type = type(connector).__name__

            try:
                connected = await connector.connect()
            except Exception as exc:
                connected = False
                self.logger.error(
                    "Connector connection raised exception",
                    system_id=system_id,
                    connector_type=connector_type,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

            if not connected:
                self.logger.warning(
                    "Skipping unavailable connector",
                    system_id=system_id,
                    connector_type=connector_type,
                )
                if ComponentBuilder.should_collect(self.config, "core_framework", self.metrics):
                    self.metrics.increment(
                        "polaris.core.connector_unavailable",
                        tags={"system_id": system_id, "connector_type": connector_type},
                    )
                continue

            try:
                sys_cfg = sys_cfg_map.get(system_id)
                deps = sys_cfg.dependencies if sys_cfg else ()
                contract = await build_system_contract(
                    connector, logger=self.logger, dependencies=deps
                )
                await self.registry.register(connector, contract=contract)
                self.logger.info(
                    "Connected to system",
                    system_id=system_id,
                    connector_type=connector_type,
                )
            except Exception as exc:
                self.logger.error(
                    "Connected connector failed registration and will be dropped",
                    system_id=system_id,
                    connector_type=connector_type,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                try:
                    await connector.disconnect()
                except Exception:
                    # Best effort cleanup if registration fails after connect.
                    pass
                if ComponentBuilder.should_collect(self.config, "core_framework", self.metrics):
                    self.metrics.increment(
                        "polaris.core.connector_registration_errors",
                        tags={"system_id": system_id, "connector_type": connector_type},
                    )

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
            fallback_strategy = None
            fallback_cfg = getattr(self.config.strategy, "fallback", None)
            if fallback_cfg:
                from polaris.infrastructure.config import StrategyConfig

                fallback_obj = (
                    StrategyConfig.model_validate(fallback_cfg)
                    if isinstance(fallback_cfg, dict)
                    else fallback_cfg
                )
                fallback_strategy = ComponentBuilder.build_strategy(
                    fallback_obj,
                    self.logger,
                    self.metrics,
                    self.knowledge_store,
                    self.world_model,
                    self.registry,
                    self.config,
                )

            pipeline = AdaptationPipeline(
                strategy=self.strategy,
                knowledge_store=self.knowledge_store,
                world_model=self.world_model,
                event_bus=self.event_bus,
                logger=self.logger,
                metrics=self.metrics,
                config=self.config,
                dry_run=bool(self.cli_overrides.get("dry_run", False)),
                fallback_strategy=fallback_strategy,
                circuit_breaker_threshold=getattr(
                    self.config.strategy, "circuit_breaker_threshold", 3
                ),
                circuit_breaker_recovery_seconds=getattr(
                    self.config.strategy, "circuit_breaker_recovery_seconds", 60.0
                ),
                topology=self._topology,
                safety_engine=self.safety_engine,
                verifier=self.verifier,
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

        self._monitoring_loop = monitoring
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

        # Start OpenTelemetry receiver if enabled
        if self._otel_receiver and self._otel_receiver.config.enabled:
            await self._otel_receiver.start()

        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            if self._otel_receiver:
                try:
                    await self._otel_receiver.stop()
                except Exception:
                    pass
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = []

    async def stop(self) -> None:
        """Stop the framework gracefully."""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping Polaris framework")

        if self._otel_receiver:
            try:
                await self._otel_receiver.stop()
            except Exception as exc:
                self.logger.error("OTel receiver stop failed", error=str(exc))

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

        disconnect_errors = 0
        for connector in self.registry.all():
            system_id = "unknown"
            try:
                system_id = await connector.get_system_id()
            except Exception:
                pass

            try:
                await connector.disconnect()
            except Exception as exc:
                disconnect_errors += 1
                self.logger.error(
                    "Connector disconnect failed",
                    system_id=system_id,
                    connector_type=type(connector).__name__,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        if disconnect_errors and ComponentBuilder.should_collect(
            self.config, "core_framework", self.metrics
        ):
            self.metrics.increment("polaris.core.disconnect_errors", value=disconnect_errors)

        try:
            await self.event_bus.stop()
        except Exception as exc:
            self.logger.error(
                "Event bus stop failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

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

    @property
    def monitoring_loop(self) -> Optional[Any]:
        """Return the active monitoring loop instance if running."""
        return self._monitoring_loop

    @property
    def topology(self) -> Any:
        """Return the system dependency topology graph."""
        return self._topology

    @property
    def safety_policy_engine(self) -> Optional[Any]:
        """Return the cluster safety guardrail engine if configured."""
        return self.safety_engine

    @property
    def otel_receiver(self) -> Optional[Any]:
        """Return the OpenTelemetry receiver instance if configured."""
        return self._otel_receiver

    async def ingest_telemetry(self, state: Any) -> bool:
        """Ingest push-based telemetry into the running framework.

        Allows external systems (webhooks, OpenTelemetry receivers, streaming pipelines)
        to submit telemetry asynchronously, triggering instant evaluation.

        Args:
            state: Pushed SystemState snapshot.

        Returns:
            True if queued or processed, False otherwise.
        """
        if self._monitoring_loop is not None:
            return bool(await self._monitoring_loop.ingest_telemetry(state))

        # Fallback if loop has not started: persist state and update world model
        if self.knowledge_store:
            try:
                await self.knowledge_store.store_state(state)
            except Exception:
                pass
        if self.world_model:
            try:
                await self.world_model.update(state)
            except Exception:
                pass
        return True

    async def register_system(
        self,
        connector: "Connector",
        contract: Optional["SystemContract"] = None,
        dependencies: Optional[Sequence[str]] = None,
    ) -> bool:
        """Dynamically register a new managed system connector at runtime.

        Connects the connector if needed, resolves or synthesizes its SystemContract,
        registers it in the connector registry, updates the topology graph, and
        synchronizes the updated topology with the knowledge store.

        Args:
            connector: Managed system connector instance.
            contract: Optional explicit SystemContract. If omitted, built via contract builder.
            dependencies: Optional sequence of downstream system IDs this system depends on.

        Returns:
            True if registration succeeded, False otherwise.
        """
        from polaris.infrastructure.contract_builder import build_system_contract

        system_id = await connector.get_system_id()
        connector_type = type(connector).__name__

        # Connect if not connected
        try:
            connected = await connector.connect()
        except Exception as exc:
            self.logger.error(
                "Dynamic connector connection raised exception",
                system_id=system_id,
                connector_type=connector_type,
                error=str(exc),
            )
            connected = False

        if not connected:
            self.logger.warning(
                "Failed to connect dynamic connector",
                system_id=system_id,
                connector_type=connector_type,
            )
            return False

        # Build contract if not provided
        if contract is None:
            try:
                contract = await build_system_contract(
                    connector, logger=self.logger, dependencies=dependencies or ()
                )
            except Exception as exc:
                self.logger.error(
                    "Failed to synthesize contract for dynamic connector",
                    system_id=system_id,
                    error=str(exc),
                )
                return False

        await self.registry.register(connector, contract=contract)

        # Update topology
        self._topology.add_node(system_id, dependencies=dependencies)
        if self.knowledge_store and hasattr(self.knowledge_store, "store_topology"):
            try:
                await self.knowledge_store.store_topology(self._topology)
            except Exception as exc:
                self.logger.debug(
                    "Failed to persist updated topology after dynamic registration",
                    error=str(exc),
                )

        self.logger.info(
            "Dynamically registered system connector",
            system_id=system_id,
            connector_type=connector_type,
            dependencies=list(dependencies or []),
        )
        if ComponentBuilder.should_collect(self.config, "core_framework", self.metrics):
            self.metrics.increment(
                "polaris.core.dynamic_system_registered",
                tags={"system_id": system_id, "connector_type": connector_type},
            )
        return True

    async def unregister_system(
        self,
        system_id: str,
        disconnect: bool = True,
    ) -> bool:
        """Dynamically unregister a managed system connector at runtime.

        Removes the connector and contract from the registry, updates the topology graph,
        and optionally disconnects the connector.

        Args:
            system_id: ID of the system to unregister.
            disconnect: If True, calls connector.disconnect().

        Returns:
            True if system was found and removed, False otherwise.
        """
        connector = self.registry.unregister(system_id)
        if connector is None:
            return False

        if disconnect:
            try:
                await connector.disconnect()
            except Exception as exc:
                self.logger.warning(
                    "Error disconnecting connector during dynamic unregistration",
                    system_id=system_id,
                    error=str(exc),
                )

        self._topology.remove_node(system_id)
        if self._monitoring_loop is not None and hasattr(
            self._monitoring_loop, "unregister_system"
        ):
            self._monitoring_loop.unregister_system(system_id)
        if self.knowledge_store and hasattr(self.knowledge_store, "store_topology"):
            try:
                await self.knowledge_store.store_topology(self._topology)
            except Exception as exc:
                self.logger.debug(
                    "Failed to persist updated topology after unregistration",
                    error=str(exc),
                )

        self.logger.info("Dynamically unregistered system connector", system_id=system_id)
        if ComponentBuilder.should_collect(self.config, "core_framework", self.metrics):
            self.metrics.increment(
                "polaris.core.dynamic_system_unregistered",
                tags={"system_id": system_id},
            )
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Async context manager
    # ──────────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "Polaris":
        """Async context manager entry (does not start the run loop)."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit — stops the framework."""
        await self.stop()
