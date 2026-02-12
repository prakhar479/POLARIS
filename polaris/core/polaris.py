"""Main Polaris framework orchestrator."""

import asyncio
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from polaris.core.events import AdaptationEvent, EventBus, TelemetryEvent
from polaris.core.factories import get_connector_factory, get_strategy_factory
from polaris.core.registry import ConnectorRegistry
from polaris.infrastructure.config import PolarisConfig
from polaris.knowledge import InMemoryKnowledgeStore
from polaris.world_model import StatisticalWorldModel

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


# Runtime imports to avoid circular imports
def _get_meta_learner_class() -> type:
    from polaris.abstractions import MetaLearner

    return MetaLearner


class Polaris:
    """
    Main Polaris framework class - modular and extensible.

    Simple usage (all defaults):
        polaris = Polaris(config_path="config.yaml")
        await polaris.run()

    Custom components:
        polaris = Polaris(
            strategy=MyStrategy(),
            world_model=MyWorldModel(),
            meta_learner=MyMetaLearner()
        )
        await polaris.run()
    """

    def __init__(
        self,
        # Configuration
        config_path: Optional[str] = None,
        config: Optional[PolarisConfig] = None,
        cli_overrides: Optional[dict] = None,
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
    ):
        """Initialize Polaris with custom or default components."""
        # Type declarations
        self._config_path: Optional[str] = None
        self._config_mtime: Optional[float] = None

        # Load configuration if provided
        if config_path:
            from polaris.infrastructure.config import load_config

            loaded_config = load_config(config_path)
            self.config = loaded_config
            self._config_path = config_path
            try:
                self._config_mtime = os.path.getmtime(config_path)
            except Exception:
                self._config_mtime = None
        else:
            self.config = config or PolarisConfig()
            self._config_path = None
            self._config_mtime = None

        # Store CLI overrides
        self.cli_overrides = cli_overrides or {}

        # Set up infrastructure with defaults
        self.logger = logger or self._create_logger_from_config()
        self.metrics = metrics or self._create_metrics_from_config()
        self.event_bus = event_bus or EventBus(
            metrics=self.metrics if self._should_collect_component_metrics("event_bus") else None,
            logger=self.logger,
        )

        # Set up core components with defaults
        knowledge_metrics = (
            self.metrics if self._should_collect_component_metrics("knowledge_store") else None
        )
        self.knowledge_store = knowledge_store or InMemoryKnowledgeStore(
            logger=self.logger,
            metrics=knowledge_metrics,
        )
        world_model_metrics = (
            self.metrics if self._should_collect_component_metrics("world_model") else None
        )
        self.world_model = world_model or StatisticalWorldModel(
            self.knowledge_store, logger=self.logger, metrics=world_model_metrics
        )

        # Set up registry before strategy so factories can depend on it
        registry_metrics = (
            self.metrics if self._should_collect_component_metrics("registry") else None
        )
        self.registry = ConnectorRegistry(metrics=registry_metrics)
        self._connectors = connectors or []

        # Strategy - use provided or create from config or default
        self.strategy = strategy
        if not self.strategy and hasattr(self.config, "strategy") and self.config.strategy:
            self.strategy = self._create_strategy_from_config(self.config.strategy)
        elif not self.strategy:
            # Create default threshold strategy if none provided
            from polaris.strategies import ThresholdReactiveStrategy

            self.strategy = ThresholdReactiveStrategy()

        # Set up meta-learner (explicit instance takes precedence over config)
        self.meta_learner = meta_learner if meta_learner is not None else None

        # Default meta-learning interval (seconds)
        self._meta_learning_interval_seconds: float = 3600.0

        # If no explicit meta-learner was provided, optionally create from config
        if self.meta_learner is None:
            meta_cfg = getattr(self.config, "meta_learner", None)
            meta_enabled = False
            if isinstance(meta_cfg, dict):
                meta_enabled = bool(meta_cfg.get("enabled", False))

            # Backwards-compatible behavior: allow enable_meta_learning flag
            if enable_meta_learning or meta_enabled:
                self.meta_learner = self._create_meta_learner_from_config(meta_cfg)

        # Load connectors from config if available
        if hasattr(self.config, "systems") and self.config.systems and not connectors:
            self._connectors = self._create_connectors_from_config(self.config.systems)

        # Internal state
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Get monitoring interval from config and allow CLI override
        self._monitoring_interval: float = 30.0
        if hasattr(self.config, "monitoring") and self.config.monitoring:
            self._monitoring_interval = self.config.monitoring.get(
                "interval_seconds", self._monitoring_interval
            )
        if "monitoring_interval" in self.cli_overrides:
            self._monitoring_interval = self.cli_overrides["monitoring_interval"]

        # Guard against invalid/non-positive intervals
        try:
            self._monitoring_interval = float(self._monitoring_interval)
        except Exception:
            self.logger.warning(
                f"Invalid monitoring interval {self._monitoring_interval}"
                "falling back to 30 seconds"
            )
            self._monitoring_interval = 30.0

        if self._monitoring_interval <= 0:
            self.logger.warning(
                f"Non-positive monitoring interval {self._monitoring_interval}"
                "falling back to 30 seconds"
            )
            self._monitoring_interval = 30.0

        # Setup metrics configuration
        self._setup_metrics_auto_export()

        # Log component configuration summary
        self.logger.info(
            "Polaris components initialized",
            has_strategy=self.strategy is not None,
            has_world_model=self.world_model is not None,
            has_meta_learner=self.meta_learner is not None,
            metrics_enabled=self.metrics is not None,
            monitoring_interval_seconds=self._monitoring_interval,
        )

        if self.metrics and self._should_collect_component_metrics("core_framework"):
            self.metrics.increment("polaris.core.initialized")
            self.metrics.gauge(
                "polaris.core.monitoring_interval_seconds", self._monitoring_interval
            )

    def _create_logger_from_config(self) -> "Logger":
        """Create logger from configuration with CLI overrides."""
        from polaris.infrastructure.observability.logger import create_logger

        # Default values
        logger_type = "structured"
        level = "INFO"
        console = True
        log_file = None
        use_colors = True

        # Get values from config
        if hasattr(self.config, "observability") and self.config.observability:
            logging_config = self.config.observability.get("logging", {})
            logger_type = logging_config.get("type", logger_type)
            level = logging_config.get("level", level)
            console = logging_config.get("console", console)
            use_colors = logging_config.get("use_colors", use_colors)

            if logging_config.get("file", False):
                log_file = logging_config.get("file_path", "./logs/polaris.log")

        # Apply CLI overrides
        if "log_format" in self.cli_overrides:
            logger_type = self.cli_overrides["log_format"]
        if "log_level" in self.cli_overrides:
            level = self.cli_overrides["log_level"]
        # Allow CLI to disable console logging (e.g. for dashboard mode)
        if "console_logging" in self.cli_overrides:
            console = bool(self.cli_overrides["console_logging"])
        if "log_file" in self.cli_overrides:
            log_file = self.cli_overrides["log_file"]

        return create_logger(
            logger_type=logger_type,
            name="polaris",
            level=level,
            log_file=log_file,
            console=console,
            use_colors=use_colors,
        )

    def _create_metrics_from_config(self) -> Optional["MetricsCollector"]:
        """Create metrics collector from configuration with CLI overrides."""
        # Check if metrics are disabled
        if self.cli_overrides.get("metrics_enabled", True) is False:
            return None

        # Get metrics config
        metrics_config = {}
        if hasattr(self.config, "observability") and self.config.observability:
            metrics_config = self.config.observability.get("metrics", {})

        # Check if metrics are enabled in config
        if not metrics_config.get("enabled", True):
            return None

        # Create metrics collector based on type
        collector_type = metrics_config.get("collector_type", "simple")

        if collector_type == "simple":
            from polaris.infrastructure.observability.metrics import SimpleMetricsCollector

            # Get simple collector settings
            # Note: simple_config not used as SimpleMetricsCollector doesn't support configuration

            collector = SimpleMetricsCollector()
            # Note: histogram_max_values configuration not supported by SimpleMetricsCollector

            return collector
        else:
            # For other collector types (prometheus, datadog, etc.)
            # Return simple collector as fallback
            from polaris.infrastructure.observability.metrics import SimpleMetricsCollector

            return SimpleMetricsCollector()

    def _should_collect_component_metrics(self, component_name: str) -> bool:
        """Check if metrics should be collected for a specific component."""
        if not self.metrics:
            return False

        # Get component metrics config
        metrics_config = {}
        if hasattr(self.config, "observability") and self.config.observability:
            metrics_config = self.config.observability.get("metrics", {})

        components_config = metrics_config.get("components", {})
        return bool(components_config.get(component_name, True))

    def _setup_metrics_auto_export(self) -> None:
        """Set up automatic metrics export if configured."""
        if not self.metrics or not hasattr(self.metrics, "export_to_file"):
            return

        # Get export config
        export_config = {}
        if hasattr(self.config, "observability") and self.config.observability:
            metrics_config = self.config.observability.get("metrics", {})
            export_config = metrics_config.get("export", {})

        # Check CLI overrides
        export_enabled = export_config.get("enabled", False)
        export_dir = self.cli_overrides.get("metrics_export_dir") or export_config.get(
            "output_dir", "./metrics"
        )
        auto_interval = self.cli_overrides.get("metrics_auto_export_interval")
        if auto_interval is None:
            auto_interval = export_config.get("auto_export_interval_minutes", 0)

        # Setup auto-export if enabled and interval > 0
        if export_enabled and auto_interval is not None and auto_interval > 0:
            self._metrics_export_config = {
                "enabled": True,
                "interval_minutes": auto_interval,
                "output_dir": export_dir,
                "formats": self.cli_overrides.get("metrics_export_formats")
                or export_config.get("formats", ["json"]),
                "experiment_name": self.cli_overrides.get("metrics_experiment_name")
                or export_config.get("experiment_name"),
                "include_timestamp": export_config.get("include_timestamp", True),
            }
        else:
            self._metrics_export_config = {"enabled": False}

    def _create_strategy_from_config(self, strategy_config: Any) -> Optional["AdaptationStrategy"]:
        """Create strategy from configuration."""
        strategy_metrics = (
            self.metrics if self._should_collect_component_metrics("strategy") else None
        )

        factory = get_strategy_factory(strategy_config.type)
        if not factory:
            self.logger.warning(
                f"No strategy factory registered for type '{strategy_config.type}'"
                "using threshold strategy instead"
            )
            from polaris.strategies import ThresholdReactiveStrategy

            return ThresholdReactiveStrategy(logger=self.logger, metrics=strategy_metrics)

        try:
            return factory(
                strategy_config,
                self.logger,
                strategy_metrics,
                self.knowledge_store,
                self.world_model,
                self.registry,
            )
        except Exception as e:
            from polaris.strategies import ThresholdReactiveStrategy

            self.logger.warning(
                f"Failed to initialize strategy of type '{strategy_config.type}'"
                f"from config: {e}. Falling back to threshold.",
            )
            return ThresholdReactiveStrategy(logger=self.logger, metrics=strategy_metrics)

    def _create_meta_learner_from_config(
        self, meta_config: Optional[Dict[str, Any]]
    ) -> Optional["MetaLearner"]:
        """Create meta-learner from configuration.

        Supports both statistical and LLM-based meta-learners. When an
        analysis_interval_hours value is provided, it is used to configure
        the background meta-learning loop interval.
        """
        if not isinstance(meta_config, dict):
            return None

        # Configure meta-learning loop interval in seconds (default 1 hour)
        try:
            interval_hours = float(meta_config.get("analysis_interval_hours", 1.0))
            if interval_hours > 0:
                self._meta_learning_interval_seconds = interval_hours * 3600.0
        except Exception:
            # Keep default on any parsing issue
            self._meta_learning_interval_seconds = 3600.0

        meta_type = meta_config.get("type", "statistical")

        # Reuse metrics component flag for meta-learner
        meta_metrics = (
            self.metrics if self._should_collect_component_metrics("meta_learner") else None
        )

        if meta_type == "statistical":
            from polaris.meta_learner import StatisticalMetaLearner
            from polaris.meta_learner.bayesian_optimizer import AcquisitionFunction

            # Get statistical configuration
            stat_cfg = meta_config.get("statistical", {}) or {}
            conservative_mode = bool(stat_cfg.get("conservative_mode", True))
            enable_bayesian = bool(stat_cfg.get("enable_bayesian_optimization", True))
            min_samples = int(stat_cfg.get("min_samples_for_optimization", 10))

            # Get acquisition function
            acq_func_str = stat_cfg.get("acquisition_function", "expected_improvement")
            try:
                acquisition_function = AcquisitionFunction(acq_func_str)
            except ValueError:
                self.logger.warning(f"Unknown acquisition function '{acq_func_str}', using default")
                acquisition_function = AcquisitionFunction.EXPECTED_IMPROVEMENT

            exploration_weight = float(stat_cfg.get("exploration_weight", 0.1))

            return StatisticalMetaLearner(
                knowledge_store=self.knowledge_store,
                logger=self.logger,
                conservative_mode=conservative_mode,
                world_model=self.world_model,
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

                temperature = float(llm_cfg.get("temperature", 0.1))
                auto_apply = bool(llm_cfg.get("auto_apply", False))

                analysis_prompt = llm_cfg.get("analysis_system_prompt")
                optimization_prompt = llm_cfg.get("optimization_system_prompt")
                per_system_prompts = llm_cfg.get("per_system_prompts")

                return LLMMetaLearner(
                    llm_client=llm_client,
                    knowledge_store=self.knowledge_store,
                    logger=self.logger,
                    auto_apply=auto_apply,
                    temperature=temperature,
                    analysis_system_prompt=analysis_prompt,
                    optimization_system_prompt=optimization_prompt,
                    per_system_prompts=per_system_prompts,
                    metrics=meta_metrics,
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM meta-learner from config: {e}")
                return None

        self.logger.warning(
            f"Unknown meta_learner type '{meta_type}'. Meta-learning will be disabled."
        )
        return None

    def _create_connectors_from_config(self, systems_config: Any) -> List[Any]:
        """Create connectors from configuration."""
        connectors = []
        for system in systems_config:
            if not system.enabled:
                continue

            factory = get_connector_factory(system.connector_type)
            if not factory:
                self.logger.error(
                    f"No connector factory registered for type '{system.connector_type}'"
                    f"skipping system '{system.id}'"
                )
                continue

            connector_metrics = (
                self.metrics if self._should_collect_component_metrics("connectors") else None
            )
            try:
                connector = factory(system, self.logger, connector_metrics)
            except Exception as e:
                self.logger.error(
                    f"Failed to create connector for system '{system.id}' (type '{system.connector_type}'): {e}"
                )
                continue

            connectors.append(connector)

        return connectors

    async def run(self) -> None:
        """Start the framework and run until stopped."""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting Polaris framework")

        # Start event bus
        await self.event_bus.start()

        # Register and connect to all systems
        for connector in self._connectors:
            system_id = await connector.get_system_id()
            await self.registry.register(connector)
            connected = await connector.connect()

            if connected:
                self.logger.info(f"Connected to system: {system_id}")
            else:
                self.logger.error(f"Failed to connect to system: {system_id}")

        # Start monitoring loop
        self._tasks.append(asyncio.create_task(self._monitoring_loop()))

        # Start metrics auto-export if configured
        if self._metrics_export_config.get("enabled", False):
            self._tasks.append(asyncio.create_task(self._metrics_export_loop()))

        # Start meta-learner if enabled
        if self.meta_learner and self.strategy:
            self._tasks.append(asyncio.create_task(self._meta_learning_loop()))

        # Wait for shutdown
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel all tasks gracefully
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete cancellation
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _monitoring_loop(self) -> None:
        """Run main monitoring and adaptation loop."""
        self.logger.info("Starting monitoring loop")
        if self.metrics and self._should_collect_component_metrics("core_framework"):
            self.metrics.increment("polaris.monitoring.started")

        while self._running:
            try:
                await self._maybe_hot_reload_config()
                loop_start = datetime.now(timezone.utc)
                systems_processed = 0
                adaptations_executed = 0

                for connector in self.registry.all():
                    result = await self._process_system_iteration(connector)
                    systems_processed += result["systems_processed"]
                    adaptations_executed += result["adaptations_executed"]

                self._record_monitoring_loop_metrics(
                    loop_start,
                    systems_processed,
                    adaptations_executed,
                )

                # Aim for approximately self._monitoring_interval seconds between
                # the *starts* of successive iterations.
                loop_duration = (datetime.now(timezone.utc) - loop_start).total_seconds()
                sleep_for = max(0.0, float(self._monitoring_interval) - loop_duration)
                await asyncio.sleep(sleep_for)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if self.metrics and self._should_collect_component_metrics("monitoring_loop"):
                    self.metrics.increment("polaris.monitoring.loop_errors")
                await asyncio.sleep(float(self._monitoring_interval))

        self.logger.info("Monitoring loop stopped")
        if self.metrics and self._should_collect_component_metrics("core_framework"):
            self.metrics.increment("polaris.monitoring.stopped")

    async def _process_system_iteration(self, connector: "Connector") -> Dict[str, int]:
        systems_processed = 0
        adaptations_executed = 0

        try:
            state = await connector.collect_telemetry()
            systems_processed = 1

            if self.metrics and self._should_collect_component_metrics("monitoring_loop"):
                self.metrics.increment(
                    "polaris.telemetry.collected",
                    tags={"system_id": state.system_id},
                )

            if self.knowledge_store:
                await self.knowledge_store.store_state(state)
                if self.metrics and self._should_collect_component_metrics("knowledge_store"):
                    self.metrics.increment(
                        "polaris.knowledge.state_stored",
                        tags={"system_id": state.system_id},
                    )

            if self.world_model:
                await self.world_model.update(state)
                if self.metrics and self._should_collect_component_metrics("world_model"):
                    self.metrics.increment(
                        "polaris.world_model.updated",
                        tags={"system_id": state.system_id},
                    )

            await self.event_bus.publish(
                TelemetryEvent(
                    system_id=state.system_id,
                    state=state,
                    timestamp=state.timestamp,
                )
            )
            if self.metrics and self._should_collect_component_metrics("event_bus"):
                self.metrics.increment(
                    "polaris.events.telemetry_published",
                    tags={"system_id": state.system_id},
                )

            if self.strategy:
                from polaris.abstractions.strategy import AdaptationContext

                context = AdaptationContext(
                    system_id=state.system_id,
                    historical_states=[],
                    world_model_insights=await self.world_model.get_insights()
                    if self.world_model
                    else None,
                )

                action = await self.strategy.assess(state, context)
                if self.metrics and self._should_collect_component_metrics("strategy"):
                    self.metrics.increment(
                        "polaris.strategy.assessments",
                        tags={"system_id": state.system_id},
                    )

                if action:
                    self.logger.info(
                        f"Adaptation proposed for {state.system_id}: {action.action_type}",
                        action_id=action.action_id,
                    )
                    if self.metrics and self._should_collect_component_metrics("core_framework"):
                        self.metrics.increment(
                            "polaris.adaptations.proposed",
                            tags={
                                "system_id": state.system_id,
                                "action_type": action.action_type,
                            },
                        )

                    if await connector.validate_action(action):
                        result = await connector.execute_action(action)
                        adaptations_executed = 1

                        if self.metrics and self._should_collect_component_metrics(
                            "core_framework"
                        ):
                            self.metrics.increment(
                                "polaris.adaptations.executed",
                                tags={
                                    "system_id": state.system_id,
                                    "action_type": action.action_type,
                                    "status": result.status.value,
                                },
                            )

                        if self.knowledge_store:
                            await self.knowledge_store.store_action(action, result)

                        await self.strategy.on_action_executed(action, result)

                        await self.event_bus.publish(
                            AdaptationEvent(
                                action=action,
                                result=result,
                                timestamp=result.completed_at or datetime.now(timezone.utc),
                            )
                        )
                        if self.metrics and self._should_collect_component_metrics("event_bus"):
                            self.metrics.increment(
                                "polaris.events.adaptation_published",
                                tags={"system_id": state.system_id},
                            )

                        self.logger.info(
                            f"Adaptation executed: {action.action_type} -> {result.status.value}",
                            action_id=action.action_id,
                        )
                    else:
                        self.logger.warning(
                            f"Action validation failed for {action.action_type}",
                            action_id=action.action_id,
                        )
                        if self.metrics and self._should_collect_component_metrics(
                            "core_framework"
                        ):
                            self.metrics.increment(
                                "polaris.adaptations.validation_errors",
                                tags={
                                    "system_id": state.system_id,
                                    "action_type": action.action_type,
                                },
                            )
        except Exception as e:
            system_id = await connector.get_system_id()
            self.logger.error(f"Error monitoring system {system_id}: {e}")
            if self.metrics and self._should_collect_component_metrics("monitoring_loop"):
                self.metrics.increment(
                    "polaris.monitoring.errors",
                    tags={"system_id": system_id},
                )

        return {
            "systems_processed": systems_processed,
            "adaptations_executed": adaptations_executed,
        }

    def _record_monitoring_loop_metrics(
        self,
        loop_start: datetime,
        systems_processed: int,
        adaptations_executed: int,
    ) -> None:
        if not self.metrics or not self._should_collect_component_metrics("monitoring_loop"):
            return

        loop_duration = (datetime.now(timezone.utc) - loop_start).total_seconds()
        self.metrics.histogram(
            "polaris.monitoring.loop_duration_seconds",
            loop_duration,
        )
        self.metrics.gauge(
            "polaris.monitoring.systems_processed",
            systems_processed,
        )
        self.metrics.gauge(
            "polaris.monitoring.adaptations_executed",
            adaptations_executed,
        )
        self.metrics.gauge(
            "polaris.monitoring.last_iteration_timestamp",
            datetime.now(timezone.utc).timestamp(),
        )

    async def _maybe_hot_reload_config(self) -> None:
        """Check for config changes and apply strategy/resilience updates."""
        if not self._config_path:
            return
        try:
            mtime = os.path.getmtime(self._config_path)
        except Exception:
            return
        if self._config_mtime is not None and mtime <= self._config_mtime:
            return
        # Reload and apply
        if self.metrics and self._should_collect_component_metrics("core_framework"):
            self.metrics.increment("polaris.config.hot_reload.attempts")
        try:
            from polaris.infrastructure.config import load_config

            new_conf = load_config(self._config_path)
            await self._apply_strategy_hot_reload(new_conf.strategy)
            # update stored
            self.config = new_conf
            self._config_mtime = mtime
            if self.metrics and self._should_collect_component_metrics("core_framework"):
                self.metrics.increment("polaris.config.hot_reload.success")
            self.logger.info("Applied hot-reload from updated configuration")
        except Exception as e:
            if self.metrics and self._should_collect_component_metrics("core_framework"):
                self.metrics.increment("polaris.config.hot_reload.errors")
            self.logger.warning(f"Hot-reload skipped due to error: {e}")

    async def _apply_strategy_hot_reload(self, strategy_config: Any) -> None:
        """Apply parameter updates for current strategy from new config."""
        if not self.strategy or not strategy_config:
            return
        # If strategy type differs, skip (avoid disruptive replacement)
        current_type = type(self.strategy).__name__
        desired_type = strategy_config.type
        if desired_type == "threshold" and current_type != "ThresholdReactiveStrategy":
            self.logger.info("Strategy type changed in config; restart required to apply.")
            return
        if desired_type == "llm_reasoning" and current_type != "LLMReasoningStrategy":
            self.logger.info("Strategy type changed in config; restart required to apply.")
            return
        if desired_type == "hybrid" and current_type != "HybridStrategy":
            self.logger.info("Strategy type changed in config; restart required to apply.")
            return
        if desired_type == "agentic_llm" and current_type != "AgenticLLMStrategy":
            self.logger.info("Strategy type changed in config; restart required to apply.")
            return

        # Build a type-specific configuration payload and delegate to the strategy
        config_payload: Dict[str, Any]
        if desired_type == "threshold":
            config_payload = strategy_config.threshold or {}
        elif desired_type == "llm_reasoning":
            config_payload = strategy_config.llm or {}
        elif desired_type == "hybrid":
            config_payload = strategy_config.hybrid or {}
        elif desired_type == "agentic_llm":
            config_payload = strategy_config.agentic or {}
        else:
            config_payload = {}

        try:
            await self.strategy.apply_config_update(config_payload)
        except Exception as e:
            self.logger.warning(f"Failed to apply strategy config update: {e}")

    async def _meta_learning_loop(self) -> None:
        """Meta-learning loop for autonomous optimization."""
        self.logger.info("Starting meta-learning loop")
        if self.metrics and self._should_collect_component_metrics("meta_learner"):
            self.metrics.increment("polaris.meta_learning.started")

        meta_learner_config = getattr(self.config, "meta_learner", {})
        if not meta_learner_config:
            self.logger.warning("Meta-learner config not found; skipping meta-learning loop")
            return
        meta_loop_interval = meta_learner_config["analysis_interval_hours"]
        if self.meta_learner is None:
            self.logger.warning("Meta-learner not initialized; skipping meta-learning loop")
            return

        while self._running:
            try:
                await asyncio.sleep(meta_loop_interval * 3600)

                if not self.strategy:
                    continue

                # Analyze performance for each system
                for system_id in self.registry.system_ids():
                    try:
                        analysis = await self.meta_learner.analyze_performance(system_id)
                        if self.metrics and self._should_collect_component_metrics("meta_learner"):
                            self.metrics.increment(
                                "polaris.meta_learning.analysis_completed",
                                tags={"system_id": system_id},
                            )

                        # Get proposals
                        proposals = await self.meta_learner.propose_strategy_updates(
                            self.strategy, analysis
                        )
                        if self.metrics and self._should_collect_component_metrics("meta_learner"):
                            self.metrics.gauge(
                                "polaris.meta_learning.proposals_generated",
                                len(proposals),
                                tags={"system_id": system_id},
                            )

                        if proposals:
                            # Validate proposals
                            validated = await self.meta_learner.validate_proposals(proposals)
                            if self.metrics and self._should_collect_component_metrics(
                                "meta_learner"
                            ):
                                self.metrics.gauge(
                                    "polaris.meta_learning.proposals_validated",
                                    len(validated),
                                    tags={"system_id": system_id},
                                )

                            # Apply approved proposals
                            applied = await self.meta_learner.apply_proposals(
                                self.strategy, validated
                            )
                            if self.metrics and self._should_collect_component_metrics(
                                "meta_learner"
                            ):
                                self.metrics.gauge(
                                    "polaris.meta_learning.proposals_applied",
                                    len(applied),
                                    tags={"system_id": system_id},
                                )

                            self.logger.info(
                                f"Meta-learner applied {len(applied)} parameter updates"
                            )

                    except Exception as e:
                        self.logger.error(f"Error in meta-learning for {system_id}: {e}")
                        if self.metrics and self._should_collect_component_metrics("meta_learner"):
                            self.metrics.increment(
                                "polaris.meta_learning.errors", tags={"system_id": system_id}
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in meta-learning loop: {e}")
                if self.metrics and self._should_collect_component_metrics("meta_learner"):
                    self.metrics.increment("polaris.meta_learning.loop_errors")

        self.logger.info("Meta-learning loop stopped")
        if self.metrics and self._should_collect_component_metrics("meta_learner"):
            self.metrics.increment("polaris.meta_learning.stopped")

    async def _metrics_export_loop(self) -> None:
        """Auto-export metrics loop."""
        if not self.metrics or not hasattr(self.metrics, "export_to_file"):
            return

        export_config = self._metrics_export_config
        interval_seconds = export_config["interval_minutes"] * 60

        self.logger.info(
            f"Starting metrics auto-export every {export_config['interval_minutes']} minutes"
        )
        if self.metrics and self._should_collect_component_metrics("core_framework"):
            self.metrics.increment("polaris.metrics.auto_export_started")

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)

                # Check if we should continue running
                if not self._running:
                    break  # type: ignore[unreachable]

                # Export metrics
                from polaris.infrastructure.observability.export import export_polaris_metrics

                try:
                    export_start = datetime.now(timezone.utc)
                    exported_files = export_polaris_metrics(
                        metrics_collector=self.metrics,  # type: ignore[arg-type]
                        output_dir=export_config["output_dir"],
                        experiment_name=export_config.get("experiment_name"),
                        formats=export_config["formats"],
                    )

                    self.logger.info(f"Auto-exported metrics to {len(exported_files)} files")
                    if self.metrics and self._should_collect_component_metrics("core_framework"):
                        self.metrics.increment("polaris.metrics.auto_exports_completed")
                        export_duration = (
                            datetime.now(timezone.utc) - export_start
                        ).total_seconds()
                        self.metrics.histogram(
                            "polaris.metrics.auto_export_duration_seconds", export_duration
                        )

                except Exception as e:
                    self.logger.error(f"Failed to auto-export metrics: {e}")
                    if self.metrics and self._should_collect_component_metrics("core_framework"):
                        self.metrics.increment("polaris.metrics.auto_export_errors")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics export loop: {e}")
                if self.metrics and self._should_collect_component_metrics("core_framework"):
                    self.metrics.increment("polaris.metrics.export_loop_errors")

        self.logger.info("Metrics auto-export loop stopped")

    async def stop(self) -> None:
        """Stop the framework gracefully."""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping Polaris framework")

        if self.metrics and self._should_collect_component_metrics("core_framework"):
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

        # Disconnect connectors
        for connector in self.registry.all():
            await connector.disconnect()

        # Stop event bus
        await self.event_bus.stop()

    def register_connector(self, connector: "Connector") -> None:
        """Register a new managed system connector."""
        self._connectors.append(connector)

    def get_knowledge_store(self) -> Optional["KnowledgeStore"]:
        """Access knowledge store for querying."""
        return self.knowledge_store

    def get_world_model(self) -> Optional["WorldModel"]:
        """Access world model for insights."""
        return self.world_model

    def is_running(self) -> bool:
        """Check if framework is running."""
        return self._running

    def export_metrics(self, file_path: str, format: str = "json") -> None:
        """
        Export collected metrics to file.

        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        if self.metrics and hasattr(self.metrics, "export_to_file"):
            self.metrics.export_to_file(file_path, format)
        else:
            raise NotImplementedError("Metrics collector does not support export")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if self.metrics:
            return self.metrics.get_summary()
        return {}

    async def __aenter__(self) -> "Polaris":
        """Async context manager entry."""
        # Don't call run() here as it blocks
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
