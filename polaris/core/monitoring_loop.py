"""Main monitoring and adaptation loop.

Extracted from ``Polaris._monitoring_loop`` and ``Polaris._process_system_iteration`` so
the per-cycle logic can be tested independently of the Polaris orchestrator.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    from polaris.abstractions import Connector, KnowledgeStore, Logger, MetricsCollector, WorldModel
    from polaris.core.adaptation_pipeline import AdaptationPipeline
    from polaris.core.config_reloader import ConfigReloader
    from polaris.core.events import EventBus
    from polaris.core.registry import ConnectorRegistry
    from polaris.infrastructure.config import PolarisConfig


class MonitoringLoop:
    """Runs the main monitoring and adaptation cycle.

    Each iteration:

    1. Optionally hot-reloads the config via :class:`ConfigReloader`. 2. Iterates over
    all registered connectors. 3. Collects telemetry, stores state, updates the world
    model, publishes a ``TelemetryEvent``, then delegates to
    :class:`AdaptationPipeline`. 4. Records loop-level metrics and sleeps for the
    remainder of the interval.
    """

    def __init__(
        self,
        registry: "ConnectorRegistry",
        adaptation_pipeline: "AdaptationPipeline",
        config_reloader: "ConfigReloader",
        knowledge_store: Optional["KnowledgeStore"],
        world_model: Optional["WorldModel"],
        event_bus: "EventBus",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        interval_seconds: float,
        config: "PolarisConfig",
    ) -> None:
        """Initialize the monitoring loop."""
        self._registry = registry
        self._pipeline = adaptation_pipeline
        self._reloader = config_reloader
        self._knowledge_store = knowledge_store
        self._world_model = world_model
        self._event_bus = event_bus
        self._logger = logger
        self._metrics = metrics
        self._interval = interval_seconds
        self._config = config
        self._running = False

    async def run(self) -> None:
        """Run the monitoring loop until cancelled."""
        self._running = True
        self._logger.info("Starting monitoring loop")
        self._emit("polaris.monitoring.started", component="core_framework")

        # Concurrency cap: process at most this many connectors in parallel.
        max_concurrent = getattr(self._config, "max_concurrent_connectors", 10)
        semaphore = asyncio.Semaphore(max_concurrent)
        error_backoff = 5.0  # initial backoff on loop-level errors

        while self._running:
            try:
                new_config = await self._reloader.maybe_reload()
                if new_config is not None:
                    self._config = new_config

                loop_start = datetime.now(timezone.utc)

                connectors = list(self._registry.all())

                async def _bounded(connector: "Connector") -> Dict[str, int]:
                    async with semaphore:
                        return await self._process_system(connector)

                results: List[Union[Dict[str, int], BaseException]] = await asyncio.gather(
                    *[_bounded(c) for c in connectors], return_exceptions=True
                )

                systems_processed = 0
                adaptations_executed = 0
                for r in results:
                    if isinstance(r, dict):
                        systems_processed += r["systems_processed"]
                        adaptations_executed += r["adaptations_executed"]

                self._record_loop_metrics(loop_start, systems_processed, adaptations_executed)
                error_backoff = 5.0  # reset on success

                loop_duration = (datetime.now(timezone.utc) - loop_start).total_seconds()
                sleep_for = max(0.0, float(self._interval) - loop_duration)
                await asyncio.sleep(sleep_for)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}")
                self._emit("polaris.monitoring.loop_errors", component="monitoring_loop")
                # Exponential backoff capped at the monitoring interval.
                await asyncio.sleep(error_backoff)
                error_backoff = min(error_backoff * 2, float(self._interval))

        self._logger.info("Monitoring loop stopped")
        self._emit("polaris.monitoring.stopped", component="core_framework")

    async def _process_system(self, connector: "Connector") -> Dict[str, int]:
        """Run one monitoring + adaptation cycle for a single connector."""
        from polaris.core.events import TelemetryEvent

        systems_processed = 0
        adaptations_executed = 0

        try:
            state = await connector.collect_telemetry()
            systems_processed = 1
            self._emit_tagged(
                "polaris.telemetry.collected",
                state.system_id,
                component="monitoring_loop",
            )

            if self._knowledge_store:
                await self._knowledge_store.store_state(state)
                self._emit_tagged(
                    "polaris.knowledge.state_stored",
                    state.system_id,
                    component="knowledge_store",
                )

            if self._world_model:
                await self._world_model.update(state)
                self._emit_tagged(
                    "polaris.world_model.updated",
                    state.system_id,
                    component="world_model",
                )

            await self._event_bus.publish(
                TelemetryEvent(
                    system_id=state.system_id,
                    state=state,
                    timestamp=state.timestamp,
                )
            )
            self._emit_tagged(
                "polaris.events.telemetry_published",
                state.system_id,
                component="event_bus",
            )

            system_contract = self._registry.get_contract(state.system_id)
            executed = await self._pipeline.run(
                state,
                connector,
                system_contract=system_contract,
            )
            if executed:
                adaptations_executed = 1

        except Exception as e:
            system_id = await connector.get_system_id()
            self._logger.error(f"Error monitoring system {system_id}: {e}")
            self._emit_tagged(
                "polaris.monitoring.errors",
                system_id,
                component="monitoring_loop",
            )

        return {
            "systems_processed": systems_processed,
            "adaptations_executed": adaptations_executed,
        }

    def _record_loop_metrics(
        self,
        loop_start: datetime,
        systems_processed: int,
        adaptations_executed: int,
    ) -> None:
        if not self._metrics or not self._should_collect("monitoring_loop"):
            return

        loop_duration = (datetime.now(timezone.utc) - loop_start).total_seconds()
        self._metrics.histogram("polaris.monitoring.loop_duration_seconds", loop_duration)
        self._metrics.gauge("polaris.monitoring.systems_processed", systems_processed)
        self._metrics.gauge("polaris.monitoring.adaptations_executed", adaptations_executed)
        self._metrics.gauge(
            "polaris.monitoring.last_iteration_timestamp",
            datetime.now(timezone.utc).timestamp(),
        )

    # Metric helpers

    def _should_collect(self, component: str) -> bool:
        from polaris.core.component_builder import ComponentBuilder

        return ComponentBuilder.should_collect(self._config, component, self._metrics)

    def _emit(self, metric: str, component: str) -> None:
        if self._metrics and self._should_collect(component):
            self._metrics.increment(metric)

    def _emit_tagged(self, metric: str, system_id: str, component: str) -> None:
        if self._metrics and self._should_collect(component):
            self._metrics.increment(metric, tags={"system_id": system_id})
