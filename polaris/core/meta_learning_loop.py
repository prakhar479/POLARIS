"""Meta-learning background loop.

Extracted from ``Polaris._meta_learning_loop`` so the meta-learning cycle
can be tested and reused independently of the monitoring loop.
"""

import asyncio
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from polaris.abstractions import AdaptationStrategy, Logger, MetaLearner, MetricsCollector
    from polaris.core.registry import ConnectorRegistry
    from polaris.infrastructure.config import PolarisConfig


class MetaLearningLoop:
    """Background loop that periodically analyses performance and tunes the strategy.

    The loop sleeps for ``interval_seconds``, then for each registered system:

    1. Calls ``meta_learner.analyze_performance(system_id)``.
    2. Calls ``meta_learner.propose_strategy_updates(strategy, analysis)``.
    3. Validates and applies approved proposals via the meta-learner.
    """

    def __init__(
        self,
        meta_learner: "MetaLearner",
        strategy: "AdaptationStrategy",
        registry: "ConnectorRegistry",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        interval_seconds: float,
        config: "PolarisConfig",
    ) -> None:
        """Initialize the meta-learning loop."""
        self._meta_learner = meta_learner
        self._strategy = strategy
        self._registry = registry
        self._logger = logger
        self._metrics = metrics
        self._interval_seconds = interval_seconds
        self._config = config
        self._running = False

    async def run(self) -> None:
        """Run the meta-learning loop until cancelled."""
        self._running = True
        self._logger.info("Starting meta-learning loop")
        self._emit("polaris.meta_learning.started")

        while self._running:
            try:
                await asyncio.sleep(self._interval_seconds)

                for system_id in self._registry.system_ids():
                    await self._run_for_system(system_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in meta-learning loop: {e}")
                self._emit("polaris.meta_learning.loop_errors")

        self._logger.info("Meta-learning loop stopped")
        self._emit("polaris.meta_learning.stopped")

    async def _run_for_system(self, system_id: str) -> None:
        """Run one meta-learning cycle for a single system."""
        try:
            analysis = await self._meta_learner.analyze_performance(system_id)
            self._emit_tagged("polaris.meta_learning.analysis_completed", system_id)

            proposals = await self._meta_learner.propose_strategy_updates(self._strategy, analysis)
            self._gauge_tagged(
                "polaris.meta_learning.proposals_generated", len(proposals), system_id
            )

            if not proposals:
                return

            validated = await self._meta_learner.validate_proposals(proposals)
            self._gauge_tagged(
                "polaris.meta_learning.proposals_validated", len(validated), system_id
            )

            applied = await self._meta_learner.apply_proposals(self._strategy, validated)
            self._gauge_tagged("polaris.meta_learning.proposals_applied", len(applied), system_id)

            self._logger.info(f"Meta-learner applied {len(applied)} parameter updates")

        except Exception as e:
            self._logger.error(f"Error in meta-learning for {system_id}: {e}")
            self._emit_tagged("polaris.meta_learning.errors", system_id)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _should_collect(self) -> bool:
        from polaris.core.component_builder import ComponentBuilder

        return ComponentBuilder.should_collect(self._config, "meta_learner", self._metrics)

    def _emit(self, metric: str) -> None:
        if self._metrics and self._should_collect():
            self._metrics.increment(metric)

    def _emit_tagged(self, metric: str, system_id: str) -> None:
        if self._metrics and self._should_collect():
            self._metrics.increment(metric, tags={"system_id": system_id})

    def _gauge_tagged(self, metric: str, value: float, system_id: str) -> None:
        if self._metrics and self._should_collect():
            self._metrics.gauge(metric, value, tags={"system_id": system_id})
