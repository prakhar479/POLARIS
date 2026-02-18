"""Adaptation pipeline: assess → validate → execute → store → notify.

Extracted from ``Polaris._process_system_iteration`` so the decision-and-
execution logic can be tested and reused independently of the monitoring loop.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from polaris.abstractions import (
        AdaptationStrategy,
        Connector,
        KnowledgeStore,
        Logger,
        MetricsCollector,
        WorldModel,
    )
    from polaris.core.events import EventBus
    from polaris.core.models import SystemState
    from polaris.infrastructure.config import PolarisConfig


class AdaptationPipeline:
    """Runs the full adaptation cycle for a single system state.

    Given a ``SystemState`` and the connector that produced it, the pipeline:

    1. Builds an ``AdaptationContext`` (with world-model insights).
    2. Asks the strategy to ``assess`` the state.
    3. If an action is proposed, validates it against the connector.
    4. Executes the action and stores the result in the knowledge store.
    5. Notifies the strategy via ``on_action_executed``.
    6. Publishes an ``AdaptationEvent`` on the event bus.

    Returns ``True`` if an action was successfully executed, ``False``
    otherwise (including the case where no action was proposed).
    """

    def __init__(
        self,
        strategy: "AdaptationStrategy",
        knowledge_store: Optional["KnowledgeStore"],
        world_model: Optional["WorldModel"],
        event_bus: "EventBus",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        config: "PolarisConfig",
    ) -> None:
        """Initialize the pipeline."""
        self._strategy = strategy
        self._knowledge_store = knowledge_store
        self._world_model = world_model
        self._event_bus = event_bus
        self._logger = logger
        self._metrics = metrics
        self._config = config

    async def run(
        self,
        state: "SystemState",
        connector: "Connector",
    ) -> bool:
        """Execute the full assess→execute pipeline.

        Args:
            state: Current system state (already collected by the caller).
            connector: The connector for the managed system.

        Returns:
            ``True`` if an adaptation action was executed, ``False`` otherwise.
        """
        from polaris.abstractions.strategy import AdaptationContext
        from polaris.core.events import AdaptationEvent

        # Build context
        context = AdaptationContext(
            system_id=state.system_id,
            historical_states=[],
            world_model_insights=(
                await self._world_model.get_insights() if self._world_model else None
            ),
        )

        # Assess
        action = await self._strategy.assess(state, context)
        self._emit(
            "polaris.strategy.assessments",
            tags={"system_id": state.system_id},
            component="strategy",
        )

        if not action:
            return False

        self._logger.info(
            f"Adaptation proposed for {state.system_id}: {action.action_type}",
            action_id=action.action_id,
        )
        self._emit(
            "polaris.adaptations.proposed",
            tags={"system_id": state.system_id, "action_type": action.action_type},
            component="core_framework",
        )

        # Validate
        if not await connector.validate_action(action):
            self._logger.warning(
                f"Action validation failed for {action.action_type}",
                action_id=action.action_id,
            )
            self._emit(
                "polaris.adaptations.validation_errors",
                tags={"system_id": state.system_id, "action_type": action.action_type},
                component="core_framework",
            )
            return False

        # Execute
        result = await connector.execute_action(action)
        self._emit(
            "polaris.adaptations.executed",
            tags={
                "system_id": state.system_id,
                "action_type": action.action_type,
                "status": result.status.value,
            },
            component="core_framework",
        )

        # Store
        if self._knowledge_store:
            await self._knowledge_store.store_action(action, result)

        # Notify strategy
        await self._strategy.on_action_executed(action, result)

        # Publish event
        await self._event_bus.publish(
            AdaptationEvent(
                action=action,
                result=result,
                timestamp=result.completed_at or datetime.now(timezone.utc),
            )
        )
        self._emit(
            "polaris.events.adaptation_published",
            tags={"system_id": state.system_id},
            component="event_bus",
        )

        self._logger.info(
            f"Adaptation executed: {action.action_type} -> {result.status.value}",
            action_id=action.action_id,
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, metric: str, tags: dict, component: str) -> None:
        """Increment a counter metric if the component is enabled."""
        if not self._metrics:
            return
        from polaris.core.component_builder import ComponentBuilder

        if ComponentBuilder.should_collect(self._config, component, self._metrics):
            self._metrics.increment(metric, tags=tags)
