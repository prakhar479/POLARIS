"""Adaptation pipeline: assess → validate → execute → store → notify.

Extracted from ``Polaris._process_system_iteration`` so the decision-and- execution
logic can be tested and reused independently of the monitoring loop.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

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

from polaris.infrastructure.observability.null_metrics import NullMetricsCollector


class AdaptationPipeline:
    """Runs the full adaptation cycle for a single system state.

    Given a ``SystemState`` and the connector that produced it, the pipeline:

    1. Builds an ``AdaptationContext`` (with world-model insights). 2. Asks the strategy
    to ``assess`` the state. 3. If actions are proposed, validates each against the
    connector. 4. Executes the actions and stores the results in the knowledge store. 5.
    Notifies the strategy via ``on_action_executed`` for each. 6. Publishes
    ``AdaptationEvent``s on the event bus.

    Returns ``True`` if at least one action was successfully executed (or would have
    been in dry-run mode), ``False`` otherwise.
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
        dry_run: bool = False,
    ) -> None:
        """Initialize the pipeline."""
        self._strategy = strategy
        self._knowledge_store = knowledge_store
        self._world_model = world_model
        self._event_bus = event_bus
        self._logger = logger
        self._metrics = metrics or NullMetricsCollector()
        self._config = config
        self._dry_run = dry_run

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
            ``True`` if at least one adaptation action was executed, ``False``
                otherwise.
        """
        from polaris.abstractions.strategy import AdaptationContext
        from polaris.core.events import AdaptationEvent

        # Fetch recent history so strategies can reason about trends.
        historical_states = []
        if self._knowledge_store:
            from datetime import timedelta

            now = state.timestamp
            start = now - timedelta(hours=1)
            try:
                historical_states = await self._knowledge_store.query_states(
                    state.system_id, start, now
                )
                # Exclude the current state (it was just stored by the monitoring loop)
                historical_states = [s for s in historical_states if s.timestamp < now][-10:]
            except Exception:
                historical_states = []

        # Build context
        context = AdaptationContext(
            system_id=state.system_id,
            historical_states=historical_states,
            world_model_insights=(
                await self._world_model.get_insights() if self._world_model else None
            ),
        )

        # Assess
        actions = await self._strategy.assess(state, context)
        self._emit(
            "polaris.strategy.assessments",
            tags={"system_id": state.system_id},
            component="strategy",
        )

        # Wildfire simulations are often step-driven: we may need to advance the
        # simulation clock even when adaptations happen.
        # This behavior is opt-in via config to avoid affecting other connectors.
        actions = self._apply_wildfire_step_policy(state, actions)

        if not actions:
            return False

        executed_any = False
        for action in actions:
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
                continue

            # Execute (or skip in dry-run mode)
            if self._dry_run:
                self._logger.info(
                    f"[DRY-RUN] Would execute {action.action_type} on {state.system_id} "
                    f"(action_id={action.action_id})",
                    parameters=action.parameters,
                )
                self._emit(
                    "polaris.adaptations.dry_run_skipped",
                    tags={"system_id": state.system_id, "action_type": action.action_type},
                    component="core_framework",
                )
                executed_any = True
                continue

            try:
                result = await connector.execute_action(action)
                executed_any = True

                self._logger.info(
                    f"Adaptation executed: {action.action_type} -> {result.status.value}",
                    action_id=action.action_id,
                )
                self._emit(
                    "polaris.adaptations.executed",
                    tags={
                        "system_id": state.system_id,
                        "action_type": action.action_type,
                        "status": result.status.value,
                    },
                    component="core_framework",
                )

                # Store result
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
            except Exception as e:
                self._logger.error(
                    f"Error executing adaptation {action.action_type} on {state.system_id}: {e}",
                    action_id=action.action_id,
                )
                self._emit(
                    "polaris.adaptations.execution_errors",
                    tags={"system_id": state.system_id, "action_type": action.action_type},
                    component="core_framework",
                )

        return executed_any

    def _apply_wildfire_step_policy(self, state: "SystemState", actions: Any) -> Any:
        """Apply wildfire step policy.

        Supported policies (wildfire config):
        - always_step_each_cycle: when true, append wildfire_step every cycle
        - auto_step_when_no_adaptation: when true, inject wildfire_step only when
        strategy returned no actions (legacy behavior)
        """
        if state.system_id.lower() != "wildfire":
            return actions

        try:
            wildfire_cfg = None
            # PolarisConfig is a pydantic model; connector-specific blocks like `wildfire:`
            # are preserved in config.extra.
            extra = getattr(self._config, "extra", {})
            if isinstance(extra, dict) and isinstance(extra.get("wildfire"), dict):
                wildfire_cfg = extra.get("wildfire")
            wildfire_cfg = wildfire_cfg or {}
        except Exception:
            wildfire_cfg = {}

        always_step = bool(wildfire_cfg.get("always_step_each_cycle", False))
        step_when_no_actions = bool(wildfire_cfg.get("auto_step_when_no_adaptation", False))

        import uuid

        from polaris.core.models import AdaptationAction

        def make_step(reason: str) -> AdaptationAction:
            return AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type="wildfire_step",
                target_system=state.system_id,
                parameters={"reason": reason},
            )

        if always_step:
            if actions is None:
                actions = []
            if not isinstance(actions, list):
                return actions
            actions = list(actions)
            step_action = make_step("always_step_each_cycle")
            actions.append(step_action)
            self._logger.info(
                "Wildfire step appended",
                system_id=state.system_id,
                reason="always_step_each_cycle",
                total_actions=len(actions),
                actions=[a.action_type for a in actions if hasattr(a, "action_type")],
            )
            return actions

        if not actions and step_when_no_actions:
            self._logger.debug(
                "Auto-stepping wildfire (no adaptation proposed)",
                system_id=state.system_id,
            )
            return [make_step("auto_step_when_no_adaptation")]

        return actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, metric: str, tags: dict, component: str) -> None:
        """Increment a counter metric if the component is enabled."""
        from polaris.core.component_builder import ComponentBuilder

        if ComponentBuilder.should_collect(self._config, component, self._metrics):
            self._metrics.increment(metric, tags=tags)
