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
    from polaris.abstractions.system_contract import SystemContract
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
        system_contract: Optional["SystemContract"] = None,
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
        from polaris.strategies.action_resolution import StrictContractViolation

        if getattr(self._strategy, "requires_system_contract", False):
            supported = (
                list(system_contract.supported_action_types) if system_contract is not None else []
            )
            if not supported:
                raise StrictContractViolation(
                    "Missing connector-supported action contract for strict strategy "
                    f"{type(self._strategy).__name__} (system_id='{state.system_id}')"
                )

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
            system_contract=system_contract,
        )

        # Assess
        try:
            actions = await self._strategy.assess(state, context)
        except StrictContractViolation:
            # Fatal contract errors should propagate
            raise
        except Exception as exc:
            self._logger.error(
                "Error in adaptation assessment", system_id=state.system_id, error=str(exc)
            )
            self._emit(
                "polaris.adaptations.assessment_errors",
                tags={"system_id": state.system_id},
                component="core_framework",
            )
            return False

        self._emit(
            "polaris.strategy.assessments",
            tags={"system_id": state.system_id},
            component="strategy",
        )

        # Apply per-system action policies (for example, optional action injection).
        actions = self._apply_action_policies(state, actions)

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

    def _apply_action_policies(self, state: "SystemState", actions: Any) -> Any:
        """Apply optional per-system action policies.

        Supported policies (under ``systems[].action_policy``):
        - ``append_each_cycle``: append one configured action every cycle.
        - ``inject_when_no_actions``: inject one configured action only when
          the strategy returned no actions.
        """
        policy = self._resolve_system_action_policy(state.system_id)
        if not policy:
            return actions

        import uuid

        from polaris.core.models import AdaptationAction

        def _make_policy_action(
            policy_block: Any, default_reason: str
        ) -> Optional[AdaptationAction]:
            if not isinstance(policy_block, dict):
                return None
            action_cfg = policy_block.get("action")
            if not isinstance(action_cfg, dict):
                return None

            action_type = action_cfg.get("type")
            if not isinstance(action_type, str) or not action_type.strip():
                return None

            parameters = action_cfg.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}

            final_parameters = dict(parameters)
            if "reason" not in final_parameters:
                final_parameters["reason"] = default_reason

            return AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type=action_type.strip(),
                target_system=state.system_id,
                parameters=final_parameters,
            )

        append_policy = policy.get("append_each_cycle")
        if isinstance(append_policy, dict) and bool(append_policy.get("enabled", False)):
            if actions is None:
                actions = []
            if not isinstance(actions, list):
                return actions

            actions = list(actions)
            appended_action = _make_policy_action(append_policy, "append_each_cycle")
            if appended_action is not None:
                actions.append(appended_action)
                self._logger.info(
                    "Action policy appended action",
                    system_id=state.system_id,
                    action_type=appended_action.action_type,
                    policy="append_each_cycle",
                    total_actions=len(actions),
                )

        inject_policy = policy.get("inject_when_no_actions")
        if (
            not actions
            and isinstance(inject_policy, dict)
            and bool(inject_policy.get("enabled", False))
        ):
            injected_action = _make_policy_action(inject_policy, "inject_when_no_actions")
            if injected_action is not None:
                self._logger.debug(
                    "Action policy injected action",
                    system_id=state.system_id,
                    action_type=injected_action.action_type,
                    policy="inject_when_no_actions",
                )
                return [injected_action]

        return actions

    def _resolve_system_action_policy(self, system_id: str) -> dict:
        """Resolve action policy config for a system ID."""
        systems = getattr(self._config, "systems", None)
        if not isinstance(systems, list):
            return {}

        for system_cfg in systems:
            cfg_id = getattr(system_cfg, "id", None)
            if not isinstance(cfg_id, str) or cfg_id.lower() != system_id.lower():
                continue

            action_policy = getattr(system_cfg, "action_policy", None)
            if action_policy is None:
                return {}

            # Pydantic model case.
            if hasattr(action_policy, "model_dump"):
                try:
                    dumped = action_policy.model_dump(exclude_none=True)
                except Exception:
                    return {}
                return dumped if isinstance(dumped, dict) else {}

            # Fallback for tests/custom config objects.
            return action_policy if isinstance(action_policy, dict) else {}

        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, metric: str, tags: dict, component: str) -> None:
        """Increment a counter metric if the component is enabled."""
        from polaris.core.component_builder import ComponentBuilder

        if ComponentBuilder.should_collect(self._config, component, self._metrics):
            self._metrics.increment(metric, tags=tags)
