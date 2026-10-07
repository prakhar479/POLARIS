"""Adaptation pipeline: assess → validate → execute → store → notify.

Extracted from ``Polaris._process_system_iteration`` so the decision-and- execution
logic can be tested and reused independently of the monitoring loop.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
        config: "PolarisConfig",
        metrics: Optional["MetricsCollector"] = None,
        dry_run: bool = False,
        fallback_strategy: Optional["AdaptationStrategy"] = None,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_recovery_seconds: float = 60.0,
        topology: Optional[Any] = None,
        safety_engine: Optional[Any] = None,
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
        self._fallback_strategy = fallback_strategy
        self._circuit_breaker_threshold = max(1, int(circuit_breaker_threshold))
        self._circuit_breaker_recovery_seconds = max(0.001, float(circuit_breaker_recovery_seconds))
        self._topology = topology
        self._safety_engine = safety_engine
        self._consecutive_failures = 0
        self._circuit_breaker_open_until: Optional[datetime] = None
        self._circuit_breaker_state: str = "CLOSED"

    @property
    def safety_engine(self) -> Optional[Any]:
        """Optional safety guardrail policy engine."""
        return self._safety_engine

    @property
    def circuit_breaker_state(self) -> str:
        """Current state of strategy circuit breaker ('CLOSED', 'OPEN', 'HALF_OPEN')."""
        return self._circuit_breaker_state

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive assessment failures on the primary strategy."""
        return self._consecutive_failures

    @property
    def fallback_strategy(self) -> Optional["AdaptationStrategy"]:
        """Optional fallback strategy instance."""
        return self._fallback_strategy

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
        from polaris.core.models import (
            ActionWorkflow,
            AdaptationAction,
            ExecutionStatus,
            WorkflowStatus,
        )
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
            now = state.timestamp
            start = now - timedelta(hours=1)
            try:
                historical_states = await self._knowledge_store.query_states(
                    state.system_id, start, now
                )
                # Exclude the current state (it was just stored by the monitoring loop)
                historical_states = [s for s in historical_states if s.timestamp < now][-10:]
            except Exception as exc:
                self._logger.debug(
                    "Failed to fetch historical states for adaptation context",
                    system_id=state.system_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                historical_states = []

        # Resolve topology and peer context
        topology = self._topology
        if (
            topology is None
            and self._knowledge_store
            and hasattr(self._knowledge_store, "get_topology")
        ):
            try:
                topology = await self._knowledge_store.get_topology()
            except Exception as exc:
                self._logger.debug(
                    "Failed to retrieve topology from knowledge store",
                    system_id=state.system_id,
                    error=str(exc),
                )
                topology = None

        from polaris.core.topology import SystemTopology

        if not isinstance(topology, SystemTopology):
            topology = None

        upstream_systems: List[str] = []
        downstream_systems: List[str] = []
        peer_states: Dict[str, SystemState] = {}

        if topology is not None:
            try:
                upstream_systems = list(topology.get_dependents(state.system_id))
                downstream_systems = list(topology.get_dependencies(state.system_id))
                impact_radius = topology.get_impact_radius(state.system_id)
                if self._knowledge_store and hasattr(self._knowledge_store, "get_latest_state"):
                    for peer_id in impact_radius:
                        if peer_id != state.system_id:
                            peer_st = await self._knowledge_store.get_latest_state(peer_id)
                            if peer_st is not None:
                                peer_states[peer_id] = peer_st
            except Exception as exc:
                self._logger.debug(
                    "Failed to resolve topology context for adaptation",
                    system_id=state.system_id,
                    error=str(exc),
                )

        # Build context
        context = AdaptationContext(
            system_id=state.system_id,
            historical_states=historical_states,
            world_model_insights=(
                await self._world_model.get_insights() if self._world_model else None
            ),
            system_contract=system_contract,
            connector=connector,
            metadata={"connector": connector},
            topology=topology,
            peer_states=peer_states,
            upstream_systems=upstream_systems,
            downstream_systems=downstream_systems,
        )

        # Circuit breaker & Assess
        actions = []
        now_dt = datetime.now(timezone.utc)
        primary_eligible = True

        if self._circuit_breaker_state == "OPEN":
            if self._circuit_breaker_open_until and now_dt >= self._circuit_breaker_open_until:
                self._circuit_breaker_state = "HALF_OPEN"
                self._logger.info(
                    "Circuit breaker entered HALF_OPEN state; probing primary strategy",
                    system_id=state.system_id,
                )
                self._emit(
                    "polaris.circuit_breaker.half_open",
                    tags={"system_id": state.system_id},
                    component="core_framework",
                )
            else:
                primary_eligible = False
                self._logger.warning(
                    "Circuit breaker is OPEN for primary strategy",
                    system_id=state.system_id,
                    open_until=(
                        self._circuit_breaker_open_until.isoformat()
                        if self._circuit_breaker_open_until
                        else None
                    ),
                )
                self._emit(
                    "polaris.circuit_breaker.open_skip",
                    tags={"system_id": state.system_id},
                    component="core_framework",
                )
                if self._fallback_strategy is not None:
                    self._logger.info(
                        "Circuit breaker OPEN; delegating to fallback strategy",
                        system_id=state.system_id,
                        fallback=type(self._fallback_strategy).__name__,
                    )
                    self._emit(
                        "polaris.circuit_breaker.fallback_delegated",
                        tags={"system_id": state.system_id},
                        component="core_framework",
                    )
                    try:
                        actions = await self._fallback_strategy.assess(state, context)
                    except Exception as fb_exc:
                        self._logger.error(
                            "Fallback strategy assessment failed",
                            system_id=state.system_id,
                            error=str(fb_exc),
                        )
                        return False
                else:
                    return False

        if primary_eligible:
            try:
                actions = await self._strategy.assess(state, context)
                if self._circuit_breaker_state == "HALF_OPEN" or self._consecutive_failures > 0:
                    self._logger.info(
                        "Circuit breaker reset to CLOSED after successful primary assessment",
                        system_id=state.system_id,
                    )
                    self._consecutive_failures = 0
                    self._circuit_breaker_state = "CLOSED"
                    self._circuit_breaker_open_until = None
                    self._emit(
                        "polaris.circuit_breaker.reset",
                        tags={"system_id": state.system_id},
                        component="core_framework",
                    )
            except StrictContractViolation:
                # Fatal contract errors should propagate
                raise
            except Exception as exc:
                self._consecutive_failures += 1
                self._logger.error(
                    "Error in adaptation assessment",
                    system_id=state.system_id,
                    error=str(exc),
                    consecutive_failures=self._consecutive_failures,
                )
                self._emit(
                    "polaris.adaptations.assessment_errors",
                    tags={"system_id": state.system_id},
                    component="core_framework",
                )
                if self._consecutive_failures >= self._circuit_breaker_threshold:
                    self._circuit_breaker_state = "OPEN"
                    self._circuit_breaker_open_until = datetime.now(timezone.utc) + timedelta(
                        seconds=self._circuit_breaker_recovery_seconds
                    )
                    self._logger.warning(
                        "Circuit breaker TRIPPED to OPEN state",
                        system_id=state.system_id,
                        threshold=self._circuit_breaker_threshold,
                        recovery_seconds=self._circuit_breaker_recovery_seconds,
                    )
                    self._emit(
                        "polaris.circuit_breaker.tripped",
                        tags={"system_id": state.system_id},
                        component="core_framework",
                    )

                if self._fallback_strategy is not None:
                    self._logger.info(
                        "Invoking fallback strategy due to primary strategy failure",
                        system_id=state.system_id,
                        fallback=type(self._fallback_strategy).__name__,
                    )
                    self._emit(
                        "polaris.circuit_breaker.fallback_invoked",
                        tags={"system_id": state.system_id},
                        component="core_framework",
                    )
                    try:
                        actions = await self._fallback_strategy.assess(state, context)
                    except Exception as fb_exc:
                        self._logger.error(
                            "Fallback strategy also failed",
                            system_id=state.system_id,
                            error=str(fb_exc),
                        )
                        return False
                else:
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

        import uuid

        executed_any = False
        for action in actions:
            # Check cluster safety guardrails
            if self._safety_engine is not None:
                is_safe, safety_reason = self._safety_engine.check_action_safety(
                    action, topology=topology
                )
                if not is_safe:
                    self._logger.warning(
                        f"Safety guardrail rejected action '{action.action_type}' for "
                        f"{state.system_id}: {safety_reason}",
                        action_id=action.action_id,
                        system_id=state.system_id,
                        reason=safety_reason,
                    )
                    self._emit(
                        "polaris.safety.action_rejected",
                        tags={"system_id": state.system_id, "action_type": action.action_type},
                        component="safety_engine",
                    )
                    continue

            self._logger.info(
                f"Adaptation proposed for {state.system_id}: {action.action_type}",
                action_id=action.action_id,
            )
            self._emit(
                "polaris.adaptations.proposed",
                tags={"system_id": state.system_id, "action_type": action.action_type},
                component="core_framework",
            )

            # Resolve schema defaults for rollback and verification window if not specified
            rollback_act = action.rollback_action
            verif_window = getattr(action, "verification_window_seconds", 0.0)
            if system_contract:
                schema = system_contract.get_action_schema(action.action_type)
                if schema:
                    if verif_window == 0.0 and schema.default_verification_window_seconds > 0.0:
                        verif_window = schema.default_verification_window_seconds
                    if rollback_act is None and schema.rollback_action:
                        rollback_act = AdaptationAction(
                            action_id=str(uuid.uuid4()),
                            action_type=schema.rollback_action,
                            target_system=action.target_system,
                            parameters={"reason": f"Contract rollback for {action.action_type}"},
                        )

            workflow = ActionWorkflow(
                action=action,
                rollback_action=rollback_act,
                verification_window_seconds=verif_window,
            )

            # Validate
            workflow.transition_to(WorkflowStatus.VALIDATING)
            await self._publish_workflow_event(
                workflow, WorkflowStatus.PENDING, WorkflowStatus.VALIDATING
            )
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
                workflow.transition_to(
                    WorkflowStatus.FAILED,
                    error=f"Action validation failed for {action.action_type}",
                )
                await self._publish_workflow_event(
                    workflow, WorkflowStatus.VALIDATING, WorkflowStatus.FAILED
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
                workflow.transition_to(WorkflowStatus.COMPLETED)
                await self._publish_workflow_event(
                    workflow, WorkflowStatus.VALIDATING, WorkflowStatus.COMPLETED
                )
                executed_any = True
                continue

            workflow.transition_to(WorkflowStatus.EXECUTING)
            await self._publish_workflow_event(
                workflow, WorkflowStatus.VALIDATING, WorkflowStatus.EXECUTING
            )
            if self._safety_engine is not None:
                self._safety_engine.record_action_start(action, topology=topology)
            action_success = False
            try:
                result = await connector.execute_action(action)
                workflow.execution_result = result
                executed_any = True
                if result.status == ExecutionStatus.SUCCESS:
                    action_success = True

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
                    if hasattr(self._knowledge_store, "store_workflow"):
                        await self._knowledge_store.store_workflow(workflow)

                # Notify strategy
                await self._strategy.on_action_executed(action, result)

                # Verification Window & Rollback Check
                if result.status == ExecutionStatus.SUCCESS:
                    if workflow.verification_window_seconds > 0:
                        workflow.transition_to(WorkflowStatus.VERIFYING)
                        await self._publish_workflow_event(
                            workflow, WorkflowStatus.EXECUTING, WorkflowStatus.VERIFYING
                        )
                        is_verified = await self._verify_action(
                            workflow, connector, state, system_contract
                        )
                        if not is_verified:
                            self._logger.warning(
                                f"Post-adaptation verification failed for {action.action_type}",
                                action_id=action.action_id,
                                system_id=state.system_id,
                            )
                            if workflow.rollback_action:
                                await self._execute_rollback(workflow, connector)
                            else:
                                workflow.transition_to(
                                    WorkflowStatus.FAILED,
                                    error="Post-adaptation verification failed",
                                )
                                await self._publish_workflow_event(
                                    workflow, WorkflowStatus.VERIFYING, WorkflowStatus.FAILED
                                )
                        else:
                            workflow.transition_to(WorkflowStatus.COMPLETED)
                            await self._publish_workflow_event(
                                workflow, WorkflowStatus.VERIFYING, WorkflowStatus.COMPLETED
                            )
                            self._emit(
                                "polaris.workflow.verified",
                                tags={
                                    "system_id": state.system_id,
                                    "action_type": action.action_type,
                                },
                                component="core_framework",
                            )
                    else:
                        workflow.transition_to(WorkflowStatus.COMPLETED)
                        await self._publish_workflow_event(
                            workflow, WorkflowStatus.EXECUTING, WorkflowStatus.COMPLETED
                        )
                else:
                    if workflow.rollback_action:
                        await self._execute_rollback(workflow, connector)
                    else:
                        workflow.transition_to(
                            WorkflowStatus.FAILED,
                            error=result.error_message
                            or f"Execution returned {result.status.value}",
                        )
                        await self._publish_workflow_event(
                            workflow, WorkflowStatus.EXECUTING, WorkflowStatus.FAILED
                        )

                # Publish events
                await self._event_bus.publish(
                    AdaptationEvent(
                        action=action,
                        result=result,
                        timestamp=result.completed_at or datetime.now(timezone.utc),
                        workflow=workflow,
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
                if workflow.rollback_action:
                    try:
                        await self._execute_rollback(workflow, connector)
                    except Exception as rb_exc:
                        self._logger.error(f"Rollback execution failed: {rb_exc}")
                else:
                    workflow.transition_to(WorkflowStatus.FAILED, error=str(e))
                    await self._publish_workflow_event(
                        workflow, WorkflowStatus.EXECUTING, WorkflowStatus.FAILED
                    )
            finally:
                if self._safety_engine is not None:
                    self._safety_engine.record_action_end(action, success=action_success)

        return executed_any

    async def _verify_action(
        self,
        workflow: Any,
        connector: "Connector",
        initial_state: "SystemState",
        system_contract: Optional["SystemContract"],
    ) -> bool:
        """Verify action effects over verification_window_seconds.

        Returns True if the system is healthy and satisfies SLOs post-adaptation,
        False if health degraded to CRITICAL/UNHEALTHY or an SLO was breached.
        """
        import asyncio

        if workflow.verification_window_seconds > 0:
            await asyncio.sleep(workflow.verification_window_seconds)

        try:
            post_state = await connector.collect_telemetry()
        except Exception as exc:
            self._logger.warning(
                "Telemetry collection failed during verification window",
                system_id=workflow.action.target_system,
                error=str(exc),
            )
            return False

        if self._knowledge_store:
            try:
                await self._knowledge_store.store_state(post_state)
            except Exception:
                pass
        if self._world_model:
            try:
                await self._world_model.update(post_state)
            except Exception:
                pass

        from polaris.core.models import HealthStatus

        if getattr(post_state, "health_status", None) in (
            HealthStatus.CRITICAL,
            HealthStatus.UNHEALTHY,
        ):
            return False

        if system_contract is not None:
            violated = system_contract.get_violated_slos(post_state)
            if violated:
                self._logger.warning(
                    f"SLO violated after action {workflow.action.action_type}: "
                    f"{[s.metric_name for s in violated]}",
                    system_id=post_state.system_id,
                )
                return False

        return True

    async def _execute_rollback(
        self,
        workflow: Any,
        connector: "Connector",
    ) -> None:
        """Execute automated compensation/rollback action."""
        if not workflow.rollback_action:
            return

        from polaris.core.models import ExecutionStatus, WorkflowStatus

        rb_action = workflow.rollback_action
        prev_status = workflow.status
        workflow.transition_to(WorkflowStatus.ROLLING_BACK)
        await self._publish_workflow_event(workflow, prev_status, WorkflowStatus.ROLLING_BACK)
        self._logger.warning(
            f"Triggering automated rollback {rb_action.action_type} for "
            f"action {workflow.action.action_type}",
            system_id=rb_action.target_system,
            workflow_id=workflow.workflow_id,
        )
        self._emit(
            "polaris.workflow.rolling_back",
            tags={"system_id": rb_action.target_system, "action_type": rb_action.action_type},
            component="core_framework",
        )

        try:
            if await connector.validate_action(rb_action):
                rb_result = await connector.execute_action(rb_action)
                workflow.rollback_result = rb_result
                if rb_result.status == ExecutionStatus.SUCCESS:
                    workflow.transition_to(WorkflowStatus.ROLLED_BACK)
                    await self._publish_workflow_event(
                        workflow, WorkflowStatus.ROLLING_BACK, WorkflowStatus.ROLLED_BACK
                    )
                    self._logger.info(
                        f"Rollback succeeded for {workflow.action.action_type}",
                        workflow_id=workflow.workflow_id,
                    )
                    self._emit(
                        "polaris.workflow.rolled_back",
                        tags={
                            "system_id": rb_action.target_system,
                            "action_type": rb_action.action_type,
                        },
                        component="core_framework",
                    )
                else:
                    workflow.transition_to(
                        WorkflowStatus.FAILED,
                        error=f"Rollback returned status {rb_result.status.value}",
                    )
                    await self._publish_workflow_event(
                        workflow, WorkflowStatus.ROLLING_BACK, WorkflowStatus.FAILED
                    )
                    self._emit(
                        "polaris.workflow.rollback_failed",
                        tags={
                            "system_id": rb_action.target_system,
                            "action_type": rb_action.action_type,
                        },
                        component="core_framework",
                    )
            else:
                workflow.transition_to(
                    WorkflowStatus.FAILED,
                    error=f"Rollback action {rb_action.action_type} validation failed",
                )
                await self._publish_workflow_event(
                    workflow, WorkflowStatus.ROLLING_BACK, WorkflowStatus.FAILED
                )
        except Exception as exc:
            workflow.transition_to(
                WorkflowStatus.FAILED,
                error=f"Exception during rollback: {exc}",
            )
            await self._publish_workflow_event(
                workflow, WorkflowStatus.ROLLING_BACK, WorkflowStatus.FAILED
            )
            self._logger.error(
                f"Exception during rollback execution: {exc}",
                workflow_id=workflow.workflow_id,
            )

    async def _publish_workflow_event(
        self,
        workflow: Any,
        previous_status: Any,
        current_status: Any,
    ) -> None:
        """Publish workflow transition event to event bus."""
        from polaris.core.events import WorkflowEvent

        try:
            await self._event_bus.publish(
                WorkflowEvent(
                    workflow=workflow,
                    previous_status=previous_status,
                    current_status=current_status,
                    timestamp=datetime.now(timezone.utc),
                )
            )
        except Exception:
            pass

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
                except (AttributeError, TypeError, ValueError) as exc:
                    self._logger.debug(
                        "Failed to dump action policy model",
                        system_id=system_id,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
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
