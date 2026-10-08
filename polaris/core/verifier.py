"""Formal Neuro-Symbolic Verifier implementation for POLARIS.

Enforces static contract adherence, continuous parameter safety envelopes,
rate-of-change (delta) clamping, metric temporal logic (MTL) anti-flapping,
and topological blast-radius protection.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.verifier import (
    InvariantSeverity,
    InvariantViolation,
    SafetyInvariant,
    VerificationContext,
    VerificationDecision,
    VerificationResult,
    Verifier,
)
from polaris.core.models import AdaptationAction, HealthStatus

if TYPE_CHECKING:
    from polaris.core.safety import SafetyPolicyEngine


class ContractSchemaInvariant(SafetyInvariant):
    """Enforces that action types are supported and parameter types conform to ActionSchema."""

    @property
    def name(self) -> str:
        """Unique identifier of the invariant."""
        return "contract_schema"

    @property
    def severity(self) -> InvariantSeverity:
        """Severity level when this invariant fails."""
        return InvariantSeverity.CRITICAL

    def evaluate(
        self,
        action: AdaptationAction,
        context: VerificationContext,
    ) -> Tuple[bool, Optional[InvariantViolation], Optional[Dict[str, Any]]]:
        """Evaluate action against contract schema and types."""
        contract = context.system_contract
        if contract is None:
            return True, None, None

        action_type = action.action_type.strip()
        schema = contract.get_action_schema(action_type)
        if schema is None and action_type.lower() not in [
            a.lower() for a in contract.supported_action_types
        ]:
            canonical = contract.action_aliases.get(action_type.lower())
            if not canonical or canonical.lower() not in [
                a.lower() for a in contract.supported_action_types
            ]:
                violation = InvariantViolation(
                    invariant_name=self.name,
                    severity=self.severity,
                    message=(
                        f"Action type '{action_type}' is not supported by contract "
                        f"for system '{context.system_id}'"
                    ),
                    violating_parameter="action_type",
                    current_value=action_type,
                    admissible_bound=list(contract.supported_action_types),
                )
                return False, violation, None

        if schema:
            params = action.parameters or {}
            # Check required parameters
            for req in schema.required_parameters:
                if req not in params:
                    violation = InvariantViolation(
                        invariant_name=self.name,
                        severity=self.severity,
                        message=f"Missing required parameter '{req}' for action '{action_type}'",
                        violating_parameter=req,
                        current_value=None,
                        admissible_bound="required",
                    )
                    return False, violation, None

            # Check parameter types
            for key, spec in schema.parameters_schema.items():
                if key in params and isinstance(spec, dict):
                    val = params[key]
                    exp_type = spec.get("type")
                    type_valid = True
                    if exp_type == "integer" and (
                        isinstance(val, bool) or not isinstance(val, int)
                    ):
                        type_valid = False
                    elif exp_type == "number" and (
                        isinstance(val, bool) or not isinstance(val, (int, float))
                    ):
                        type_valid = False
                    elif exp_type == "string" and not isinstance(val, str):
                        type_valid = False
                    elif exp_type == "boolean" and not isinstance(val, bool):
                        type_valid = False

                    if not type_valid:
                        violation = InvariantViolation(
                            invariant_name=self.name,
                            severity=self.severity,
                            message=f"Parameter '{key}' has invalid type; expected '{exp_type}'",
                            violating_parameter=key,
                            current_value=type(val).__name__,
                            admissible_bound=exp_type,
                        )
                        return False, violation, None

        return True, None, None


class ParameterBoundsClampingInvariant(SafetyInvariant):
    """Enforces continuous numeric bounds and projects/clamps out-of-bounds parameters."""

    def __init__(self, allow_clamping: bool = True):
        """Initialize parameter bounds clamping invariant."""
        self._allow_clamping = allow_clamping

    @property
    def name(self) -> str:
        """Unique identifier of the invariant."""
        return "parameter_bounds_clamping"

    @property
    def severity(self) -> InvariantSeverity:
        """Severity level when this invariant fails."""
        return InvariantSeverity.WARNING if self._allow_clamping else InvariantSeverity.CRITICAL

    def evaluate(
        self,
        action: AdaptationAction,
        context: VerificationContext,
    ) -> Tuple[bool, Optional[InvariantViolation], Optional[Dict[str, Any]]]:
        """Evaluate action against parameter bounds and clamp if soft violation."""
        contract = context.system_contract
        if contract is None:
            return True, None, None

        schema = contract.get_action_schema(action.action_type)
        if not schema or not action.parameters:
            return True, None, None

        clamped_params: Dict[str, Any] = {}
        first_violation: Optional[InvariantViolation] = None

        for key, spec in schema.parameters_schema.items():
            if key in action.parameters and isinstance(spec, dict):
                val = action.parameters[key]
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    target_val = float(val)
                    clamped_val = target_val

                    if "minimum" in spec and target_val < spec["minimum"]:
                        min_bound = spec["minimum"]
                        clamped_val = float(min_bound) if isinstance(val, float) else int(min_bound)
                        if first_violation is None:
                            first_violation = InvariantViolation(
                                invariant_name=self.name,
                                severity=self.severity,
                                message=f"Parameter '{key}' value {val} is below minimum {min_bound}",
                                violating_parameter=key,
                                current_value=val,
                                admissible_bound=min_bound,
                            )

                    if "maximum" in spec and target_val > spec["maximum"]:
                        max_bound = spec["maximum"]
                        clamped_val = float(max_bound) if isinstance(val, float) else int(max_bound)
                        if first_violation is None:
                            first_violation = InvariantViolation(
                                invariant_name=self.name,
                                severity=self.severity,
                                message=f"Parameter '{key}' value {val} exceeds maximum {max_bound}",
                                violating_parameter=key,
                                current_value=val,
                                admissible_bound=max_bound,
                            )

                    if clamped_val != val:
                        clamped_params[key] = clamped_val

        if clamped_params:
            if self._allow_clamping:
                return False, first_violation, clamped_params
            else:
                return False, first_violation, None

        return True, None, None


class RateOfChangeClampingInvariant(SafetyInvariant):
    """Enforces maximum step change (delta envelope) per adaptation cycle for continuous parameters."""

    def __init__(self, max_deltas: Optional[Dict[str, float]] = None):
        """Initialize rate-of-change clamping invariant."""
        # Maps parameter name to maximum allowed change per cycle (e.g. {"dimmer": 0.25})
        self._max_deltas: Dict[str, float] = {
            "dimmer": 0.25,
            **(max_deltas or {}),
        }

    @property
    def name(self) -> str:
        """Unique identifier of the invariant."""
        return "rate_of_change_clamping"

    @property
    def severity(self) -> InvariantSeverity:
        """Severity level when this invariant fails."""
        return InvariantSeverity.WARNING

    def evaluate(
        self,
        action: AdaptationAction,
        context: VerificationContext,
    ) -> Tuple[bool, Optional[InvariantViolation], Optional[Dict[str, Any]]]:
        """Evaluate action against rate-of-change envelope and clamp if soft violation."""
        if not action.parameters:
            return True, None, None

        clamped_params: Dict[str, Any] = {}
        first_violation: Optional[InvariantViolation] = None

        # Inspect current state metrics or most recent action to find baseline parameter value
        curr_metrics = getattr(context.system_state, "metrics", {}) or {}

        for param_name, max_step in self._max_deltas.items():
            if param_name in action.parameters:
                candidate_val = action.parameters[param_name]
                if isinstance(candidate_val, (int, float)) and not isinstance(candidate_val, bool):
                    prev_val: Optional[float] = None
                    if param_name in curr_metrics:
                        mv = curr_metrics[param_name]
                        if isinstance(mv.value, (int, float)) and not isinstance(mv.value, bool):
                            prev_val = float(mv.value)

                    # Fallback: check recent actions
                    if prev_val is None and context.recent_actions:
                        for prev_act in reversed(context.recent_actions):
                            if (
                                prev_act.parameters
                                and param_name in prev_act.parameters
                                and isinstance(prev_act.parameters[param_name], (int, float))
                            ):
                                prev_val = float(prev_act.parameters[param_name])
                                break

                    if prev_val is not None:
                        delta = float(candidate_val) - prev_val
                        if abs(delta) > max_step:
                            clamped = prev_val + (max_step if delta > 0 else -max_step)
                            clamped_params[param_name] = round(clamped, 4)
                            if first_violation is None:
                                first_violation = InvariantViolation(
                                    invariant_name=self.name,
                                    severity=self.severity,
                                    message=(
                                        f"Parameter '{param_name}' delta {delta:+.2f} exceeds max "
                                        f"rate-of-change envelope {max_step:.2f}; clamped to {clamped_params[param_name]}"
                                    ),
                                    violating_parameter=param_name,
                                    current_value=candidate_val,
                                    admissible_bound=f"[{prev_val - max_step:.2f}, {prev_val + max_step:.2f}]",
                                )

        if clamped_params:
            return False, first_violation, clamped_params

        return True, None, None


class TemporalDwellInvariant(SafetyInvariant):
    """Enforces minimum dwell time between conflicting adaptation actions (Metric Temporal Logic)."""

    def __init__(
        self,
        opposing_actions: Optional[Dict[str, str]] = None,
        cooldown_seconds: float = 30.0,
    ):
        """Initialize temporal dwell anti-flapping invariant."""
        self._opposing: Dict[str, str] = {
            "scale_up": "scale_down",
            "scale_down": "scale_up",
            "scale_out": "scale_in",
            "scale_in": "scale_out",
            "throttle": "unthrottle",
            "unthrottle": "throttle",
            **(opposing_actions or {}),
        }
        self._cooldown_seconds = max(0.1, float(cooldown_seconds))

    @property
    def name(self) -> str:
        """Unique identifier of the invariant."""
        return "temporal_dwell_anti_flapping"

    @property
    def severity(self) -> InvariantSeverity:
        """Severity level when this invariant fails."""
        return InvariantSeverity.CRITICAL

    def evaluate(
        self,
        action: AdaptationAction,
        context: VerificationContext,
    ) -> Tuple[bool, Optional[InvariantViolation], Optional[Dict[str, Any]]]:
        """Evaluate action against temporal dwell anti-flapping constraints."""
        act_type = action.action_type.strip().lower()
        opposing_type = self._opposing.get(act_type)
        if not opposing_type or not context.recent_actions:
            return True, None, None

        now = action.created_at.timestamp() if action.created_at else time.time()

        for prev_act in reversed(context.recent_actions):
            prev_type = prev_act.action_type.strip().lower()
            if prev_type == opposing_type:
                prev_time = prev_act.created_at.timestamp() if prev_act.created_at else now
                elapsed = now - prev_time
                if elapsed < self._cooldown_seconds:
                    remaining = self._cooldown_seconds - elapsed
                    violation = InvariantViolation(
                        invariant_name=self.name,
                        severity=self.severity,
                        message=(
                            f"Action '{action.action_type}' opposes recent '{prev_act.action_type}' "
                            f"within dwell window ({elapsed:.1f}s < {self._cooldown_seconds:.1f}s cooldown; "
                            f"{remaining:.1f}s remaining)"
                        ),
                        violating_parameter="action_type",
                        current_value=action.action_type,
                        admissible_bound=f"dwell >= {self._cooldown_seconds}s",
                    )
                    return False, violation, None

        return True, None, None


class TopologicalSafetyInvariant(SafetyInvariant):
    """Enforces dependency safety: suppresses upstream capacity expansions when downstream dependencies are stressed."""

    @property
    def name(self) -> str:
        """Unique identifier of the invariant."""
        return "topological_safety_backpressure"

    @property
    def severity(self) -> InvariantSeverity:
        """Severity level when this invariant fails."""
        return InvariantSeverity.CRITICAL

    def evaluate(
        self,
        action: AdaptationAction,
        context: VerificationContext,
    ) -> Tuple[bool, Optional[InvariantViolation], Optional[Dict[str, Any]]]:
        """Evaluate action against topological dependency health."""
        if context.topology is None or not context.peer_states:
            return True, None, None

        downstream = context.topology.get_dependencies(context.system_id)
        if not downstream:
            return True, None, None

        # Check if downstream dependency is currently critical or unhealthy
        stressed_downstream: List[Tuple[str, str]] = []
        for dep_id in downstream:
            peer_st = context.peer_states.get(dep_id)
            if peer_st and peer_st.health_status in (
                HealthStatus.CRITICAL,
                HealthStatus.UNHEALTHY,
            ):
                stressed_downstream.append((dep_id, peer_st.health_status.value))

        if not stressed_downstream:
            return True, None, None

        # Actions that increase upstream load/throughput exacerbate downstream overload
        act_type = action.action_type.strip().lower()
        is_load_amplifying = False
        if act_type in ("scale_up", "scale_out"):
            is_load_amplifying = True
        elif act_type in ("set_dimmer", "dimmer"):
            dim_val = (action.parameters or {}).get("dimmer")
            if isinstance(dim_val, (int, float)) and dim_val > 0.8:
                is_load_amplifying = True

        if is_load_amplifying:
            dep_names = ", ".join(f"{d} ({h})" for d, h in stressed_downstream)
            violation = InvariantViolation(
                invariant_name=self.name,
                severity=self.severity,
                message=(
                    f"Action '{action.action_type}' amplifies downstream workload while downstream "
                    f"dependencies [{dep_names}] are degraded; cascade backpressure requires load-shedding"
                ),
                violating_parameter="action_type",
                current_value=action.action_type,
                admissible_bound="load_shedding_only",
            )
            return False, violation, None

        return True, None, None


class NeuroSymbolicVerifier(Verifier):
    """Primary Neuro-Symbolic Verification gatekeeper for POLARIS.

    Evaluates candidate actions against registered invariants and either accepts,
    clamps to the nearest safe envelope, or formally rejects with a counterexample.
    """

    def __init__(
        self,
        invariants: Optional[Sequence[SafetyInvariant]] = None,
        safety_engine: Optional[SafetyPolicyEngine] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
        allow_clamping: bool = True,
    ):
        """Initialize NeuroSymbolicVerifier."""
        self._logger = logger
        self._metrics = metrics
        self._safety_engine = safety_engine
        self._allow_clamping = allow_clamping
        self._invariants: List[SafetyInvariant] = []

        if invariants is not None:
            for inv in invariants:
                self.register_invariant(inv)
        else:
            # Default standard invariants suite
            self.register_invariant(ContractSchemaInvariant())
            self.register_invariant(ParameterBoundsClampingInvariant(allow_clamping=allow_clamping))
            self.register_invariant(RateOfChangeClampingInvariant())
            self.register_invariant(TemporalDwellInvariant())
            self.register_invariant(TopologicalSafetyInvariant())

    def register_invariant(self, invariant: SafetyInvariant) -> None:
        """Register a new safety invariant."""
        self._invariants.append(invariant)

    def get_invariants(self) -> Sequence[SafetyInvariant]:
        """Return registered invariants."""
        return list(self._invariants)

    async def verify(
        self,
        action: AdaptationAction,
        context: VerificationContext,
    ) -> VerificationResult:
        """Formally verify and optionally clamp candidate action."""
        start_t = time.perf_counter()
        violations: List[InvariantViolation] = []
        accumulated_clamped_params: Dict[str, Any] = dict(action.parameters or {})
        has_clamping = False
        rejection_violation: Optional[InvariantViolation] = None

        # 1. External safety policy engine check (rate limits, blast radius locks)
        if self._safety_engine is not None:
            is_safe, reason = self._safety_engine.check_action_safety(
                action, topology=context.topology
            )
            if not is_safe:
                violation = InvariantViolation(
                    invariant_name="cluster_safety_policy",
                    severity=InvariantSeverity.CRITICAL,
                    message=reason or "Cluster safety policy check rejected action",
                    violating_parameter="action",
                )
                rejection_violation = violation
                violations.append(violation)

        # 2. Sequential evaluation of all registered invariants
        if rejection_violation is None:
            current_action = action
            for invariant in self._invariants:
                satisfied, viol, clamped = invariant.evaluate(current_action, context)
                if not satisfied and viol is not None:
                    violations.append(viol)
                    if viol.severity == InvariantSeverity.CRITICAL:
                        rejection_violation = viol
                        break
                    elif clamped is not None:
                        accumulated_clamped_params.update(clamped)
                        has_clamping = True
                        current_action = AdaptationAction(
                            action_id=action.action_id,
                            action_type=action.action_type,
                            target_system=action.target_system,
                            parameters=dict(accumulated_clamped_params),
                            priority=action.priority,
                            timeout_seconds=action.timeout_seconds,
                            created_at=action.created_at,
                            rollback_action=action.rollback_action,
                            verification_window_seconds=action.verification_window_seconds,
                            metadata=action.metadata,
                        )

            # Safety projection operator: guarantee parameters remain within schema [min, max]
            if has_clamping and context.system_contract:
                schema = context.system_contract.get_action_schema(action.action_type)
                if schema:
                    for param_k, param_spec in schema.parameters_schema.items():
                        if param_k in accumulated_clamped_params and isinstance(param_spec, dict):
                            val = accumulated_clamped_params[param_k]
                            if isinstance(val, (int, float)) and not isinstance(val, bool):
                                if "minimum" in param_spec and val < param_spec["minimum"]:
                                    accumulated_clamped_params[param_k] = param_spec["minimum"]
                                if "maximum" in param_spec and val > param_spec["maximum"]:
                                    accumulated_clamped_params[param_k] = param_spec["maximum"]

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        # 3. Formulate formal decision
        if rejection_violation is not None:
            explanation = (
                f"REJECTED by invariant '{rejection_violation.invariant_name}': "
                f"{rejection_violation.message}"
            )
            if self._logger:
                self._logger.warning(
                    f"Verifier rejected action: {explanation}",
                    system_id=context.system_id,
                    action_id=action.action_id,
                    action_type=action.action_type,
                )
            if self._metrics:
                self._metrics.increment(
                    "polaris.verifier.rejected",
                    tags={"system_id": context.system_id, "action_type": action.action_type},
                )
            return VerificationResult(
                decision=VerificationDecision.REJECTED,
                original_action=action,
                verified_action=None,
                violations=violations,
                explanation=explanation,
                latency_ms=latency_ms,
            )

        if has_clamping and self._allow_clamping:
            # Construct new verified action with clamped parameters
            clamped_action = AdaptationAction(
                action_id=action.action_id,
                action_type=action.action_type,
                target_system=action.target_system,
                parameters=accumulated_clamped_params,
                priority=action.priority,
                timeout_seconds=action.timeout_seconds,
                created_at=action.created_at,
                rollback_action=action.rollback_action,
                verification_window_seconds=action.verification_window_seconds,
                metadata={
                    **(action.metadata or {}),
                    "verifier_clamped": True,
                    "original_parameters": action.parameters,
                },
            )
            explanation = "CLAMPED onto safety envelope: " + "; ".join(
                v.message for v in violations if v.severity == InvariantSeverity.WARNING
            )
            if self._logger:
                self._logger.info(
                    f"Verifier clamped action: {explanation}",
                    system_id=context.system_id,
                    action_id=action.action_id,
                    action_type=action.action_type,
                    clamped_parameters=accumulated_clamped_params,
                )
            if self._metrics:
                self._metrics.increment(
                    "polaris.verifier.clamped",
                    tags={"system_id": context.system_id, "action_type": action.action_type},
                )
            return VerificationResult(
                decision=VerificationDecision.CLAMPED,
                original_action=action,
                verified_action=clamped_action,
                violations=violations,
                explanation=explanation,
                latency_ms=latency_ms,
            )

        # Action fully accepted
        explanation = "ACCEPTED: All formal safety invariants verified successfully"
        if self._metrics:
            self._metrics.increment(
                "polaris.verifier.accepted",
                tags={"system_id": context.system_id, "action_type": action.action_type},
            )
        return VerificationResult(
            decision=VerificationDecision.ACCEPTED,
            original_action=action,
            verified_action=action,
            violations=[],
            explanation=explanation,
            latency_ms=latency_ms,
        )
