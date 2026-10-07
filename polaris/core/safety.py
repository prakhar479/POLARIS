"""Cluster-wide safety guardrails and blast-radius policy engine for Polaris.

Provides multi-system rate-limiting, blast-radius mutual exclusion,
oscillation/flapping prevention, and conflict checks for autonomous adaptations.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.core.models import AdaptationAction
from polaris.core.topology import SystemTopology

DEFAULT_OPPOSING_ACTIONS: Dict[str, str] = {
    "scale_up": "scale_down",
    "scale_down": "scale_up",
    "scale_out": "scale_in",
    "scale_in": "scale_out",
    "throttle": "unthrottle",
    "unthrottle": "throttle",
    "enable": "disable",
    "disable": "enable",
}


@dataclass
class SafetyConfig:
    """Configuration for cluster safety guardrails."""

    enabled: bool = True
    max_cluster_adaptations_per_window: int = 0  # 0 = unlimited
    cluster_window_seconds: float = 60.0
    max_system_adaptations_per_window: int = 0  # 0 = unlimited
    system_window_seconds: float = 60.0
    max_concurrent_cluster_adaptations: int = 0  # 0 = unlimited
    concurrency_mode: str = (
        "blast_radius_isolated"  # "parallel", "serialized", "blast_radius_isolated"
    )
    flapping_window_seconds: float = 120.0
    flapping_threshold: int = 3
    flapping_cooldown_seconds: float = 180.0
    opposing_actions: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_OPPOSING_ACTIONS))
    mutual_exclusions: List[List[str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SafetyConfig":
        """Instantiate config from dictionary with defaults."""
        if not data:
            return cls()

        opposing = dict(DEFAULT_OPPOSING_ACTIONS)
        custom_opposing = data.get("opposing_actions")
        if isinstance(custom_opposing, dict):
            for k, v in custom_opposing.items():
                if isinstance(k, str) and isinstance(v, str):
                    opposing[k.strip().lower()] = v.strip().lower()
                    opposing[v.strip().lower()] = k.strip().lower()

        exclusions: List[List[str]] = []
        raw_exclusions = data.get("mutual_exclusions")
        if isinstance(raw_exclusions, list):
            for group in raw_exclusions:
                if isinstance(group, list):
                    exclusions.append([str(item).strip().lower() for item in group if item])

        return cls(
            enabled=bool(data.get("enabled", True)),
            max_cluster_adaptations_per_window=int(
                data.get("max_cluster_adaptations_per_window", 0)
            ),
            cluster_window_seconds=float(data.get("cluster_window_seconds", 60.0)),
            max_system_adaptations_per_window=int(data.get("max_system_adaptations_per_window", 0)),
            system_window_seconds=float(data.get("system_window_seconds", 60.0)),
            max_concurrent_cluster_adaptations=int(
                data.get("max_concurrent_cluster_adaptations", 0)
            ),
            concurrency_mode=str(data.get("concurrency_mode", "blast_radius_isolated")).strip(),
            flapping_window_seconds=float(data.get("flapping_window_seconds", 120.0)),
            flapping_threshold=int(data.get("flapping_threshold", 3)),
            flapping_cooldown_seconds=float(data.get("flapping_cooldown_seconds", 180.0)),
            opposing_actions=opposing,
            mutual_exclusions=exclusions,
        )


class SafetyPolicyEngine:
    """Cluster-wide safety guardrail policy engine.

    Enforces rate limits, topological blast radius isolation, and anti-flapping
    protections before adaptations are dispatched to managed systems.
    """

    def __init__(
        self,
        config: Optional[SafetyConfig | Dict[str, Any]] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize safety engine."""
        if isinstance(config, SafetyConfig):
            self.config = config
        else:
            self.config = SafetyConfig.from_dict(config)

        self.logger = logger
        self.metrics = metrics

        # State tracking
        self._cluster_history: List[float] = []  # Monotonic timestamps
        self._system_history: Dict[str, List[Tuple[float, str]]] = (
            {}
        )  # system -> [(ts, action_type)]
        self._active_actions: Dict[str, Tuple[AdaptationAction, Set[str]]] = (
            {}
        )  # action_id -> (action, impact_nodes)
        self._flapping_cooldowns: Dict[Tuple[str, str], float] = (
            {}
        )  # (system_id, action_type) -> expires_monotonic

    def _prune_history(self, now: float) -> None:
        """Prune sliding window histories and expired cooldowns."""
        # Prune cluster history
        cluster_cutoff = now - self.config.cluster_window_seconds
        self._cluster_history = [ts for ts in self._cluster_history if ts >= cluster_cutoff]

        # Prune system history
        system_cutoff = now - max(
            self.config.system_window_seconds, self.config.flapping_window_seconds
        )
        for sys_id, records in list(self._system_history.items()):
            valid = [(ts, act) for ts, act in records if ts >= system_cutoff]
            if valid:
                self._system_history[sys_id] = valid
            else:
                self._system_history.pop(sys_id, None)

        # Prune cooldowns
        for key, expires_at in list(self._flapping_cooldowns.items()):
            if now >= expires_at:
                self._flapping_cooldowns.pop(key, None)

    def check_action_safety(
        self,
        action: AdaptationAction,
        topology: Optional[SystemTopology] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate whether candidate action satisfies safety invariants.

        Args:
            action: Candidate adaptation action to inspect.
            topology: Optional current system dependency graph.

        Returns:
            (True, None) if action is safe to execute.
            (False, rejection_reason) if action breaches safety policy.
        """
        if not self.config.enabled:
            return True, None

        now = time.monotonic()
        self._prune_history(now)

        sys_id = action.target_system.strip()
        act_type = action.action_type.strip().lower()

        # 1. Flapping Cooldown Check
        cooldown_key = (sys_id, act_type)
        if cooldown_key in self._flapping_cooldowns:
            remaining = self._flapping_cooldowns[cooldown_key] - now
            if remaining > 0:
                reason = (
                    f"Action '{action.action_type}' on '{sys_id}' blocked due to flapping "
                    f"({remaining:.1f}s remaining in cooldown)"
                )
                self._emit_rejection(sys_id, act_type, "flapping_cooldown")
                return False, reason
            else:
                self._flapping_cooldowns.pop(cooldown_key, None)

        # 2. Cluster Rate Limit Check
        if self.config.max_cluster_adaptations_per_window > 0:
            cluster_count = len(self._cluster_history)
            if cluster_count >= self.config.max_cluster_adaptations_per_window:
                reason = (
                    f"Cluster adaptation rate limit reached ({cluster_count}/"
                    f"{self.config.max_cluster_adaptations_per_window} in "
                    f"{self.config.cluster_window_seconds}s)"
                )
                self._emit_rejection(sys_id, act_type, "cluster_rate_limit")
                return False, reason

        # 3. System Rate Limit Check
        if self.config.max_system_adaptations_per_window > 0:
            sys_records = self._system_history.get(sys_id, [])
            sys_cutoff = now - self.config.system_window_seconds
            sys_count = sum(1 for ts, _ in sys_records if ts >= sys_cutoff)
            if sys_count >= self.config.max_system_adaptations_per_window:
                reason = (
                    f"System adaptation rate limit reached for '{sys_id}' ({sys_count}/"
                    f"{self.config.max_system_adaptations_per_window} in "
                    f"{self.config.system_window_seconds}s)"
                )
                self._emit_rejection(sys_id, act_type, "system_rate_limit")
                return False, reason

        # 4. Max Concurrent Cluster Adaptations
        if self.config.max_concurrent_cluster_adaptations > 0:
            if len(self._active_actions) >= self.config.max_concurrent_cluster_adaptations:
                reason = (
                    f"Cluster max concurrent adaptations reached ({len(self._active_actions)}/"
                    f"{self.config.max_concurrent_cluster_adaptations})"
                )
                self._emit_rejection(sys_id, act_type, "max_concurrent_cluster")
                return False, reason

        # 5. Concurrency Mode & Blast Radius Isolation Check
        if self.config.concurrency_mode == "serialized":
            if self._active_actions:
                active_types = [act.action_type for act, _ in self._active_actions.values()]
                reason = (
                    f"Cluster concurrency mode is 'serialized'; waiting for active adaptation(s) "
                    f"{active_types} to complete"
                )
                self._emit_rejection(sys_id, act_type, "serialized_lock")
                return False, reason

        elif self.config.concurrency_mode == "blast_radius_isolated":
            target_impact = {sys_id}
            if topology is not None:
                target_impact = topology.get_impact_radius(sys_id)

            for active_act, active_impact in self._active_actions.values():
                # Direct system conflict
                if active_act.target_system.strip() == sys_id:
                    reason = (
                        f"System '{sys_id}' already has an active in-flight adaptation "
                        f"'{active_act.action_type}'"
                    )
                    self._emit_rejection(sys_id, act_type, "system_busy")
                    return False, reason

                # Topological impact overlap
                overlap = target_impact.intersection(active_impact)
                if overlap:
                    reason = (
                        f"Blast radius of '{sys_id}' overlaps with active adaptation "
                        f"'{active_act.action_type}' on '{active_act.target_system}' "
                        f"(shared nodes: {sorted(overlap)})"
                    )
                    self._emit_rejection(sys_id, act_type, "blast_radius_overlap")
                    return False, reason

        # 6. Mutual Exclusion Conflict Check with In-Flight Actions
        if self.config.mutual_exclusions:
            for active_act, _ in self._active_actions.values():
                if active_act.target_system.strip() == sys_id:
                    active_type = active_act.action_type.strip().lower()
                    for group in self.config.mutual_exclusions:
                        if act_type in group and active_type in group:
                            reason = (
                                f"Action '{action.action_type}' is mutually exclusive with active "
                                f"action '{active_act.action_type}' on '{sys_id}'"
                            )
                            self._emit_rejection(sys_id, act_type, "mutual_exclusion")
                            return False, reason

        # 7. Flapping / Oscillation Pattern Detection
        opposing = self.config.opposing_actions.get(act_type)
        if opposing:
            sys_records = self._system_history.get(sys_id, [])
            flap_cutoff = now - self.config.flapping_window_seconds
            recent_opposing = [
                act for ts, act in sys_records if ts >= flap_cutoff and act in (act_type, opposing)
            ]

            if len(recent_opposing) >= self.config.flapping_threshold:
                # Check if there is alternating pattern
                transitions = 0
                for i in range(1, len(recent_opposing)):
                    if recent_opposing[i] != recent_opposing[i - 1]:
                        transitions += 1

                if transitions >= (self.config.flapping_threshold - 1):
                    expires = now + self.config.flapping_cooldown_seconds
                    self._flapping_cooldowns[(sys_id, act_type)] = expires
                    self._flapping_cooldowns[(sys_id, opposing)] = expires
                    reason = (
                        f"Flapping detected on '{sys_id}' between '{act_type}' and '{opposing}'. "
                        f"Activated {self.config.flapping_cooldown_seconds}s cooldown."
                    )
                    if self.logger:
                        self.logger.warning(
                            "Safety guardrail tripped anti-flapping cooldown",
                            system_id=sys_id,
                            action_type=action.action_type,
                            opposing_action=opposing,
                            cooldown_seconds=self.config.flapping_cooldown_seconds,
                        )
                    self._emit_rejection(sys_id, act_type, "flapping_tripped")
                    return False, reason

        return True, None

    def record_action_start(
        self,
        action: AdaptationAction,
        topology: Optional[SystemTopology] = None,
    ) -> None:
        """Record the start of an action execution."""
        if not self.config.enabled:
            return

        now = time.monotonic()
        sys_id = action.target_system.strip()
        act_type = action.action_type.strip().lower()

        self._cluster_history.append(now)
        if sys_id not in self._system_history:
            self._system_history[sys_id] = []
        self._system_history[sys_id].append((now, act_type))

        impact_nodes = {sys_id}
        if topology is not None:
            impact_nodes = topology.get_impact_radius(sys_id)

        self._active_actions[action.action_id] = (action, impact_nodes)

        if self.metrics:
            self.metrics.increment(
                "polaris.safety.action_started",
                tags={"system_id": sys_id, "action_type": act_type},
            )
            self.metrics.gauge("polaris.safety.active_adaptations", len(self._active_actions))

    def record_action_end(
        self,
        action: AdaptationAction,
        success: bool = True,
    ) -> None:
        """Record the completion of an action execution."""
        if not self.config.enabled:
            return

        self._active_actions.pop(action.action_id, None)

        if self.metrics:
            self.metrics.increment(
                "polaris.safety.action_completed",
                tags={
                    "system_id": action.target_system.strip(),
                    "action_type": action.action_type.strip().lower(),
                    "success": str(success).lower(),
                },
            )
            self.metrics.gauge("polaris.safety.active_adaptations", len(self._active_actions))

    def get_active_actions(self) -> List[AdaptationAction]:
        """Return list of currently in-flight adaptation actions."""
        return [act for act, _ in self._active_actions.values()]

    def is_system_active(self, system_id: str) -> bool:
        """Check if a specific system has an active adaptation in flight."""
        target = system_id.strip()
        return any(act.target_system.strip() == target for act, _ in self._active_actions.values())

    def get_active_cooldowns(self, system_id: str) -> Dict[str, float]:
        """Return remaining seconds of any active cooldowns for a system."""
        now = time.monotonic()
        res: Dict[str, float] = {}
        target = system_id.strip()
        for (sid, act), expires in self._flapping_cooldowns.items():
            if sid == target and expires > now:
                res[act] = expires - now
        return res

    def reset(self) -> None:
        """Reset all in-flight tracking and history (useful in tests)."""
        self._cluster_history.clear()
        self._system_history.clear()
        self._active_actions.clear()
        self._flapping_cooldowns.clear()

    def _emit_rejection(self, system_id: str, action_type: str, reason_code: str) -> None:
        if self.metrics:
            self.metrics.increment(
                "polaris.safety.action_rejected",
                tags={
                    "system_id": system_id,
                    "action_type": action_type,
                    "reason": reason_code,
                },
            )
