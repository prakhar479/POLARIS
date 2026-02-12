"""In-memory knowledge store implementation."""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.core.models import AdaptationAction, ExecutionResult, SystemState


class InMemoryKnowledgeStore(KnowledgeStore):
    """
    Simple in-memory knowledge store.

    Good for testing and simple deployments. Data is not persisted.
    """

    def __init__(
        self,
        max_states_per_system: int = 1000,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize in-memory knowledge store with capacity limits."""
        self.max_states = max_states_per_system
        self._states: Dict[str, List[SystemState]] = defaultdict(list)
        self._actions: Dict[str, List[Tuple[AdaptationAction, ExecutionResult]]] = defaultdict(list)
        self._logger = logger
        self._metrics = metrics

    async def store_state(self, state: SystemState) -> None:
        """Store system state (keeping max_states most recent)."""
        states = self._states[state.system_id]
        states.append(state)

        if self._metrics:
            self._metrics.increment(
                "polaris.knowledge.inmemory.states_stored",
                tags={"system_id": state.system_id},
            )

        # Keep only most recent states
        if len(states) > self.max_states:
            if self._logger:
                self._logger.debug(
                    "InMemoryKnowledgeStore trimming states",
                    system_id=state.system_id,
                    max_states=self.max_states,
                    previous_count=len(states),
                )
            self._states[state.system_id] = states[-self.max_states :]

        if self._metrics:
            self._metrics.gauge(
                "polaris.knowledge.inmemory.states_per_system",
                len(self._states[state.system_id]),
                tags={"system_id": state.system_id},
            )

    async def store_action(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Store adaptation action and result."""
        system_id = action.target_system
        self._actions[system_id].append((action, result))

        if self._metrics:
            status = (
                result.status.value
                if hasattr(result, "status") and hasattr(result.status, "value")
                else "unknown"
            )
            self._metrics.increment(
                "polaris.knowledge.inmemory.actions_stored",
                tags={"system_id": system_id, "status": status},
            )

        # Keep only most recent actions
        if len(self._actions[system_id]) > self.max_states:
            if self._logger:
                self._logger.debug(
                    "InMemoryKnowledgeStore trimming actions",
                    system_id=system_id,
                    max_actions=self.max_states,
                    previous_count=len(self._actions[system_id]),
                )
            self._actions[system_id] = self._actions[system_id][-self.max_states :]

        if self._metrics:
            self._metrics.gauge(
                "polaris.knowledge.inmemory.actions_per_system",
                len(self._actions[system_id]),
                tags={"system_id": system_id},
            )

    async def query_states(
        self, system_id: str, start_time: datetime, end_time: datetime
    ) -> List[SystemState]:
        """Query states in time range."""
        if self._metrics:
            self._metrics.increment(
                "polaris.knowledge.inmemory.state_queries",
                tags={"system_id": system_id},
            )

        states = self._states.get(system_id, [])
        results = [s for s in states if start_time <= s.timestamp <= end_time]

        if self._metrics:
            self._metrics.gauge(
                "polaris.knowledge.inmemory.state_query_results",
                len(results),
                tags={"system_id": system_id},
            )

        return results

    async def query_actions(
        self, system_id: str, start_time: Optional[datetime], end_time: Optional[datetime]
    ) -> List[Tuple[AdaptationAction, ExecutionResult]]:
        """Query adaptation history."""
        if self._metrics:
            self._metrics.increment(
                "polaris.knowledge.inmemory.action_queries",
                tags={"system_id": system_id},
            )

        actions = self._actions.get(system_id, [])
        results = [
            (action, result)
            for action, result in actions
            if (
                start_time is None
                or action.created_at is not None
                and start_time <= action.created_at
            )
            and (
                end_time is None or action.created_at is not None and action.created_at <= end_time
            )
        ]

        if self._metrics:
            self._metrics.gauge(
                "polaris.knowledge.inmemory.action_query_results",
                len(results),
                tags={"system_id": system_id},
            )

        return results
