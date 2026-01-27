"""
In-memory knowledge store implementation.
"""

from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.core.models import SystemState, AdaptationAction, ExecutionResult


class InMemoryKnowledgeStore(KnowledgeStore):
    """
    Simple in-memory knowledge store.

    Good for testing and simple deployments. Data is not persisted.
    """

    def __init__(self, max_states_per_system: int = 1000):
        self.max_states = max_states_per_system
        self._states: Dict[str, List[SystemState]] = defaultdict(list)
        self._actions: Dict[str, List[tuple]] = defaultdict(list)

    async def store_state(self, state: SystemState) -> None:
        """Store system state (keeping max_states most recent)."""
        states = self._states[state.system_id]
        states.append(state)

        # Keep only most recent states
        if len(states) > self.max_states:
            self._states[state.system_id] = states[-self.max_states:]

    async def store_action(
        self,
        action: AdaptationAction,
        result: ExecutionResult
    ) -> None:
        """Store adaptation action and result."""
        self._actions[action.target_system].append((action, result))

        # Keep only most recent actions
        if len(self._actions[action.target_system]) > self.max_states:
            self._actions[action.target_system] = \
                self._actions[action.target_system][-self.max_states:]

    async def query_states(
        self,
        system_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[SystemState]:
        """Query states in time range."""
        states = self._states.get(system_id, [])
        return [
            s for s in states
            if start_time <= s.timestamp <= end_time
        ]

    async def query_actions(
        self,
        system_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[tuple]:
        """Query adaptation history."""
        actions = self._actions.get(system_id, [])
        return [
            (action, result) for action, result in actions
            if start_time <= action.created_at <= end_time
        ]
