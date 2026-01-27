"""Tests for in-memory knowledge store."""

import pytest
from datetime import datetime, timezone, timedelta

from polaris.knowledge.memory import InMemoryKnowledgeStore
from polaris.core.models import SystemState, AdaptationAction, ExecutionResult, MetricValue, HealthStatus, ExecutionStatus


class TestInMemoryKnowledgeStore:
    """Test InMemoryKnowledgeStore functionality."""
    
    @pytest.fixture
    def knowledge_store(self):
        """Create knowledge store with small max size for testing."""
        return InMemoryKnowledgeStore(max_states_per_system=3)
    
    @pytest.fixture
    def sample_states(self):
        """Create sample system states."""
        base_time = datetime.now(timezone.utc)
        states = []
        
        for i in range(5):
            state = SystemState(
                system_id="test-system",
                timestamp=base_time + timedelta(minutes=i),
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", 50.0 + i * 10, "percent")
                },
                health_status=HealthStatus.HEALTHY
            )
            states.append(state)
        
        return states
    
    @pytest.fixture
    def sample_actions(self):
        """Create sample actions and results."""
        base_time = datetime.now(timezone.utc)
        actions_results = []
        
        for i in range(3):
            action = AdaptationAction(
                action_id=f"action-{i}",
                action_type="scale_up",
                target_system="test-system",
                parameters={"instances": i + 1},
                created_at=base_time + timedelta(minutes=i)
            )
            
            result = ExecutionResult(
                action_id=f"action-{i}",
                status=ExecutionStatus.SUCCESS,
                result_data={"message": f"Action {i} completed"}
            )
            
            actions_results.append((action, result))
        
        return actions_results
    
    @pytest.mark.asyncio
    async def test_store_single_state(self, knowledge_store, sample_states):
        """Test storing a single system state."""
        state = sample_states[0]
        
        await knowledge_store.store_state(state)
        
        # Verify state is stored
        assert "test-system" in knowledge_store._states
        assert len(knowledge_store._states["test-system"]) == 1
        assert knowledge_store._states["test-system"][0] == state
    
    @pytest.mark.asyncio
    async def test_store_multiple_states(self, knowledge_store, sample_states):
        """Test storing multiple states."""
        for state in sample_states[:2]:
            await knowledge_store.store_state(state)
        
        stored_states = knowledge_store._states["test-system"]
        assert len(stored_states) == 2
        assert stored_states[0] == sample_states[0]
        assert stored_states[1] == sample_states[1]
    
    @pytest.mark.asyncio
    async def test_state_limit_enforcement(self, knowledge_store, sample_states):
        """Test that state limit is enforced."""
        # Store more states than the limit (3)
        for state in sample_states:  # 5 states
            await knowledge_store.store_state(state)
        
        stored_states = knowledge_store._states["test-system"]
        
        # Should only keep the last 3 states
        assert len(stored_states) == 3
        assert stored_states[0] == sample_states[2]  # States 2, 3, 4 should remain
        assert stored_states[1] == sample_states[3]
        assert stored_states[2] == sample_states[4]
    
    @pytest.mark.asyncio
    async def test_store_action(self, knowledge_store, sample_actions):
        """Test storing action and result."""
        action, result = sample_actions[0]
        
        await knowledge_store.store_action(action, result)
        
        # Verify action is stored
        assert "test-system" in knowledge_store._actions
        assert len(knowledge_store._actions["test-system"]) == 1
        assert knowledge_store._actions["test-system"][0] == (action, result)
    
    @pytest.mark.asyncio
    async def test_action_limit_enforcement(self, knowledge_store, sample_actions):
        """Test that action limit is enforced."""
        # Create more actions than the limit
        base_time = datetime.now(timezone.utc)
        for i in range(5):  # More than limit of 3
            action = AdaptationAction(
                action_id=f"action-{i}",
                action_type="scale_up",
                target_system="test-system",
                parameters={},
                created_at=base_time + timedelta(minutes=i)
            )
            result = ExecutionResult(
                action_id=f"action-{i}",
                status=ExecutionStatus.SUCCESS,
                result_data={}
            )
            await knowledge_store.store_action(action, result)
        
        stored_actions = knowledge_store._actions["test-system"]
        
        # Should only keep the last 3 actions
        assert len(stored_actions) == 3
        assert stored_actions[0][0].action_id == "action-2"
        assert stored_actions[1][0].action_id == "action-3"
        assert stored_actions[2][0].action_id == "action-4"
    
    @pytest.mark.asyncio
    async def test_query_states_time_range(self, knowledge_store, sample_states):
        """Test querying states within time range."""
        # Store all states
        for state in sample_states:
            await knowledge_store.store_state(state)
        
        # Query for states in middle time range
        start_time = sample_states[1].timestamp
        end_time = sample_states[3].timestamp
        
        results = await knowledge_store.query_states("test-system", start_time, end_time)
        
        # Should return states 1, 2, 3 (but only 2, 3, 4 are stored due to limit)
        # So should return states 2, 3
        assert len(results) == 2
        assert results[0].timestamp >= start_time
        assert results[1].timestamp <= end_time
    
    @pytest.mark.asyncio
    async def test_query_states_no_results(self, knowledge_store, sample_states):
        """Test querying states with no matching time range."""
        # Store states
        await knowledge_store.store_state(sample_states[0])
        
        # Query for future time range
        future_start = datetime.now(timezone.utc) + timedelta(hours=1)
        future_end = datetime.now(timezone.utc) + timedelta(hours=2)
        
        results = await knowledge_store.query_states("test-system", future_start, future_end)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_query_states_nonexistent_system(self, knowledge_store):
        """Test querying states for non-existent system."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        results = await knowledge_store.query_states("nonexistent-system", start_time, end_time)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_query_actions_time_range(self, knowledge_store, sample_actions):
        """Test querying actions within time range."""
        # Store all actions
        for action, result in sample_actions:
            await knowledge_store.store_action(action, result)
        
        # Query for actions in time range
        start_time = sample_actions[0][0].created_at
        end_time = sample_actions[2][0].created_at
        
        results = await knowledge_store.query_actions("test-system", start_time, end_time)
        
        assert len(results) == 3
        for i, (action, result) in enumerate(results):
            assert action.action_id == f"action-{i}"
    
    @pytest.mark.asyncio
    async def test_query_actions_no_results(self, knowledge_store, sample_actions):
        """Test querying actions with no matching time range."""
        # Store action
        await knowledge_store.store_action(sample_actions[0][0], sample_actions[0][1])
        
        # Query for past time range
        past_start = datetime.now(timezone.utc) - timedelta(hours=2)
        past_end = datetime.now(timezone.utc) - timedelta(hours=1)
        
        results = await knowledge_store.query_actions("test-system", past_start, past_end)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_systems(self, knowledge_store):
        """Test storing data for multiple systems."""
        # Create states for different systems
        state1 = SystemState(
            system_id="system-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu": MetricValue("cpu", 50.0)},
            health_status=HealthStatus.HEALTHY
        )
        
        state2 = SystemState(
            system_id="system-2",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu": MetricValue("cpu", 60.0)},
            health_status=HealthStatus.HEALTHY
        )
        
        await knowledge_store.store_state(state1)
        await knowledge_store.store_state(state2)
        
        # Verify both systems are stored separately
        assert len(knowledge_store._states) == 2
        assert "system-1" in knowledge_store._states
        assert "system-2" in knowledge_store._states
        assert knowledge_store._states["system-1"][0] == state1
        assert knowledge_store._states["system-2"][0] == state2
    
    def test_initialization_with_custom_limit(self):
        """Test initialization with custom max states limit."""
        store = InMemoryKnowledgeStore(max_states_per_system=100)
        assert store.max_states == 100
    
    def test_default_initialization(self):
        """Test default initialization."""
        store = InMemoryKnowledgeStore()
        assert store.max_states == 1000