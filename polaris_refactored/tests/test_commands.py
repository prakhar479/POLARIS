"""
Tests for the POLARIS Command System

Comprehensive tests for command pattern implementation, validation, execution,
and integration with the event system.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from typing import List

from ..src.framework.commands import (
    PolarisAdaptationCommand, CommandStatus, CommandPriority, CommandContext,
    CommandResult, CommandQueue, PolarisCommandProcessor, CommandValidationError,
    CommandExecutionError
)
from ..src.framework.events import PolarisEventBus, ExecutionResultEvent
from ..src.domain.models import (
    AdaptationAction, ExecutionResult, SystemState, MetricValue, HealthStatus, ExecutionStatus
)
from ..src.domain.interfaces import ManagedSystemConnector


class MockManagedSystemConnector(ManagedSystemConnector):
    """Mock connector for testing commands."""
    
    def __init__(self, system_id: str = "test_system"):
        self.system_id = system_id
        self.connected = False
        self.should_validate = True
        self.should_fail_execution = False
        self.execution_delay = 0.0
        self.executed_actions: List[AdaptationAction] = []
    
    async def connect(self) -> bool:
        self.connected = True
        return True
    
    async def disconnect(self) -> bool:
        self.connected = False
        return True
    
    async def get_system_id(self) -> str:
        return self.system_id
    
    async def collect_metrics(self) -> dict:
        return {
            "cpu_usage": MetricValue("cpu_usage", 50.0, "percent"),
            "memory_usage": MetricValue("memory_usage", 60.0, "percent"),
            "instance_count": MetricValue("instance_count", 3, "count")
        }
    
    async def get_system_state(self) -> SystemState:
        metrics = await self.collect_metrics()
        return SystemState(
            system_id=self.system_id,
            health_status=HealthStatus.HEALTHY,
            metrics=metrics,
            timestamp=datetime.utcnow()
        )
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        if self.execution_delay > 0:
            await asyncio.sleep(self.execution_delay)
        
        if self.should_fail_execution:
            raise Exception("Mock execution failure")
        
        self.executed_actions.append(action)
        
        return ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.SUCCESS,
            result_data={"executed": True, "system_id": self.system_id}
        )
    
    async def validate_action(self, action: AdaptationAction) -> bool:
        return self.should_validate
    
    async def get_supported_actions(self) -> List[str]:
        return ["SCALE_UP", "SCALE_DOWN", "RESTART"]


class TestPolarisAdaptationCommand:
    """Test the PolarisAdaptationCommand class."""
    
    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for testing."""
        return MockManagedSystemConnector()
    
    @pytest.fixture
    def sample_action(self):
        """Create a sample adaptation action."""
        return AdaptationAction(
            action_id="test_action_001",
            action_type="SCALE_UP",
            target_system="test_system",
            parameters={"instances": 2}
        )
    
    def test_command_creation(self, sample_action, mock_connector):
        """Test basic command creation."""
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector,
            priority=CommandPriority.HIGH
        )
        
        assert command.command_id is not None
        assert command.action == sample_action
        assert command.connector == mock_connector
        assert command.priority == CommandPriority.HIGH
        assert command.status == CommandStatus.PENDING
        assert command.created_at is not None
    
    def test_get_action(self, sample_action, mock_connector):
        """Test getting the associated action."""
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector
        )
        
        assert command.get_action() == sample_action
    
    @pytest.mark.asyncio
    async def test_can_execute_valid(self, sample_action, mock_connector):
        """Test validation of a valid command."""
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector
        )
        
        can_execute = await command.can_execute()
        assert can_execute is True
        assert len(command.validation_errors) == 0
        assert command.status == CommandStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_can_execute_invalid_connector_validation(self, sample_action, mock_connector):
        """Test validation failure at connector level."""
        mock_connector.should_validate = False
        
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector
        )
        
        can_execute = await command.can_execute()
        assert can_execute is False
        assert len(command.validation_errors) > 0
        assert command.status == CommandStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_can_execute_timeout_exceeded(self, sample_action, mock_connector):
        """Test validation failure due to timeout."""
        context = CommandContext(timeout=timedelta(seconds=0.1))
        
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector,
            context=context
        )
        
        # Wait for timeout to exceed
        await asyncio.sleep(0.2)
        
        can_execute = await command.can_execute()
        assert can_execute is False
        assert any("timeout" in error.lower() for error in command.validation_errors)
    
    @pytest.mark.asyncio
    async def test_execute_success(self, sample_action, mock_connector):
        """Test successful command execution."""
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector
        )
        
        result = await command.execute()
        
        assert result is not None
        assert result.status == ExecutionStatus.SUCCESS
        assert command.status == CommandStatus.COMPLETED
        assert len(command.execution_history) == 1
        assert len(mock_connector.executed_actions) == 1
    
    @pytest.mark.asyncio
    async def test_execute_failure(self, sample_action, mock_connector):
        """Test command execution failure."""
        mock_connector.should_fail_execution = True
        
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector
        )
        
        with pytest.raises(CommandExecutionError):
            await command.execute()
        
        assert command.status == CommandStatus.FAILED
        assert len(command.execution_history) == 1
        assert command.execution_history[0].error is not None
    
    @pytest.mark.asyncio
    async def test_system_state_validation_unhealthy(self, sample_action, mock_connector):
        """Test validation failure for unhealthy system state."""
        # Create unhealthy system state
        unhealthy_state = SystemState(
            system_id="test_system",
            health_status=HealthStatus.UNHEALTHY,
            metrics={},
            timestamp=datetime.utcnow()
        )
        
        context = CommandContext(system_state=unhealthy_state)
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector,
            context=context
        )
        
        can_execute = await command.can_execute()
        assert can_execute is False
        assert any("unhealthy" in error.lower() for error in command.validation_errors)
    
    @pytest.mark.asyncio
    async def test_scale_up_validation_high_cpu(self, mock_connector):
        """Test scale up validation with high CPU usage."""
        # Create action for scaling up
        scale_up_action = AdaptationAction(
            action_id="scale_up_001",
            action_type="SCALE_UP",
            target_system="test_system",
            parameters={"instances": 2}
        )
        
        # Mock high CPU usage
        async def mock_get_system_state():
            return SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", 98.0, "percent")
                },
                timestamp=datetime.utcnow()
            )
        
        mock_connector.get_system_state = mock_get_system_state
        
        command = PolarisAdaptationCommand(
            action=scale_up_action,
            connector=mock_connector
        )
        
        can_execute = await command.can_execute()
        assert can_execute is False
        assert any("cpu usage too high" in error.lower() for error in command.validation_errors)
    
    @pytest.mark.asyncio
    async def test_scale_down_validation_minimum_instances(self, mock_connector):
        """Test scale down validation with minimum instances."""
        # Create action for scaling down
        scale_down_action = AdaptationAction(
            action_id="scale_down_001",
            action_type="SCALE_DOWN",
            target_system="test_system",
            parameters={"instances": 1}
        )
        
        # Mock minimum instance count
        async def mock_get_system_state():
            return SystemState(
                system_id="test_system",
                health_status=HealthStatus.HEALTHY,
                metrics={
                    "instance_count": MetricValue("instance_count", 1, "count")
                },
                timestamp=datetime.utcnow()
            )
        
        mock_connector.get_system_state = mock_get_system_state
        
        command = PolarisAdaptationCommand(
            action=scale_down_action,
            connector=mock_connector
        )
        
        can_execute = await command.can_execute()
        assert can_execute is False
        assert any("minimum instances" in error.lower() for error in command.validation_errors)
    
    def test_undo_actions(self, sample_action, mock_connector):
        """Test undo action management."""
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector
        )
        
        undo_called = False
        
        async def mock_undo():
            nonlocal undo_called
            undo_called = True
        
        command.add_undo_action(mock_undo)
        assert len(command._undo_actions) == 1
    
    @pytest.mark.asyncio
    async def test_undo_execution(self, sample_action, mock_connector):
        """Test undo execution."""
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector
        )
        
        undo_called = False
        
        async def mock_undo():
            nonlocal undo_called
            undo_called = True
        
        command.add_undo_action(mock_undo)
        
        # Execute command first
        await command.execute()
        assert command.status == CommandStatus.COMPLETED
        
        # Now undo
        success = await command.undo()
        assert success is True
        assert undo_called is True
        assert command.status == CommandStatus.UNDONE
    
    def test_execution_summary(self, sample_action, mock_connector):
        """Test getting execution summary."""
        command = PolarisAdaptationCommand(
            action=sample_action,
            connector=mock_connector,
            priority=CommandPriority.HIGH
        )
        
        summary = command.get_execution_summary()
        
        assert "command_id" in summary
        assert "action_id" in summary
        assert "action_type" in summary
        assert "status" in summary
        assert "priority" in summary
        assert summary["action_type"] == "SCALE_UP"
        assert summary["priority"] == CommandPriority.HIGH.value


class TestCommandQueue:
    """Test the CommandQueue class."""
    
    @pytest.fixture
    def command_queue(self):
        """Create a command queue for testing."""
        return CommandQueue()
    
    @pytest.fixture
    def sample_commands(self):
        """Create sample commands with different priorities."""
        mock_connector = MockManagedSystemConnector()
        
        commands = []
        for i, priority in enumerate([CommandPriority.LOW, CommandPriority.HIGH, CommandPriority.NORMAL]):
            action = AdaptationAction(
                action_id=f"action_{i}",
                action_type="SCALE_UP",
                target_system="test_system",
                parameters={}
            )
            command = PolarisAdaptationCommand(
                action=action,
                connector=mock_connector,
                priority=priority
            )
            commands.append(command)
        
        return commands
    
    @pytest.mark.asyncio
    async def test_enqueue_and_dequeue(self, command_queue, sample_commands):
        """Test basic enqueue and dequeue operations."""
        # Enqueue commands
        for command in sample_commands:
            await command_queue.enqueue(command)
        
        assert await command_queue.size() == 3
        
        # Dequeue should return highest priority first
        first_command = await command_queue.dequeue()
        assert first_command.priority == CommandPriority.HIGH
        
        second_command = await command_queue.dequeue()
        assert second_command.priority == CommandPriority.NORMAL
        
        third_command = await command_queue.dequeue()
        assert third_command.priority == CommandPriority.LOW
        
        assert await command_queue.size() == 0
    
    @pytest.mark.asyncio
    async def test_peek(self, command_queue, sample_commands):
        """Test peek operation."""
        await command_queue.enqueue(sample_commands[0])  # LOW priority
        
        peeked_command = await command_queue.peek()
        assert peeked_command == sample_commands[0]
        assert await command_queue.size() == 1  # Should not remove from queue
    
    @pytest.mark.asyncio
    async def test_clear(self, command_queue, sample_commands):
        """Test clearing the queue."""
        for command in sample_commands:
            await command_queue.enqueue(command)
        
        assert await command_queue.size() == 3
        
        await command_queue.clear()
        assert await command_queue.size() == 0


class TestPolarisCommandProcessor:
    """Test the PolarisCommandProcessor class."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create an event bus for testing."""
        bus = PolarisEventBus(worker_count=1)
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    async def command_processor(self, event_bus):
        """Create a command processor for testing."""
        processor = PolarisCommandProcessor(
            event_bus=event_bus,
            max_concurrent_commands=2,
            default_timeout=timedelta(seconds=10)
        )
        await processor.start()
        yield processor
        await processor.stop()
    
    @pytest.fixture
    def sample_command(self):
        """Create a sample command for testing."""
        mock_connector = MockManagedSystemConnector()
        action = AdaptationAction(
            action_id="test_action",
            action_type="SCALE_UP",
            target_system="test_system",
            parameters={"instances": 2}
        )
        return PolarisAdaptationCommand(
            action=action,
            connector=mock_connector
        )
    
    @pytest.mark.asyncio
    async def test_processor_lifecycle(self, event_bus):
        """Test processor start and stop."""
        processor = PolarisCommandProcessor(event_bus=event_bus)
        
        assert not processor._running
        
        await processor.start()
        assert processor._running
        assert processor._processor_task is not None
        
        await processor.stop()
        assert not processor._running
    
    @pytest.mark.asyncio
    async def test_submit_and_execute_command(self, command_processor, sample_command):
        """Test submitting and executing a command."""
        # Submit command
        command_id = await command_processor.submit_command(sample_command)
        assert command_id == sample_command.command_id
        
        # Wait for execution
        await asyncio.sleep(0.2)
        
        # Check status
        status = await command_processor.get_command_status(command_id)
        assert status == CommandStatus.COMPLETED
        
        # Check result
        result = await command_processor.get_command_result(command_id)
        assert result is not None
        assert result.is_successful
    
    @pytest.mark.asyncio
    async def test_command_validation_failure(self, command_processor):
        """Test handling of command validation failure."""
        mock_connector = MockManagedSystemConnector()
        mock_connector.should_validate = False
        
        action = AdaptationAction(
            action_id="invalid_action",
            action_type="INVALID_TYPE",
            target_system="test_system",
            parameters={}
        )
        
        command = PolarisAdaptationCommand(
            action=action,
            connector=mock_connector
        )
        
        command_id = await command_processor.submit_command(command)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Check status
        status = await command_processor.get_command_status(command_id)
        assert status == CommandStatus.FAILED
        
        result = await command_processor.get_command_result(command_id)
        assert result is not None
        assert not result.is_successful
    
    @pytest.mark.asyncio
    async def test_command_execution_failure_with_retry(self, command_processor):
        """Test command execution failure with retry."""
        mock_connector = MockManagedSystemConnector()
        mock_connector.should_fail_execution = True
        
        action = AdaptationAction(
            action_id="failing_action",
            action_type="SCALE_UP",
            target_system="test_system",
            parameters={}
        )
        
        context = CommandContext(max_retries=2)
        command = PolarisAdaptationCommand(
            action=action,
            connector=mock_connector,
            context=context
        )
        
        command_id = await command_processor.submit_command(command)
        
        # Wait for processing and retries
        await asyncio.sleep(0.5)
        
        # Should eventually fail after retries
        status = await command_processor.get_command_status(command_id)
        assert status == CommandStatus.FAILED
        
        result = await command_processor.get_command_result(command_id)
        assert result is not None
        assert not result.is_successful
    
    @pytest.mark.asyncio
    async def test_concurrent_command_execution(self, command_processor):
        """Test concurrent execution of multiple commands."""
        mock_connector = MockManagedSystemConnector()
        mock_connector.execution_delay = 0.1  # Add small delay
        
        commands = []
        for i in range(3):
            action = AdaptationAction(
                action_id=f"concurrent_action_{i}",
                action_type="SCALE_UP",
                target_system="test_system",
                parameters={}
            )
            command = PolarisAdaptationCommand(
                action=action,
                connector=mock_connector
            )
            commands.append(command)
        
        # Submit all commands
        command_ids = []
        for command in commands:
            command_id = await command_processor.submit_command(command)
            command_ids.append(command_id)
        
        # Wait for execution
        await asyncio.sleep(0.5)
        
        # All commands should be completed
        for command_id in command_ids:
            status = await command_processor.get_command_status(command_id)
            assert status == CommandStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_event_publishing(self, command_processor, sample_command):
        """Test that execution results are published as events."""
        handled_events = []
        
        async def event_handler(event):
            handled_events.append(event)
        
        # Subscribe to execution result events
        subscription_id = command_processor.event_bus.subscribe(ExecutionResultEvent, event_handler)
        
        # Submit and execute command
        command_id = await command_processor.submit_command(sample_command)
        
        # Wait for execution and event processing
        await asyncio.sleep(0.3)
        
        # Should have received execution result event
        assert len(handled_events) == 1
        assert isinstance(handled_events[0], ExecutionResultEvent)
        assert handled_events[0].action_id == sample_command.action.action_id
        
        # Cleanup
        await command_processor.event_bus.unsubscribe(subscription_id)
    
    @pytest.mark.asyncio
    async def test_processing_stats(self, command_processor, sample_command):
        """Test getting processing statistics."""
        # Submit and execute command
        await command_processor.submit_command(sample_command)
        
        # Wait for execution
        await asyncio.sleep(0.2)
        
        # Get stats
        stats = command_processor.get_processing_stats()
        
        assert "active_commands" in stats
        assert "total_processed" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert "success_rate" in stats
        assert "is_running" in stats
        assert stats["is_running"] is True
        assert stats["total_processed"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])