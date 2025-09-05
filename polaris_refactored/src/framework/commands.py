"""
Command Pattern Implementation for POLARIS

Provides a comprehensive command system for adaptation actions with validation,
execution tracking, undo capabilities, and integration with the event system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..domain.models import AdaptationAction, ExecutionResult, SystemState
from ..domain.interfaces import AdaptationCommand, ManagedSystemConnector
from ..infrastructure.di import Injectable
from ..infrastructure.exceptions import PolarisException
from .events import PolarisEventBus, ExecutionResultEvent, EventMetadata

logger = logging.getLogger(__name__)


class CommandStatus(Enum):
    """Status of command execution."""
    PENDING = "pending"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNDOING = "undoing"
    UNDONE = "undone"


class CommandPriority(Enum):
    """Priority levels for command execution."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CommandContext:
    """Context information for command execution."""
    system_state: Optional[SystemState] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class CommandResult:
    """Result of command execution."""
    command_id: str
    status: CommandStatus
    execution_result: Optional[ExecutionResult] = None
    error: Optional[Exception] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[timedelta] = None
    context: Optional[CommandContext] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if command execution was successful."""
        return self.status == CommandStatus.COMPLETED and self.error is None
    
    @property
    def execution_time(self) -> Optional[timedelta]:
        """Get execution time if available."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return self.duration


class CommandValidationError(PolarisException):
    """Exception raised when command validation fails."""
    
    def __init__(self, message: str, command_id: str, validation_errors: List[str], **kwargs):
        context = kwargs.pop('context', {})
        context.update({
            'command_id': command_id,
            'validation_errors': validation_errors
        })
        
        super().__init__(
            message=message,
            error_code="COMMAND_VALIDATION_ERROR",
            context=context,
            **kwargs
        )
        self.command_id = command_id
        self.validation_errors = validation_errors


class CommandExecutionError(PolarisException):
    """Exception raised when command execution fails."""
    
    def __init__(self, message: str, command_id: str, cause: Exception = None, **kwargs):
        context = kwargs.pop('context', {})
        context['command_id'] = command_id
        
        super().__init__(
            message=message,
            error_code="COMMAND_EXECUTION_ERROR",
            context=context,
            cause=cause,
            **kwargs
        )
        self.command_id = command_id


class PolarisAdaptationCommand(AdaptationCommand):
    """
    Concrete implementation of AdaptationCommand with comprehensive features.
    
    Features:
    - Validation and pre-execution checks
    - Execution tracking and monitoring
    - Undo/rollback capabilities
    - Integration with event system
    - Timeout and retry handling
    """
    
    def __init__(
        self,
        action: AdaptationAction,
        connector: ManagedSystemConnector,
        command_id: Optional[str] = None,
        priority: CommandPriority = CommandPriority.NORMAL,
        context: Optional[CommandContext] = None
    ):
        self.command_id = command_id or str(uuid.uuid4())
        self.action = action
        self.connector = connector
        self.priority = priority
        self.context = context or CommandContext()
        self.status = CommandStatus.PENDING
        self.created_at = datetime.utcnow()
        self.validation_errors: List[str] = []
        self.execution_history: List[CommandResult] = []
        self._undo_actions: List[Callable] = []
        
        logger.info(f"Created command {self.command_id} for action {action.action_id}")
    
    async def execute(self) -> ExecutionResult:
        """Execute the adaptation command with full lifecycle management."""
        result = CommandResult(
            command_id=self.command_id,
            status=CommandStatus.EXECUTING,
            started_at=datetime.utcnow(),
            context=self.context
        )
        
        try:
            # Update status
            self.status = CommandStatus.EXECUTING
            
            # Execute the action
            logger.info(f"Executing command {self.command_id}")
            execution_result = await self.connector.execute_action(self.action)
            
            # Update result
            result.execution_result = execution_result
            result.status = CommandStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.duration = result.completed_at - result.started_at
            
            self.status = CommandStatus.COMPLETED
            
            logger.info(f"Command {self.command_id} completed successfully")
            return execution_result
            
        except Exception as e:
            # Handle execution failure
            result.error = e
            result.status = CommandStatus.FAILED
            result.completed_at = datetime.utcnow()
            result.duration = result.completed_at - result.started_at
            
            self.status = CommandStatus.FAILED
            
            logger.error(f"Command {self.command_id} failed: {e}")
            raise CommandExecutionError(
                f"Command execution failed: {e}",
                self.command_id,
                e
            )
        
        finally:
            # Record execution history
            self.execution_history.append(result)
    
    async def can_execute(self) -> bool:
        """Check if the command can be executed."""
        try:
            self.status = CommandStatus.VALIDATING
            
            # Basic validation
            if not self.action:
                self.validation_errors.append("No action specified")
                return False
            
            if not self.connector:
                self.validation_errors.append("No connector specified")
                return False
            
            # Validate with connector
            is_valid = await self.connector.validate_action(self.action)
            if not is_valid:
                self.validation_errors.append("Action validation failed at connector level")
                return False
            
            # Always validate against current system state
            if not await self._validate_system_state():
                return False
            
            # Check timeout constraints
            if self.context.timeout:
                if datetime.utcnow() - self.created_at > self.context.timeout:
                    self.validation_errors.append("Command has exceeded timeout before execution")
                    return False
            
            logger.info(f"Command {self.command_id} validation passed")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Validation error: {e}")
            logger.error(f"Command {self.command_id} validation failed: {e}")
            return False
        
        finally:
            if self.validation_errors:
                self.status = CommandStatus.FAILED
            else:
                self.status = CommandStatus.PENDING
    
    async def _validate_system_state(self) -> bool:
        """Validate command against current system state."""
        try:
            # Use provided system state or get current state from connector
            if self.context.system_state:
                current_state = self.context.system_state
            else:
                current_state = await self.connector.get_system_state()
            
            # Basic health check
            if hasattr(current_state, 'health_status'):
                # Handle both enum and string values
                health_status = current_state.health_status
                if hasattr(health_status, 'value'):
                    health_value = health_status.value.lower()
                elif hasattr(health_status, 'name'):
                    health_value = health_status.name.lower()
                else:
                    health_value = str(health_status).lower()
                
                if health_value == 'unhealthy':
                    self.validation_errors.append("System is in unhealthy state")
                    return False
            
            # Action-specific validation
            if self.action.action_type == "SCALE_UP":
                # Check if system can handle more load
                if hasattr(current_state, 'metrics') and current_state.metrics:
                    cpu_metric = current_state.metrics.get('cpu_usage')
                    if cpu_metric and hasattr(cpu_metric, 'value') and cpu_metric.value > 95:
                        self.validation_errors.append("CPU usage too high for scaling up")
                        return False
            
            elif self.action.action_type == "SCALE_DOWN":
                # Check if system has minimum resources
                if hasattr(current_state, 'metrics') and current_state.metrics:
                    instance_metric = current_state.metrics.get('instance_count')
                    if instance_metric and hasattr(instance_metric, 'value') and instance_metric.value <= 1:
                        self.validation_errors.append("Cannot scale down below minimum instances")
                        return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"System state validation error: {e}")
            return False
    
    def get_action(self) -> AdaptationAction:
        """Get the adaptation action associated with this command."""
        return self.action
    
    async def undo(self) -> bool:
        """Attempt to undo the command execution."""
        if self.status != CommandStatus.COMPLETED:
            logger.warning(f"Cannot undo command {self.command_id} - not in completed state")
            return False
        
        try:
            self.status = CommandStatus.UNDOING
            
            # Execute undo actions in reverse order
            for undo_action in reversed(self._undo_actions):
                await undo_action()
            
            self.status = CommandStatus.UNDONE
            logger.info(f"Command {self.command_id} undone successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to undo command {self.command_id}: {e}")
            return False
    
    def add_undo_action(self, undo_action: Callable) -> None:
        """Add an undo action to be executed if rollback is needed."""
        self._undo_actions.append(undo_action)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of command execution."""
        return {
            "command_id": self.command_id,
            "action_id": self.action.action_id,
            "action_type": self.action.action_type,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "execution_count": len(self.execution_history),
            "validation_errors": self.validation_errors,
            "has_undo_actions": len(self._undo_actions) > 0
        }


class CommandQueue:
    """Priority queue for managing command execution order."""
    
    def __init__(self):
        self._commands: List[PolarisAdaptationCommand] = []
        self._lock = asyncio.Lock()
    
    async def enqueue(self, command: PolarisAdaptationCommand) -> None:
        """Add a command to the queue with priority ordering."""
        async with self._lock:
            # Insert command based on priority (higher priority first)
            inserted = False
            for i, existing_command in enumerate(self._commands):
                if command.priority.value > existing_command.priority.value:
                    self._commands.insert(i, command)
                    inserted = True
                    break
            
            if not inserted:
                self._commands.append(command)
            
            logger.debug(f"Enqueued command {command.command_id} with priority {command.priority.value}")
    
    async def dequeue(self) -> Optional[PolarisAdaptationCommand]:
        """Remove and return the highest priority command."""
        async with self._lock:
            if self._commands:
                command = self._commands.pop(0)
                logger.debug(f"Dequeued command {command.command_id}")
                return command
            return None
    
    async def peek(self) -> Optional[PolarisAdaptationCommand]:
        """Look at the next command without removing it."""
        async with self._lock:
            return self._commands[0] if self._commands else None
    
    async def size(self) -> int:
        """Get the number of commands in the queue."""
        async with self._lock:
            return len(self._commands)
    
    async def clear(self) -> None:
        """Clear all commands from the queue."""
        async with self._lock:
            self._commands.clear()


class PolarisCommandProcessor(Injectable):
    """
    Command processor that handles command execution with queuing, validation,
    and integration with the event system.
    
    Features:
    - Priority-based command queuing
    - Concurrent command execution with limits
    - Command validation and error handling
    - Integration with event system for result publishing
    - Command history and monitoring
    - Retry and timeout handling
    """
    
    def __init__(
        self,
        event_bus: Optional[PolarisEventBus] = None,
        max_concurrent_commands: int = 5,
        default_timeout: timedelta = timedelta(minutes=5)
    ):
        self.event_bus = event_bus
        self.max_concurrent_commands = max_concurrent_commands
        self.default_timeout = default_timeout
        
        self._command_queue = CommandQueue()
        self._active_commands: Dict[str, PolarisAdaptationCommand] = {}
        self._command_history: Dict[str, CommandResult] = {}
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_commands)
        
        logger.info(f"CommandProcessor initialized with max {max_concurrent_commands} concurrent commands")
    
    async def start(self) -> None:
        """Start the command processor."""
        if self._running:
            logger.warning("Command processor is already running")
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_commands())
        logger.info("Command processor started")
    
    async def stop(self) -> None:
        """Stop the command processor."""
        if not self._running:
            return
        
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active commands to complete
        if self._active_commands:
            logger.info(f"Waiting for {len(self._active_commands)} active commands to complete")
            await asyncio.sleep(1.0)  # Give commands time to finish
        
        logger.info("Command processor stopped")
    
    async def submit_command(self, command: PolarisAdaptationCommand) -> str:
        """Submit a command for execution."""
        if not self._running:
            raise RuntimeError("Command processor is not running")
        
        # Set default timeout if not specified
        if not command.context.timeout:
            command.context.timeout = self.default_timeout
        
        await self._command_queue.enqueue(command)
        logger.info(f"Submitted command {command.command_id} for execution")
        
        return command.command_id
    
    async def _process_commands(self) -> None:
        """Main command processing loop."""
        logger.info("Command processing loop started")
        
        while self._running:
            try:
                # Check if we can process more commands
                if len(self._active_commands) >= self.max_concurrent_commands:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next command
                command = await self._command_queue.dequeue()
                if not command:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process command asynchronously
                task = asyncio.create_task(self._execute_command(command))
                self._active_commands[command.command_id] = command
                
                # Don't await here - let it run concurrently
                
            except Exception as e:
                logger.error(f"Error in command processing loop: {e}")
                await asyncio.sleep(1.0)
        
        logger.info("Command processing loop stopped")
    
    async def _execute_command(self, command: PolarisAdaptationCommand) -> None:
        """Execute a single command with full lifecycle management."""
        async with self._semaphore:
            try:
                # Validate command
                if not await command.can_execute():
                    await self._handle_command_failure(
                        command,
                        CommandValidationError(
                            "Command validation failed",
                            command.command_id,
                            command.validation_errors
                        )
                    )
                    return
                
                # Execute command
                execution_result = await command.execute()
                
                # Handle success
                await self._handle_command_success(command, execution_result)
                
            except CommandExecutionError as e:
                # Handle command execution failure
                await self._handle_command_failure(command, e)
            except Exception as e:
                # Handle unexpected failure
                wrapped_error = CommandExecutionError(
                    f"Unexpected command execution error: {e}",
                    command.command_id,
                    e
                )
                await self._handle_command_failure(command, wrapped_error)
            
            finally:
                # Remove from active commands
                self._active_commands.pop(command.command_id, None)
    
    async def _handle_command_success(
        self,
        command: PolarisAdaptationCommand,
        execution_result: ExecutionResult
    ) -> None:
        """Handle successful command execution."""
        logger.info(f"Command {command.command_id} executed successfully")
        
        # Create command result
        result = CommandResult(
            command_id=command.command_id,
            status=CommandStatus.COMPLETED,
            execution_result=execution_result,
            completed_at=datetime.utcnow(),
            context=command.context
        )
        
        # Store in history
        self._command_history[command.command_id] = result
        
        # Publish event if event bus is available
        if self.event_bus:
            event = ExecutionResultEvent(
                execution_result=execution_result,
                correlation_id=command.context.correlation_id,
                metadata=EventMetadata(
                    source=f"command_processor:{command.command_id}",
                    tags={"command_id": command.command_id, "action_type": command.action.action_type}
                )
            )
            await self.event_bus.publish_execution_result(event)
    
    async def _handle_command_failure(
        self,
        command: PolarisAdaptationCommand,
        error: Exception
    ) -> None:
        """Handle command execution failure."""
        logger.error(f"Command {command.command_id} failed: {error}")
        
        # Check if we should retry
        if (command.context.retry_count < command.context.max_retries and
            not isinstance(error, CommandValidationError)):
            
            command.context.retry_count += 1
            logger.info(f"Retrying command {command.command_id} (attempt {command.context.retry_count})")
            
            # Re-queue for retry
            await self._command_queue.enqueue(command)
            return
        
        # Create failure result
        result = CommandResult(
            command_id=command.command_id,
            status=CommandStatus.FAILED,
            error=error,
            completed_at=datetime.utcnow(),
            context=command.context
        )
        
        # Store in history
        self._command_history[command.command_id] = result
        
        # Publish failure event if event bus is available
        if self.event_bus:
            execution_result = ExecutionResult(
                action_id=command.action.action_id,
                status="failed",
                result_data={"error": str(error), "command_id": command.command_id}
            )
            
            event = ExecutionResultEvent(
                execution_result=execution_result,
                correlation_id=command.context.correlation_id,
                metadata=EventMetadata(
                    source=f"command_processor:{command.command_id}",
                    tags={"command_id": command.command_id, "action_type": command.action.action_type, "status": "failed"}
                )
            )
            await self.event_bus.publish_execution_result(event)
    
    async def get_command_status(self, command_id: str) -> Optional[CommandStatus]:
        """Get the status of a command."""
        # Check active commands
        if command_id in self._active_commands:
            return self._active_commands[command_id].status
        
        # Check history
        if command_id in self._command_history:
            return self._command_history[command_id].status
        
        return None
    
    async def get_command_result(self, command_id: str) -> Optional[CommandResult]:
        """Get the result of a command."""
        return self._command_history.get(command_id)
    
    async def cancel_command(self, command_id: str) -> bool:
        """Cancel a pending or active command."""
        # Check if command is active
        if command_id in self._active_commands:
            command = self._active_commands[command_id]
            command.status = CommandStatus.CANCELLED
            logger.info(f"Cancelled active command {command_id}")
            return True
        
        # Note: Cannot cancel commands already in queue easily without more complex queue management
        logger.warning(f"Cannot cancel command {command_id} - not found in active commands")
        return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get command processing statistics."""
        total_commands = len(self._command_history)
        successful_commands = sum(1 for result in self._command_history.values() 
                                if result.status == CommandStatus.COMPLETED)
        failed_commands = sum(1 for result in self._command_history.values() 
                            if result.status == CommandStatus.FAILED)
        
        # Get queue size synchronously by accessing the internal list
        queue_size = len(self._command_queue._commands)
        
        return {
            "active_commands": len(self._active_commands),
            "queued_commands": queue_size,
            "total_processed": total_commands,
            "successful": successful_commands,
            "failed": failed_commands,
            "success_rate": successful_commands / total_commands if total_commands > 0 else 0,
            "is_running": self._running,
            "max_concurrent": self.max_concurrent_commands
        }