"""
Generic Execution Adapter for POLARIS Framework.

This adapter executes control actions on any managed system using the plugin
architecture and publishes execution results to NATS.
"""

import asyncio
import logging
import time
import re
import operator
from datetime import datetime, timezone
from typing import Any, Dict, Union, Optional

from nats.aio.msg import Msg

from polaris.adapters.core import ExternalAdapter
from polaris.models.actions import ControlAction, ExecutionResult, ActionStatus
from polaris.knowledge_base.models import KBEntry, KBDataType


class ExecutionAdapter(ExternalAdapter):
    """Generic execution adapter that executes control actions on any managed system.

    This adapter provides a plugin-driven approach to action execution, supporting:
    - Configurable action types and parameters from plugin configuration
    - Parameter validation and type checking
    - Precondition evaluation before action execution
    - Concurrent execution control with throttling
    - Comprehensive result reporting and metrics
    - Queue management with configurable constraints

    The adapter uses the plugin configuration to determine:
    - Which actions are available and their parameters
    - How to validate action parameters
    - What preconditions must be met
    - How to execute actions on the managed system
    - What constraints apply to execution

    Example:
        adapter = ExecutionAdapter(
            polaris_config_path="config/polaris.yaml",
            plugin_dir="plugins/my_system"
        )

        async with adapter:
            # Adapter listens for actions on NATS
            await asyncio.sleep(60)
    """

    def __init__(
        self, polaris_config_path: str, plugin_dir: str, logger: Optional[logging.Logger] = None
    ):
        """Initialize the execution adapter.

        Args:
            polaris_config_path: Path to POLARIS framework configuration
            plugin_dir: Directory containing the managed system plugin
            logger: Logger instance (created if not provided)
        """
        super().__init__(polaris_config_path, plugin_dir, logger)

        # Get execution configuration from plugin
        self.execution_config = self.config_manager.get_execution_config()

        # Extract configuration values
        self.actions_config = self.execution_config.get("actions", [])
        self.constraints = self.execution_config.get("constraints", {})

        # Execution constraints
        self.min_interval = self.constraints.get("min_interval", 0.0)
        self.max_concurrent = self.constraints.get("max_concurrent", 1)
        self.queue_size = self.constraints.get("queue_size", 1000)

        # Framework configuration
        execution_framework_config = self.framework_config.get("execution", {})
        self.action_subject = execution_framework_config.get(
            "action_subject", "polaris.execution.actions"
        )
        self.result_subject = execution_framework_config.get(
            "result_subject", "polaris.execution.results"
        )
        self.metrics_subject = execution_framework_config.get(
            "metrics_subject", "polaris.execution.metrics"
        )
        self.decisions_subject = execution_framework_config.get(
            "decisions_subject", "polaris.execution.decisions"
        )

        # Runtime state
        self.action_queue: asyncio.Queue[ControlAction] = asyncio.Queue(
            maxsize=self.queue_size if self.queue_size > 0 else 0
        )
        self.worker_task: Optional[asyncio.Task] = None
        self._last_action_finished_at: float = 0.0
        self._execution_semaphore = asyncio.Semaphore(self.max_concurrent)

        # Build action type to config mapping
        self.action_configs = {}
        for action_config in self.actions_config:
            self.action_configs[action_config["type"]] = action_config

        self.logger.info(
            "Execution adapter initialized",
            extra={
                "system_name": self.plugin_config.get("system_name"),
                "actions_count": len(self.actions_config),
                "min_interval": self.min_interval,
                "max_concurrent": self.max_concurrent,
                "queue_size": self.queue_size,
            },
        )

    async def _publish_metric(self, metric_name: str, payload: Dict[str, Any]):
        """Publish execution metrics to NATS."""
        try:
            metric_data = {
                "metric": metric_name,
                "timestamp": time.time(),
                "system": self.plugin_config.get("system_name"),
                **payload,
            }

            await self.nats_client.publish_json(self.metrics_subject, metric_data)

        except Exception as e:
            self.logger.warning(
                "Metric publish failed", extra={"metric": metric_name, "error": str(e)}
            )

    def _validate_action_parameters(
        self, action: ControlAction, action_config: Dict[str, Any]
    ) -> Optional[str]:
        """Validate action parameters against configuration.

        Args:
            action: The control action to validate
            action_config: Action configuration from plugin

        Returns:
            Error message if validation fails, None if valid
        """
        required_params = action_config.get("parameters", [])

        for param_config in required_params:
            param_name = param_config["name"]
            param_type = param_config["type"]
            required = param_config.get("required", False)
            validation = param_config.get("validation", {})

            # Check if required parameter is present
            if required and param_name not in action.params:
                return f"Missing required parameter: {param_name}"

            # Skip validation if parameter not provided and not required
            if param_name not in action.params:
                continue

            param_value = action.params[param_name]

            # Type validation
            try:
                if param_type == "float":
                    param_value = float(param_value)
                elif param_type == "integer":
                    param_value = int(param_value)
                elif param_type == "boolean":
                    param_value = bool(param_value)
                # string needs no conversion
            except (ValueError, TypeError):
                return f"Parameter {param_name} must be of type {param_type}"

            # Range validation
            if "min" in validation and param_value < validation["min"]:
                return f"Parameter {param_name} must be >= {validation['min']}"

            if "max" in validation and param_value > validation["max"]:
                return f"Parameter {param_name} must be <= {validation['max']}"

            # Enum validation
            if "enum" in validation and param_value not in validation["enum"]:
                return f"Parameter {param_name} must be one of: {validation['enum']}"

            # Pattern validation (for strings)
            if "pattern" in validation and param_type == "string":
                import re

                if not re.match(validation["pattern"], str(param_value)):
                    return f"Parameter {param_name} does not match required pattern"

        return None

    async def _check_preconditions(self, action_config: Dict[str, Any]) -> Optional[str]:
        """Check action preconditions with support for multiple mathematical operations.

        Supports comparison operators: <, >, <=, >=, ==, !=
        Supports equality operators: =, ==
        Supports containment operators: in, not in

        The right side of operations can be:
        - Another connector command that returns a value
        - A constant (string, number, boolean)
        - For 'in'/'not in' operations, a comma-separated list of values

        Args:
            action_config: Action configuration from plugin containing preconditions list

        Returns:
            Error message if any precondition fails, None if all pass
        """
        preconditions = action_config.get("preconditions", [])

        # Define supported operators with their corresponding Python operators
        operators = {
            "<=": operator.le,
            ">=": operator.ge,
            "<": operator.lt,
            ">": operator.gt,
            "==": operator.eq,
            "=": operator.eq,  # Support single = as well
            "!=": operator.ne,
            "not in": lambda x, y: x not in y,
            "in": lambda x, y: x in y,
        }

        for precondition in preconditions:
            check = precondition["check"]
            message = precondition["message"]

            try:
                # Find the operator in the check string
                op_found = None
                op_func = None

                # Sort operators by length (descending) to match longer operators first
                for op in sorted(operators.keys(), key=len, reverse=True):
                    if op in check:
                        op_found = op
                        op_func = operators[op]
                        break

                if not op_found:
                    self.logger.warning(
                        "Precondition check error: No supported operator found",
                        extra={"check": check},
                    )
                    return f"Precondition check error: No supported operator found in '{check}'"

                # Split the expression by the operator
                parts = check.split(op_found, 1)
                if len(parts) != 2:
                    self.logger.warning(
                        "Precondition check error: Invalid expression format",
                        extra={"check": check},
                    )
                    return f"Precondition check error: Invalid expression format '{check}'"

                left_expr = parts[0].strip()
                right_expr = parts[1].strip()

                # Evaluate left side (always treated as a command)
                left_val = await self.connector.execute_command(left_expr)

                # Evaluate right side (command or constant)
                right_val = await self._evaluate_expression(right_expr)

                # Special handling for 'in' and 'not in' operations
                if op_found in ["in", "not in"]:
                    # If right_val is a string containing commas, split it into a list
                    if isinstance(right_val, str) and "," in right_val:
                        right_val = [item.strip() for item in right_val.split(",")]
                    elif not isinstance(right_val, (list, tuple, set)):
                        # Convert single value to list for consistency
                        right_val = [right_val]

                # Type conversion for numeric comparisons
                if op_found in ["<", ">", "<=", ">="]:
                    try:
                        left_val = float(left_val)
                        # Only convert right_val to float if it's not already a number
                        if not isinstance(right_val, (int, float)):
                            right_val = float(right_val)
                    except (ValueError, TypeError):
                        # If conversion fails, proceed with string comparison
                        self.logger.warning(
                            "Precondition check error: Failed to convert values for comparison",
                            extra={"check": check},
                        )
                        pass

                # Perform the comparison
                result = op_func(left_val, right_val)

                if not result:
                    return message

            except Exception as e:
                self.logger.warning(
                    "Precondition check failed",
                    extra={
                        "check": check,
                        "error": str(e),
                        "operator": op_found if "op_found" in locals() else "unknown",
                    },
                )
                return f"Precondition check error: {e}"

        return None

    async def _evaluate_expression(self, expression: str) -> Union[str, int, float, bool]:
        """Evaluates an expression string, determining if it's a literal or a command.

        The method follows a specific parsing order:
        1. Checks for boolean literals ('true'/'false', case-insensitive)
        2. Attempts to parse as a number (integer or float)
        3. Checks for quoted strings (strips quotes)
        4. If none of the above, assumes it's a command and executes it via connector

        Args:
            expression: The expression string to evaluate
                    (e.g., "10", "true", "'running'", "get_max_servers")

        Returns:
            The evaluated value (boolean, number, or string).
            Connector command results are always returned as raw strings.
        """
        # Define regex patterns for different literal types
        boolean_pattern = re.compile(r"^(true|false)$", re.IGNORECASE)
        integer_pattern = re.compile(r"^[+-]?\d+$")
        float_pattern = re.compile(r"^[+-]?\d+\.\d*$|^[+-]?\d*\.\d+$|^[+-]?\d+\.?\d*[eE][+-]?\d+$")
        quoted_string_pattern = re.compile(r'^(["\'])(.*)(\1)$', re.DOTALL)

        expr = expression.strip()

        # 1. Check for boolean literals
        if boolean_pattern.match(expr):
            return expr.lower() == "true"

        # 2. Check for numeric literals
        try:
            # Check for integer pattern first
            if integer_pattern.match(expr):
                return int(expr)

            # Check for float pattern
            if float_pattern.match(expr):
                num = float(expr)
                # Return as integer if it represents a whole number
                if num.is_integer():
                    return int(num)
                return num

        except (ValueError, OverflowError):
            # Handle edge cases like extremely large numbers
            pass

        # 3. Check for quoted string literals
        quoted_match = quoted_string_pattern.match(expr)
        if quoted_match:
            return quoted_match.group(2)  # Return content between quotes

        # 4. Treat as command if no literal pattern matches
        self.logger.debug("Evaluating expression as a command", extra={"expression": expr})

        # The result from the connector is always a string.
        # The calling function (_check_preconditions) handles further type conversion.
        return await self.connector.execute_command(expr)

    async def execute_action(self, action: ControlAction) -> ExecutionResult:
        """Execute a control action on the managed system.

        Args:
            action: The control action to execute

        Returns:
            Execution result with status and details
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc).isoformat()

        ctx = {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "source": action.source,
        }

        self.logger.info("Executing action", extra=ctx)

        try:
            # Find action configuration
            action_config = self.action_configs.get(action.action_type)
            if not action_config:
                raise ValueError(f"Unknown action type: {action.action_type}")

            # Validate parameters
            param_error = self._validate_action_parameters(action, action_config)
            if param_error:
                raise ValueError(param_error)

            # Check preconditions
            precondition_error = await self._check_preconditions(action_config)
            if precondition_error:
                raise ValueError(precondition_error)

            # Execute the command
            command_template = action_config["command"]
            response = await self.connector.execute_command(command_template, action.params)

            # Check if response indicates an error
            success = not (response.lower().startswith("error") or "error" in response.lower())

            end_time = time.time()
            finished_at = datetime.now(timezone.utc).isoformat()
            duration = end_time - start_time

            result = ExecutionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                status=ActionStatus.SUCCESS if success else ActionStatus.FAILED,
                success=success,
                message=response,
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=duration,
                output={"response": response} if success else None,
                error=response if not success else None,
            )

            # Publish metrics
            await self._publish_metric(
                "action_duration",
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "duration_sec": duration,
                    "success": success,
                },
            )

            self.logger.info(
                "Action execution completed",
                extra={
                    **ctx,
                    "success": success,
                    "duration_sec": round(duration, 3),
                    "response_message": response[:100] + "..." if len(response) > 100 else response,
                },
            )

            return result

        except Exception as e:
            end_time = time.time()
            finished_at = datetime.now(timezone.utc).isoformat()
            duration = end_time - start_time

            error_msg = str(e)

            result = ExecutionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                status=ActionStatus.FAILED,
                success=False,
                message=error_msg,
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=duration,
                error=error_msg,
            )

            self.logger.error(
                f"Action execution failed {error_msg},{action}",
                extra={**ctx, "error": error_msg, "duration_sec": round(duration, 3)},
            )

            return result

    async def _publish_execution_result(self, result: ExecutionResult):
        """Publish execution result to NATS."""
        try:
            await self.nats_client.publish_json(self.result_subject, result.to_dict())

            self.logger.info(
                "Execution result published",
                extra={
                    "action_id": result.action_id,
                    "action_type": result.action_type,
                    "success": result.success,
                    "duration_sec": round(result.duration_sec, 6),
                },
            )

        except Exception as e:
            self.logger.error(
                "Failed to publish execution result",
                extra={
                    "action_id": result.action_id,
                    "action_type": result.action_type,
                    "error": str(e),
                },
            )

    async def _publish_adaptation_decision(self, action: ControlAction, result: ExecutionResult):
        """Publish adaptation decision to knowledge base via NATS.

        Args:
            action: The control action that was executed
            result: The execution result of the action
        """
        try:
            # Create a KBEntry for the adaptation decision
            decision_entry = KBEntry(
                data_type=KBDataType.ADAPTATION_DECISION,
                summary=(
                    f"Action {action.action_type} (ID: {action.action_id[:8]}) "
                    f"{'succeeded' if result.success else 'failed'} "
                    f"in {result.duration_sec:.3f}s"
                ),
                content={
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "action_params": action.params,
                    "action_source": action.source,
                    "action_target": action.target,
                    "action_priority": (
                        action.priority.value
                        if hasattr(action.priority, "value")
                        else action.priority
                    ),
                    "execution_status": (
                        result.status.value if hasattr(result.status, "value") else result.status
                    ),
                    "execution_success": result.success,
                    "execution_message": result.message,
                    "execution_duration_sec": result.duration_sec,
                    "execution_started_at": result.started_at,
                    "execution_finished_at": result.finished_at,
                    "execution_output": result.output,
                    "execution_error": result.error,
                },
                tags=[
                    "adaptation",
                    "decision",
                    action.action_type.lower(),
                    action.source.lower(),
                    "success" if result.success else "failed",
                    self.plugin_config.get("system_name", "unknown").lower(),
                ],
                timestamp=result.finished_at,
                metadata={
                    "system": self.plugin_config.get("system_name"),
                    "adapter": "execution",
                },
            )

            # Publish to NATS for knowledge base consumption
            await self.nats_client.publish_json(self.decisions_subject, decision_entry.model_dump())

            self.logger.info(
                "Adaptation decision published to knowledge base",
                extra={
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "success": result.success,
                    "subject": self.decisions_subject,
                },
            )

        except Exception as e:
            self.logger.warning(
                "Failed to publish adaptation decision",
                extra={
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "error": str(e),
                },
            )

    async def _enqueue_action(self, action: ControlAction):
        """Enqueue an action for execution."""
        queue_size_before = self.action_queue.qsize()

        try:
            await self.action_queue.put(action)
            queue_size_after = self.action_queue.qsize()

            self.logger.info(
                "Action enqueued",
                extra={
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "source": action.source,
                    "queue_size_before": queue_size_before,
                    "queue_size_after": queue_size_after,
                },
            )

            # Publish queue length metric
            await self._publish_metric("queue_length", {"queue_length": queue_size_after})

        except asyncio.QueueFull:
            self.logger.error(
                "Action queue full, dropping action",
                extra={
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "queue_size": queue_size_before,
                },
            )

    async def _action_handler(self, msg: Msg):
        """Handle incoming action messages from NATS."""
        try:
            # Parse the action
            action_data = msg.data.decode()
            action = ControlAction.from_json(action_data)

            # Enqueue for processing
            await self._enqueue_action(action)

        except Exception as e:
            self.logger.error(
                f"Failed to parse action message {action_data}",
                extra={"error": str(e), "raw_data": msg.data[:256].decode(errors="replace")},
            )

    async def _worker(self):
        """Main worker loop that processes actions from the queue."""
        self.logger.info("Execution worker started")

        try:
            while self.running:
                # Get next action from queue
                action: ControlAction = await self.action_queue.get()

                ctx = {"action_id": action.action_id, "action_type": action.action_type}

                # Respect minimum interval between actions
                if self.min_interval > 0 and self._last_action_finished_at > 0:
                    elapsed = time.time() - self._last_action_finished_at
                    remaining = self.min_interval - elapsed

                    if remaining > 0:
                        self.logger.info(
                            "Throttling action execution",
                            extra={**ctx, "wait_sec": round(remaining, 3)},
                        )
                        await asyncio.sleep(remaining)

                # Execute with concurrency control
                async with self._execution_semaphore:
                    queue_size_before = self.action_queue.qsize()

                    self.logger.info(
                        "Starting action execution",
                        extra={**ctx, "queue_size_before": queue_size_before},
                    )

                    try:
                        # Execute the action
                        result = await self.execute_action(action)

                        # Publish the result
                        await self._publish_execution_result(result)

                        # Publish adaptation decision to knowledge base
                        await self._publish_adaptation_decision(action, result)

                    except Exception as e:
                        # This should rarely happen as execute_action handles most exceptions
                        self.logger.exception(
                            "Unexpected error in action processing", extra={**ctx, "error": str(e)}
                        )

                    finally:
                        # Update timing and queue metrics
                        self._last_action_finished_at = time.time()
                        queue_size_after = self.action_queue.qsize()

                        self.logger.info(
                            "Action processing completed",
                            extra={**ctx, "queue_size_after": queue_size_after},
                        )

                        await self._publish_metric(
                            "queue_length", {"queue_length": queue_size_after}
                        )

                        # Mark task as done
                        self.action_queue.task_done()

        except asyncio.CancelledError:
            self.logger.info("Execution worker cancelled")
        finally:
            self.logger.info("Execution worker stopped")

    async def _start_processing(self) -> None:
        """Start execution-specific processing."""
        # Subscribe to action messages
        await self.nats_client.subscribe(self.action_subject, self._action_handler)

        # Start worker task
        self.worker_task = asyncio.create_task(self._worker())
        self._tasks.append(self.worker_task)

        self.logger.info(
            "Execution processing started",
            extra={
                "action_subject": self.action_subject,
                "max_concurrent": self.max_concurrent,
                "min_interval": self.min_interval,
            },
        )

    async def _stop_processing(self) -> None:
        """Stop execution-specific processing."""
        # Stop accepting new actions (unsubscribe handled by base class)

        # Drain the action queue with timeout
        try:
            await asyncio.wait_for(self.action_queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            self.logger.warning(
                "Action queue drain timeout", extra={"remaining": self.action_queue.qsize()}
            )

        # Cancel worker
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()

        self.logger.info("Execution processing stopped")
