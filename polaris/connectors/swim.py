"""SWIM system connector.

Connects Polaris to the SWIM exemplar system for self-adaptation.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import List, Optional

from polaris.abstractions.connector import Connector
from polaris.abstractions.connector_capabilities import ConnectorCapabilities
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.infrastructure.constants import DEFAULT_SWIM_TIMEOUT


class SWIMConnector(Connector):
    """Connector for SWIM (Simulated Web Infrastructure Manager).

    SWIM is a reference implementation for self-adaptive systems research. Communicates
    via TCP socket using line-based protocol.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4242,
        timeout: float = DEFAULT_SWIM_TIMEOUT,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        """Initialize SWIM connector with connection parameters."""
        self.host = host
        self.port = port
        self.timeout = timeout
        self._connected = False
        self._logger = logger
        self._metrics = metrics

    async def connect(self) -> bool:
        """Connect to SWIM system by testing connectivity."""
        attempts = 3
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                response = await self._send_command("get_servers")
                int(response)  # Verify response is numeric
                self._connected = True
                if self._logger:
                    self._logger.info(
                        "SWIMConnector connected",
                        host=self.host,
                        port=self.port,
                    )
                if self._metrics:
                    self._metrics.increment("polaris.connector.swim.connected")
                return True
            except Exception as exc:
                # Broad catch is intentional: SWIM socket connections can fail in
                # unpredictable ways (connection refused, timeout, malformed response).
                last_error = exc
                self._connected = False
                if self._logger and attempt < attempts:
                    self._logger.warning(
                        "SWIMConnector connection attempt failed",
                        host=self.host,
                        port=self.port,
                        attempt=attempt,
                        error=str(exc),
                    )
                if attempt < attempts:
                    await asyncio.sleep(0.2 * attempt)

        if self._logger:
            self._logger.error(
                "SWIMConnector connection failed",
                error=str(last_error) if last_error else "unknown",
            )
        if self._metrics:
            self._metrics.increment("polaris.connector.swim.connection_errors")
        return False

    async def disconnect(self) -> bool:
        """Disconnect from SWIM system."""
        self._connected = False
        if self._logger:
            self._logger.info(
                "SWIMConnector disconnected",
                host=self.host,
                port=self.port,
            )
        if self._metrics:
            self._metrics.increment("polaris.connector.swim.disconnected")
        return True

    async def get_system_id(self) -> str:
        """Get SWIM system identifier."""
        return "swim"

    async def collect_telemetry(self) -> SystemState:
        """Collect current state from SWIM."""
        if self._metrics:
            self._metrics.increment("polaris.connector.swim.telemetry_calls")
        if not self._connected:
            if self._logger:
                self._logger.warning("SWIMConnector collect_telemetry called while not connected")
            if self._metrics:
                self._metrics.increment("polaris.connector.swim.telemetry_not_connected")
            return SystemState(
                system_id="swim",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
                metadata={"error": "Not connected"},
            )

        start_time = time.monotonic()
        try:
            metrics = {}

            # Collect server metrics
            server_count = int(await self._send_command("get_servers"))
            active_servers = int(await self._send_command("get_active_servers"))
            max_servers = int(await self._send_command("get_max_servers"))
            dimmer = float(await self._send_command("get_dimmer"))

            metrics["server_count"] = MetricValue(
                name="server_count",
                value=server_count,
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )

            metrics["active_servers"] = MetricValue(
                name="active_servers",
                value=active_servers,
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )

            metrics["max_servers"] = MetricValue(
                name="max_servers",
                value=max_servers,
                unit="count",
                timestamp=datetime.now(timezone.utc),
            )

            metrics["dimmer"] = MetricValue(
                name="dimmer", value=dimmer, unit="ratio", timestamp=datetime.now(timezone.utc)
            )

            # Try to get response time metrics (optional)
            try:
                basic_rt = float(await self._send_command("get_basic_rt"))
                metrics["basic_response_time"] = MetricValue(
                    name="basic_response_time",
                    value=basic_rt,
                    unit="ms",
                    timestamp=datetime.now(timezone.utc),
                )
            except (ConnectionError, TimeoutError, ValueError, TypeError):
                # Skip optional metric if unavailable
                pass

            try:
                basic_throughput = float(await self._send_command("get_basic_throughput"))
                metrics["basic_throughput"] = MetricValue(
                    name="basic_throughput",
                    value=basic_throughput,
                    unit="req/s",
                    timestamp=datetime.now(timezone.utc),
                )
            except (ConnectionError, TimeoutError, ValueError, TypeError):
                # Skip optional metric if unavailable
                pass

            try:
                opt_rt = float(await self._send_command("get_opt_rt"))
                metrics["optional_response_time"] = MetricValue(
                    name="optional_response_time",
                    value=opt_rt,
                    unit="ms",
                    timestamp=datetime.now(timezone.utc),
                )
            except (ConnectionError, TimeoutError, ValueError, TypeError):
                # Skip optional metric if unavailable
                pass

            try:
                opt_throughput = float(await self._send_command("get_opt_throughput"))
                metrics["optional_throughput"] = MetricValue(
                    name="optional_throughput",
                    value=opt_throughput,
                    unit="req/s",
                    timestamp=datetime.now(timezone.utc),
                )
            except (ConnectionError, TimeoutError, ValueError, TypeError):
                # Skip optional metric if unavailable
                pass

            # Weighted average response time across all services
            if (
                "basic_response_time" in metrics
                and "optional_response_time" in metrics
                and "basic_throughput" in metrics
                and "optional_throughput" in metrics
            ):
                basic_throughput_val = float(metrics["basic_throughput"].value)
                optional_throughput_val = float(metrics["optional_throughput"].value)
                total_throughput = basic_throughput_val + optional_throughput_val
                if total_throughput > 0:
                    avg_rt = (
                        float(metrics["basic_response_time"].value) * basic_throughput_val
                        + float(metrics["optional_response_time"].value) * optional_throughput_val
                    ) / total_throughput
                    metrics["average_response_time"] = MetricValue(
                        name="average_response_time",
                        value=avg_rt,
                        unit="ms",
                        timestamp=datetime.now(timezone.utc),
                    )

            # Calculate average utilization
            if active_servers > 0:
                total_util = 0.0
                success_count = 0
                for server_id in range(1, active_servers + 1):
                    try:
                        util = float(await self._send_command(f"get_utilization server{server_id}"))
                        total_util += util
                        success_count += 1
                    except (ConnectionError, TimeoutError, ValueError, TypeError):
                        # Skip server if utilization unavailable
                        pass

                if success_count > 0:
                    metrics["average_utilization"] = MetricValue(
                        name="average_utilization",
                        value=total_util / success_count,
                        unit="ratio",
                        timestamp=datetime.now(timezone.utc),
                    )

            if self._metrics:
                duration = time.monotonic() - start_time
                self._metrics.histogram(
                    "polaris.connector.swim.telemetry_duration_seconds",
                    duration,
                )
            return SystemState(
                system_id="swim",
                timestamp=datetime.now(timezone.utc),
                metrics=metrics,
                health_status=HealthStatus.HEALTHY,
            )

        except Exception as exc:
            # Broad catch is intentional: SWIM telemetry collection can fail due to
            # socket errors, connection issues, or malformed responses.
            if self._logger:
                self._logger.error(
                    "SWIMConnector telemetry collection failed",
                    error=str(exc),
                )
            if self._metrics:
                self._metrics.increment("polaris.connector.swim.telemetry_errors")
            return SystemState(
                system_id="swim",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
                metadata={"error": str(exc)},
            )

    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute adaptation action on SWIM."""
        if not self._connected:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message="Not connected to SWIM",
            )

        start_time = time.monotonic()
        try:
            # Map action types to SWIM commands
            command = None

            if action.action_type.upper() in ["ADD_SERVER", "SCALE_UP"]:
                # Check if we can add
                current = int(await self._send_command("get_servers"))
                max_servers = int(await self._send_command("get_max_servers"))
                if current >= max_servers:
                    if self._metrics:
                        self._metrics.increment("polaris.connector.swim.actions_validation_failed")
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        result_data={},
                        error_message=f"Already at maximum servers ({max_servers})",
                    )
                command = "add_server"

            elif action.action_type.upper() in ["REMOVE_SERVER", "SCALE_DOWN"]:
                # Check if we can remove
                current = int(await self._send_command("get_servers"))
                if current <= 1:
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        result_data={},
                        error_message="Cannot remove last server",
                    )
                command = "remove_server"

            elif action.action_type.upper() in ["SET_DIMMER", "ADJUST_QOS"]:
                dimmer_value = (action.parameters or {}).get("value", 1.0)
                if not 0.0 <= dimmer_value <= 1.0:
                    if self._metrics:
                        self._metrics.increment("polaris.connector.swim.actions_validation_failed")
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        result_data={},
                        error_message=f"Invalid dimmer value: {dimmer_value}",
                    )
                command = f"set_dimmer {dimmer_value}"

            else:
                if self._metrics:
                    self._metrics.increment("polaris.connector.swim.actions_unsupported")
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    result_data={},
                    error_message=f"Unsupported action type: {action.action_type}",
                )

            # Execute command
            response = await self._send_command(command)

            if self._metrics:
                duration = time.monotonic() - start_time
                self._metrics.histogram(
                    "polaris.connector.swim.action_execution_duration_seconds",
                    duration,
                    tags={"action_type": action.action_type},
                )
                self._metrics.increment(
                    "polaris.connector.swim.actions_executed",
                    tags={
                        "action_type": action.action_type,
                        "status": ExecutionStatus.SUCCESS.value,
                    },
                )
            if self._logger:
                self._logger.info(
                    "SWIMConnector action executed successfully",
                    action_type=action.action_type,
                    command=command,
                )

            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.SUCCESS,
                result_data={
                    "swim_response": response,
                    "action_type": action.action_type,
                    "command": command,
                },
            )

        except Exception as exc:
            # Broad catch is intentional: SWIM action execution can fail due to
            # socket errors, connection issues, or protocol violations.
            if self._metrics:
                duration = time.monotonic() - start_time
                self._metrics.histogram(
                    "polaris.connector.swim.action_execution_duration_seconds",
                    duration,
                    tags={"action_type": action.action_type},
                )
                self._metrics.increment(
                    "polaris.connector.swim.actions_executed",
                    tags={
                        "action_type": action.action_type,
                        "status": ExecutionStatus.FAILED.value,
                    },
                )
            if self._logger:
                self._logger.error(
                    "SWIMConnector action execution failed",
                    action_type=action.action_type,
                    error=str(exc),
                )
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=str(exc),
            )

    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if action can be executed on SWIM."""
        valid_types = [
            "add_server",
            "remove_server",
            "scale_up",
            "scale_down",
            "set_dimmer",
            "adjust_qos",
        ]
        return action.action_type.lower() in valid_types

    async def get_supported_actions(self) -> List[AdaptationAction]:
        """Get list of actions supported by SWIM."""
        return [
            AdaptationAction(
                action_id="", action_type="scale_up", target_system="swim", parameters={}
            ),
            AdaptationAction(
                action_id="", action_type="scale_down", target_system="swim", parameters={}
            ),
            AdaptationAction(
                action_id="",
                action_type="set_dimmer",
                target_system="swim",
                parameters={"value": 1.0},
            ),
        ]

    async def get_capabilities(self) -> ConnectorCapabilities:
        """Expose normalized SWIM capability metadata."""
        actions = await self.get_supported_actions()
        action_types = [
            action.action_type
            for action in actions
            if isinstance(action.action_type, str) and action.action_type.strip()
        ]
        return ConnectorCapabilities.from_supported_action_types(
            action_types,
            action_aliases={
                "add_server": "scale_up",
                "remove_server": "scale_down",
                "adjust_qos": "set_dimmer",
            },
            metadata={"system_family": "swim"},
        )

    async def _send_command(self, command: str) -> str:
        """Send command to SWIM via TCP and receive response.

        Args:
            command: Command to send

        Returns:
            Response from SWIM
        """
        start_time = time.monotonic()
        writer = None
        try:
            # Open connection
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=self.timeout
            )

            # Send command
            writer.write((command + "\n").encode())
            await asyncio.wait_for(writer.drain(), timeout=self.timeout)

            # Receive response
            line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
            response = line.decode(errors="replace").strip()

            if not response:
                raise ConnectionError(f"Command '{command}' returned empty response")

            if self._metrics:
                duration = time.monotonic() - start_time
                self._metrics.histogram(
                    "polaris.connector.swim.command_duration_seconds",
                    duration,
                    tags={"command": command},
                )
            return response

        except asyncio.TimeoutError as exc:
            if self._metrics:
                self._metrics.increment(
                    "polaris.connector.swim.command_timeouts",
                    tags={"command": command},
                )
            raise TimeoutError(f"Command '{command}' timed out") from exc
        except Exception as exc:
            if self._metrics:
                self._metrics.increment(
                    "polaris.connector.swim.command_errors",
                    tags={"command": command},
                )
            raise ConnectionError(f"Command '{command}' failed: {exc}") from exc
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    # Best effort cleanup; close failures should not mask primary errors.
                    pass
