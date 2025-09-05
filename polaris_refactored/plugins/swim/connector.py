"""
SWIM TCP Connector for POLARIS Framework (Refactored Version).

This module implements the managed system connector for SWIM,
handling TCP communication with retry logic and error handling.
Migrated from POC to work with the refactored POLARIS architecture.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, List
from datetime import datetime

from ...src.domain.interfaces import ManagedSystemConnector
from ...src.domain.models import (
    SystemState, AdaptationAction, ExecutionResult, MetricValue, HealthStatus, ExecutionStatus
)


class SwimTCPConnector(ManagedSystemConnector):
    """
    TCP connector for SWIM (Simulated Web Infrastructure Manager).
    
    This connector implements the communication protocol for SWIM's
    external control interface via TCP socket connections.
    Refactored to work with the new POLARIS architecture.
    """
    
    def __init__(self, system_config: Optional[Dict[str, Any]] = None):
        """Initialize the SWIM TCP connector.
        
        Args:
            system_config: Complete configuration for SWIM (optional for basic usage)
        """
        self.config = system_config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Extract connection parameters
        connection_config = self.config.get("connection", {})
        self.host = connection_config.get("host", "localhost")
        self.port = connection_config.get("port", 4242)
        
        # Extract implementation parameters
        implementation_config = self.config.get("implementation", {})
        self.timeout = implementation_config.get("timeout", 30.0)
        self.max_retries = implementation_config.get("max_retries", 3)
        self.retry_base_delay = implementation_config.get("retry_base_delay", 1.0)
        self.retry_max_delay = implementation_config.get("retry_max_delay", 5.0)
        
        # Connection state
        self._connected = False
        self._system_id = self.config.get("system_name", "swim")
        
        self.logger.info(
            "SWIM connector initialized",
            extra={
                "host": self.host,
                "port": self.port,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "system_id": self._system_id
            }
        )
    
    async def connect(self) -> bool:
        """Establish connection to SWIM.
        
        For SWIM, we don't maintain a persistent connection.
        Each command opens a new TCP connection.
        """
        try:
            # Test connection with a simple command
            response = await self._execute_swim_command("get_servers")
            self._connected = True
            self.logger.info(
                "SWIM connection verified",
                extra={
                    "host": self.host,
                    "port": self.port,
                    "test_response": response
                }
            )
            return True
        except Exception as e:
            self._connected = False
            self.logger.error(
                "Failed to connect to SWIM",
                extra={
                    "host": self.host,
                    "port": self.port,
                    "error": str(e)
                }
            )
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from SWIM.
        
        Since we don't maintain persistent connections, this is a no-op.
        """
        self._connected = False
        self.logger.info("SWIM connector disconnected")
        return True
    
    async def get_system_id(self) -> str:
        """Get the unique identifier for this managed system."""
        return self._system_id
    
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        """Collect current metrics from the managed system."""
        try:
            metrics = {}
            
            # Collect basic SWIM metrics
            server_count = int(await self._execute_swim_command("get_servers"))
            active_servers = int(await self._execute_swim_command("get_active_servers"))
            max_servers = int(await self._execute_swim_command("get_max_servers"))
            dimmer = float(await self._execute_swim_command("dimmer"))
            
            # Create metric values
            metrics["server_count"] = MetricValue(
                name="server_count",
                value=server_count,
                unit="count",
                timestamp=datetime.utcnow()
            )
            
            metrics["active_servers"] = MetricValue(
                name="active_servers", 
                value=active_servers,
                unit="count",
                timestamp=datetime.utcnow()
            )
            
            metrics["max_servers"] = MetricValue(
                name="max_servers",
                value=max_servers,
                unit="count", 
                timestamp=datetime.utcnow()
            )
            
            metrics["dimmer"] = MetricValue(
                name="dimmer",
                value=dimmer,
                unit="ratio",
                timestamp=datetime.utcnow()
            )
            
            # Try to collect performance metrics (may not be available in all SWIM versions)
            try:
                basic_rt = float(await self._execute_swim_command("get_basic_rt"))
                metrics["basic_response_time"] = MetricValue(
                    name="basic_response_time",
                    value=basic_rt,
                    unit="ms",
                    timestamp=datetime.utcnow()
                )
            except Exception:
                pass  # Optional metric
            
            try:
                opt_rt = float(await self._execute_swim_command("get_opt_rt"))
                metrics["optional_response_time"] = MetricValue(
                    name="optional_response_time",
                    value=opt_rt,
                    unit="ms",
                    timestamp=datetime.utcnow()
                )
            except Exception:
                pass  # Optional metric
            
            # Calculate derived metrics
            if active_servers > 0 and max_servers > 0:
                utilization = active_servers / max_servers
                metrics["server_utilization"] = MetricValue(
                    name="server_utilization",
                    value=utilization,
                    unit="ratio",
                    timestamp=datetime.utcnow()
                )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return {}
    
    async def get_system_state(self) -> SystemState:
        """Get the current state of the managed system."""
        try:
            metrics = await self.collect_metrics()
            
            # Determine health status based on connectivity and basic metrics
            health_status = HealthStatus.HEALTHY
            if not self._connected:
                health_status = HealthStatus.UNHEALTHY
            elif not metrics:
                health_status = HealthStatus.WARNING
            
            return SystemState(
                system_id=self._system_id,
                timestamp=datetime.utcnow(),
                metrics=metrics,
                health_status=health_status,
                metadata={
                    "host": self.host,
                    "port": self.port,
                    "connected": self._connected
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system state: {e}")
            return SystemState(
                system_id=self._system_id,
                timestamp=datetime.utcnow(),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
                metadata={"error": str(e)}
            )
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute an adaptation action on the managed system."""
        try:
            action_type = action.action_type.upper()
            
            # Map action types to SWIM commands
            if action_type == "ADD_SERVER" or action_type == "SCALE_UP":
                response = await self._execute_swim_command("add_server")
                
            elif action_type == "REMOVE_SERVER" or action_type == "SCALE_DOWN":
                response = await self._execute_swim_command("remove_server")
                
            elif action_type == "SET_DIMMER" or action_type == "ADJUST_QOS":
                dimmer_value = action.parameters.get("value", 1.0)
                if not 0.0 <= dimmer_value <= 1.0:
                    raise ValueError(f"Dimmer value must be between 0.0 and 1.0, got {dimmer_value}")
                response = await self._execute_swim_command(f"set_dimmer {dimmer_value}")
                
            else:
                raise ValueError(f"Unsupported action type: {action_type}")
            
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.SUCCESS,
                result_data={
                    "swim_response": response,
                    "action_type": action_type,
                    "parameters": action.parameters
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute action {action.action_id}: {e}")
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={"error": str(e)}
            )
    
    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if an adaptation action can be executed."""
        try:
            action_type = action.action_type.upper()
            
            if action_type == "ADD_SERVER" or action_type == "SCALE_UP":
                # Check if we can add servers
                current_servers = int(await self._execute_swim_command("get_servers"))
                max_servers = int(await self._execute_swim_command("get_max_servers"))
                return current_servers < max_servers
                
            elif action_type == "REMOVE_SERVER" or action_type == "SCALE_DOWN":
                # Check if we can remove servers (must keep at least 1)
                current_servers = int(await self._execute_swim_command("get_servers"))
                return current_servers > 1
                
            elif action_type == "SET_DIMMER" or action_type == "ADJUST_QOS":
                # Check dimmer value is valid
                dimmer_value = action.parameters.get("value", 1.0)
                return 0.0 <= dimmer_value <= 1.0
                
            else:
                return False  # Unsupported action type
                
        except Exception as e:
            self.logger.error(f"Failed to validate action {action.action_id}: {e}")
            return False
    
    async def get_supported_actions(self) -> List[str]:
        """Get the list of action types supported by this managed system."""
        return [
            "ADD_SERVER",
            "REMOVE_SERVER", 
            "SCALE_UP",
            "SCALE_DOWN",
            "SET_DIMMER",
            "ADJUST_QOS"
        ]
    
    # SWIM-specific helper methods (migrated from POC)
    
    async def _execute_swim_command(self, command: str) -> str:
        """Execute a command on SWIM via TCP with retry logic."""
        return await self._send_with_retries(command)
    
    async def _send_recv(self, command: str) -> str:
        """Send command and receive response via TCP."""
        start_time = time.perf_counter()
        
        try:
            # Open TCP connection with timeout
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Connection to {self.host}:{self.port} timed out")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")
        
        try:
            # Send command
            writer.write((command + "\n").encode())
            await asyncio.wait_for(writer.drain(), timeout=self.timeout)
            
            # Receive response
            line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
            response = line.decode(errors="replace").strip()
            
            elapsed = time.perf_counter() - start_time
            self.logger.debug(
                "SWIM command executed",
                extra={
                    "command": command,
                    "response": response,
                    "elapsed_ms": round(elapsed * 1000, 3)
                }
            )
            
            return response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command '{command}' timed out after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Command '{command}' failed: {e}")
        finally:
            # Always close the connection
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
    
    async def _send_with_retries(self, command: str) -> str:
        """Send command with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(
                    "Sending SWIM command",
                    extra={
                        "command": command,
                        "attempt": attempt,
                        "max_retries": self.max_retries
                    }
                )
                
                response = await self._send_recv(command)
                
                self.logger.debug(
                    "SWIM command successful",
                    extra={
                        "command": command,
                        "response": response,
                        "attempt": attempt
                    }
                )
                
                return response
                
            except (TimeoutError, ConnectionError, OSError, asyncio.TimeoutError) as e:
                last_error = e
                
                if attempt >= self.max_retries:
                    self.logger.error(
                        "SWIM command failed after max retries",
                        extra={
                            "command": command,
                            "attempts": attempt + 1,
                            "error": str(e)
                        }
                    )
                    raise
                
                # Calculate retry delay with jitter
                delay = min(
                    self.retry_base_delay * (2 ** attempt) + (time.time() % 1),
                    self.retry_max_delay
                )
                
                self.logger.warning(
                    "SWIM command failed, retrying",
                    extra={
                        "command": command,
                        "attempt": attempt,
                        "error": str(e),
                        "retry_in_sec": round(delay, 3)
                    }
                )
                
                await asyncio.sleep(delay)
        
        # This should not be reached, but just in case
        raise last_error or Exception(f"Command '{command}' failed after {self.max_retries + 1} attempts")
    
    # Additional SWIM-specific helper methods for backward compatibility
    
    async def get_server_count(self) -> int:
        """Get the current number of servers."""
        response = await self._execute_swim_command("get_servers")
        return int(response)
    
    async def get_max_servers(self) -> int:
        """Get the maximum number of servers allowed."""
        response = await self._execute_swim_command("get_max_servers")
        return int(response)
    
    async def get_active_servers(self) -> int:
        """Get the number of active servers."""
        response = await self._execute_swim_command("get_active_servers")
        return int(response)
    
    async def get_dimmer(self) -> float:
        """Get the current dimmer value."""
        response = await self._execute_swim_command("dimmer")
        return float(response)
    
    async def set_dimmer(self, value: float) -> str:
        """Set the dimmer value."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Dimmer value must be between 0.0 and 1.0, got {value}")
        return await self._execute_swim_command(f"set_dimmer {value}")
    
    async def add_server(self) -> str:
        """Add a server to the pool."""
        # Check if we can add a server
        current = await self.get_server_count()
        maximum = await self.get_max_servers()
        
        if current >= maximum:
            raise ValueError(f"Cannot add server: already at maximum ({maximum})")
        
        return await self._execute_swim_command("add_server")
    
    async def remove_server(self) -> str:
        """Remove a server from the pool."""
        # Check if we can remove a server
        current = await self.get_server_count()
        
        if current <= 1:
            raise ValueError(f"Cannot remove server: only {current} server(s) remaining")
        
        return await self._execute_swim_command("remove_server")