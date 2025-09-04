"""
SWIM TCP Connector for POLARIS Framework.

This module implements the managed system connector for SWIM,
handling TCP communication with retry logic and error handling.
"""

import asyncio
import time
from typing import Any, Dict, Optional

import sys
from pathlib import Path
# Add parent directory to path to import from polaris
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.adapters.core import ManagedSystemConnector
from polaris.common.nats_client import jittered_backoff


class SwimTCPConnector(ManagedSystemConnector):
    """
    TCP connector for SWIM (Simulated Web Infrastructure Manager).
    
    This connector implements the communication protocol for SWIM's
    external control interface via TCP socket connections.
    """
    
    def __init__(self, system_config: Dict[str, Any], logger):
        """Initialize the SWIM TCP connector.
        
        Args:
            system_config: Complete configuration for SWIM
            logger: Logger instance for structured logging
        """
        super().__init__(system_config, logger)
        
        # Extract connection parameters
        self.host = self.connection_config.get("host", "localhost")
        self.port = self.connection_config.get("port", 4242)
        
        # Extract retry parameters
        self.timeout = self.get_timeout()
        self.max_retries = self.get_max_retries()
        self.retry_base_delay = self.implementation_config.get("retry_base_delay", 1.0)
        self.retry_max_delay = self.implementation_config.get("retry_max_delay", 5.0)
        
        # Connection state
        self._connected = False
        
        self.logger.info(
            "SWIM connector initialized",
            extra={
                "host": self.host,
                "port": self.port,
                "timeout": self.timeout,
                "max_retries": self.max_retries
            }
        )
    
    async def connect(self) -> None:
        """Establish connection to SWIM.
        
        For SWIM, we don't maintain a persistent connection.
        Each command opens a new TCP connection.
        """
        # Test connection with a simple command
        try:
            response = await self.execute_command("get_servers")
            self._connected = True
            self.logger.info(
                "SWIM connection verified",
                extra={
                    "host": self.host,
                    "port": self.port,
                    "test_response": response
                }
            )
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
            raise ConnectionError(f"Cannot connect to SWIM at {self.host}:{self.port}: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from SWIM.
        
        Since we don't maintain persistent connections, this is a no-op.
        """
        self._connected = False
        self.logger.info("SWIM connector disconnected")
    
    async def execute_command(
        self,
        command_template: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a command on SWIM via TCP.
        
        Args:
            command_template: Command template (may include {placeholders})
            params: Parameters to substitute in the template
            
        Returns:
            Raw response from SWIM
            
        Raises:
            TimeoutError: If command times out
            ConnectionError: If connection fails
            Exception: For other errors
        """
        # Format command with parameters
        command = self._format_command(command_template, params)
        
        # Execute with retry logic
        return await self._send_with_retries(command)
    
    async def health_check(self) -> bool:
        """Check if SWIM is healthy and responsive.
        
        Returns:
            True if SWIM responds to a simple query, False otherwise
        """
        try:
            response = await self.execute_command("get_servers")
            # SWIM should return a number
            int(response)
            return True
        except Exception as e:
            self.logger.warning(
                "SWIM health check failed",
                extra={"error": str(e)}
            )
            return False
    
    def _format_command(self, template: str, params: Optional[Dict[str, Any]]) -> str:
        """Format command template with parameters.
        
        Args:
            template: Command template with {placeholders}
            params: Parameters to substitute
            
        Returns:
            Formatted command string
        """
        if not params:
            return template
        
        try:
            return template.format(**params)
        except KeyError as e:
            self.logger.error(
                "Missing parameter for command template",
                extra={
                    "template": template,
                    "params": params,
                    "missing": str(e)
                }
            )
            raise ValueError(f"Missing parameter {e} for command template: {template}")
    
    async def _send_recv(self, command: str) -> str:
        """Send command and receive response via TCP.
        
        Args:
            command: Command to send
            
        Returns:
            Response from SWIM
            
        Raises:
            TimeoutError: If operation times out
            ConnectionError: If connection fails
        """
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
        """Send command with exponential backoff retry.
        
        Args:
            command: Command to send
            
        Returns:
            Response from SWIM
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(
                    "Sending SWIM command",
                    extra={
                        "command": command,
                        "attempt": attempt,
                        "max_retries": self.max_retries
                    }
                )
                
                response = await self._send_recv(command)
                
                self.logger.info(
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
                delay = jittered_backoff(
                    attempt,
                    self.retry_base_delay,
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
    
    # Additional SWIM-specific helper methods
    
    async def get_server_count(self) -> int:
        """Get the current number of servers.
        
        Returns:
            Number of servers
        """
        response = await self.execute_command("get_servers")
        return int(response)
    
    async def get_max_servers(self) -> int:
        """Get the maximum number of servers allowed.
        
        Returns:
            Maximum server count
        """
        response = await self.execute_command("get_max_servers")
        return int(response)
    
    async def get_active_servers(self) -> int:
        """Get the number of active servers.
        
        Returns:
            Active server count
        """
        response = await self.execute_command("get_active_servers")
        return int(response)
    
    async def get_dimmer(self) -> float:
        """Get the current dimmer value.
        
        Returns:
            Dimmer value (0.0 to 1.0)
        """
        response = await self.execute_command("dimmer")
        return float(response)
    
    async def set_dimmer(self, value: float) -> str:
        """Set the dimmer value.
        
        Args:
            value: Dimmer value (0.0 to 1.0)
            
        Returns:
            Response from SWIM
        """
        
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Dimmer value must be between 0.0 and 1.0, got {value}")
        
        return await self.execute_command("set_dimmer {value}", {"value": value})
    
    async def add_server(self) -> str:
        """Add a server to the pool.
        
        Returns:
            Response from SWIM
        """
        # Check if we can add a server
        current = await self.get_server_count()
        maximum = await self.get_max_servers()
        
        if current >= maximum:
            raise ValueError(f"Cannot add server: already at maximum ({maximum})")
        
        return await self.execute_command("add_server")
    
    async def remove_server(self) -> str:
        """Remove a server from the pool.
        
        Returns:
            Response from SWIM
        """
        # Check if we can remove a server
        current = await self.get_server_count()
        
        if current <= 1:
            raise ValueError(f"Cannot remove server: only {current} server(s) remaining")
        
        return await self.execute_command("remove_server")
    
    async def get_utilization(self, server_id: int) -> float:
        """Get utilization for a specific server.
        
        Args:
            server_id: Server ID (1-based)
            
        Returns:
            Server utilization
        """
        response = await self.execute_command(
            "get_utilization server{id}",
            {"id": server_id}
        )
        return float(response)
    
    async def get_total_utilization(self) -> float:
        """Get total utilization across all active servers.
        
        Returns:
            Total utilization
        """
        active_servers = await self.get_active_servers()
        total = 0.0
        
        for server_id in range(1, active_servers + 1):
            try:
                util = await self.get_utilization(server_id)
                total += util
            except Exception as e:
                self.logger.warning(
                    "Failed to get utilization for server",
                    extra={
                        "server_id": server_id,
                        "error": str(e)
                    }
                )
        
        return total
