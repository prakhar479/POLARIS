"""
SWIM Execution Adapter for POLARIS POC

This adapter receives control actions from the POLARIS system via NATS
and executes them on the SWIM system via TCP connection.
"""

import asyncio
import json
import logging
import socket
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from nats.aio.client import Client as NATS, Msg
from nats.aio.errors import ErrConnectionClosed, ErrTimeout


class ActionType(Enum):
    """Supported action types"""
    ADD_SERVER = "ADD_SERVER"
    REMOVE_SERVER = "REMOVE_SERVER" 
    ADJUST_QOS = "ADJUST_QOS"
    SET_DIMMER = "SET_DIMMER"


@dataclass
class ControlAction:
    """Structure for control actions from POLARIS"""
    action_type: str
    timestamp: str
    source: str
    params: Dict[str, Any]
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ControlAction':
        data = json.loads(json_str)
        return cls(
            action_type=data.get("action_type", ""),
            timestamp=data.get("timestamp", ""),
            source=data.get("source", ""),
            params=data.get("params", {})
        )


@dataclass
class ExecutionResult:
    """Result of action execution"""
    action_type: str
    success: bool
    message: str
    timestamp: str
    
    def to_json(self) -> str:
        return json.dumps({
            "action_type": self.action_type,
            "success": self.success,
            "message": self.message,
            "timestamp": self.timestamp
        })


class SwimExecutionClient:
    """TCP client for executing commands on SWIM"""
    
    def __init__(self, host: str = "localhost", port: int = 6565):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        self.strike_count = 0
    
    def connect(self) -> bool:
        """Establish connection to SWIM"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.logger.info(f"Connected to SWIM at {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to SWIM: {e}")
            self.connected = False
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def send_command(self, command: str) -> str:
        """Send command to SWIM and receive response"""
        if not self.connected or not self.socket:
            if not self.connect():
                return "Error: Not connected to SWIM"
        
        try:
            self.socket.sendall((command + "\n").encode())
            data = self.socket.recv(1024)
            response = data.decode().strip()
            
            self.logger.info(f"Executed SWIM command: '{command}' -> '{response}'")
            return response
            
        except Exception as e:
            self.logger.error(f"Error executing command '{command}': {e}")
            self.connected = False
            return f"Error: {str(e)}"
    
    def get_current_servers(self) -> int:
        """Get current number of servers"""
        try:
            response = self.send_command("get_servers")
            return int(response)
        except:
            return 0
    
    def get_max_servers(self) -> int:
        """Get maximum number of servers"""
        try:
            response = self.send_command("get_max_servers")
            return int(response)
        except:
            return 0
    
    def add_server(self) -> str:
        """Add a server to SWIM"""
        current_servers = self.get_current_servers()
        max_servers = self.get_max_servers()
        
        if current_servers < max_servers:
            return self.send_command("add_server")
        else:
            self.strike_count += 1
            error_msg = f"Strike {self.strike_count} - Cannot add server: already at maximum ({max_servers})"
            self.logger.warning(error_msg)
            return error_msg
    
    def remove_server(self) -> str:
        """Remove a server from SWIM"""
        current_servers = self.get_current_servers()
        
        if current_servers > 1:
            return self.send_command("remove_server")
        else:
            self.strike_count += 1
            error_msg = f"Strike {self.strike_count} - Cannot remove server: only 1 server remaining"
            self.logger.warning(error_msg)
            return error_msg
    
    def set_dimmer(self, dimmer_value: float) -> str:
        """Set dimmer value (QoS adjustment)"""
        if not (0.0 <= dimmer_value <= 1.0):
            self.strike_count += 1
            error_msg = f"Strike {self.strike_count} - Invalid dimmer value: {dimmer_value} (must be 0.0-1.0)"
            self.logger.warning(error_msg)
            return error_msg
        
        return self.send_command(f"set_dimmer {dimmer_value}")
    
    def close(self):
        """Close connection to SWIM"""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False


class SwimExecutionAdapter:
    """Main execution adapter class"""
    
    def __init__(self, 
                 nats_url: str = "nats://localhost:4222",
                 swim_host: str = "localhost",
                 swim_port: int = 6565):
        
        self.nats_url = nats_url
        self.swim_client = SwimExecutionClient(swim_host, swim_port)
        
        self.nc: Optional[NATS] = None
        self.running = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def connect_nats(self) -> bool:
        """Connect to NATS server"""
        try:
            self.nc = NATS()
            await self.nc.connect(self.nats_url)
            self.logger.info(f"Connected to NATS at {self.nats_url}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to NATS: {e}")
            return False
    
    def execute_action(self, action: ControlAction) -> ExecutionResult:
        """Execute a control action on SWIM"""
        self.logger.info(f"Executing action: {action.action_type}")
        
        try:
            if action.action_type == ActionType.ADD_SERVER.value:
                result = self.swim_client.add_server()
                success = not result.startswith("Strike") and not result.startswith("Error")
                
            elif action.action_type == ActionType.REMOVE_SERVER.value:
                result = self.swim_client.remove_server()
                success = not result.startswith("Strike") and not result.startswith("Error")
                
            elif action.action_type in [ActionType.ADJUST_QOS.value, ActionType.SET_DIMMER.value]:
                dimmer_value = action.params.get("value", action.params.get("dimmer_value"))
                if dimmer_value is None:
                    result = "Error: Missing dimmer value in parameters"
                    success = False
                else:
                    result = self.swim_client.set_dimmer(float(dimmer_value))
                    success = not result.startswith("Strike") and not result.startswith("Error")
            
            else:
                result = f"Error: Unknown action type '{action.action_type}'"
                success = False
            
            return ExecutionResult(
                action_type=action.action_type,
                success=success,
                message=result,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Exception executing action: {e}")
            return ExecutionResult(
                action_type=action.action_type,
                success=False,
                message=f"Exception: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    async def publish_execution_result(self, result: ExecutionResult):
        """Publish execution result to NATS"""
        if not self.nc:
            return
        
        try:
            await self.nc.publish("polaris.execution.results", result.to_json().encode())
            self.logger.debug(f"Published execution result: {result.action_type} - {result.success}")
        except Exception as e:
            self.logger.error(f"Failed to publish execution result: {e}")
    
    async def action_handler(self, msg: Msg):
        """Handle incoming control actions"""
        try:
            # Decode and parse action
            action_data = msg.data.decode()
            action = ControlAction.from_json(action_data)
            
            self.logger.info(f"Received action: {action.action_type} from {action.source}")
            
            # Execute the action
            result = self.execute_action(action)
            
            # Log result
            status = "SUCCESS" if result.success else "FAILED"
            self.logger.info(f"Action {action.action_type} {status}: {result.message}")
            
            # Publish result
            await self.publish_execution_result(result)
            
        except Exception as e:
            self.logger.error(f"Error handling action message: {e}")
    
    async def start(self):
        """Start the execution adapter"""
        self.logger.info("Starting SWIM Execution Adapter")
        
        # Connect to NATS
        if not await self.connect_nats():
            self.logger.error("Failed to connect to NATS, exiting")
            return
        
        # Connect to SWIM
        if not self.swim_client.connect():
            self.logger.error("Failed to connect to SWIM, exiting")
            return
        
        # Subscribe to action topics
        await self.nc.subscribe("polaris.actions.swim_adapter", cb=self.action_handler)
        await self.nc.subscribe("polaris.actions.swim", cb=self.action_handler)  # Alternative topic name
        
        self.logger.info("Subscribed to action topics, waiting for commands...")
        self.running = True
        
        try:
            # Keep running
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the execution adapter"""
        self.logger.info("Stopping SWIM Execution Adapter")
        self.running = False
        
        if self.nc:
            await self.nc.close()
        
        self.swim_client.close()


async def main():
    """Main entry point"""
    adapter = SwimExecutionAdapter()
    await adapter.start()


if __name__ == "__main__":
    asyncio.run(main())