"""
SWIM Monitor Adapter for POLARIS POC

This adapter connects to the SWIM system via TCP, continuously monitors
key metrics, and publishes telemetry events to the NATS message bus.
"""

import asyncio
import json
import logging
import socket
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout


@dataclass
class TelemetryEvent:
    """Structured telemetry event for POLARIS"""
    event_type: str
    timestamp: str
    source: str
    payload: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps({
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "source": self.source,
            "payload": self.payload
        })


class SwimTCPClient:
    """TCP client for communicating with SWIM system"""
    
    def __init__(self, host: str = "localhost", port: int = 6565):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Establish connection to SWIM"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second timeout
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
    
    def send_command(self, command: str) -> Optional[str]:
        """Send command to SWIM and receive response"""
        if not self.connected or not self.socket:
            self.logger.warning("Not connected to SWIM")
            return None
        
        try:
            # Send command with newline
            self.socket.sendall((command + "\n").encode())
            
            # Receive response
            data = self.socket.recv(1024)
            response = data.decode().strip()
            
            self.logger.debug(f"SWIM command '{command}' -> '{response}'")
            return response
            
        except Exception as e:
            self.logger.error(f"Error sending command '{command}': {e}")
            self.connected = False
            return None
    
    def get_metric(self, command: str) -> Optional[float]:
        """Get a numeric metric from SWIM"""
        response = self.send_command(command)
        if response is None:
            return None
        
        try:
            return float(response)
        except ValueError:
            self.logger.warning(f"Non-numeric response for '{command}': {response}")
            return None
    
    def close(self):
        """Close connection to SWIM"""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False


class SwimMonitorAdapter:
    """Main monitor adapter class"""
    
    def __init__(self, 
                 nats_url: str = "nats://localhost:4222",
                 swim_host: str = "localhost", 
                 swim_port: int = 6565,
                 monitoring_interval: float = 5.0):
        
        self.nats_url = nats_url
        self.swim_client = SwimTCPClient(swim_host, swim_port)
        self.monitoring_interval = monitoring_interval
        
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
    
    async def collect_metrics(self) -> Optional[Dict[str, float]]:
        """Collect all relevant metrics from SWIM"""
        if not self.swim_client.connected:
            if not self.swim_client.connect():
                return None
        
        metrics = {}
        
        # Define metrics to collect with their SWIM commands
        metric_commands = {
            "dimmer": "get_dimmer",
            "active_servers": "get_active_servers", 
            "max_servers": "get_max_servers",
            "servers": "get_servers",
            "basic_response_time": "get_basic_rt",
            "optional_response_time": "get_opt_rt",
            "basic_throughput": "get_basic_throughput",
            "optional_throughput": "get_opt_throughput",
            "arrival_rate": "get_arrival_rate"
        }
        
        # Collect basic metrics
        for metric_name, command in metric_commands.items():
            value = self.swim_client.get_metric(command)
            if value is not None:
                metrics[metric_name] = value
        
        # Calculate derived metrics
        if all(key in metrics for key in ["basic_response_time", "optional_response_time", 
                                          "basic_throughput", "optional_throughput"]):
            total_throughput = metrics["basic_throughput"] + metrics["optional_throughput"]
            if total_throughput > 0:
                avg_response_time = (
                    metrics["basic_response_time"] * metrics["basic_throughput"] +
                    metrics["optional_response_time"] * metrics["optional_throughput"]
                ) / total_throughput
                metrics["average_response_time"] = avg_response_time
        
        # Get total utilization
        if "active_servers" in metrics:
            total_util = 0.0
            active_servers = int(metrics["active_servers"])
            
            for server_id in range(1, active_servers + 1):
                util = self.swim_client.get_metric(f"get_utilization server{server_id}")
                if util is not None:
                    total_util += util
            
            metrics["total_utilization"] = total_util
        
        return metrics if metrics else None
    
    async def publish_telemetry_event(self, metric_name: str, value: float):
        """Publish a single telemetry event to NATS"""
        if not self.nc:
            self.logger.warning("NATS not connected")
            return
        
        event = TelemetryEvent(
            event_type="TelemetryEvent",
            timestamp=datetime.now().isoformat(),
            source="swim_monitor",
            payload={
                "name": f"swim.{metric_name}",
                "value": value,
                "unit": self._get_metric_unit(metric_name)
            }
        )
        
        try:
            await self.nc.publish("polaris.telemetry.events", event.to_json().encode())
            self.logger.debug(f"Published telemetry: {metric_name}={value}")
        except Exception as e:
            self.logger.error(f"Failed to publish telemetry event: {e}")
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for metric"""
        unit_mapping = {
            "dimmer": "ratio",
            "active_servers": "count",
            "max_servers": "count", 
            "servers": "count",
            "basic_response_time": "ms",
            "optional_response_time": "ms",
            "average_response_time": "ms",
            "basic_throughput": "req/s",
            "optional_throughput": "req/s",
            "arrival_rate": "req/s",
            "total_utilization": "ratio"
        }
        return unit_mapping.get(metric_name, "unknown")
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info(f"Starting monitoring loop (interval: {self.monitoring_interval}s)")
        
        while self.running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                if metrics:
                    # Publish each metric as a separate telemetry event
                    for metric_name, value in metrics.items():
                        await self.publish_telemetry_event(metric_name, value)
                    
                    self.logger.info(f"Published {len(metrics)} telemetry events")
                else:
                    self.logger.warning("Failed to collect metrics from SWIM")
                
                # Wait for next collection cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def start(self):
        """Start the monitor adapter"""
        self.logger.info("Starting SWIM Monitor Adapter")
        
        # Connect to NATS
        if not await self.connect_nats():
            self.logger.error("Failed to connect to NATS, exiting")
            return
        
        # Connect to SWIM
        if not self.swim_client.connect():
            self.logger.error("Failed to connect to SWIM, exiting")
            return
        
        # Start monitoring
        self.running = True
        try:
            await self.monitoring_loop()
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the monitor adapter"""
        self.logger.info("Stopping SWIM Monitor Adapter")
        self.running = False
        
        if self.nc:
            await self.nc.close()
        
        self.swim_client.close()


async def main():
    """Main entry point"""
    adapter = SwimMonitorAdapter()
    await adapter.start()


if __name__ == "__main__":
    asyncio.run(main())