"""
Switch System Connector for POLARIS Framework.

This module implements the managed system connector for the Switch system,
handling HTTP API communication with the YOLO model switching system.
"""

import asyncio
import aiohttp
import time
import csv
import os
from typing import Any, Dict, Optional, List
from pathlib import Path
import sys

# Add parent directory to path to import from polaris
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.adapters.core import ManagedSystemConnector
from polaris.common.nats_client import jittered_backoff


class SwitchSystemConnector(ManagedSystemConnector):
    """
    HTTP connector for Switch System (YOLO Model Switching System).
    
    This connector implements the communication protocol for the Switch system's
    REST API interface for model switching and monitoring.
    """
    
    def __init__(self, system_config: Dict[str, Any], logger):
        """Initialize the Switch system connector.
        
        Args:
            system_config: Complete configuration for Switch system
            logger: Logger instance for structured logging
        """
        super().__init__(system_config, logger)
        
        # Extract connection parameters
        self.host = self.connection_config.get("host", "localhost")
        self.port = self.connection_config.get("port", 3001)
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Extract retry parameters
        self.timeout = self.get_timeout()
        self.max_retries = self.get_max_retries()
        self.retry_base_delay = self.implementation_config.get("retry_base_delay", 1.0)
        self.retry_max_delay = self.implementation_config.get("retry_max_delay", 5.0)
        
        # Switch system specific configuration
        self.switch_config = system_config.get("switch_system", {})
        self.model_file_path = self.switch_config.get("model_file_path", "model.csv")
        self.monitor_file_path = self.switch_config.get("monitor_file_path", "monitor.csv")
        self.knowledge_file_path = self.switch_config.get("knowledge_file_path", "knowledge.csv")
        self.metrics_file_path = self.switch_config.get("metrics_file_path", "metrics.csv")
        
        # Available YOLO models
        self.available_models = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]
        self.model_action_map = {
            "yolov5n": 1,
            "yolov5s": 2, 
            "yolov5m": 3,
            "yolov5l": 4,
            "yolov5x": 5
        }
        
        # Connection state
        self._connected = False
        self._session = None
        
        self.logger.info(
            "Switch system connector initialized",
            extra={
                "host": self.host,
                "port": self.port,
                "base_url": self.base_url,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "available_models": self.available_models
            }
        )
    
    async def connect(self) -> None:
        """Establish connection to Switch system.
        
        Creates an aiohttp session and tests connectivity.
        """
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            # Test connection with health check
            is_healthy = await self.health_check()
            if not is_healthy:
                raise ConnectionError("Switch system health check failed")
            
            self._connected = True
            self.logger.info(
                "Switch system connection established",
                extra={
                    "host": self.host,
                    "port": self.port
                }
            )
        except Exception as e:
            self._connected = False
            if self._session:
                await self._session.close()
                self._session = None
            
            self.logger.error(
                "Failed to connect to Switch system",
                extra={
                    "host": self.host,
                    "port": self.port,
                    "error": str(e)
                }
            )
            raise ConnectionError(f"Cannot connect to Switch system at {self.host}:{self.port}: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Switch system."""
        if self._session:
            await self._session.close()
            self._session = None
        
        self._connected = False
        self.logger.info("Switch system connector disconnected")
    
    async def execute_command(
        self,
        command_template: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a command on Switch system.
        
        Args:
            command_template: Command template (e.g., "switch_model", "get_metrics")
            params: Parameters for the command
            
        Returns:
            Response from Switch system
            
        Raises:
            TimeoutError: If command times out
            ConnectionError: If connection fails
            Exception: For other errors
        """
        if not self._connected or not self._session:
            raise ConnectionError("Not connected to Switch system")
        
        # Route command to appropriate handler
        if command_template == "switch_model":
            return await self._switch_model(params)
        elif command_template == "get_current_model":
            return await self._get_current_model()
        elif command_template == "get_metrics":
            return await self._get_metrics(params)
        elif command_template == "get_latest_metrics":
            return await self._get_latest_metrics()
        elif command_template == "get_latest_logs":
            return await self._get_latest_logs()
        elif command_template == "update_knowledge":
            return await self._update_knowledge(params)
        elif command_template == "start_processing":
            return await self._start_processing()
        elif command_template == "stop_processing":
            return await self._stop_processing()
        elif command_template == "restart_processing":
            return await self._restart_processing()
        else:
            raise ValueError(f"Unknown command: {command_template}")
    
    async def health_check(self) -> bool:
        """Check if Switch system is healthy and responsive.
        
        Returns:
            True if Switch system responds, False otherwise
        """
        try:
            if not self._session:
                return False
            
            # Try to get latest metrics as a health check
            async with self._session.post(f"{self.base_url}/api/latest_metrics_data") as response:
                if response.status == 200:
                    return True
                else:
                    self.logger.warning(
                        "Switch system health check failed",
                        extra={"status_code": response.status}
                    )
                    return False
                    
        except Exception as e:
            self.logger.warning(
                "Switch system health check failed",
                extra={"error": str(e)}
            )
            return False
    
    # Switch system specific methods
    
    async def _switch_model(self, params: Optional[Dict[str, Any]]) -> str:
        """Switch to a different YOLO model.
        
        Args:
            params: Dictionary containing 'model' key with model name
            
        Returns:
            Success message
        """
        if not params or "model" not in params:
            raise ValueError("Model parameter required for switch_model command")
        
        model_name = params["model"]
        if model_name not in self.available_models:
            raise ValueError(f"Invalid model: {model_name}. Available: {self.available_models}")
        
        try:
            # Write model to CSV file (this is how the Switch system works)
            await self._write_model_file(model_name)
            
            self.logger.info(
                "Model switched successfully",
                extra={
                    "new_model": model_name,
                    "action_id": self.model_action_map[model_name]
                }
            )
            
            return f"Successfully switched to model: {model_name}"
            
        except Exception as e:
            self.logger.error(
                "Failed to switch model",
                extra={
                    "target_model": model_name,
                    "error": str(e)
                }
            )
            raise
    
    async def _get_current_model(self) -> str:
        """Get the currently active model.
        
        Returns:
            Current model name
        """
        try:
            # Read from model.csv file
            if os.path.exists(self.model_file_path):
                with open(self.model_file_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if rows:
                        current_model = rows[0][0].strip()
                        self.logger.debug(f"Current model: {current_model}")
                        return current_model
            
            # Default model if file doesn't exist
            return "yolov5n"
            
        except Exception as e:
            self.logger.error(f"Failed to get current model: {e}")
            return "yolov5n"  # Default fallback
    
    async def _get_metrics(self, params: Optional[Dict[str, Any]]) -> str:
        """Get system metrics.
        
        Args:
            params: Optional parameters for metric filtering
            
        Returns:
            JSON string of metrics data
        """
        try:
            async with self._session.post(f"{self.base_url}/api/latest_metrics_data") as response:
                if response.status == 200:
                    data = await response.json()
                    return str(data.get("message", {}))
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise
    
    async def _get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics from the system.
        
        Returns:
            Dictionary containing latest metrics
        """
        try:
            async with self._session.post(f"{self.base_url}/api/latest_metrics_data") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("message", {})
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get latest metrics: {e}")
            return {}
    
    async def _get_latest_logs(self) -> Dict[str, Any]:
        """Get the latest logs from the system.
        
        Returns:
            Dictionary containing latest log data
        """
        try:
            async with self._session.post(f"{self.base_url}/api/latest_logs") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("message", {})
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get latest logs: {e}")
            return {}
    
    async def _update_knowledge(self, params: Dict[str, Any]) -> str:
        """Update the knowledge base (thresholds) for model switching.
        
        Args:
            params: Dictionary containing threshold values for each model
            
        Returns:
            Success message
        """
        try:
            # Prepare the knowledge update payload
            knowledge_data = {}
            for model in self.available_models:
                lower_key = f"{model}Lower"
                upper_key = f"{model}Upper"
                if lower_key in params and upper_key in params:
                    knowledge_data[lower_key] = params[lower_key]
                    knowledge_data[upper_key] = params[upper_key]
            
            async with self._session.post(
                f"{self.base_url}/api/changeKnowledge",
                json=knowledge_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.info("Knowledge base updated successfully")
                    return result.get("message", "Knowledge updated")
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update knowledge: {e}")
            raise
    
    async def _start_processing(self) -> str:
        """Start the image processing pipeline.
        
        Returns:
            Success message
        """
        try:
            async with self._session.post(f"{self.base_url}/execute-python-script") as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.info("Processing started successfully")
                    return result.get("message", "Processing started")
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            raise
    
    async def _stop_processing(self) -> str:
        """Stop the image processing pipeline.
        
        Returns:
            Success message
        """
        try:
            async with self._session.post(f"{self.base_url}/api/stopProcess") as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.info("Processing stopped successfully")
                    return result.get("message", "Processing stopped")
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            self.logger.error(f"Failed to stop processing: {e}")
            raise
    
    async def _restart_processing(self) -> str:
        """Restart the image processing pipeline.
        
        Returns:
            Success message
        """
        try:
            async with self._session.post(f"{self.base_url}/api/newProcess") as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.info("Processing restarted successfully")
                    return result.get("message", "Processing restarted")
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            self.logger.error(f"Failed to restart processing: {e}")
            raise
    
    async def _write_model_file(self, model_name: str) -> None:
        """Write model name to the model.csv file.
        
        Args:
            model_name: Name of the model to write
        """
        try:
            with open(self.model_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_name])
                
        except Exception as e:
            self.logger.error(f"Failed to write model file: {e}")
            raise
    
    # High-level convenience methods
    
    async def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state including current model and metrics.
        
        Returns:
            Dictionary containing system state information
        """
        try:
            current_model = await self._get_current_model()
            latest_metrics = await self._get_latest_metrics()
            latest_logs = await self._get_latest_logs()
            
            return {
                "current_model": current_model,
                "metrics": latest_metrics,
                "logs": latest_logs,
                "available_models": self.available_models,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system state: {e}")
            return {
                "current_model": "unknown",
                "metrics": {},
                "logs": {},
                "available_models": self.available_models,
                "timestamp": time.time(),
                "error": str(e)
            }
    
    async def switch_to_optimal_model(self, input_rate: float) -> str:
        """Switch to the optimal model based on input rate.
        
        This method implements the logic from the original Switch system
        to determine which model is best for a given input rate.
        
        Args:
            input_rate: Current input rate (images per second)
            
        Returns:
            Name of the model switched to
        """
        try:
            # Read knowledge base to get thresholds
            knowledge = await self._read_knowledge_file()
            
            # Find the appropriate model for the input rate
            for i, (model_name, min_rate, max_rate) in enumerate(knowledge):
                if min_rate <= input_rate <= max_rate:
                    await self._switch_model({"model": model_name})
                    return model_name
            
            # If no model fits, use default logic
            if input_rate < 1.0:
                target_model = "yolov5n"
            elif input_rate < 5.0:
                target_model = "yolov5s"
            elif input_rate < 10.0:
                target_model = "yolov5m"
            elif input_rate < 20.0:
                target_model = "yolov5l"
            else:
                target_model = "yolov5x"
            
            await self._switch_model({"model": target_model})
            return target_model
            
        except Exception as e:
            self.logger.error(f"Failed to switch to optimal model: {e}")
            raise
    
    async def _read_knowledge_file(self) -> List[tuple]:
        """Read the knowledge.csv file containing model thresholds.
        
        Returns:
            List of tuples (model_name, min_rate, max_rate)
        """
        try:
            knowledge = []
            if os.path.exists(self.knowledge_file_path):
                with open(self.knowledge_file_path, 'r') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if len(row) >= 3:
                            model_name = self.available_models[i]
                            min_rate = float(row[1])
                            max_rate = float(row[2])
                            knowledge.append((model_name, min_rate, max_rate))
            
            return knowledge
            
        except Exception as e:
            self.logger.error(f"Failed to read knowledge file: {e}")
            # Return default thresholds
            return [
                ("yolov5n", 0.0, 2.0),
                ("yolov5s", 2.0, 5.0),
                ("yolov5m", 5.0, 10.0),
                ("yolov5l", 10.0, 20.0),
                ("yolov5x", 20.0, 100.0)
            ]
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get key performance metrics from the system.
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            metrics = await self._get_latest_metrics()
            
            # Extract key metrics based on the Switch system's metric structure
            performance = {
                "image_processing_time": float(metrics.get("image_processing_time", 0.0)),
                "confidence": float(metrics.get("confidence", 0.0)),
                "utility": float(metrics.get("utility", 0.0)),
                "cpu_usage": float(metrics.get("cpu_usage", 0.0)),
                "detection_boxes": int(metrics.get("detection_boxes", 0)),
                "total_processed": int(metrics.get("total_processed", 0))
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {
                "image_processing_time": 0.0,
                "confidence": 0.0,
                "utility": 0.0,
                "cpu_usage": 0.0,
                "detection_boxes": 0,
                "total_processed": 0
            }