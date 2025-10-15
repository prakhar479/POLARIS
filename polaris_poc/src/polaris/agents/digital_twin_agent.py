"""
Digital Twin Agent for POLARIS Framework.

This module implements the core Digital Twin Agent that serves as the central
"World Model" for the POLARIS framework, providing both asynchronous ingestion
of system updates via NATS and synchronous query capabilities via gRPC.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from nats.aio.msg import Msg
import grpc
from concurrent import futures

from polaris.common.nats_client import NATSClient
from polaris.models.digital_twin_events import KnowledgeEvent, CalibrationEvent
from polaris.models.world_model import (
    WorldModel, WorldModelFactory, WorldModelError,
    QueryRequest, SimulationRequest, DiagnosisRequest
)
from polaris.services.digital_twin_service import DigitalTwinService
from polaris.proto import digital_twin_pb2_grpc


# Import World Model implementation for registration 
# add other implementations here to register
from polaris.models.mock_world_model import MockWorldModel
from polaris.models.gemini_world_model import GeminiWorldModel
from polaris.models.bayesian_world_model import BayesianWorldModel

class DigitalTwinAgent:
    """
    Digital Twin Agent implementing hybrid interface architecture.
    
    This agent provides:
    - Asynchronous NATS ingestion for system updates
    - Synchronous gRPC interface for queries and simulations
    - Implementation-agnostic World Model integration
    - Comprehensive error handling and observability
    """
    
    def __init__(
        self,
        config_path: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the Digital Twin Agent.
        
        Args:
            config_path: Path to POLARIS framework configuration
            logger: Logger instance (created if not provided)
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Load configuration using Digital Twin specific manager
        from polaris.common.digital_twin_config import DigitalTwinConfigManager
        self.config_manager = DigitalTwinConfigManager(self.logger)
        
        # Load complete Digital Twin configuration (no legacy fallback)
        complete_config = self.config_manager.load_configuration(config_path)
        self.framework_config = complete_config["framework"]
        self.dt_config = complete_config["digital_twin"]
        self.world_model_config = complete_config["world_model"]
        
        # Extract configuration sections
        self.nats_config = self.dt_config.get("nats", {})
        self.grpc_config = self.dt_config.get("grpc", {})
        
        # Initialize NATS client
        nats_url = self.framework_config.get("nats", {}).get("url", "nats://localhost:4222")
        self.nats_client = NATSClient(
            nats_url=nats_url,
            logger=self.logger,
            name="digital-twin-agent",
            max_reconnect_attempts=self.nats_config.get("max_reconnect_attempts", 10),
            reconnect_base_delay=self.nats_config.get("reconnect_wait_sec", 2)
        )
        
        # Initialize World Model
        self.world_model: Optional[WorldModel] = None
        
        # gRPC server components
        self.grpc_server: Optional[grpc.aio.Server] = None
        self.grpc_service: Optional[DigitalTwinService] = None
        
        # Runtime state
        self.running = False
        self._tasks = []
        self._message_queue = asyncio.Queue(maxsize=self.nats_config.get("queue_maxsize", 1000))
        self._batch_size = self.nats_config.get("batch_size", 10)
        self._batch_timeout = self.nats_config.get("batch_timeout_sec", 1.0)
        
        # Performance metrics
        self._metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "batches_processed": 0,
            "last_batch_time": None,
            "queue_high_water_mark": 0
        }
        
        # NATS subscription tracking
        self._subscriptions = {}
        
        self.logger.info(
            "Digital Twin Agent initialized",
            extra={
                "nats_url": nats_url,
                "grpc_port": self.grpc_config.get("port", 50051),
                "world_model_type": self.world_model_config.get("implementation", "mock")
            }
        )
    
    async def start(self) -> None:
        """Start the Digital Twin Agent.
        
        This method establishes connections and starts processing.
        
        Raises:
            Exception: If startup fails
        """
        if self.running:
            self.logger.warning("Digital Twin Agent is already running")
            return
        
        self.logger.info("Starting Digital Twin Agent")
        
        try:
            # Connect to NATS
            await self.nats_client.connect()
            
            # Load and initialize World Model
            await self._load_world_model()
            
            # Start NATS ingestion engine
            await self._start_nats_ingestion()
            
            # Start gRPC server
            await self._start_grpc_server()
            
            # Start message processing
            await self._start_processing()
            
            self.running = True
            
            self.logger.info(
                "Digital Twin Agent started successfully",
                extra={
                    "status": "running",
                    "world_model": self.world_model_config.get("implementation", "mock"),
                    "nats_connected": self.nats_client.is_connected,
                    "grpc_port": self.grpc_config.get("port", 50051)
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to start Digital Twin Agent",
                extra={"error": str(e)}
            )
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the Digital Twin Agent gracefully.
        
        This method stops processing and closes connections.
        """
        if not self.running:
            self.logger.debug("Digital Twin Agent is not running")
            return
        
        self.logger.info("Stopping Digital Twin Agent")
        
        # Clear running flag
        self.running = False
        
        # Stop message processing
        await self._stop_processing()
        
        # Stop gRPC server
        await self._stop_grpc_server()
        
        # Stop NATS ingestion
        await self._stop_nats_ingestion()
        
        # Shutdown World Model
        if self.world_model:
            try:
                await self.world_model.shutdown()
            except Exception as e:
                self.logger.error(
                    "Error shutting down World Model",
                    extra={"error": str(e)}
                )
        
        # Close NATS connection
        await self.nats_client.close()
        
        self.logger.info("Digital Twin Agent stopped")
    
    async def _load_world_model(self) -> None:
        """Load and initialize the World Model implementation with enhanced error handling."""
        # Get model type and config from the flat config structure
        model_type = self.world_model_config.get("implementation", "mock")
        # Pass only the implementation-specific 'config' section to the model
        model_config = self.world_model_config.get("config", {})
        
        # Validate configuration before attempting to create model
        from polaris.models.world_model import ConfigurationValidator
        
        config_validation = ConfigurationValidator.validate_world_model_config(
            self.world_model_config
        )
        
        if not config_validation["valid"]:
            error_msg = f"Invalid World Model configuration: {config_validation['errors']}"
            self.logger.error(error_msg)
            raise WorldModelError(error_msg)
        
        # Log any configuration warnings
        for warning in config_validation["warnings"]:
            self.logger.warning(f"World Model configuration warning: {warning}")
        
        # Log any configuration recommendations
        for recommendation in config_validation["recommendations"]:
            self.logger.info(f"World Model configuration recommendation: {recommendation}")
        
        try:
            # Check if model type is registered
            if not WorldModelFactory.is_registered(model_type):
                available_types = WorldModelFactory.get_registered_types()
                raise WorldModelError(
                    f"Unknown World Model type '{model_type}'. "
                    f"Available types: {available_types}"
                )
            
            # Create model instance - pass only implementation-specific config
            self.world_model = WorldModelFactory.create_model(
                model_type=model_type,
                config=model_config,
                logger=self.logger
            )
            
            # Initialize the model
            await self.world_model.initialize()
            
            # Validate interface compliance if in debug mode
            debug_config = self.dt_config.get("debugging", {})
            if debug_config.get("validate_interface", False):
                from polaris.models.world_model import WorldModelValidator
                
                validation_results = await WorldModelValidator.validate_interface_compliance(
                    self.world_model
                )
                
                if not validation_results["compliant"]:
                    self.logger.warning(
                        "World Model interface compliance issues detected",
                        extra={"validation_errors": validation_results["errors"]}
                    )
                else:
                    self.logger.debug("World Model interface compliance validated successfully")
            
            self.logger.info(
                "World Model loaded successfully",
                extra={
                    "model_type": model_type,
                    "model_class": self.world_model.__class__.__name__,
                    "initialized": self.world_model.is_initialized,
                    "config_warnings": len(config_validation["warnings"]),
                    "config_recommendations": len(config_validation["recommendations"])
                }
            )
            
        except WorldModelError:
            # Re-raise WorldModelError as-is
            raise
        except Exception as e:
            error_msg = f"Failed to load {model_type} World Model: {str(e)}"
            self.logger.error(
                "World Model instantiation failed",
                extra={
                    "model_type": model_type,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise WorldModelError(error_msg) from e
    
    async def _start_processing(self) -> None:
        """Start adapter-specific processing tasks."""
        # Start message processing task
        message_processor = asyncio.create_task(self._process_messages())
        self._tasks.append(message_processor)
        
        self.logger.debug("Message processing started")
    
    async def _stop_processing(self) -> None:
        """Stop adapter-specific processing tasks."""
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        
        self.logger.debug("Message processing stopped")
    
    async def _send_to_dead_letter_queue(self, msg: Msg, message_type: str, error: str) -> None:
        """Send failed message to dead letter queue.
        
        Args:
            msg: Original NATS message that failed
            message_type: Type of message (update/calibrate)
            error: Error description
        """
        try:
            error_subject = self.nats_config.get("error_subject", "polaris.digitaltwin.errors")
            
            # Create error envelope
            error_envelope = {
                "original_subject": msg.subject,
                "original_data": msg.data.decode() if msg.data else None,
                "message_type": message_type,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retry_count": 0  # Could be enhanced with retry logic
            }
            
            # Publish to error topic
            await self.nats_client.publish_json(
                subject=error_subject,
                data=error_envelope
            )
            
            self.logger.warning(
                "Message sent to dead letter queue",
                extra={
                    "original_subject": msg.subject,
                    "error_subject": error_subject,
                    "message_type": message_type
                }
            )
            
        except Exception as dlq_error:
            self.logger.error(
                "Failed to send message to dead letter queue",
                extra={
                    "original_error": error,
                    "dlq_error": str(dlq_error)
                }
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()    

    # NATS Ingestion Engine Methods
    
    async def _start_nats_ingestion(self) -> None:
        """Start NATS subscription handlers for async updates."""
        try:
            queue_group = self.nats_config.get("queue_group", "digital_twin_workers")
            
            # Subscribe to telemetry batch topic (from Monitor adapters)
            telemetry_config = self.framework_config.get("telemetry", {})
            telemetry_batch_subject = telemetry_config.get("batch_subject", "polaris.telemetry.events.batch")
            telemetry_stream_subject = telemetry_config.get("stream_subject", "polaris.telemetry.events.stream")
            
            # Subscribe to batch telemetry events
            batch_sid = await self.nats_client.subscribe(
                subject=telemetry_batch_subject,
                callback=self._handle_telemetry_batch_message,
                queue=queue_group
            )
            self._subscriptions["telemetry_batch"] = batch_sid
            
            # Subscribe to stream telemetry events
            stream_sid = await self.nats_client.subscribe(
                subject=telemetry_stream_subject,
                callback=self._handle_telemetry_stream_message,
                queue=queue_group
            )
            self._subscriptions["telemetry_stream"] = stream_sid
            
            # Subscribe to execution results (from Execution adapters)
            execution_config = self.framework_config.get("execution", {})
            execution_result_subject = execution_config.get("result_subject", "polaris.execution.results")
            
            execution_sid = await self.nats_client.subscribe(
                subject=execution_result_subject,
                callback=self._handle_execution_result_message,
                queue=queue_group
            )
            self._subscriptions["execution_results"] = execution_sid
            
            # Subscribe to calibrate topic (for future use)
            calibrate_subject = self.nats_config.get("calibrate_subject", "polaris.digitaltwin.calibrate")
            
            calibrate_sid = await self.nats_client.subscribe(
                subject=calibrate_subject,
                callback=self._handle_calibrate_message,
                queue=queue_group
            )
            self._subscriptions["calibrate"] = calibrate_sid
            
            self.logger.info(
                "NATS ingestion engine started",
                extra={
                    "telemetry_batch_subject": telemetry_batch_subject,
                    "telemetry_stream_subject": telemetry_stream_subject,
                    "execution_result_subject": execution_result_subject,
                    "calibrate_subject": calibrate_subject,
                    "queue_group": queue_group
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to start NATS ingestion engine",
                extra={"error": str(e)}
            )
            raise
    
    async def _stop_nats_ingestion(self) -> None:
        """Stop NATS subscription handlers."""
        try:
            for sub_name, sid in self._subscriptions.items():
                await self.nats_client.unsubscribe(sid)
                self.logger.debug(f"Unsubscribed from {sub_name} (sid: {sid})")
            
            self._subscriptions.clear()
            
            self.logger.info("NATS ingestion engine stopped")
            
        except Exception as e:
            self.logger.error(
                "Error stopping NATS ingestion engine",
                extra={"error": str(e)}
            )
    
    async def _handle_telemetry_batch_message(self, msg: Msg) -> None:
        """Handle TelemetryBatch messages from Monitor adapters.
        
        Args:
            msg: NATS message containing TelemetryBatch
        """
        try:
            # Parse message
            data = json.loads(msg.data.decode())
            
            # Import here to avoid circular imports
            from polaris.models.telemetry import TelemetryBatch
            batch = TelemetryBatch(**data)
            
            # Convert each telemetry event to KnowledgeEvent and queue
            for telemetry_event in batch.events:
                knowledge_event = KnowledgeEvent(
                    source=f"{telemetry_event.source}_monitor",
                    event_type="telemetry",
                    data=telemetry_event,
                    metadata={
                        "batch_id": batch.batch_id,
                        "adapter_type": "monitor",
                        "original_source": telemetry_event.source
                    }
                )
                
                await self._message_queue.put(("update", knowledge_event))
            
            self.logger.debug(
                "Telemetry batch processed",
                extra={
                    "batch_id": batch.batch_id,
                    "event_count": len(batch.events),
                    "subject": msg.subject
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to handle telemetry batch message",
                extra={
                    "error": str(e),
                    "subject": msg.subject,
                    "reply": msg.reply
                }
            )
            # Send to dead letter queue
            await self._send_to_dead_letter_queue(msg, "telemetry_batch", str(e))

    async def _handle_telemetry_stream_message(self, msg: Msg) -> None:
        """Handle individual TelemetryEvent messages from Monitor adapters.
        
        Args:
            msg: NATS message containing TelemetryEvent
        """
        try:
            # Parse message
            data = json.loads(msg.data.decode())
            
            # Import here to avoid circular imports
            from polaris.models.telemetry import TelemetryEvent
            telemetry_event = TelemetryEvent(**data)
            
            # Convert to KnowledgeEvent and queue
            knowledge_event = KnowledgeEvent(
                source=f"{telemetry_event.source}_monitor",
                event_type="telemetry",
                data=telemetry_event,
                metadata={
                    "adapter_type": "monitor",
                    "original_source": telemetry_event.source
                }
            )
            
            await self._message_queue.put(("update", knowledge_event))
            
            self.logger.debug(
                "Telemetry stream event processed",
                extra={
                    "event_id": knowledge_event.event_id,
                    "metric_name": telemetry_event.name,
                    "subject": msg.subject
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to handle telemetry stream message",
                extra={
                    "error": str(e),
                    "subject": msg.subject,
                    "reply": msg.reply
                }
            )
            # Send to dead letter queue
            await self._send_to_dead_letter_queue(msg, "telemetry_stream", str(e))

    async def _handle_execution_result_message(self, msg: Msg) -> None:
        """Handle ExecutionResult messages from Execution adapters.
        
        Args:
            msg: NATS message containing ExecutionResult
        """
        try:
            # Parse message
            data = json.loads(msg.data.decode())
            
            # Import here to avoid circular imports
            from polaris.models.actions import ExecutionResult
            execution_result = ExecutionResult(**data)
            
            # Convert to KnowledgeEvent and queue
            knowledge_event = KnowledgeEvent(
                source=f"execution_adapter",
                event_type="execution_status",
                data=execution_result,
                metadata={
                    "adapter_type": "execution",
                    "action_id": execution_result.action_id,
                    "action_type": execution_result.action_type
                }
            )
            
            await self._message_queue.put(("update", knowledge_event))
            
            self.logger.debug(
                "Execution result processed",
                extra={
                    "event_id": knowledge_event.event_id,
                    "action_id": execution_result.action_id,
                    "action_type": execution_result.action_type,
                    "success": execution_result.success,
                    "subject": msg.subject
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to handle execution result message",
                extra={
                    "error": str(e),
                    "subject": msg.subject,
                    "reply": msg.reply
                }
            )
            # Send to dead letter queue
            await self._send_to_dead_letter_queue(msg, "execution_result", str(e))

    async def _handle_update_message(self, msg: Msg) -> None:
        """Handle KnowledgeEvent messages from NATS (legacy support).
        
        Args:
            msg: NATS message containing KnowledgeEvent
        """
        try:
            # Parse message
            data = json.loads(msg.data.decode())
            event = KnowledgeEvent(**data)
            
            # Add to processing queue
            await self._message_queue.put(("update", event))
            
            self.logger.debug(
                "Update message queued",
                extra={
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "source": event.source
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to handle update message",
                extra={
                    "error": str(e),
                    "subject": msg.subject,
                    "reply": msg.reply
                }
            )
            # Send to dead letter queue
            await self._send_to_dead_letter_queue(msg, "update", str(e))
    
    async def _handle_calibrate_message(self, msg: Msg) -> None:
        """Handle CalibrationEvent messages from NATS.
        
        Args:
            msg: NATS message containing CalibrationEvent
        """
        try:
            # Parse message
            data = json.loads(msg.data.decode())
            event = CalibrationEvent(**data)
            
            # Add to processing queue
            await self._message_queue.put(("calibrate", event))
            
            self.logger.debug(
                "Calibration message queued",
                extra={
                    "calibration_id": event.calibration_id,
                    "prediction_id": event.prediction_id,
                    "accuracy": event.calculate_accuracy_score()
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to handle calibrate message",
                extra={
                    "error": str(e),
                    "subject": msg.subject,
                    "reply": msg.reply
                }
            )
            # Send to dead letter queue
            await self._send_to_dead_letter_queue(msg, "calibrate", str(e))
    
    async def _process_messages(self) -> None:
        """Process messages from the queue in batches with high-throughput optimization."""
        batch = []
        last_batch_time = datetime.now(timezone.utc)
        
        while self.running:
            try:
                # Update queue metrics
                current_queue_size = self._message_queue.qsize()
                self._metrics["queue_high_water_mark"] = max(
                    self._metrics["queue_high_water_mark"],
                    current_queue_size
                )
                
                # Wait for messages with adaptive timeout
                try:
                    message = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=self._batch_timeout
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    # Process any pending batch on timeout
                    if batch:
                        await self._process_batch(batch)
                        batch.clear()
                        last_batch_time = datetime.now(timezone.utc)
                    continue
                
                # Check if we should process batch (size or time-based)
                current_time = datetime.now(timezone.utc)
                time_since_last_batch = (current_time - last_batch_time).total_seconds()
                
                should_process = (
                    len(batch) >= self._batch_size or
                    time_since_last_batch >= self._batch_timeout
                )
                
                if should_process:
                    await self._process_batch(batch)
                    batch.clear()
                    last_batch_time = current_time
                
            except Exception as e:
                self.logger.error(
                    "Error in message processing loop",
                    extra={"error": str(e)}
                )
                # Clear batch on error to prevent stuck messages
                batch.clear()
                last_batch_time = datetime.now(timezone.utc)
                await asyncio.sleep(1.0)
        
        # Process any remaining messages on shutdown
        if batch:
            await self._process_batch(batch)
    
    async def _process_batch(self, batch: list) -> None:
        """Process a batch of messages.
        
        Args:
            batch: List of (message_type, event) tuples
        """
        if not batch or not self.world_model:
            return
        
        start_time = datetime.now(timezone.utc)
        processed = 0
        errors = 0
        
        for message_type, event in batch:
            try:
                if message_type == "update":
                    await self.world_model.update_state(event)
                elif message_type == "calibrate":
                    await self.world_model.calibrate(event)
                
                processed += 1
                
            except Exception as e:
                errors += 1
                self.logger.error(
                    f"Failed to process {message_type} message",
                    extra={
                        "error": str(e),
                        "event_id": getattr(event, "event_id", "unknown")
                    }
                )
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Update metrics
        self._metrics["messages_processed"] += processed
        self._metrics["messages_failed"] += errors
        self._metrics["batches_processed"] += 1
        self._metrics["last_batch_time"] = start_time.isoformat()
        
        self.logger.debug(
            "Batch processed",
            extra={
                "batch_size": len(batch),
                "processed": processed,
                "errors": errors,
                "processing_time_sec": round(processing_time, 3),
                "throughput_msg_per_sec": round(processed / processing_time, 2) if processing_time > 0 else 0
            }
        )    
   
 # gRPC Server Infrastructure Methods
    
    async def _start_grpc_server(self) -> None:
        """Initialize and start the gRPC server."""
        try:
            # Create gRPC service
            self.grpc_service = DigitalTwinService(
                world_model=self.world_model,
                logger=self.logger
            )
            
            # Create server
            self.grpc_server = grpc.aio.server(
                futures.ThreadPoolExecutor(
                    max_workers=self.grpc_config.get("max_workers", 10)
                )
            )
            
            # Add service to server
            digital_twin_pb2_grpc.add_DigitalTwinServicer_to_server(
                self.grpc_service,
                self.grpc_server
            )
            
            # Configure server address
            host = self.grpc_config.get("host", "0.0.0.0")
            port = self.grpc_config.get("port", 50051)
            listen_addr = f"{host}:{port}"
            
            self.grpc_server.add_insecure_port(listen_addr)
            
            # Start server
            await self.grpc_server.start()
            
            self.logger.info(
                "gRPC server started",
                extra={
                    "address": listen_addr,
                    "max_workers": self.grpc_config.get("max_workers", 10)
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to start gRPC server",
                extra={"error": str(e)}
            )
            raise
    
    async def _stop_grpc_server(self) -> None:
        """Stop the gRPC server gracefully."""
        if self.grpc_server:
            try:
                # Graceful shutdown with timeout
                await self.grpc_server.stop(grace=5.0)
                
                self.logger.info("gRPC server stopped")
                
            except Exception as e:
                self.logger.error(
                    "Error stopping gRPC server",
                    extra={"error": str(e)}
                )
            finally:
                self.grpc_server = None
                self.grpc_service = None
    
    # Health and Status Methods
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the Digital Twin Agent.
        
        Returns:
            Dictionary containing health status information
        """
        status = {
            "agent_status": "healthy" if self.running else "stopped",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nats_connected": self.nats_client.is_connected,
            "grpc_running": self.grpc_server is not None,
            "world_model_initialized": self.world_model.is_initialized if self.world_model else False,
            "message_queue_size": self._message_queue.qsize(),
            "active_subscriptions": len(self._subscriptions),
            "active_tasks": len([t for t in self._tasks if not t.done()]),
            "performance_metrics": self._metrics.copy()
        }
        
        # Add World Model factory information
        status["world_model_factory"] = {
            "registered_types": WorldModelFactory.get_registered_types(),
            "current_type": self.world_model_config.get("implementation", "none"),
            "current_class": self.world_model.__class__.__name__ if self.world_model else None
        }
        
        # Add World Model health if available
        if self.world_model:
            try:
                wm_health = await self.world_model.get_health_status()
                status["world_model_health"] = wm_health
            except Exception as e:
                status["world_model_health"] = {"error": str(e)}
        else:
            status["world_model_health"] = {"status": "not_loaded"}
        
        return status
    
    async def reload_world_model(self, force_recreate: bool = False) -> bool:
        """Reload the World Model implementation with factory support.
        
        Args:
            force_recreate: If True, recreate the model instance entirely
            
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            if not self.world_model:
                self.logger.warning("No World Model to reload, attempting to load new instance")
                try:
                    await self._load_world_model()
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to load new World Model instance: {str(e)}")
                    return False
            
            if force_recreate:
                # Completely recreate the World Model instance
                self.logger.info("Force recreating World Model instance")
                
                # Shutdown current model
                try:
                    await self.world_model.shutdown()
                except Exception as e:
                    self.logger.warning(f"Error shutting down current model: {str(e)}")
                
                # Clear current model
                self.world_model = None
                
                # Load new instance
                try:
                    await self._load_world_model()
                    self.logger.info("World Model recreated successfully")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to recreate World Model: {str(e)}")
                    return False
            else:
                # Use the model's built-in reload capability
                success = await self.world_model.reload_model()
                if success:
                    self.logger.info("World Model reloaded successfully using built-in reload")
                else:
                    self.logger.warning("World Model built-in reload returned False")
                return success
                
        except Exception as e:
            self.logger.error(
                "Failed to reload World Model",
                extra={
                    "error": str(e),
                    "force_recreate": force_recreate,
                    "error_type": type(e).__name__
                }
            )
            return False
    
    async def validate_world_model(self) -> Dict[str, Any]:
        """Validate the current World Model implementation.
        
        Returns:
            Dictionary containing validation results
        """
        if not self.world_model:
            return {
                "valid": False,
                "error": "No World Model loaded"
            }
        
        try:
            from polaris.models.world_model import WorldModelValidator
            
            # Run comprehensive validation
            validation_results = await WorldModelValidator.run_comprehensive_test(
                self.world_model
            )
            
            self.logger.info(
                "World Model validation completed",
                extra={
                    "overall_passed": validation_results["overall_passed"],
                    "model_type": validation_results["model_type"],
                    "test_count": len(validation_results["tests"])
                }
            )
            
            return validation_results
            
        except Exception as e:
            error_result = {
                "valid": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.error(
                "World Model validation failed",
                extra={"error": str(e)}
            )
            
            return error_result
    
    async def change_world_model_type(self, new_model_type: str, new_config: Optional[Dict[str, Any]] = None) -> bool:
        """Dynamically change the World Model implementation type.
        
        Args:
            new_model_type: New model type to load
            new_config: Optional new configuration for the model
            
        Returns:
            True if change was successful, False otherwise
        """
        try:
            # Validate new model type is registered
            if not WorldModelFactory.is_registered(new_model_type):
                available_types = WorldModelFactory.get_registered_types()
                self.logger.error(
                    f"Cannot change to unknown model type '{new_model_type}'. "
                    f"Available types: {available_types}"
                )
                return False

            
            # Update configuration
            old_model_type = self.world_model_config.get("implementation", "unknown")
            old_config = self.world_model_config.get("config", {})
            # Apply candidate changes locally
            self.world_model_config["implementation"] = new_model_type
            if new_config is not None:
                self.world_model_config["config"] = new_config
            
            self.logger.info(
                f"Changing World Model from '{old_model_type}' to '{new_model_type}'"
            )
            
            # Validate the updated world model config before reload
            from polaris.models.world_model import ConfigurationValidator
            validation = ConfigurationValidator.validate_world_model_config(self.world_model_config)
            if not validation["valid"]:
                # Revert and abort
                self.world_model_config["implementation"] = old_model_type
                self.world_model_config["config"] = old_config
                self.logger.error(
                    "New World Model configuration is invalid",
                    extra={"errors": validation.get("errors", [])}
                )
                return False

            # Reload with new configuration (force recreate)
            success = await self.reload_world_model(force_recreate=True)
            
            if success:
                self.logger.info(
                    f"Successfully changed World Model to '{new_model_type}'"
                )
            else:
                # Revert configuration on failure
                self.world_model_config["implementation"] = old_model_type
                self.world_model_config["config"] = old_config
                self.logger.error(
                    f"Failed to change World Model to '{new_model_type}', "
                    f"reverted to '{old_model_type}'"
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Error changing World Model type",
                extra={
                    "new_model_type": new_model_type,
                    "error": str(e)
                }
            )
            return False