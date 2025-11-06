"""
Improved GRPC Digital Twin Client with robust timeout handling and connection management.

This module provides an enhanced GRPC client that addresses timeout issues,
implements connection pooling, circuit breaker patterns, and proper error handling.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import grpc
from grpc import aio as grpc_aio

from .reasoner_agent import (
    DigitalTwinInterface, DTQuery, DTSimulation, DTDiagnosis, DTResponse
)
from polaris.proto import digital_twin_pb2, digital_twin_pb2_grpc


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ConnectionMetrics:
    """Metrics for connection monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    average_response_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


class CircuitBreaker:
    """Circuit breaker for GRPC calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class ImprovedGRPCDigitalTwinClient(DigitalTwinInterface):
    """
    Improved gRPC implementation for querying the Digital Twin with:
    - Configurable timeouts
    - Connection pooling
    - Circuit breaker pattern
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Connection health monitoring
    """
    
    def __init__(
        self,
        grpc_address: str,
        logger: Optional[logging.Logger] = None,
        # Timeout configurations
        default_timeout: float = 30.0,
        query_timeout: float = 15.0,
        simulation_timeout: float = 60.0,
        diagnosis_timeout: float = 30.0,
        connection_timeout: float = 10.0,
        # Retry configurations
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        # Circuit breaker configurations
        circuit_breaker_enabled: bool = True,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        # Connection configurations
        max_receive_message_length: int = 100 * 1024 * 1024,  # 100MB
        max_send_message_length: int = 100 * 1024 * 1024,     # 100MB
        keepalive_time_ms: int = 30000,
        keepalive_timeout_ms: int = 5000,
        keepalive_permit_without_calls: bool = True,
        max_connection_idle_ms: int = 300000,  # 5 minutes
    ):
        self.grpc_address = grpc_address
        self.logger = logger or logging.getLogger(f"ImprovedDigitalTwinClient")
        
        # Timeout configurations
        self.default_timeout = default_timeout
        self.query_timeout = query_timeout
        self.simulation_timeout = simulation_timeout
        self.diagnosis_timeout = diagnosis_timeout
        self.connection_timeout = connection_timeout
        
        # Retry configurations
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        
        # Connection state
        self.channel: Optional[grpc_aio.Channel] = None
        self.stub: Optional[digital_twin_pb2_grpc.DigitalTwinStub] = None
        self._connection_lock = asyncio.Lock()
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        
        # Circuit breaker
        self.circuit_breaker = None
        if circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=(grpc_aio.AioRpcError, asyncio.TimeoutError, Exception)
            )
        
        # Connection metrics
        self.metrics = ConnectionMetrics()
        
        # Channel options for better reliability (fixed keepalive settings)
        self.channel_options = [
            ('grpc.max_receive_message_length', max_receive_message_length),
            ('grpc.max_send_message_length', max_send_message_length),
            ('grpc.keepalive_time_ms', 120000),  # Increased to 2 minutes
            ('grpc.keepalive_timeout_ms', 20000),  # Increased timeout
            ('grpc.keepalive_permit_without_calls', False),  # Disable keepalive without calls
            ('grpc.http2.max_pings_without_data', 2),  # Limit pings without data
            ('grpc.http2.min_time_between_pings_ms', 60000),  # Min 1 minute between pings
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),  # 5 minutes without data
            ('grpc.max_connection_idle_ms', 600000),  # 10 minutes idle timeout
        ]
        
        self.logger.info(
            f"Initialized ImprovedGRPCDigitalTwinClient",
            extra={
                "grpc_address": grpc_address,
                "query_timeout": query_timeout,
                "simulation_timeout": simulation_timeout,
                "diagnosis_timeout": diagnosis_timeout,
                "max_retries": max_retries,
                "circuit_breaker_enabled": circuit_breaker_enabled
            }
        )
    
    async def connect(self) -> None:
        """Establish the gRPC connection with improved error handling."""
        async with self._connection_lock:
            if self.channel and self.stub:
                # Check if existing connection is healthy
                if await self._is_connection_healthy():
                    self.logger.debug("Existing connection is healthy, reusing")
                    return
                else:
                    self.logger.info("Existing connection is unhealthy, reconnecting")
                    await self._close_connection()
            
            self.logger.info(f"Connecting to Digital Twin gRPC server at {self.grpc_address}")
            
            try:
                # Create channel with options
                self.channel = grpc_aio.insecure_channel(
                    self.grpc_address,
                    options=self.channel_options
                )
                
                # Create stub
                self.stub = digital_twin_pb2_grpc.DigitalTwinStub(self.channel)
                
                # Test the connection
                await asyncio.wait_for(
                    self._test_connection(),
                    timeout=self.connection_timeout
                )
                
                self.logger.info("Digital Twin gRPC connection established successfully")
                self.metrics.last_success_time = time.time()
                
            except Exception as e:
                self.logger.error(f"Failed to connect to Digital Twin: {e}")
                await self._close_connection()
                self.metrics.last_failure_time = time.time()
                raise
    
    async def disconnect(self) -> None:
        """Close the gRPC connection."""
        async with self._connection_lock:
            await self._close_connection()
    
    async def _close_connection(self) -> None:
        """Internal method to close connection."""
        if self.channel:
            self.logger.info("Closing Digital Twin gRPC connection")
            try:
                await self.channel.close()
            except Exception as e:
                self.logger.warning(f"Error closing gRPC channel: {e}")
            finally:
                self.channel = None
                self.stub = None
    
    async def _is_connection_healthy(self) -> bool:
        """Check if the current connection is healthy."""
        if not self.channel or not self.stub:
            return False
        
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return True  # Skip frequent health checks
        
        try:
            await asyncio.wait_for(
                self._test_connection(),
                timeout=5.0
            )
            self._last_health_check = current_time
            return True
        except Exception:
            return False
    
    async def _test_connection(self) -> None:
        """Test the gRPC connection with a simple health check."""
        if not self.stub:
            raise Exception("gRPC stub not initialized")
        
        try:
            request = digital_twin_pb2.ManagementRequest(
                request_id=str(uuid.uuid4()),
                operation="health_check",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            await self.stub.Manage(request, timeout=5.0)
            
        except grpc_aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                # Health check not implemented, but connection works
                self.logger.debug("Health check not implemented, but connection is working")
                return
            raise
    
    async def _execute_with_retry(
        self,
        operation_name: str,
        operation_func,
        timeout: float,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with retry logic and circuit breaker."""
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Ensure connection is established
                if not self.stub:
                    await self.connect()
                
                # Execute with circuit breaker if enabled
                if self.circuit_breaker:
                    result = await self.circuit_breaker.call(
                        self._execute_operation,
                        operation_func,
                        timeout,
                        *args,
                        **kwargs
                    )
                else:
                    result = await self._execute_operation(
                        operation_func,
                        timeout,
                        *args,
                        **kwargs
                    )
                
                # Update metrics on success
                self.metrics.total_requests += 1
                self.metrics.successful_requests += 1
                self.metrics.last_success_time = time.time()
                
                # Update average response time
                response_time = time.time() - start_time
                if self.metrics.average_response_time == 0:
                    self.metrics.average_response_time = response_time
                else:
                    self.metrics.average_response_time = (
                        0.9 * self.metrics.average_response_time + 0.1 * response_time
                    )
                
                self.logger.debug(
                    f"{operation_name} completed successfully",
                    extra={
                        "attempt": attempt + 1,
                        "response_time_sec": response_time,
                        "success_rate": self.metrics.success_rate
                    }
                )
                
                return result
                
            except (grpc_aio.AioRpcError, asyncio.TimeoutError, Exception) as e:
                last_exception = e
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self.metrics.last_failure_time = time.time()
                
                if isinstance(e, asyncio.TimeoutError):
                    self.metrics.timeout_requests += 1
                
                self.logger.warning(
                    f"{operation_name} failed on attempt {attempt + 1}",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": attempt + 1,
                        "max_attempts": self.max_retries + 1
                    }
                )
                
                # Don't retry on the last attempt
                if attempt == self.max_retries:
                    break
                
                # Handle specific error types
                if isinstance(e, grpc_aio.AioRpcError):
                    if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                        # Connection issues, try to reconnect
                        await self._close_connection()
                    elif e.code() in [grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.NOT_FOUND]:
                        # Don't retry on client errors
                        break
                
                # Exponential backoff
                delay = self.retry_delay * (self.retry_backoff_factor ** attempt)
                await asyncio.sleep(delay)
        
        # All retries failed
        self.logger.error(
            f"{operation_name} failed after {self.max_retries + 1} attempts",
            extra={
                "last_error": str(last_exception),
                "success_rate": self.metrics.success_rate,
                "total_failures": self.metrics.failed_requests
            }
        )
        
        return None
    
    async def _execute_operation(
        self,
        operation_func,
        timeout: float,
        *args,
        **kwargs
    ) -> Any:
        """Execute a single operation with timeout."""
        return await asyncio.wait_for(
            operation_func(*args, **kwargs),
            timeout=timeout
        )
    
    async def query(self, query: DTQuery) -> Optional[DTResponse]:
        """Send a query to the Digital Twin with improved error handling."""
        async def _query_operation():
            # Convert parameters to strings but preserve type information for debugging
            converted_params = {}
            for k, v in (query.parameters or {}).items():
                converted_params[k] = str(v)
                self.logger.debug(f"Query parameter conversion: {k}={v} ({type(v).__name__}) -> {str(v)} (str)")
            
            request_pb = digital_twin_pb2.QueryRequest(
                query_id=query.query_id,
                query_type=query.query_type,
                query_content=query.query_content,
                parameters=converted_params,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            
            response_pb = await self.stub.Query(request_pb)
            
            # Parse result
            result = None
            if response_pb.result:
                try:
                    result = json.loads(response_pb.result)
                except json.JSONDecodeError:
                    result = response_pb.result
            
            return DTResponse(
                success=response_pb.success,
                result=result,
                confidence=response_pb.confidence,
                explanation=response_pb.explanation,
                metadata=dict(response_pb.metadata),
            )
        
        return await self._execute_with_retry(
            "Query",
            _query_operation,
            self.query_timeout
        )
    
    async def simulate(self, simulation: DTSimulation) -> Optional[DTResponse]:
        """Run a simulation in the Digital Twin with improved error handling."""
        async def _simulate_operation():
            pb_actions = []
            for i, action in enumerate(simulation.actions):
                self.logger.debug(f"Processing action {i}: {action}")
                
                # Convert action params to strings but log the conversion
                converted_params = {}
                for k, v in action.get("params", {}).items():
                    converted_params[k] = str(v)
                    self.logger.debug(f"Action {i} param conversion: {k}={v} ({type(v).__name__}) -> {str(v)} (str)")
                
                pb_action = digital_twin_pb2.ControlAction(
                    action_id=action.get("action_id", str(uuid.uuid4())),
                    action_type=action.get("action_type", ""),
                    target=action.get("target", ""),
                    params=converted_params,
                )
                pb_actions.append(pb_action)
            
            # Convert parameters to strings but preserve type information for debugging
            converted_params = {}
            for k, v in (simulation.parameters or {}).items():
                converted_params[k] = str(v)
                self.logger.debug(f"Simulation parameter conversion: {k}={v} ({type(v).__name__}) -> {str(v)} (str)")
            
            request_pb = digital_twin_pb2.SimulationRequest(
                simulation_id=simulation.simulation_id,
                simulation_type=simulation.simulation_type,
                actions=pb_actions,
                horizon_minutes=simulation.horizon_minutes,
                parameters=converted_params,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            
            response_pb = await self.stub.Simulate(request_pb)
            
            # Parse future states
            future_states = []
            for state in response_pb.future_states:
                future_state = {
                    "timestamp": state.timestamp,
                    "metrics": dict(state.metrics),
                    "confidence": state.confidence,
                    "description": state.description
                }
                future_states.append(future_state)
            
            return DTResponse(
                success=response_pb.success,
                confidence=response_pb.confidence,
                explanation=response_pb.explanation,
                metadata=dict(response_pb.metadata),
                future_states=future_states,
            )
        
        return await self._execute_with_retry(
            "Simulate",
            _simulate_operation,
            self.simulation_timeout
        )
    
    async def diagnose(self, diagnosis: DTDiagnosis) -> Optional[DTResponse]:
        """Request a diagnosis from the Digital Twin with improved error handling."""
        async def _diagnose_operation():
            # Convert context to strings but preserve type information for debugging
            converted_context = {}
            for k, v in (diagnosis.context or {}).items():
                converted_context[k] = str(v)
                self.logger.debug(f"Diagnosis context conversion: {k}={v} ({type(v).__name__}) -> {str(v)} (str)")
            
            request_pb = digital_twin_pb2.DiagnosisRequest(
                diagnosis_id=diagnosis.diagnosis_id,
                anomaly_description=diagnosis.anomaly_description,
                context=converted_context,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            
            response_pb = await self.stub.Diagnose(request_pb)
            
            hypotheses = [
                {
                    "hypothesis": h.hypothesis,
                    "probability": h.probability,
                    "reasoning": h.reasoning,
                    "evidence": list(h.evidence),
                }
                for h in response_pb.hypotheses
            ]
            
            return DTResponse(
                success=response_pb.success,
                confidence=response_pb.confidence,
                explanation=response_pb.explanation,
                metadata=dict(response_pb.metadata),
                hypotheses=hypotheses,
            )
        
        return await self._execute_with_retry(
            "Diagnose",
            _diagnose_operation,
            self.diagnosis_timeout
        )
    
    def get_connection_metrics(self) -> Dict[str, Any]:
        """Get connection metrics for monitoring."""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "timeout_requests": self.metrics.timeout_requests,
            "success_rate": self.metrics.success_rate,
            "failure_rate": self.metrics.failure_rate,
            "average_response_time_sec": self.metrics.average_response_time,
            "last_success_time": self.metrics.last_success_time,
            "last_failure_time": self.metrics.last_failure_time,
            "circuit_breaker_state": self.circuit_breaker.state.value if self.circuit_breaker else "disabled",
            "connection_healthy": self.channel is not None and self.stub is not None,
            "grpc_address": self.grpc_address
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        is_healthy = await self._is_connection_healthy()
        
        return {
            "healthy": is_healthy,
            "connected": self.channel is not None and self.stub is not None,
            "metrics": self.get_connection_metrics(),
            "configuration": {
                "grpc_address": self.grpc_address,
                "query_timeout": self.query_timeout,
                "simulation_timeout": self.simulation_timeout,
                "diagnosis_timeout": self.diagnosis_timeout,
                "max_retries": self.max_retries,
                "circuit_breaker_enabled": self.circuit_breaker is not None
            }
        }