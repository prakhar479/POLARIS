"""
Resilience Patterns

Implements common resilience patterns including Circuit Breaker, Retry Policy,
and Bulkhead for protecting against failures and resource exhaustion.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
import logging

from .exceptions import PolarisException

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying half-open
    success_threshold: int = 3  # Successes needed in half-open to close
    timeout: float = 30.0  # Operation timeout in seconds


class CircuitBreakerError(PolarisException):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            message=f"Circuit breaker is open for service '{service_name}'",
            error_code="CIRCUIT_BREAKER_OPEN",
            context={"service_name": service_name},
            **kwargs
        )


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for protecting external services.
    
    Prevents cascading failures by monitoring failure rates and temporarily
    blocking calls to failing services.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
                else:
                    raise CircuitBreakerError(self.name)
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(func(), timeout=self.config.timeout)
            await self._on_success()
            return result
            
        except asyncio.TimeoutError as e:
            await self._on_failure()
            raise PolarisException(
                f"Operation timed out after {self.config.timeout}s",
                error_code="OPERATION_TIMEOUT",
                context={"service_name": self.name, "timeout": self.config.timeout},
                cause=e
            )
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful operation."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to CLOSED")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
    
    async def _on_failure(self):
        """Handle failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' moved to OPEN after {self.failure_count} failures")
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' moved back to OPEN")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state for monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retryable_exceptions: tuple = (Exception,)  # Exceptions that trigger retry


class RetryPolicy:
    """
    Retry Policy with exponential backoff and jitter.
    
    Automatically retries failed operations with increasing delays
    to handle transient failures gracefully.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    async def execute(self, func: Callable[[], Awaitable[T]], operation_name: str = "operation") -> T:
        """Execute function with retry policy."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await func()
                if attempt > 1:
                    logger.info(f"Operation '{operation_name}' succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not isinstance(e, self.config.retryable_exceptions):
                    logger.debug(f"Non-retryable exception for '{operation_name}': {e}")
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts:
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Operation '{operation_name}' failed on attempt {attempt}/{self.config.max_attempts}, "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise PolarisException(
            f"Operation '{operation_name}' failed after {self.config.max_attempts} attempts",
            error_code="RETRY_EXHAUSTED",
            context={
                "operation_name": operation_name,
                "max_attempts": self.config.max_attempts,
                "last_exception": str(last_exception)
            },
            cause=last_exception
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter."""
        # Exponential backoff: base_delay * (exponential_base ^ (attempt - 1))
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        
        # Cap at max_delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            # Add random jitter up to 25% of the delay
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead pattern."""
    max_concurrent: int = 10  # Maximum concurrent operations
    queue_size: int = 100  # Maximum queue size for waiting operations
    timeout: float = 30.0  # Timeout for acquiring semaphore


class BulkheadError(PolarisException):
    """Raised when bulkhead limits are exceeded."""
    
    def __init__(self, resource_name: str, reason: str, **kwargs):
        super().__init__(
            message=f"Bulkhead limit exceeded for resource '{resource_name}': {reason}",
            error_code="BULKHEAD_LIMIT_EXCEEDED",
            context={"resource_name": resource_name, "reason": reason},
            **kwargs
        )


class Bulkhead:
    """
    Bulkhead pattern implementation for resource isolation.
    
    Prevents resource exhaustion by limiting concurrent operations
    and isolating different types of operations.
    """
    
    def __init__(self, name: str, config: Optional[BulkheadConfig] = None):
        self.name = name
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._queue_size = 0
        self._active_operations = 0
        self._total_operations = 0
        self._rejected_operations = 0
        self._lock = asyncio.Lock()
    
    async def execute(self, func: Callable[[], Awaitable[T]], operation_name: str = "operation") -> T:
        """Execute function with bulkhead protection."""
        # Check queue size before attempting to acquire semaphore
        async with self._lock:
            if self._queue_size >= self.config.queue_size:
                self._rejected_operations += 1
                raise BulkheadError(
                    self.name, 
                    f"Queue full ({self._queue_size}/{self.config.queue_size})"
                )
            self._queue_size += 1
            self._total_operations += 1
        
        semaphore_acquired = False
        try:
            # Try to acquire semaphore with timeout
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(), 
                    timeout=self.config.timeout
                )
                semaphore_acquired = True
            except asyncio.TimeoutError:
                async with self._lock:
                    self._queue_size -= 1
                raise BulkheadError(
                    self.name,
                    f"Timeout waiting for resource (timeout: {self.config.timeout}s)"
                )
            
            # Update counters
            async with self._lock:
                self._queue_size -= 1
                self._active_operations += 1
            
            try:
                # Execute the operation
                result = await func()
                logger.debug(f"Bulkhead '{self.name}' completed operation '{operation_name}'")
                return result
                
            finally:
                # Always release semaphore and update counters
                async with self._lock:
                    self._active_operations -= 1
                self._semaphore.release()
                
        except Exception as e:
            # Remove from queue if we never acquired the semaphore
            if not semaphore_acquired:
                async with self._lock:
                    if self._queue_size > 0:
                        self._queue_size -= 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics for monitoring."""
        return {
            "name": self.name,
            "max_concurrent": self.config.max_concurrent,
            "active_operations": self._active_operations,
            "queue_size": self._queue_size,
            "max_queue_size": self.config.queue_size,
            "total_operations": self._total_operations,
            "rejected_operations": self._rejected_operations,
            "utilization": self._active_operations / self.config.max_concurrent
        }


class ResilienceManager:
    """
    Centralized manager for resilience patterns.
    
    Provides a unified interface for managing circuit breakers,
    retry policies, and bulkheads across the system.
    """
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._bulkheads: Dict[str, Bulkhead] = {}
        self._default_retry_policy = RetryPolicy()
    
    def get_circuit_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]
    
    def get_bulkhead(
        self, 
        name: str, 
        config: Optional[BulkheadConfig] = None
    ) -> Bulkhead:
        """Get or create a bulkhead."""
        if name not in self._bulkheads:
            self._bulkheads[name] = Bulkhead(name, config)
        return self._bulkheads[name]
    
    def get_retry_policy(self, config: Optional[RetryConfig] = None) -> RetryPolicy:
        """Get a retry policy (creates new instance or returns default)."""
        if config:
            return RetryPolicy(config)
        return self._default_retry_policy
    
    async def execute_with_resilience(
        self,
        func: Callable[[], Awaitable[T]],
        operation_name: str,
        circuit_breaker_name: Optional[str] = None,
        bulkhead_name: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        bulkhead_config: Optional[BulkheadConfig] = None
    ) -> T:
        """
        Execute function with multiple resilience patterns applied.
        
        Applies circuit breaker, bulkhead, and retry policy as configured.
        """
        # Start with the original function
        current_func = func
        
        # Apply bulkhead if specified (innermost wrapper)
        if bulkhead_name:
            bulkhead = self.get_bulkhead(bulkhead_name, bulkhead_config)
            # Use default parameter to capture the current value
            current_func = (lambda f=current_func: bulkhead.execute(f, operation_name))
        
        # Apply circuit breaker if specified (middle wrapper)
        if circuit_breaker_name:
            cb = self.get_circuit_breaker(circuit_breaker_name, circuit_breaker_config)
            # Use default parameter to capture the current value
            current_func = (lambda f=current_func: cb.call(f))
        
        # Apply retry policy (outermost wrapper)
        retry_policy = self.get_retry_policy(retry_config)
        return await retry_policy.execute(current_func, operation_name)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all resilience components."""
        return {
            "circuit_breakers": {
                name: cb.get_state() 
                for name, cb in self._circuit_breakers.items()
            },
            "bulkheads": {
                name: bulkhead.get_stats() 
                for name, bulkhead in self._bulkheads.items()
            }
        }


# Global resilience manager instance
resilience_manager = ResilienceManager()