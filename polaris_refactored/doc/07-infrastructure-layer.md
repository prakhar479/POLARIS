# Infrastructure Layer

## Overview

The Infrastructure Layer provides the foundational technical services that support all other layers in the POLARIS framework. This layer implements cross-cutting concerns such as messaging, data persistence, dependency injection, resilience patterns, and observability.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  Message Bus    │ │  Data Storage   │ │   Resilience    │   │
│  │ - NATS Adapter  │ │ - Multi-backend │ │ - Circuit Break │   │
│  │ - Middleware    │ │ - Repositories  │ │ - Retry Policy  │   │
│  │ - Compression   │ │ - Unit of Work  │ │ - Bulkhead      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │Dependency Inject│ │  Observability  │ │   Exceptions    │   │
│  │ - Container     │ │ - Logging       │ │ - Hierarchy     │   │
│  │ - Auto-wiring   │ │ - Metrics       │ │ - Context       │   │
│  │ - Lifecycle     │ │ - Tracing       │ │ - Structured    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Message Bus (PolarisMessageBus)

### Purpose
Provides reliable, scalable message passing with middleware support for event-driven architecture.

### Key Features
- **NATS Integration**: High-performance message broker with clustering support
- **Middleware Pipeline**: Extensible message processing chain
- **Type Safety**: Strongly typed event publishing and subscription
- **Reliability**: Automatic reconnection and error handling

### Implementation

#### Core Message Bus
```python
class PolarisMessageBus(Injectable):
    """Main message bus implementation with NATS backend."""
    
    def __init__(self, broker: MessageBroker, middleware_chain: MiddlewareChain):
        self._broker = broker
        self._middleware_chain = middleware_chain
        self._subscriptions: Dict[str, List[EventHandler]] = {}
    
    async def publish(self, topic: str, event: Any) -> None:
        """Publish an event to a topic."""
        context = MessageContext(topic=topic, event=event)
        await self._middleware_chain.process(context, self._publish_internal)
    
    async def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribe to events on a topic."""
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
            await self._broker.subscribe(topic, self._handle_message)
        
        self._subscriptions[topic].append(handler)
    
    async def publish_telemetry(self, telemetry_event: TelemetryEvent) -> None:
        """Publish telemetry event with system-specific routing."""
        topic = f"polaris.telemetry.{telemetry_event.system_state.system_id}"
        await self.publish(topic, telemetry_event)
```

#### NATS Message Broker
```python
class NATSMessageBroker(MessageBroker):
    """NATS-specific message broker implementation."""
    
    def __init__(self, config: NATSConfig):
        self._config = config
        self._nc: Optional[NATS] = None
        self._connection_lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Connect to NATS servers with retry logic."""
        async with self._connection_lock:
            if self._nc and self._nc.is_connected:
                return
            
            self._nc = NATS()
            await self._nc.connect(
                servers=self._config.servers,
                name=self._config.name,
                max_reconnect_attempts=self._config.max_reconnect_attempts,
                reconnect_time_wait=self._config.reconnect_time_wait,
                error_cb=self._error_callback,
                reconnected_cb=self._reconnected_callback
            )
    
    async def publish(self, topic: str, data: bytes) -> None:
        """Publish message to NATS."""
        if not self._nc or not self._nc.is_connected:
            await self.connect()
        
        await self._nc.publish(topic, data)
    
    async def subscribe(self, topic: str, handler: Callable[[bytes], Awaitable[None]]) -> None:
        """Subscribe to NATS topic."""
        if not self._nc or not self._nc.is_connected:
            await self.connect()
        
        await self._nc.subscribe(topic, cb=handler)
```

### Middleware System

#### Middleware Chain
```python
class MiddlewareChain:
    """Processes messages through a chain of middleware components."""
    
    def __init__(self, middlewares: List[MessageMiddleware]):
        self._middlewares = middlewares
    
    async def process(self, context: MessageContext, 
                     final_handler: Callable[[MessageContext], Awaitable[Any]]) -> Any:
        """Process message through middleware chain."""
        if not self._middlewares:
            return await final_handler(context)
        
        async def create_next(index: int):
            if index >= len(self._middlewares):
                return await final_handler(context)
            
            middleware = self._middlewares[index]
            return await middleware.process(context, lambda ctx: create_next(index + 1))
        
        return await create_next(0)
```

#### Logging Middleware
```python
class LoggingMiddleware(MessageMiddleware):
    """Logs message processing with correlation IDs."""
    
    def __init__(self, logger: Logger):
        self._logger = logger
    
    async def process(self, context: MessageContext, 
                     next_handler: Callable[[MessageContext], Awaitable[Any]]) -> Any:
        correlation_id = str(uuid.uuid4())
        context.correlation_id = correlation_id
        
        self._logger.info(
            "Processing message",
            extra={
                "correlation_id": correlation_id,
                "topic": context.topic,
                "event_type": type(context.event).__name__
            }
        )
        
        start_time = time.time()
        try:
            result = await next_handler(context)
            duration = time.time() - start_time
            
            self._logger.info(
                "Message processed successfully",
                extra={
                    "correlation_id": correlation_id,
                    "duration_ms": duration * 1000
                }
            )
            return result
        except Exception as e:
            self._logger.error(
                "Message processing failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise
```

#### Metrics Middleware
```python
class MetricsMiddleware(MessageMiddleware):
    """Collects metrics for message processing."""
    
    def __init__(self):
        self.message_count = Counter(
            'polaris_messages_total',
            'Total number of messages processed',
            ['topic', 'status', 'event_type']
        )
        self.processing_time = Histogram(
            'polaris_message_processing_seconds',
            'Time spent processing messages',
            ['topic', 'event_type']
        )
    
    async def process(self, context: MessageContext, 
                     next_handler: Callable[[MessageContext], Awaitable[Any]]) -> Any:
        event_type = type(context.event).__name__
        
        with self.processing_time.labels(
            topic=context.topic, 
            event_type=event_type
        ).time():
            try:
                result = await next_handler(context)
                self.message_count.labels(
                    topic=context.topic,
                    status="success",
                    event_type=event_type
                ).inc()
                return result
            except Exception:
                self.message_count.labels(
                    topic=context.topic,
                    status="error",
                    event_type=event_type
                ).inc()
                raise
```

## Data Storage (PolarisDataStore)

### Purpose
Provides multi-backend data persistence with repository pattern abstraction.

### Storage Backends

#### Time-Series Backend
```python
class TimeSeriesStorageBackend(StorageBackend):
    """Storage backend optimized for time-series data."""
    
    async def store_time_series(self, collection: str, timestamp: datetime,
                               tags: Dict[str, str], fields: Dict[str, Any]) -> None:
        """Store time-series data point."""
        pass
    
    async def query_time_series(self, collection: str, 
                               time_range: TimeRange,
                               tags: Optional[Dict[str, str]] = None) -> List[TimeSeriesPoint]:
        """Query time-series data."""
        pass
    
    async def aggregate_time_series(self, collection: str,
                                   time_range: TimeRange,
                                   aggregation: AggregationType,
                                   group_by: Optional[List[str]] = None) -> List[AggregatedPoint]:
        """Aggregate time-series data."""
        pass
```

#### Document Backend
```python
class DocumentStorageBackend(StorageBackend):
    """Storage backend for document-oriented data."""
    
    async def insert_document(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert a document and return its ID."""
        pass
    
    async def find_documents(self, collection: str, 
                           query: Dict[str, Any],
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find documents matching query."""
        pass
    
    async def update_document(self, collection: str, document_id: str,
                            updates: Dict[str, Any]) -> bool:
        """Update a document."""
        pass
    
    async def create_index(self, collection: str, fields: List[str]) -> None:
        """Create an index on specified fields."""
        pass
```

#### Graph Backend
```python
class GraphStorageBackend(StorageBackend):
    """Storage backend for graph data structures."""
    
    async def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Add a node to the graph."""
        pass
    
    async def add_edge(self, source: str, target: str, 
                      relationship_type: str, properties: Dict[str, Any] = None) -> None:
        """Add an edge between nodes."""
        pass
    
    async def get_neighbors(self, node_id: str, 
                           direction: str = "both",
                           relationship_type: Optional[str] = None) -> List[str]:
        """Get neighboring nodes."""
        pass
    
    async def find_path(self, source: str, target: str,
                       max_depth: int = 5) -> Optional[List[str]]:
        """Find path between nodes."""
        pass
```

### Repository Pattern

#### Base Repository
```python
class Repository(ABC, Generic[T]):
    """Base repository with common operations."""
    
    def __init__(self, storage_backend: StorageBackend):
        self._storage = storage_backend
    
    @abstractmethod
    async def save(self, entity: T) -> None:
        """Save an entity."""
        pass
    
    @abstractmethod
    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """Find entity by ID."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[T]:
        """Find all entities."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity."""
        pass
```

#### System State Repository
```python
class SystemStateRepository(Repository[SystemState]):
    """Repository for system state data."""
    
    async def save_current_state(self, state: SystemState) -> None:
        """Save current system state."""
        await self._storage.store_time_series(
            collection="system_states",
            timestamp=state.timestamp,
            tags={"system_id": state.system_id},
            fields={
                "health_status": state.health_status.value,
                "metrics": {k: v.to_dict() for k, v in state.metrics.items()}
            }
        )
    
    async def get_current_state(self, system_id: str) -> Optional[SystemState]:
        """Get the most recent state for a system."""
        points = await self._storage.query_time_series(
            collection="system_states",
            time_range=TimeRange.last_hour(),
            tags={"system_id": system_id}
        )
        
        if not points:
            return None
        
        latest_point = max(points, key=lambda p: p.timestamp)
        return self._point_to_system_state(latest_point)
    
    async def get_state_history(self, system_id: str, 
                               time_range: TimeRange) -> List[SystemState]:
        """Get historical states for a system."""
        points = await self._storage.query_time_series(
            collection="system_states",
            time_range=time_range,
            tags={"system_id": system_id}
        )
        
        return [self._point_to_system_state(p) for p in points]
```

## Dependency Injection (DIContainer)

### Purpose
Manages service dependencies and lifecycles with automatic resolution.

### Implementation

#### DI Container
```python
class DIContainer:
    """Dependency injection container with automatic resolution."""
    
    def __init__(self):
        self._singletons: Dict[Type, Any] = {}
        self._transient_factories: Dict[Type, Callable[[], Any]] = {}
        self._registrations: Dict[Type, ServiceRegistration] = {}
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        self._registrations[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON
        )
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service."""
        self._registrations[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT
        )
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function."""
        self._transient_factories[interface] = factory
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance."""
        if interface in self._transient_factories:
            return self._transient_factories[interface]()
        
        if interface not in self._registrations:
            raise ServiceNotRegisteredException(f"Service {interface} not registered")
        
        registration = self._registrations[interface]
        
        if registration.lifetime == ServiceLifetime.SINGLETON:
            if interface not in self._singletons:
                self._singletons[interface] = self._create_instance(registration.implementation)
            return self._singletons[interface]
        else:
            return self._create_instance(registration.implementation)
    
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create instance with dependency injection."""
        constructor = implementation.__init__
        signature = inspect.signature(constructor)
        
        args = {}
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                raise DependencyResolutionException(
                    f"Parameter {param_name} in {implementation} has no type annotation"
                )
            
            args[param_name] = self.resolve(param_type)
        
        return implementation(**args)
```

#### Injectable Base Class
```python
class Injectable:
    """Base class for services that can be injected."""
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Validate that all constructor parameters have type annotations
        constructor = cls.__init__
        signature = inspect.signature(constructor)
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation == inspect.Parameter.empty:
                raise TypeError(
                    f"Injectable class {cls.__name__} parameter {param_name} "
                    f"must have a type annotation"
                )
```

## Resilience Components

### Purpose
Implements fault tolerance patterns to ensure system reliability.

### Circuit Breaker

```python
class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self._name = name
        self._config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenException(f"Circuit breaker {self._name} is open")
        
        try:
            result = await func()
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self) -> None:
        """Handle successful operation."""
        async with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED
    
    async def _on_failure(self) -> None:
        """Handle failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()
            
            if self._failure_count >= self._config.failure_threshold:
                self._state = CircuitBreakerState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self._last_failure_time:
            return False
        
        time_since_failure = datetime.utcnow() - self._last_failure_time
        return time_since_failure >= self._config.timeout
```

### Retry Policy

```python
class RetryPolicy:
    """Retry policy with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self._config = config
    
    async def execute(self, func: Callable[[], Awaitable[T]], 
                     operation_name: str) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self._config.max_attempts):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt):
                    raise
                
                if attempt < self._config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Operation {operation_name} failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s",
                        extra={"error": str(e), "attempt": attempt + 1}
                    )
                    await asyncio.sleep(delay)
        
        raise last_exception
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self._config.max_attempts - 1:
            return False
        
        # Don't retry certain types of exceptions
        if isinstance(exception, (ValueError, TypeError)):
            return False
        
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        base_delay = self._config.base_delay * (2 ** attempt)
        max_delay = min(base_delay, self._config.max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, 0.1) * max_delay
        return max_delay + jitter
```

### Bulkhead

```python
class Bulkhead:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, name: str, config: BulkheadConfig):
        self._name = name
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self._active_calls = 0
        self._lock = asyncio.Lock()
    
    async def execute(self, func: Callable[[], Awaitable[T]], 
                     operation_name: str) -> T:
        """Execute function with bulkhead protection."""
        if not await self._semaphore.acquire():
            raise BulkheadRejectedException(
                f"Bulkhead {self._name} rejected call to {operation_name}"
            )
        
        try:
            async with self._lock:
                self._active_calls += 1
            
            return await func()
        finally:
            async with self._lock:
                self._active_calls -= 1
            self._semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self._name,
            "max_concurrent_calls": self._config.max_concurrent_calls,
            "active_calls": self._active_calls,
            "available_permits": self._semaphore._value
        }
```

## Observability

### Purpose
Provides comprehensive monitoring, logging, and tracing capabilities.

### Structured Logging

```python
class StructuredLogger:
    """Structured JSON logger with correlation ID support."""
    
    def __init__(self, name: str, config: LoggingConfig):
        self._logger = logging.getLogger(name)
        self._config = config
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup log handlers based on configuration."""
        formatter = JsonFormatter()
        
        # Console handler
        if self._config.console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
        
        # File handler
        if self._config.file_enabled:
            file_handler = RotatingFileHandler(
                self._config.file_path,
                maxBytes=self._config.max_file_size,
                backupCount=self._config.backup_count
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self._log(logging.INFO, message, kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self._log(logging.ERROR, message, kwargs)
    
    def _log(self, level: int, message: str, extra: Dict[str, Any]) -> None:
        """Log message with extra structured data."""
        # Add correlation ID if available
        correlation_id = getattr(contextvars.copy_context(), 'correlation_id', None)
        if correlation_id:
            extra['correlation_id'] = correlation_id
        
        self._logger.log(level, message, extra=extra)
```

### Metrics Collection

```python
class MetricsCollector:
    """Prometheus-compatible metrics collector."""
    
    def __init__(self):
        # Business metrics
        self.adaptation_count = Counter(
            'polaris_adaptations_total',
            'Total number of adaptations performed',
            ['system_id', 'action_type', 'status']
        )
        
        self.system_health_score = Gauge(
            'polaris_system_health_score',
            'Current system health score',
            ['system_id']
        )
        
        # Technical metrics
        self.message_processing_time = Histogram(
            'polaris_message_processing_seconds',
            'Time spent processing messages',
            ['component', 'operation']
        )
        
        self.active_connections = Gauge(
            'polaris_active_connections',
            'Number of active connections',
            ['connection_type']
        )
    
    def record_adaptation(self, system_id: str, action_type: str, success: bool) -> None:
        """Record adaptation execution."""
        status = "success" if success else "failure"
        self.adaptation_count.labels(
            system_id=system_id,
            action_type=action_type,
            status=status
        ).inc()
    
    def update_health_score(self, system_id: str, score: float) -> None:
        """Update system health score."""
        self.system_health_score.labels(system_id=system_id).set(score)
```

## Exception Handling

### Purpose
Provides structured exception hierarchy with context information.

### Exception Hierarchy

```python
class PolarisException(Exception):
    """Base exception for all POLARIS errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.utcnow()

class ConfigurationError(PolarisException):
    """Configuration-related errors."""
    pass

class ConnectorError(PolarisException):
    """Managed system connector errors."""
    pass

class AdaptationError(PolarisException):
    """Adaptation execution errors."""
    pass

class CircuitBreakerError(PolarisException):
    """Circuit breaker errors."""
    pass

class BulkheadError(PolarisException):
    """Bulkhead errors."""
    pass
```

### Error Context

```python
class ErrorContext:
    """Provides context information for errors."""
    
    @staticmethod
    def create_context(operation: str, **kwargs) -> Dict[str, Any]:
        """Create error context."""
        return {
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": getattr(contextvars.copy_context(), 'correlation_id', None),
            **kwargs
        }
    
    @staticmethod
    def wrap_exception(func: Callable) -> Callable:
        """Decorator to wrap exceptions with context."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except PolarisException:
                raise  # Re-raise POLARIS exceptions as-is
            except Exception as e:
                context = ErrorContext.create_context(
                    operation=func.__name__,
                    args=str(args),
                    kwargs=str(kwargs)
                )
                raise PolarisException(f"Unexpected error in {func.__name__}: {e}") from e
        
        return wrapper
```

---

*Continue to [Framework Layer](./08-framework-layer.md) →*