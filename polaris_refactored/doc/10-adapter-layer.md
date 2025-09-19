# Adapter Layer

## Overview

The Adapter Layer serves as the bridge between POLARIS and external managed systems, implementing the hexagonal architecture pattern to isolate core business logic from external dependencies. This layer provides standardized interfaces for system integration while accommodating diverse system types and protocols.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Adapter Layer                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  Base Adapter   │ │ Monitor Adapter │ │Execution Adapter│   │
│  │ - Lifecycle Mgmt│ │ - Telemetry     │ │ - Action Exec   │   │
│  │ - Health Checks │ │ - Collection    │ │ - Pipelines     │   │
│  │ - Error Handling│ │ - Strategies    │ │ - Validation    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Plugin Connectors                        │   │
│  │ - System-specific implementations                       │   │
│  │ - Protocol adapters                                     │   │
│  │ - Configuration validation                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Base Adapter (PolarisAdapter)

### Purpose
Provides a template method pattern implementation for consistent adapter behavior across all managed system types.

### Key Features
- **Standardized Lifecycle**: Consistent startup, operation, and shutdown procedures
- **Health Monitoring**: Continuous health assessment with automatic recovery
- **Error Handling**: Comprehensive error management with circuit breaker integration
- **Metrics Collection**: Built-in performance and operational metrics
- **Configuration Validation**: Automatic validation of adapter-specific configuration

### Implementation

#### Base Adapter Class
```python
class PolarisAdapter(Injectable, ABC):
    """Base class for all POLARIS adapters with template method pattern."""
    
    def __init__(self, system_id: str, config: Dict[str, Any], 
                 event_bus: PolarisEventBus, metrics_collector: MetricsCollector):
        self.system_id = system_id
        self._config = config
        self._event_bus = event_bus
        self._metrics = metrics_collector
        self._state = AdapterState.STOPPED
        self._health_status = HealthStatus.UNKNOWN
        self._last_error: Optional[Exception] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the adapter (template method)."""
        async with self._lock:
            if self._state != AdapterState.STOPPED:
                raise AdapterException(f"Adapter {self.system_id} is not in stopped state")
            
            try:
                logger.info(f"Starting adapter for {self.system_id}")
                self._state = AdapterState.STARTING
                
                # Template method steps
                await self._validate_configuration()
                await self._initialize_resources()
                await self._establish_connections()
                await self._start_processing()
                
                self._state = AdapterState.RUNNING
                self._health_status = HealthStatus.HEALTHY
                
                # Start health monitoring
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                
                # Record metrics
                self._metrics.adapter_start_count.labels(
                    system_id=self.system_id,
                    adapter_type=self.__class__.__name__
                ).inc()
                
                logger.info(f"Adapter for {self.system_id} started successfully")
                
            except Exception as e:
                self._state = AdapterState.ERROR
                self._last_error = e
                logger.error(f"Failed to start adapter for {self.system_id}: {e}")
                raise AdapterException(f"Adapter startup failed: {e}") from e
    
    async def stop(self) -> None:
        """Stop the adapter (template method)."""
        async with self._lock:
            if self._state == AdapterState.STOPPED:
                return
            
            try:
                logger.info(f"Stopping adapter for {self.system_id}")
                self._state = AdapterState.STOPPING
                
                # Stop health monitoring
                if self._health_check_task:
                    self._health_check_task.cancel()
                    try:
                        await self._health_check_task
                    except asyncio.CancelledError:
                        pass
                
                # Template method steps
                await self._stop_processing()
                await self._close_connections()
                await self._cleanup_resources()
                
                self._state = AdapterState.STOPPED
                self._health_status = HealthStatus.UNKNOWN
                
                logger.info(f"Adapter for {self.system_id} stopped successfully")
                
            except Exception as e:
                self._state = AdapterState.ERROR
                self._last_error = e
                logger.error(f"Error stopping adapter for {self.system_id}: {e}")
    
    async def get_health(self) -> ComponentHealth:
        """Get adapter health status."""
        return ComponentHealth(
            component_name=f"adapter_{self.system_id}",
            status=self._health_status,
            message=self._get_health_message(),
            details={
                "state": self._state.value,
                "last_error": str(self._last_error) if self._last_error else None,
                "system_id": self.system_id
            }
        )
    
    # Abstract methods for subclasses to implement
    @abstractmethod
    async def _validate_configuration(self) -> None:
        """Validate adapter-specific configuration."""
        pass
    
    @abstractmethod
    async def _initialize_resources(self) -> None:
        """Initialize adapter-specific resources."""
        pass
    
    @abstractmethod
    async def _establish_connections(self) -> None:
        """Establish connections to managed system."""
        pass
    
    @abstractmethod
    async def _start_processing(self) -> None:
        """Start adapter-specific processing."""
        pass
    
    @abstractmethod
    async def _stop_processing(self) -> None:
        """Stop adapter-specific processing."""
        pass
    
    @abstractmethod
    async def _close_connections(self) -> None:
        """Close connections to managed system."""
        pass
    
    @abstractmethod
    async def _cleanup_resources(self) -> None:
        """Clean up adapter-specific resources."""
        pass
    
    @abstractmethod
    async def _perform_health_check(self) -> HealthStatus:
        """Perform adapter-specific health check."""
        pass
    
    async def _health_check_loop(self) -> None:
        """Continuous health monitoring loop."""
        while self._state == AdapterState.RUNNING:
            try:
                health_status = await self._perform_health_check()
                
                if health_status != self._health_status:
                    old_status = self._health_status
                    self._health_status = health_status
                    
                    logger.info(
                        f"Adapter {self.system_id} health changed: {old_status.value} -> {health_status.value}"
                    )
                    
                    # Publish health change event
                    await self._event_bus.publish(AdapterHealthChangedEvent(
                        system_id=self.system_id,
                        previous_status=old_status,
                        current_status=health_status
                    ))
                
                # Record health metrics
                self._metrics.adapter_health_score.labels(
                    system_id=self.system_id
                ).set(1.0 if health_status == HealthStatus.HEALTHY else 0.0)
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check failed for adapter {self.system_id}: {e}")
                self._health_status = HealthStatus.CRITICAL
                await asyncio.sleep(60)  # Longer interval on errors
    
    def _get_health_message(self) -> str:
        """Get human-readable health message."""
        if self._state == AdapterState.RUNNING:
            return f"Adapter is running and {self._health_status.value}"
        elif self._state == AdapterState.ERROR:
            return f"Adapter error: {self._last_error}"
        else:
            return f"Adapter is {self._state.value}"
```

#### Adapter State Management
```python
class AdapterState(Enum):
    """Adapter lifecycle states."""
    
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    
    def is_operational(self) -> bool:
        """Check if adapter is in operational state."""
        return self == AdapterState.RUNNING
    
    def is_transitional(self) -> bool:
        """Check if adapter is in transitional state."""
        return self in [AdapterState.STARTING, AdapterState.STOPPING]
```

## Monitor Adapter

### Purpose
Handles telemetry collection from managed systems using configurable collection strategies.

### Key Features
- **Multiple Collection Strategies**: Pull, push, and batch collection modes
- **Configurable Intervals**: System-specific collection frequencies
- **Data Validation**: Ensures telemetry data quality and consistency
- **Error Recovery**: Automatic retry and fallback mechanisms
- **Performance Optimization**: Efficient collection and processing

### Implementation

#### Monitor Adapter Class
```python
class MonitorAdapter(PolarisAdapter):
    """Adapter for monitoring managed systems and collecting telemetry."""
    
    def __init__(self, system_id: str, config: Dict[str, Any], 
                 event_bus: PolarisEventBus, metrics_collector: MetricsCollector,
                 connector: ManagedSystemConnector):
        super().__init__(system_id, config, event_bus, metrics_collector)
        self._connector = connector
        self._collection_strategy: Optional[MetricCollectionStrategy] = None
        self._collection_task: Optional[asyncio.Task] = None
        self._validator = MetricValidator(config.get("validation", {}))
        self._transformer = MetricTransformer(config.get("transformation", {}))
    
    async def _validate_configuration(self) -> None:
        """Validate monitor adapter configuration."""
        required_fields = ["collection_strategy", "collection_interval"]
        for field in required_fields:
            if field not in self._config:
                raise ConfigurationError(f"Missing required field: {field}")
        
        # Validate collection interval
        interval = self._config["collection_interval"]
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise ConfigurationError("Collection interval must be positive number")
        
        # Validate strategy configuration
        strategy_config = self._config.get("collection_strategy", {})
        strategy_type = strategy_config.get("type", "pull")
        
        if strategy_type not in ["pull", "push", "batch"]:
            raise ConfigurationError(f"Invalid collection strategy: {strategy_type}")
    
    async def _initialize_resources(self) -> None:
        """Initialize monitoring resources."""
        # Create collection strategy
        strategy_config = self._config["collection_strategy"]
        strategy_type = strategy_config.get("type", "pull")
        
        if strategy_type == "pull":
            self._collection_strategy = PullCollectionStrategy(
                config=PullCollectionConfig.from_dict(strategy_config),
                connector=self._connector,
                event_bus=self._event_bus
            )
        elif strategy_type == "push":
            self._collection_strategy = PushCollectionStrategy(
                config=PushCollectionConfig.from_dict(strategy_config),
                connector=self._connector,
                event_bus=self._event_bus
            )
        elif strategy_type == "batch":
            self._collection_strategy = BatchCollectionStrategy(
                config=BatchCollectionConfig.from_dict(strategy_config),
                connector=self._connector,
                event_bus=self._event_bus
            )
    
    async def _establish_connections(self) -> None:
        """Establish connection to managed system."""
        connected = await self._connector.connect()
        if not connected:
            raise AdapterException(f"Failed to connect to system {self.system_id}")
        
        logger.info(f"Connected to managed system {self.system_id}")
    
    async def _start_processing(self) -> None:
        """Start telemetry collection."""
        monitoring_target = MonitoringTarget(
            system_id=self.system_id,
            connector=self._connector,
            collection_interval=self._config["collection_interval"]
        )
        
        await self._collection_strategy.start_collection(monitoring_target)
        
        # Start collection task for pull strategy
        if isinstance(self._collection_strategy, PullCollectionStrategy):
            self._collection_task = asyncio.create_task(
                self._collection_loop(monitoring_target)
            )
        
        logger.info(f"Started telemetry collection for {self.system_id}")
    
    async def _stop_processing(self) -> None:
        """Stop telemetry collection."""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        if self._collection_strategy:
            monitoring_target = MonitoringTarget(
                system_id=self.system_id,
                connector=self._connector,
                collection_interval=self._config["collection_interval"]
            )
            await self._collection_strategy.stop_collection(monitoring_target)
    
    async def _close_connections(self) -> None:
        """Close connection to managed system."""
        await self._connector.disconnect()
        logger.info(f"Disconnected from managed system {self.system_id}")
    
    async def _cleanup_resources(self) -> None:
        """Clean up monitoring resources."""
        self._collection_strategy = None
        self._collection_task = None
    
    async def _perform_health_check(self) -> HealthStatus:
        """Perform health check on managed system."""
        try:
            # Check connector health
            connector_health = await self._connector.get_health_status()
            
            # Check collection strategy health
            if self._collection_strategy and hasattr(self._collection_strategy, 'get_health'):
                strategy_health = await self._collection_strategy.get_health()
                
                # Return worst health status
                if connector_health.severity_level() > strategy_health.severity_level():
                    return connector_health
                else:
                    return strategy_health
            
            return connector_health
            
        except Exception as e:
            logger.error(f"Health check failed for {self.system_id}: {e}")
            return HealthStatus.CRITICAL
    
    async def _collection_loop(self, target: MonitoringTarget) -> None:
        """Main collection loop for pull strategy."""
        while self._state == AdapterState.RUNNING:
            try:
                start_time = time.time()
                
                # Collect metrics
                raw_metrics = await self._connector.collect_metrics()
                
                # Validate metrics
                validation_result = self._validator.validate_metrics(raw_metrics)
                if not validation_result.is_valid:
                    logger.warning(
                        f"Metric validation failed for {self.system_id}: {validation_result.errors}"
                    )
                    # Continue with valid metrics only
                    raw_metrics = {k: v for k, v in raw_metrics.items() 
                                 if k not in validation_result.invalid_metrics}
                
                # Transform metrics
                transformed_metrics = self._transformer.transform_metrics(raw_metrics)
                
                # Get system health
                health_status = await self._connector.get_health_status()
                
                # Create system state
                system_state = SystemState(
                    system_id=self.system_id,
                    timestamp=datetime.utcnow(),
                    metrics=transformed_metrics,
                    health_status=health_status
                )
                
                # Publish telemetry event
                telemetry_event = TelemetryEvent(system_state)
                await self._event_bus.publish(telemetry_event)
                
                # Record collection metrics
                collection_time = time.time() - start_time
                self._metrics.collection_duration.labels(
                    system_id=self.system_id,
                    strategy="pull"
                ).observe(collection_time)
                
                self._metrics.collection_count.labels(
                    system_id=self.system_id,
                    strategy="pull",
                    status="success"
                ).inc()
                
                # Wait for next collection
                await asyncio.sleep(target.collection_interval)
                
            except Exception as e:
                logger.error(f"Collection failed for {self.system_id}: {e}")
                
                self._metrics.collection_count.labels(
                    system_id=self.system_id,
                    strategy="pull",
                    status="error"
                ).inc()
                
                # Exponential backoff on errors
                await asyncio.sleep(min(target.collection_interval * 2, 300))
```

## Execution Adapter

### Purpose
Handles execution of adaptation actions on managed systems through a configurable pipeline.

### Key Features
- **Pipeline Processing**: Multi-stage execution with validation and verification
- **Action Validation**: Pre-execution validation of action parameters and prerequisites
- **Result Tracking**: Comprehensive execution result capture and analysis
- **Rollback Support**: Automatic rollback on execution failures
- **Concurrency Control**: Manages concurrent action execution

### Implementation

#### Execution Adapter Class
```python
class ExecutionAdapter(PolarisAdapter):
    """Adapter for executing adaptation actions on managed systems."""
    
    def __init__(self, system_id: str, config: Dict[str, Any], 
                 event_bus: PolarisEventBus, metrics_collector: MetricsCollector,
                 connector: ManagedSystemConnector):
        super().__init__(system_id, config, event_bus, metrics_collector)
        self._connector = connector
        self._execution_pipeline: Optional[ActionExecutionPipeline] = None
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_semaphore: Optional[asyncio.Semaphore] = None
        self._execution_queue: asyncio.Queue = asyncio.Queue()
        self._execution_task: Optional[asyncio.Task] = None
    
    async def _validate_configuration(self) -> None:
        """Validate execution adapter configuration."""
        # Validate concurrency limits
        max_concurrent = self._config.get("max_concurrent_executions", 5)
        if not isinstance(max_concurrent, int) or max_concurrent <= 0:
            raise ConfigurationError("max_concurrent_executions must be positive integer")
        
        # Validate timeout settings
        default_timeout = self._config.get("default_timeout", 300)
        if not isinstance(default_timeout, (int, float)) or default_timeout <= 0:
            raise ConfigurationError("default_timeout must be positive number")
    
    async def _initialize_resources(self) -> None:
        """Initialize execution resources."""
        # Create execution pipeline
        pipeline_stages = [
            ValidationStage(self._connector),
            PreConditionStage(self._connector),
            ActionExecutionStage(self._connector),
            PostVerificationStage(self._connector)
        ]
        
        self._execution_pipeline = ActionExecutionPipeline(pipeline_stages)
        
        # Create concurrency control
        max_concurrent = self._config.get("max_concurrent_executions", 5)
        self._execution_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Subscribe to adaptation action events
        await self._event_bus.subscribe(AdaptationActionEvent, self._handle_action_event)
    
    async def _establish_connections(self) -> None:
        """Establish connection to managed system."""
        connected = await self._connector.connect()
        if not connected:
            raise AdapterException(f"Failed to connect to system {self.system_id}")
        
        logger.info(f"Connected to managed system {self.system_id} for execution")
    
    async def _start_processing(self) -> None:
        """Start action execution processing."""
        self._execution_task = asyncio.create_task(self._execution_loop())
        logger.info(f"Started action execution processing for {self.system_id}")
    
    async def _stop_processing(self) -> None:
        """Stop action execution processing."""
        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active executions to complete
        if self._active_executions:
            logger.info(f"Waiting for {len(self._active_executions)} active executions to complete")
            await asyncio.gather(
                *[ctx.completion_event.wait() for ctx in self._active_executions.values()],
                return_exceptions=True
            )
    
    async def _close_connections(self) -> None:
        """Close connection to managed system."""
        await self._connector.disconnect()
        logger.info(f"Disconnected from managed system {self.system_id}")
    
    async def _cleanup_resources(self) -> None:
        """Clean up execution resources."""
        self._execution_pipeline = None
        self._active_executions.clear()
        self._execution_semaphore = None
    
    async def _perform_health_check(self) -> HealthStatus:
        """Perform health check on execution capabilities."""
        try:
            # Check connector health
            connector_health = await self._connector.get_health_status()
            
            # Check execution queue health
            queue_size = self._execution_queue.qsize()
            max_queue_size = self._config.get("max_queue_size", 100)
            
            if queue_size > max_queue_size * 0.9:
                return HealthStatus.WARNING
            elif queue_size >= max_queue_size:
                return HealthStatus.CRITICAL
            
            return connector_health
            
        except Exception as e:
            logger.error(f"Execution health check failed for {self.system_id}: {e}")
            return HealthStatus.CRITICAL
    
    async def _handle_action_event(self, event: AdaptationActionEvent) -> None:
        """Handle adaptation action event."""
        if event.system_id != self.system_id:
            return
        
        # Queue action for execution
        await self._execution_queue.put(event.action)
        
        logger.info(f"Queued action {event.action.action_id} for execution on {self.system_id}")
    
    async def _execution_loop(self) -> None:
        """Main execution processing loop."""
        while self._state == AdapterState.RUNNING:
            try:
                # Get next action from queue
                action = await asyncio.wait_for(
                    self._execution_queue.get(),
                    timeout=1.0
                )
                
                # Execute action asynchronously
                execution_task = asyncio.create_task(
                    self._execute_action_with_semaphore(action)
                )
                
                # Don't wait for completion - allows concurrent execution
                
            except asyncio.TimeoutError:
                # No actions in queue, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in execution loop for {self.system_id}: {e}")
                await asyncio.sleep(1)
    
    async def _execute_action_with_semaphore(self, action: AdaptationAction) -> None:
        """Execute action with concurrency control."""
        async with self._execution_semaphore:
            await self._execute_action(action)
    
    async def _execute_action(self, action: AdaptationAction) -> None:
        """Execute a single adaptation action."""
        execution_context = ExecutionContext(
            action=action,
            start_time=datetime.utcnow(),
            completion_event=asyncio.Event()
        )
        
        self._active_executions[action.action_id] = execution_context
        
        try:
            logger.info(f"Executing action {action.action_id} on {self.system_id}")
            
            # Execute through pipeline
            result = await self._execution_pipeline.execute(action)
            
            execution_context.result = result
            execution_context.end_time = datetime.utcnow()
            
            # Publish execution result event
            await self._event_bus.publish(ExecutionResultEvent(result))
            
            # Record metrics
            self._metrics.action_execution_count.labels(
                system_id=self.system_id,
                action_type=action.action_type,
                status=result.status.value
            ).inc()
            
            self._metrics.action_execution_duration.labels(
                system_id=self.system_id,
                action_type=action.action_type
            ).observe(result.duration_seconds)
            
            logger.info(
                f"Action {action.action_id} completed with status {result.status.value} "
                f"in {result.duration_seconds:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Action execution failed for {action.action_id}: {e}")
            
            # Create failure result
            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                start_time=execution_context.start_time,
                end_time=datetime.utcnow(),
                error_message=str(e)
            )
            
            execution_context.result = result
            execution_context.end_time = datetime.utcnow()
            
            # Publish failure event
            await self._event_bus.publish(ExecutionResultEvent(result))
            
            # Record failure metrics
            self._metrics.action_execution_count.labels(
                system_id=self.system_id,
                action_type=action.action_type,
                status="failed"
            ).inc()
            
        finally:
            # Mark execution as complete
            execution_context.completion_event.set()
            
            # Clean up after delay to allow result retrieval
            asyncio.create_task(self._cleanup_execution(action.action_id))
    
    async def _cleanup_execution(self, action_id: str) -> None:
        """Clean up completed execution after delay."""
        await asyncio.sleep(300)  # Keep execution context for 5 minutes
        
        if action_id in self._active_executions:
            del self._active_executions[action_id]
```

#### Execution Pipeline
```python
class ActionExecutionPipeline:
    """Pipeline for executing adaptation actions through multiple stages."""
    
    def __init__(self, stages: List[ExecutionStage]):
        self._stages = stages
    
    async def execute(self, action: AdaptationAction) -> ExecutionResult:
        """Execute action through all pipeline stages."""
        context = ExecutionPipelineContext(
            action=action,
            start_time=datetime.utcnow(),
            stage_results={}
        )
        
        try:
            for stage in self._stages:
                stage_result = await stage.execute(context)
                context.stage_results[stage.name] = stage_result
                
                if not stage_result.success:
                    # Stage failed, create failure result
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        start_time=context.start_time,
                        end_time=datetime.utcnow(),
                        error_message=stage_result.error_message,
                        result_data={"failed_stage": stage.name, "stage_results": context.stage_results}
                    )
            
            # All stages successful
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.SUCCESS,
                start_time=context.start_time,
                end_time=datetime.utcnow(),
                result_data={"stage_results": context.stage_results}
            )
            
        except Exception as e:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                start_time=context.start_time,
                end_time=datetime.utcnow(),
                error_message=str(e),
                result_data={"stage_results": context.stage_results}
            )

class ValidationStage(ExecutionStage):
    """Validates action parameters and prerequisites."""
    
    name = "validation"
    
    def __init__(self, connector: ManagedSystemConnector):
        self._connector = connector
    
    async def execute(self, context: ExecutionPipelineContext) -> StageResult:
        """Execute validation stage."""
        action = context.action
        
        try:
            # Validate action with connector
            is_valid = await self._connector.validate_action(action)
            
            if not is_valid:
                return StageResult(
                    success=False,
                    error_message=f"Action {action.action_type} is not valid for system"
                )
            
            # Check prerequisites
            for prerequisite in action.prerequisites:
                if not await self._check_prerequisite(prerequisite):
                    return StageResult(
                        success=False,
                        error_message=f"Prerequisite not met: {prerequisite}"
                    )
            
            return StageResult(success=True, data={"validation": "passed"})
            
        except Exception as e:
            return StageResult(
                success=False,
                error_message=f"Validation failed: {e}"
            )
    
    async def _check_prerequisite(self, prerequisite: str) -> bool:
        """Check if prerequisite is satisfied."""
        # Implementation depends on prerequisite type
        # This is a simplified example
        return True

class ActionExecutionStage(ExecutionStage):
    """Executes the actual adaptation action."""
    
    name = "execution"
    
    def __init__(self, connector: ManagedSystemConnector):
        self._connector = connector
    
    async def execute(self, context: ExecutionPipelineContext) -> StageResult:
        """Execute the adaptation action."""
        try:
            result = await self._connector.execute_action(context.action)
            
            if result.is_successful:
                return StageResult(
                    success=True,
                    data={"execution_result": result.to_dict()}
                )
            else:
                return StageResult(
                    success=False,
                    error_message=result.error_message or "Action execution failed"
                )
                
        except Exception as e:
            return StageResult(
                success=False,
                error_message=f"Action execution failed: {e}"
            )
```

## Plugin Connector Examples

### Database Connector Example
```python
class DatabaseConnector(ManagedSystemConnector):
    """Connector for database systems (PostgreSQL, MySQL, etc.)."""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._connection_pool: Optional[asyncpg.Pool] = None
        self._system_id = config["system_id"]
    
    async def connect(self) -> bool:
        """Connect to database."""
        try:
            self._connection_pool = await asyncpg.create_pool(
                host=self._config["host"],
                port=self._config["port"],
                user=self._config["user"],
                password=self._config["password"],
                database=self._config["database"],
                min_size=1,
                max_size=10
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        """Collect database metrics."""
        async with self._connection_pool.acquire() as conn:
            # Connection count
            conn_result = await conn.fetchrow(
                "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active'"
            )
            
            # Database size
            size_result = await conn.fetchrow(
                "SELECT pg_database_size(current_database()) as db_size"
            )
            
            # Query performance
            perf_result = await conn.fetchrow("""
                SELECT 
                    avg(mean_exec_time) as avg_query_time,
                    sum(calls) as total_queries
                FROM pg_stat_statements 
                WHERE last_exec > now() - interval '1 minute'
            """)
            
            return {
                "active_connections": MetricValue(
                    name="active_connections",
                    value=float(conn_result["active_connections"]),
                    unit="count"
                ),
                "database_size_bytes": MetricValue(
                    name="database_size_bytes",
                    value=float(size_result["db_size"]),
                    unit="bytes"
                ),
                "avg_query_time_ms": MetricValue(
                    name="avg_query_time_ms",
                    value=float(perf_result["avg_query_time"] or 0),
                    unit="ms"
                ),
                "queries_per_minute": MetricValue(
                    name="queries_per_minute",
                    value=float(perf_result["total_queries"] or 0),
                    unit="count"
                )
            }
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute database adaptation action."""
        start_time = datetime.utcnow()
        
        try:
            if action.action_type == "tune_parameters":
                await self._tune_parameters(action.parameters)
            elif action.action_type == "kill_long_queries":
                await self._kill_long_queries(action.parameters)
            elif action.action_type == "vacuum_analyze":
                await self._vacuum_analyze(action.parameters)
            else:
                raise ValueError(f"Unsupported action type: {action.action_type}")
            
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.SUCCESS,
                start_time=start_time,
                end_time=datetime.utcnow(),
                result_data={"message": f"Successfully executed {action.action_type}"}
            )
            
        except Exception as e:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def get_supported_actions(self) -> List[str]:
        """Get supported action types."""
        return ["tune_parameters", "kill_long_queries", "vacuum_analyze"]
```

### Kubernetes Connector Example
```python
class KubernetesConnector(ManagedSystemConnector):
    """Connector for Kubernetes clusters."""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._k8s_client: Optional[kubernetes.client.ApiClient] = None
        self._apps_v1: Optional[kubernetes.client.AppsV1Api] = None
        self._core_v1: Optional[kubernetes.client.CoreV1Api] = None
        self._system_id = config["system_id"]
    
    async def connect(self) -> bool:
        """Connect to Kubernetes cluster."""
        try:
            if self._config.get("in_cluster", False):
                kubernetes.config.load_incluster_config()
            else:
                kubernetes.config.load_kube_config(
                    config_file=self._config.get("kubeconfig_path")
                )
            
            self._k8s_client = kubernetes.client.ApiClient()
            self._apps_v1 = kubernetes.client.AppsV1Api(self._k8s_client)
            self._core_v1 = kubernetes.client.CoreV1Api(self._k8s_client)
            
            # Test connection
            await self._core_v1.list_namespace()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
            return False
    
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        """Collect Kubernetes metrics."""
        namespace = self._config.get("namespace", "default")
        deployment_name = self._config.get("deployment_name")
        
        # Get deployment info
        deployment = await self._apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )
        
        # Get pods
        pods = await self._core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={deployment_name}"
        )
        
        # Calculate metrics
        desired_replicas = deployment.spec.replicas
        ready_replicas = deployment.status.ready_replicas or 0
        available_replicas = deployment.status.available_replicas or 0
        
        pod_count = len(pods.items)
        running_pods = len([p for p in pods.items if p.status.phase == "Running"])
        
        return {
            "desired_replicas": MetricValue(
                name="desired_replicas",
                value=float(desired_replicas),
                unit="count"
            ),
            "ready_replicas": MetricValue(
                name="ready_replicas",
                value=float(ready_replicas),
                unit="count"
            ),
            "availability_ratio": MetricValue(
                name="availability_ratio",
                value=float(available_replicas / desired_replicas) if desired_replicas > 0 else 0.0,
                unit="ratio"
            ),
            "pod_count": MetricValue(
                name="pod_count",
                value=float(pod_count),
                unit="count"
            ),
            "running_pods": MetricValue(
                name="running_pods",
                value=float(running_pods),
                unit="count"
            )
        }
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute Kubernetes adaptation action."""
        start_time = datetime.utcnow()
        
        try:
            if action.action_type == "scale_deployment":
                await self._scale_deployment(action.parameters)
            elif action.action_type == "restart_deployment":
                await self._restart_deployment(action.parameters)
            elif action.action_type == "update_resources":
                await self._update_resources(action.parameters)
            else:
                raise ValueError(f"Unsupported action type: {action.action_type}")
            
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.SUCCESS,
                start_time=start_time,
                end_time=datetime.utcnow(),
                result_data={"message": f"Successfully executed {action.action_type}"}
            )
            
        except Exception as e:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _scale_deployment(self, parameters: Dict[str, Any]) -> None:
        """Scale deployment to specified replica count."""
        namespace = self._config.get("namespace", "default")
        deployment_name = self._config.get("deployment_name")
        replicas = parameters["replicas"]
        
        # Update deployment
        body = {"spec": {"replicas": replicas}}
        await self._apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=body
        )
        
        logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
```

## Configuration Examples

### Adapter Configuration
```yaml
adapters:
  database_monitor:
    type: "monitor"
    system_id: "prod_database"
    connector_type: "database"
    config:
      host: "db.example.com"
      port: 5432
      database: "production"
      user: "monitor_user"
      password: "${DB_PASSWORD}"
    collection_strategy:
      type: "pull"
      interval: 30
    validation:
      required_metrics: ["active_connections", "database_size_bytes"]
      thresholds:
        active_connections: {min: 0, max: 1000}
    transformation:
      derived_metrics:
        - name: "connection_utilization"
          formula: "active_connections / max_connections"

  kubernetes_executor:
    type: "execution"
    system_id: "prod_k8s_cluster"
    connector_type: "kubernetes"
    config:
      kubeconfig_path: "/etc/kubernetes/admin.conf"
      namespace: "production"
      deployment_name: "web-app"
    max_concurrent_executions: 3
    default_timeout: 300
    pipeline_stages:
      - validation
      - pre_condition
      - execution
      - post_verification
```

---

*Continue to [Digital Twin Layer](./11-digital-twin-layer.md) →*