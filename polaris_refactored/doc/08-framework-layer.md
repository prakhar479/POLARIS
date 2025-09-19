# Framework Layer

## Overview

The Framework Layer orchestrates system components and provides core framework services including configuration management, plugin management, event handling, and overall system coordination. This layer acts as the foundation for all higher-level functionality.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Framework Layer                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Configuration   │ │Plugin Management│ │   Event System  │   │
│  │ - Hierarchical  │ │ - Discovery     │ │ - Event Bus     │   │
│  │ - Validation    │ │ - Lifecycle     │ │ - Pub/Sub       │   │
│  │ - Hot Reload    │ │ - Isolation     │ │ - Middleware    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Framework Orchestration                    │   │
│  │ - Component Lifecycle Management                        │   │
│  │ - Startup/Shutdown Coordination                         │   │
│  │ - Health Monitoring                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Management (PolarisConfiguration)

### Purpose
Provides hierarchical configuration with validation, hot-reload capabilities, and environment-specific overrides.

### Key Features
- **Hierarchical Loading**: Multiple configuration sources with precedence
- **JSON Schema Validation**: Type-safe configuration with detailed error reporting
- **Hot Reload**: Runtime configuration updates without restart
- **Environment Overrides**: Environment variable support for deployment flexibility

### Implementation

#### Configuration Builder
```python
class ConfigurationBuilder:
    """Builder for creating configuration instances."""
    
    def __init__(self):
        self._sources: List[ConfigurationSource] = []
        self._validators: List[ConfigurationValidator] = []
    
    def add_yaml_source(self, path: str, optional: bool = False) -> 'ConfigurationBuilder':
        """Add YAML configuration source."""
        self._sources.append(YamlConfigurationSource(path, optional))
        return self
    
    def add_json_source(self, path: str, optional: bool = False) -> 'ConfigurationBuilder':
        """Add JSON configuration source."""
        self._sources.append(JsonConfigurationSource(path, optional))
        return self
    
    def add_environment_source(self, prefix: str = "POLARIS_") -> 'ConfigurationBuilder':
        """Add environment variable source."""
        self._sources.append(EnvironmentConfigurationSource(prefix))
        return self
    
    def add_validator(self, validator: ConfigurationValidator) -> 'ConfigurationBuilder':
        """Add configuration validator."""
        self._validators.append(validator)
        return self
    
    def build(self) -> 'PolarisConfiguration':
        """Build configuration instance."""
        return PolarisConfiguration(self._sources, self._validators)
```

#### Core Configuration Class
```python
class PolarisConfiguration:
    """Main configuration class with hierarchical loading and validation."""
    
    def __init__(self, sources: List[ConfigurationSource], 
                 validators: List[ConfigurationValidator]):
        self._sources = sources
        self._validators = validators
        self._config_data: Dict[str, Any] = {}
        self._change_listeners: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        self._file_watchers: List[FileWatcher] = []
    
    async def load(self) -> None:
        """Load configuration from all sources."""
        merged_config = {}
        
        # Load from sources in order (later sources override earlier ones)
        for source in self._sources:
            try:
                source_config = await source.load()
                merged_config = self._merge_configs(merged_config, source_config)
            except ConfigurationSourceException as e:
                if not source.optional:
                    raise ConfigurationError(f"Failed to load required configuration: {e}")
        
        # Validate merged configuration
        for validator in self._validators:
            validation_result = await validator.validate(merged_config)
            if not validation_result.is_valid:
                raise ConfigurationValidationError(
                    "Configuration validation failed",
                    errors=validation_result.errors
                )
        
        self._config_data = merged_config
        await self._setup_file_watchers()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        current = self._config_data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})
    
    async def add_change_listener(self, listener: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Add configuration change listener."""
        self._change_listeners.append(listener)
    
    async def _on_configuration_changed(self, changes: Dict[str, Any]) -> None:
        """Handle configuration changes."""
        # Notify all listeners
        for listener in self._change_listeners:
            try:
                await listener(changes)
            except Exception as e:
                logger.error(f"Configuration change listener failed: {e}")
```

#### Configuration Sources

```python
class YamlConfigurationSource(ConfigurationSource):
    """YAML file configuration source."""
    
    def __init__(self, file_path: str, optional: bool = False):
        self.file_path = Path(file_path)
        self.optional = optional
    
    async def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.file_path.exists():
            if self.optional:
                return {}
            raise ConfigurationSourceException(f"Configuration file not found: {self.file_path}")
        
        try:
            with open(self.file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationSourceException(f"Invalid YAML in {self.file_path}: {e}")

class EnvironmentConfigurationSource(ConfigurationSource):
    """Environment variable configuration source."""
    
    def __init__(self, prefix: str = "POLARIS_"):
        self.prefix = prefix
    
    async def load(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Convert POLARIS_MESSAGE_BUS_NATS_SERVERS to message_bus.nats.servers
                config_key = key[len(self.prefix):].lower().replace('_', '.')
                self._set_nested_value(config, config_key, self._parse_value(value))
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested configuration value."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _parse_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
```

#### Configuration Validation

```python
class JsonSchemaValidator(ConfigurationValidator):
    """JSON Schema-based configuration validator."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.validator = jsonschema.Draft7Validator(schema)
    
    async def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against JSON schema."""
        errors = []
        
        for error in self.validator.iter_errors(config):
            errors.append(ValidationError(
                path='.'.join(str(p) for p in error.absolute_path),
                message=error.message,
                value=error.instance
            ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )

# Example schema for POLARIS configuration
POLARIS_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "framework": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"}
            },
            "required": ["name", "version"]
        },
        "message_bus": {
            "type": "object",
            "properties": {
                "nats": {
                    "type": "object",
                    "properties": {
                        "servers": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "name": {"type": "string"}
                    },
                    "required": ["servers"]
                }
            }
        },
        "managed_systems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "system_id": {"type": "string"},
                    "connector_type": {"type": "string"},
                    "config": {"type": "object"}
                },
                "required": ["system_id", "connector_type"]
            }
        }
    },
    "required": ["framework", "message_bus"]
}
```

## Plugin Management (PolarisPluginRegistry)

### Purpose
Handles discovery, loading, and lifecycle management of managed system connectors with isolation and hot-reloading capabilities.

### Key Features
- **Automatic Discovery**: Scans directories for plugin definitions
- **Hot Reloading**: Update plugins without system restart
- **Isolation**: Prevents plugin failures from affecting the system
- **Validation**: Ensures plugin compatibility and configuration correctness

### Implementation

#### Plugin Registry
```python
class PolarisPluginRegistry:
    """Registry for managing managed system connector plugins."""
    
    def __init__(self, config: PolarisConfiguration):
        self._config = config
        self._plugins: Dict[str, PluginDescriptor] = {}
        self._connectors: Dict[str, ManagedSystemConnector] = {}
        self._plugin_loaders: Dict[str, PluginLoader] = {}
    
    async def discover_plugins(self, search_paths: List[Path]) -> List[PluginDescriptor]:
        """Discover plugins in specified directories."""
        discovered_plugins = []
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for plugin_dir in search_path.iterdir():
                if not plugin_dir.is_dir():
                    continue
                
                plugin_file = plugin_dir / "plugin.yaml"
                if not plugin_file.exists():
                    continue
                
                try:
                    plugin_descriptor = await self._load_plugin_descriptor(plugin_file)
                    plugin_descriptor.path = plugin_dir
                    discovered_plugins.append(plugin_descriptor)
                    
                    logger.info(f"Discovered plugin: {plugin_descriptor.name}")
                except Exception as e:
                    logger.error(f"Failed to load plugin descriptor from {plugin_file}: {e}")
        
        return discovered_plugins
    
    async def load_plugin(self, plugin_descriptor: PluginDescriptor) -> None:
        """Load a plugin and make it available."""
        try:
            # Validate plugin configuration
            await self._validate_plugin(plugin_descriptor)
            
            # Create plugin loader
            loader = PluginLoader(plugin_descriptor)
            self._plugin_loaders[plugin_descriptor.name] = loader
            
            # Register plugin
            self._plugins[plugin_descriptor.name] = plugin_descriptor
            
            logger.info(f"Loaded plugin: {plugin_descriptor.name}")
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_descriptor.name}: {e}")
            raise PluginLoadException(f"Failed to load plugin {plugin_descriptor.name}") from e
    
    def load_managed_system_connector(self, system_id: str) -> ManagedSystemConnector:
        """Load connector for a managed system."""
        system_config = self._get_system_config(system_id)
        connector_type = system_config["connector_type"]
        
        # Find plugin that supports this connector type
        plugin_descriptor = self._find_plugin_for_connector(connector_type)
        if not plugin_descriptor:
            raise ConnectorNotFoundException(f"No plugin found for connector type: {connector_type}")
        
        # Load connector if not already loaded
        if system_id not in self._connectors:
            loader = self._plugin_loaders[plugin_descriptor.name]
            connector = loader.create_connector(system_config.get("config", {}))
            self._connectors[system_id] = connector
        
        return self._connectors[system_id]
    
    async def reload_plugin(self, plugin_name: str) -> None:
        """Reload a plugin with hot-reloading."""
        if plugin_name not in self._plugins:
            raise PluginNotFoundException(f"Plugin not found: {plugin_name}")
        
        plugin_descriptor = self._plugins[plugin_name]
        
        # Unload existing connectors for this plugin
        systems_to_reload = []
        for system_id, connector in self._connectors.items():
            if isinstance(connector, self._plugin_loaders[plugin_name].connector_class):
                systems_to_reload.append(system_id)
        
        # Stop and remove old connectors
        for system_id in systems_to_reload:
            connector = self._connectors[system_id]
            if hasattr(connector, 'stop'):
                await connector.stop()
            del self._connectors[system_id]
        
        # Reload plugin
        await self.load_plugin(plugin_descriptor)
        
        # Recreate connectors
        for system_id in systems_to_reload:
            self.load_managed_system_connector(system_id)
        
        logger.info(f"Reloaded plugin: {plugin_name}")
```

#### Plugin Descriptor
```python
@dataclass
class PluginDescriptor:
    """Describes a plugin and its capabilities."""
    
    name: str
    version: str
    description: str
    connector_class: str
    supported_systems: List[str]
    configuration_schema: Dict[str, Any]
    path: Optional[Path] = None
    
    @classmethod
    async def from_file(cls, plugin_file: Path) -> 'PluginDescriptor':
        """Load plugin descriptor from YAML file."""
        with open(plugin_file, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            connector_class=data["connector_class"],
            supported_systems=data.get("supported_systems", []),
            configuration_schema=data.get("configuration_schema", {})
        )
```

#### Plugin Loader
```python
class PluginLoader:
    """Loads and manages plugin instances."""
    
    def __init__(self, plugin_descriptor: PluginDescriptor):
        self.plugin_descriptor = plugin_descriptor
        self.connector_class: Optional[Type[ManagedSystemConnector]] = None
        self._module: Optional[ModuleType] = None
    
    def create_connector(self, config: Dict[str, Any]) -> ManagedSystemConnector:
        """Create connector instance."""
        if not self.connector_class:
            self._load_connector_class()
        
        # Validate configuration against schema
        if self.plugin_descriptor.configuration_schema:
            self._validate_config(config)
        
        return self.connector_class(config)
    
    def _load_connector_class(self) -> None:
        """Load connector class from plugin module."""
        try:
            # Add plugin path to Python path
            plugin_path = str(self.plugin_descriptor.path)
            if plugin_path not in sys.path:
                sys.path.insert(0, plugin_path)
            
            # Import connector module
            module_name = self.plugin_descriptor.connector_class.split('.')[0]
            self._module = importlib.import_module(module_name)
            
            # Get connector class
            class_name = self.plugin_descriptor.connector_class.split('.')[-1]
            self.connector_class = getattr(self._module, class_name)
            
            # Verify it implements the correct interface
            if not issubclass(self.connector_class, ManagedSystemConnector):
                raise PluginLoadException(
                    f"Connector class {self.connector_class} does not implement ManagedSystemConnector"
                )
        
        except Exception as e:
            raise PluginLoadException(f"Failed to load connector class: {e}") from e
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration against plugin schema."""
        try:
            jsonschema.validate(config, self.plugin_descriptor.configuration_schema)
        except jsonschema.ValidationError as e:
            raise ConfigurationValidationError(f"Plugin configuration validation failed: {e}")
```

## Event System (PolarisEventBus)

### Purpose
Provides type-safe event handling with subscription management and middleware support for decoupled component communication.

### Key Features
- **Type Safety**: Strongly typed events with automatic serialization
- **Subscription Management**: Flexible event subscription and unsubscription
- **Middleware Support**: Extensible event processing pipeline
- **Integration**: Seamless integration with message bus

### Implementation

#### Event Bus
```python
class PolarisEventBus:
    """Type-safe event bus with subscription management."""
    
    def __init__(self, message_bus: PolarisMessageBus):
        self._message_bus = message_bus
        self._subscriptions: Dict[Type[PolarisEvent], List[EventHandler]] = {}
        self._middleware: List[EventMiddleware] = []
    
    async def publish(self, event: PolarisEvent) -> None:
        """Publish an event."""
        # Process through middleware
        context = EventContext(event=event)
        await self._process_middleware(context, self._publish_internal)
    
    async def subscribe(self, event_type: Type[T], handler: EventHandler[T]) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []
            # Subscribe to message bus topic for this event type
            topic = self._get_topic_for_event_type(event_type)
            await self._message_bus.subscribe(topic, self._handle_message)
        
        self._subscriptions[event_type].append(handler)
    
    async def unsubscribe(self, event_type: Type[T], handler: EventHandler[T]) -> None:
        """Unsubscribe from events."""
        if event_type in self._subscriptions:
            if handler in self._subscriptions[event_type]:
                self._subscriptions[event_type].remove(handler)
    
    def add_middleware(self, middleware: EventMiddleware) -> None:
        """Add event middleware."""
        self._middleware.append(middleware)
    
    async def _publish_internal(self, context: EventContext) -> None:
        """Internal event publishing."""
        event = context.event
        topic = self._get_topic_for_event_type(type(event))
        await self._message_bus.publish(topic, event)
    
    async def _handle_message(self, message_data: bytes) -> None:
        """Handle incoming message from message bus."""
        try:
            # Deserialize event
            event_data = json.loads(message_data.decode())
            event_type = self._get_event_type_from_data(event_data)
            event = self._deserialize_event(event_data, event_type)
            
            # Dispatch to handlers
            if event_type in self._subscriptions:
                for handler in self._subscriptions[event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Event handler failed: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Failed to handle event message: {e}", exc_info=True)
```

#### Event Types
```python
class PolarisEvent(ABC):
    """Base class for all POLARIS events."""
    
    def __init__(self):
        self.event_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.correlation_id = getattr(contextvars.copy_context(), 'correlation_id', None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_type": self.__class__.__name__,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            **self._to_dict_internal()
        }
    
    @abstractmethod
    def _to_dict_internal(self) -> Dict[str, Any]:
        """Serialize event-specific data."""
        pass

class TelemetryEvent(PolarisEvent):
    """Event for system telemetry data."""
    
    def __init__(self, system_state: SystemState):
        super().__init__()
        self.system_state = system_state
    
    def _to_dict_internal(self) -> Dict[str, Any]:
        return {
            "system_state": self.system_state.to_dict()
        }

class AdaptationEvent(PolarisEvent):
    """Event for adaptation decisions."""
    
    def __init__(self, system_id: str, reason: str, 
                 suggested_actions: List[AdaptationAction], severity: str):
        super().__init__()
        self.system_id = system_id
        self.reason = reason
        self.suggested_actions = suggested_actions
        self.severity = severity
    
    def _to_dict_internal(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "reason": self.reason,
            "suggested_actions": [action.to_dict() for action in self.suggested_actions],
            "severity": self.severity
        }

class ExecutionResultEvent(PolarisEvent):
    """Event for adaptation execution results."""
    
    def __init__(self, execution_result: ExecutionResult):
        super().__init__()
        self.execution_result = execution_result
    
    def _to_dict_internal(self) -> Dict[str, Any]:
        return {
            "execution_result": self.execution_result.to_dict()
        }
```

## Framework Orchestration (PolarisFramework)

### Purpose
Coordinates the entire system lifecycle, manages component dependencies, and provides centralized health monitoring.

### Key Features
- **Lifecycle Management**: Coordinated startup and shutdown of all components
- **Dependency Resolution**: Automatic dependency injection and resolution
- **Health Monitoring**: Continuous monitoring of component health
- **Graceful Shutdown**: Proper cleanup and resource release

### Implementation

#### Main Framework Class
```python
class PolarisFramework:
    """Main framework class that orchestrates all components."""
    
    def __init__(self, config: PolarisConfiguration):
        self._config = config
        self._container = DIContainer()
        self._components: List[FrameworkComponent] = []
        self._running = False
        self._health_monitor: Optional[HealthMonitor] = None
    
    async def start(self) -> None:
        """Start the POLARIS framework."""
        if self._running:
            raise FrameworkException("Framework is already running")
        
        try:
            logger.info("Starting POLARIS framework...")
            
            # Register core services
            await self._register_core_services()
            
            # Initialize components
            await self._initialize_components()
            
            # Start components in dependency order
            await self._start_components()
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            self._running = True
            logger.info("POLARIS framework started successfully")
        
        except Exception as e:
            logger.error(f"Failed to start POLARIS framework: {e}")
            await self._cleanup()
            raise
    
    async def stop(self) -> None:
        """Stop the POLARIS framework."""
        if not self._running:
            return
        
        logger.info("Stopping POLARIS framework...")
        
        try:
            # Stop health monitoring
            if self._health_monitor:
                await self._health_monitor.stop()
            
            # Stop components in reverse order
            await self._stop_components()
            
            # Cleanup resources
            await self._cleanup()
            
            self._running = False
            logger.info("POLARIS framework stopped")
        
        except Exception as e:
            logger.error(f"Error during framework shutdown: {e}")
    
    async def _register_core_services(self) -> None:
        """Register core framework services."""
        # Configuration
        self._container.register_singleton(PolarisConfiguration, lambda: self._config)
        
        # Message bus
        message_bus_config = self._config.get_section("message_bus")
        nats_config = NATSConfig.from_dict(message_bus_config.get("nats", {}))
        nats_broker = NATSMessageBroker(nats_config)
        
        middleware_chain = MiddlewareChain([
            LoggingMiddleware(logger),
            MetricsMiddleware()
        ])
        
        message_bus = PolarisMessageBus(nats_broker, middleware_chain)
        self._container.register_singleton(PolarisMessageBus, lambda: message_bus)
        
        # Data storage
        storage_config = self._config.get_section("data_storage")
        data_store = await self._create_data_store(storage_config)
        self._container.register_singleton(PolarisDataStore, lambda: data_store)
        
        # Event bus
        event_bus = PolarisEventBus(message_bus)
        self._container.register_singleton(PolarisEventBus, lambda: event_bus)
        
        # Plugin registry
        plugin_registry = PolarisPluginRegistry(self._config)
        self._container.register_singleton(PolarisPluginRegistry, lambda: plugin_registry)
    
    async def _initialize_components(self) -> None:
        """Initialize all framework components."""
        # Create component instances
        self._components = [
            self._container.resolve(MonitorAdapter),
            self._container.resolve(ExecutionAdapter),
            self._container.resolve(PolarisWorldModel),
            self._container.resolve(PolarisKnowledgeBase),
            self._container.resolve(PolarisLearningEngine),
            self._container.resolve(PolarisAdaptiveController),
            self._container.resolve(PolarisReasoningEngine)
        ]
        
        # Initialize each component
        for component in self._components:
            if hasattr(component, 'initialize'):
                await component.initialize()
    
    async def _start_components(self) -> None:
        """Start all components in dependency order."""
        for component in self._components:
            try:
                if hasattr(component, 'start'):
                    await component.start()
                logger.info(f"Started component: {component.__class__.__name__}")
            except Exception as e:
                logger.error(f"Failed to start component {component.__class__.__name__}: {e}")
                raise
    
    async def _start_health_monitoring(self) -> None:
        """Start health monitoring for all components."""
        self._health_monitor = HealthMonitor(self._components)
        await self._health_monitor.start()
```

#### Health Monitoring
```python
class HealthMonitor:
    """Monitors the health of framework components."""
    
    def __init__(self, components: List[FrameworkComponent]):
        self._components = components
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start health monitoring."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self) -> None:
        """Continuous health checking loop."""
        while self._running:
            try:
                for component in self._components:
                    health_status = await self._check_component_health(component)
                    
                    if health_status.status != HealthStatus.HEALTHY:
                        logger.warning(
                            f"Component {component.__class__.__name__} is unhealthy: {health_status.message}"
                        )
                        
                        # Attempt recovery if possible
                        if hasattr(component, 'recover'):
                            try:
                                await component.recover()
                                logger.info(f"Component {component.__class__.__name__} recovered")
                            except Exception as e:
                                logger.error(f"Failed to recover component {component.__class__.__name__}: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(30)
    
    async def _check_component_health(self, component: FrameworkComponent) -> ComponentHealth:
        """Check health of a single component."""
        try:
            if hasattr(component, 'get_health'):
                return await component.get_health()
            else:
                # Basic health check - component exists and is responsive
                return ComponentHealth(
                    component_name=component.__class__.__name__,
                    status=HealthStatus.HEALTHY,
                    message="Component is responsive"
                )
        except Exception as e:
            return ComponentHealth(
                component_name=component.__class__.__name__,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {e}"
            )
```

#### Framework Factory
```python
async def create_polaris_framework(config_path: str) -> PolarisFramework:
    """Factory function to create and configure POLARIS framework."""
    
    # Build configuration
    config = (ConfigurationBuilder()
              .add_yaml_source(config_path)
              .add_environment_source("POLARIS_")
              .add_validator(JsonSchemaValidator(POLARIS_CONFIG_SCHEMA))
              .build())
    
    await config.load()
    
    # Create framework instance
    framework = PolarisFramework(config)
    
    return framework

# Usage example
async def main():
    framework = await create_polaris_framework("config.yaml")
    
    try:
        await framework.start()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        await framework.stop()
```

---

*Continue to [Domain Layer](./09-domain-layer.md) →*