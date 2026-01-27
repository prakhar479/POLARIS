# Polaris Framework Documentation

## What Polaris Does

Polaris is a modular self-adaptive systems framework that enables autonomous management of distributed systems. It continuously monitors managed systems, makes intelligent adaptation decisions, and learns from historical behavior to optimize performance over time.

**Core Capabilities:**
- Continuous system monitoring and telemetry collection
- Autonomous adaptation decision-making via pluggable strategies
- Historical data storage and behavioral modeling
- Meta-learning for automatic parameter optimization
- Event-driven architecture with comprehensive observability

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Polaris Framework                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Main Orchestrator (Polaris)                  │   │
│  │  - Component initialization & lifecycle              │   │
│  │  - Monitoring loop (30s intervals)                   │   │
│  │  - Meta-learning loop (hourly)                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
│         ▼                ▼                ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │  Connector  │  │  Strategy   │  │ World Model  │         │
│  │  Registry   │  │  (Decision) │  │  (Learning)  │         │
│  └─────────────┘  └─────────────┘  └──────────────┘         │
│         │                │                │                  │
│         ▼                ▼                ▼                  │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Knowledge Store (History)                │       │
│  └──────────────────────────────────────────────────┘       │
│         │                │                │                  │
│         ▼                ▼                ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │ Event Bus   │  │   Logger    │  │   Metrics    │         │
│  └─────────────┘  └─────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │    Managed Systems (via Connectors) │
        └─────────────────────────────────────┘
```

**Data Flow:**
1. **Monitoring Loop** (30s): Collect telemetry → Store state → Update world model → Assess adaptation need → Execute action if needed
2. **Meta-Learning Loop** (hourly): Analyze performance → Propose parameter updates → Apply approved changes
3. **Event System**: Components communicate via async event bus (TelemetryEvent, AdaptationEvent)

## Public Interfaces / APIs

### Core Entry Point

```python
from polaris import Polaris

# Minimal usage
polaris = Polaris()
await polaris.run()

# With custom components
polaris = Polaris(
    connectors=[YourConnector()],
    strategy=YourStrategy(),
    world_model=YourWorldModel(),
    meta_learner=YourMetaLearner()
)
await polaris.run()
```

### CLI Interface

```bash
# Basic usage
polaris --config config/default.yaml

# With dashboard
polaris --config config.yaml --dashboard

# With metrics export
polaris --config config.yaml --metrics-export ./metrics --metrics-experiment exp1
```

### Core Data Models

```python
from polaris import (
    SystemState,        # Current system state with metrics
    AdaptationAction,   # Action to execute (type, parameters, priority)
    ExecutionResult,    # Result of action execution
    MetricValue,        # Metric with name, value, unit, timestamp, tags
    HealthStatus,       # HEALTHY, WARNING, CRITICAL, UNHEALTHY, UNKNOWN
    ExecutionStatus     # SUCCESS, FAILED, PARTIAL, TIMEOUT
)
```

### Default Implementations

```python
from polaris import (
    ThresholdReactiveStrategy,  # Threshold-based adaptation
    StatisticalWorldModel,      # Statistical behavior modeling
    InMemoryKnowledgeStore,     # In-memory historical storage
    StatisticalMetaLearner,     # Statistical parameter optimization
    LLMMetaLearner             # LLM-powered optimization
)
```

## Extension Points

### 1. Custom Connector (System Integration)

```python
from polaris import Connector, SystemState, AdaptationAction, ExecutionResult

class MyConnector(Connector):
    async def connect(self) -> bool:
        """Establish connection to managed system."""
        pass
    
    async def disconnect(self) -> bool:
        """Disconnect from managed system."""
        pass
    
    async def get_system_id(self) -> str:
        """Return unique system identifier."""
        pass
    
    async def collect_telemetry(self) -> SystemState:
        """Collect current system state and metrics."""
        pass
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute adaptation action on system."""
        pass
    
    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if action can be executed."""
        pass
```

### 2. Custom Strategy (Decision Making)

```python
from polaris import AdaptationStrategy, AdaptationContext, ParameterSpec

class MyStrategy(AdaptationStrategy):
    async def assess(self, state: SystemState, context: AdaptationContext) -> Optional[AdaptationAction]:
        """Decide if adaptation is needed based on current state."""
        pass
    
    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Return parameters that can be tuned by meta-learner."""
        pass
    
    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a tunable parameter."""
        pass
```

### 3. Custom World Model (Behavior Modeling)

```python
from polaris import WorldModel, PredictionResult

class MyWorldModel(WorldModel):
    async def update(self, state: SystemState) -> None:
        """Update model with new system observation."""
        pass
    
    async def predict(self, action: AdaptationAction, current_state: SystemState) -> PredictionResult:
        """Predict outcome of executing an action."""
        pass
    
    async def get_insights(self) -> Dict[str, Any]:
        """Return insights about system behavior."""
        pass
```

### 4. Custom Knowledge Store (Historical Storage)

```python
from polaris import KnowledgeStore

class MyKnowledgeStore(KnowledgeStore):
    async def store_state(self, state: SystemState) -> None:
        """Store system state."""
        pass
    
    async def store_action(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Store adaptation action and result."""
        pass
    
    async def query_states(self, system_id: str, start_time: datetime, end_time: datetime) -> List[SystemState]:
        """Query historical states for time range."""
        pass
```

### 5. Custom Meta-Learner (Parameter Optimization)

```python
from polaris import MetaLearner, PerformanceAnalysis, ParameterProposal

class MyMetaLearner(MetaLearner):
    async def analyze_performance(self, system_id: str, time_window_hours: float) -> PerformanceAnalysis:
        """Analyze recent system performance."""
        pass
    
    async def propose_strategy_updates(self, strategy: AdaptationStrategy, analysis: PerformanceAnalysis) -> List[ParameterProposal]:
        """Propose parameter updates for strategy."""
        pass
    
    async def validate_proposals(self, proposals: List[ParameterProposal]) -> List[ParameterProposal]:
        """Validate and rank proposals by safety and impact."""
        pass
```

## Configuration

### YAML Configuration Example

```yaml
systems:
  - id: my_system
    enabled: true
    connector_type: custom
    connection:
      host: localhost
      port: 8080

strategy:
  type: threshold
  threshold:
    thresholds:
      cpu_usage: {high: 80.0, low: 20.0}
      memory_usage: {high: 85.0, low: 25.0}
    cooldown_seconds: 60

observability:
  logging:
    type: structured  # or "human"
    level: INFO
    console: true
    file: true
  metrics:
    enabled: true
    collector_type: simple
```

### Component Injection

```python
# Swap any component with custom implementation
polaris = Polaris(
    strategy=MyStrategy(),
    world_model=MyWorldModel(),
    knowledge_store=MyKnowledgeStore(),
    meta_learner=MyMetaLearner(),
    logger=MyLogger(),
    metrics=MyMetricsCollector(),
    connectors=[MyConnector()]
)
```

## Development Guidelines

### Key Design Patterns
- **Dependency Injection**: All components injectable with sensible defaults
- **Interface-Based**: Major components are abstract interfaces
- **Async/Await**: Fully asynchronous for concurrent operations
- **Immutable Models**: Domain models are frozen dataclasses
- **Event-Driven**: Components communicate via event bus
- **Metrics-First**: Comprehensive metrics collection throughout

### Typical Development Flow
1. Implement required interfaces for your use case
2. Create configuration file defining systems and strategy
3. Initialize Polaris with custom components
4. Run framework: `await polaris.run()`
5. Monitor via logs, metrics, or dashboard

### Built-in Implementations
- **ThresholdReactiveStrategy**: Triggers adaptations when metrics cross thresholds
- **SWIMConnector**: Connects to SWIM exemplar system
- **StatisticalWorldModel**: Uses mean/std for simple predictions
- **InMemoryKnowledgeStore**: Non-persistent storage for testing
- **StatisticalMetaLearner**: Rule-based parameter optimization

### Extension Points Summary
- **Connector**: Integrate new managed systems
- **Strategy**: Implement custom adaptation logic
- **WorldModel**: Add advanced behavior modeling (ML, etc.)
- **KnowledgeStore**: Add persistent storage (database, etc.)
- **MetaLearner**: Add advanced optimization (Bayesian, etc.)
- **Logger**: Custom logging backends
- **MetricsCollector**: Custom metrics backends

The framework is designed for production use with clear separation of concerns and multiple extension points for customization.