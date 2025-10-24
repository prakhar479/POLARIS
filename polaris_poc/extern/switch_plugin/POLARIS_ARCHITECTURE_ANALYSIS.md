# POLARIS Framework Architecture Analysis for SWITCH Integration

## Executive Summary

This document provides a comprehensive analysis of the POLARIS (Proactive Optimization & Learning Architecture for Resilient Intelligent Systems) framework as implemented for the SWIM system, and defines the mapping for SWITCH system integration.

---

## 1. POLARIS Core Architecture

### 1.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Framework Layer                           │
│  (CLI, Configuration, Plugin Management, Monitoring)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Control & Reasoning Layer                       │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────┐      │
│  │    Kernel     │→ │ Fast Ctrl    │  │ Slow Ctrl   │      │
│  │ (Orchestrator)│→ │ (Reactive)   │  │ (Reasoner)  │      │
│  └───────────────┘  └──────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Digital Twin Layer                              │
│  ┌──────────────────┐         ┌────────────────┐           │
│  │ Bayesian World   │ ←────→  │ Knowledge Base │           │
│  │ Model (Kalman)   │         │ (Storage)      │           │
│  └──────────────────┘         └────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Adapter Layer                                │
│  ┌──────────────────┐         ┌────────────────┐           │
│  │ Monitor Adapter  │         │ Execution      │           │
│  │ (Telemetry)      │         │ Adapter        │           │
│  └──────────────────┘         └────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Infrastructure Layer                              │
│  (NATS Message Bus, Event System, Plugin Registry)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Deep Dive

### 2.1 Kernel (Orchestrator)

**Purpose**: Central orchestration component that routes telemetry to appropriate controllers and manages action verification.

**Key Implementation (`SWIMKernel`)**:
- **Base Class**: `BaseKernel`
- **Communication**: NATS message bus
- **Subscriptions**: `polaris.telemetry.events.batch`
- **Publications**: 
  - `polaris.reasoner.kernel.requests` (slow path)
  - `polaris.verification.requests` (with verification)
  - `polaris.execution.actions` (without verification)

**Decision Logic**:
```python
def process_telemetry_event(telemetry_data):
    controller = strategy.select_controller(telemetry_data)
    
    if isinstance(controller, SlowController):
        # Delegate to reasoner for deep analysis
        send_to_reasoner(telemetry_data)
    else:
        # Fast path - immediate action
        action = controller.decide_action(telemetry_data)
        execute_action_with_verification(action)
```

**Configuration**:
- `enable_verification`: Toggle verification layer
- `verification_timeout`: Max wait for verification (default: 30s)

---

### 2.2 Fast Controller (Reactive)

**Purpose**: Rule-based, reactive controller for time-critical adaptations.

**SWIM Implementation**:

**Thresholds**:
- `RT_THRESHOLD = 0.75` (response time threshold)
- `DIMMER_STEP = 0.1` (QoS adjustment increment)

**Decision Logic**:
1. **Extract telemetry**: Parse metrics from batch events
2. **Validate**: Ensure required metrics present
3. **Calculate weighted RT**: `(basic_tp * basic_rt + opt_tp * opt_rt) / total_tp`
4. **High Response Time** (RT > 0.75):
   - If `active_servers < max_servers` → `ADD_SERVER`
   - Else → `SET_DIMMER(dimmer - 0.1)`
5. **Normal Response Time** (RT ≤ 0.75):
   - If `spare_util > 1`:
     - If `dimmer < 1.0` → `SET_DIMMER(dimmer + 0.1)`
     - Else → `REMOVE_SERVER`

**Metrics Used**:
- `swim.active.servers`
- `swim.max.servers`
- `swim.basic.response.time`
- `swim.optional.response.time`
- `swim.basic.throughput`
- `swim.optional.throughput`
- `swim.dimmer`
- `swim.server.utilization`

---

### 2.3 Slow Controller (Reasoner)

**Purpose**: LLM-based agentic reasoner for optimization, learning, and complex scenarios.

**Architecture**:
- **Reasoner Agent**: NATS-based service with gRPC Digital Twin interface
- **LLM Backend**: Gemini Pro / GPT models
- **Knowledge Base Integration**: Query historical patterns
- **Digital Twin Integration**: Simulate action outcomes

**Process Flow**:
1. **Receive** telemetry from kernel via `polaris.reasoner.kernel.requests`
2. **Context Building**:
   - Query knowledge base for similar patterns
   - Query Digital Twin for current state estimation
3. **Reasoning**:
   - LLM-based analysis with system-specific prompts
   - Consider long-term optimization
   - Generate confidence scores
4. **Simulation**: Test proposed actions in Digital Twin
5. **Decision**: Publish action to verification/execution

**SWIM-Specific Prompt Structure**:
```
You are an expert in adaptive systems managing the SWIM exemplar.

SWIM System Knowledge:
- Web service simulation with servers and QoS controls
- Metrics: response_time, throughput, utilization, dimmer, servers
- Actions: ADD_SERVER, REMOVE_SERVER, SET_DIMMER
- Fast stabilization (2-10 seconds)

Adaptation Strategies:
- High RT + low util → Adjust dimmer first
- High RT + high util → Add servers
- Low util + acceptable RT → Remove servers
- Balance quality vs performance

Provide:
- Specific recommendations
- Confidence levels
- Monitoring validation points
```

---

### 2.4 Knowledge Base

**Purpose**: System-agnostic storage for observations, decisions, and learned patterns.

**Interface** (`BaseKnowledgeBase`):
- `store(entry: KBEntry)`: Store observations
- `get(entry_id)`: Retrieve by ID
- `query(query: KBQuery)`: Search entries
- `delete(entry_id)`: Remove entry
- `clear()`: Reset database
- `get_stats()`: Get statistics

**Entry Types**:
- Telemetry observations
- Action decisions
- Execution results
- Learned patterns
- System dependencies (graph-based)

**Implementation**:
- Document storage for entries
- Graph storage for dependencies
- Semantic search capabilities

---

### 2.5 Bayesian World Model

**Purpose**: Probabilistic system state modeling for prediction and simulation.

**Core Features**:

1. **Kalman Filtering**:
   - 2D state space: [position, velocity]
   - Prediction and update cycles
   - Uncertainty quantification

2. **Bayesian Statistics**:
   - Prior/posterior updating
   - Confidence intervals (95%)
   - Anomaly detection (z-score based)

3. **Correlation Analysis**:
   - Cross-metric dependencies
   - Causal relationships
   - System health scoring

4. **Simulation**:
   - Multi-step ahead prediction
   - Action effect modeling
   - Confidence decay over time

**Configuration** (SWIM):
```yaml
prediction_horizon_minutes: 30
max_history_points: 500
update_interval_seconds: 10
correlation_threshold: 0.6
anomaly_threshold: 2.0
process_noise: 0.05
measurement_noise: 0.05
learning_rate: 0.1
```

**Query Types**:
- `current_state`: Get latest estimates
- `historical`: Query past states
- `prediction`: Forecast future states
- `correlation`: Analyze relationships

---

### 2.6 Monitor Adapter

**Purpose**: Collect telemetry from managed systems via plugin connectors.

**Responsibilities**:
1. **Metric Collection**: Poll/push metrics per configuration
2. **Batch Aggregation**: Group events for efficiency
3. **Derived Metrics**: Calculate composite metrics
4. **Publication**: Send to NATS telemetry subjects

**NATS Subjects**:
- `polaris.telemetry.events.stream` (individual events)
- `polaris.telemetry.events.batch` (aggregated batches)
- `polaris.telemetry.events.snapshots` (system snapshots)

**Collection Strategies**:
- `DirectConnectorStrategy`: Direct metric reads
- `PollingStrategy`: Time-based polling
- `BatchCollectionStrategy`: Group operations
- `RetryingStrategyDecorator`: Add retry logic

---

### 2.7 Execution Adapter

**Purpose**: Execute adaptation actions on managed systems.

**Pipeline Stages**:
1. **ValidationStage**: Validate action structure and preconditions
2. **PreConditionCheckStage**: Verify system state allows action
3. **ActionExecutionStage**: Execute via connector
4. **PostExecutionVerificationStage**: Verify effects

**NATS Subscriptions**:
- `polaris.execution.actions` (direct)
- `polaris.verification.requests` (with verification)

**NATS Publications**:
- `polaris.execution.results` (execution outcomes)
- `polaris.execution.decisions` (action decisions)
- `polaris.execution.metrics` (execution metrics)

**Constraints**:
- `max_concurrent`: Limit parallel actions
- `min_interval`: Minimum time between actions
- Timeouts per action type

---

### 2.8 Verification Adapter

**Purpose**: Validate actions against safety constraints and policies before execution.

**Verification Levels**:
- `basic`: Syntax and type validation
- `policy`: Check organizational policies
- `comprehensive`: Include simulation and impact analysis

**Constraint Types**:
- **Resource**: CPU, memory, disk limits
- **Safety**: System availability, error rates
- **Performance**: Response time, throughput
- **Temporal**: Time-based restrictions
- **Governance**: Change management policies

**Flow**:
1. Receive request on `polaris.verification.requests`
2. Load constraints from configuration
3. Evaluate constraints against action + context
4. If approved → Forward to `polaris.execution.actions`
5. Publish result to `polaris.verification.results`

---

## 3. Message Flow Patterns

### 3.1 Fast Path (Reactive Adaptation)

```
Monitor Adapter → NATS (telemetry.batch)
    ↓
Kernel → Controller Strategy → Fast Controller
    ↓
Fast Controller → Decide Action
    ↓
Kernel → NATS (verification.requests) [if enabled]
    ↓
Verification Adapter → Check Constraints
    ↓
NATS (execution.actions)
    ↓
Execution Adapter → Connector → Managed System
    ↓
NATS (execution.results)
```

### 3.2 Slow Path (Reasoned Optimization)

```
Monitor Adapter → NATS (telemetry.batch)
    ↓
Kernel → Controller Strategy → Slow Controller
    ↓
Kernel → NATS (reasoner.kernel.requests)
    ↓
Reasoner Agent:
  - Query Knowledge Base (gRPC)
  - Query Digital Twin (gRPC) 
  - LLM Reasoning
  - Simulate Actions
    ↓
NATS (verification.requests) [if enabled]
    ↓
[Same as fast path from verification onward]
```

### 3.3 Digital Twin Update

```
Monitor Adapter → NATS (telemetry.batch)
    ↓
Digital Twin Agent:
  - Update Bayesian World Model
  - Detect Anomalies
  - Update Correlations
  - Store in Knowledge Base
```

---

## 4. Configuration Hierarchy

### 4.1 Framework-Level (`polaris_config.yaml`)

- NATS connection settings
- Telemetry batching and streaming
- Reasoner routing configuration
- Kernel policies
- Verification thresholds
- Digital Twin configuration
- Logging and monitoring

### 4.2 Plugin-Level (`plugin/config.yaml`)

- System identification
- Connector implementation
- Connection parameters (host, port, protocol)
- Monitoring: metrics, derived metrics, strategies, interval
- Execution: actions, parameters, constraints
- System-specific settings

### 4.3 Component-Specific

- **Bayesian World Model**: `bayesian_world_model_config.yaml`
- **Reasoner Prompts**: System-specific prompt templates
- **Verification Constraints**: Safety and policy rules

---

## 5. System-Agnostic Components (Reusable)

### ✅ Fully Reusable:
- **BaseKernel class**: Orchestration logic
- **Bayesian World Model**: Statistical modeling
- **Knowledge Base interface**: Storage abstraction
- **Monitor Adapter framework**: Collection strategies
- **Execution Adapter framework**: Pipeline stages
- **Verification Adapter**: Constraint engine
- **NATS infrastructure**: Message bus
- **gRPC interfaces**: Digital Twin protocol

### ⚙️ Configurable Per System:
- **Fast Controller logic**: System-specific rules
- **Slow Controller prompts**: Domain knowledge
- **Verification constraints**: Safety rules
- **Plugin connector**: System-specific API client
- **Thresholds and targets**: Performance goals

---

## 6. SWIM vs SWITCH Comparison

| Aspect | SWIM | SWITCH |
|--------|------|--------|
| **Domain** | Web service scaling | ML model selection |
| **Primary Metric** | Response time | Utility (RT + Confidence) |
| **Actions** | ADD/REMOVE_SERVER, SET_DIMMER | SWITCH_MODEL (5 variants) |
| **Models** | N/A | yolov5n, s, m, l, x |
| **Key Tradeoff** | Cost vs Performance | Speed vs Accuracy |
| **Telemetry Rate** | Every 5s | Every 30s (configurable) |
| **Action Effect Time** | 2-10s | 2-5s |
| **Optimization Goal** | Minimize RT, Maximize Throughput | Maximize Utility Function |

---

## 7. SWITCH-Specific Requirements

### 7.1 Utility Function

```python
def call_utility_switched(r, C, Rmin=0.1, Rmax=1.0, Cmin=0.5, Cmax=1.0,
                          wd=0.5, we=0.5, pdv=5.0, pev=5.0):
    # Normalize confidence [0, 1]
    conf_score = clip_and_normalize(C, Cmin, Cmax)
    
    # Normalize response time (lower is better)
    rt_score = 1 - clip_and_normalize(r, Rmin, Rmax)
    
    # Base utility
    base_utility = we * conf_score + wd * rt_score
    
    # Penalties for out-of-bounds
    penalty = compute_penalties(r, C, Rmin, Rmax, Cmin, Cmax, pdv, pev)
    
    return base_utility + penalty  # Range: [-inf, 1]
```

**Objective**: **Maximize Utility**

### 7.2 Model Characteristics

| Model | Response Time | Confidence | Utility Context |
|-------|--------------|------------|-----------------|
| yolov5n | ~0.05s | ~0.65 | Fast, low accuracy |
| yolov5s | ~0.10s | ~0.75 | Balanced |
| yolov5m | ~0.20s | ~0.82 | Medium accuracy |
| yolov5l | ~0.40s | ~0.88 | High accuracy |
| yolov5x | ~0.80s | ~0.92 | Highest accuracy |

### 7.3 Adaptation Scenarios

1. **Low Response Time Requirement** (Rmax < 0.2s):
   - Prefer yolov5n or yolov5s
   - Accept lower confidence

2. **High Confidence Requirement** (Cmin > 0.85):
   - Use yolov5l or yolov5x
   - Accept higher response time

3. **Balanced** (moderate R and C):
   - Use yolov5s or yolov5m
   - Optimize utility dynamically

4. **Resource Constrained** (high CPU):
   - Downgrade to lighter model
   - Trade accuracy for resource efficiency

---

## Next Steps

1. Create `SwitchKernel` class (mirror `SWIMKernel`)
2. Implement `SwitchFastController` with utility-based rules
3. Configure `SwitchSlowController` with optimization prompts
4. Define SWITCH verification constraints
5. Create SWITCH-specific configuration files
6. Integrate with existing Monitor/Execution adapters
7. Test and validate utility maximization

