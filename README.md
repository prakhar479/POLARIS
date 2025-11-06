# POLARIS
**Proactive Optimization & Learning Architecture for Resilient Intelligent Systems**


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NATS](https://img.shields.io/badge/messaging-NATS-green.svg)](https://nats.io/)

POLARIS is a comprehensive framework for building self-adaptive systems that can monitor, analyze, plan, and execute adaptations autonomously. It implements the MAPE-K (Monitor, Analyze, Plan, Execute over a Knowledge base) loop with advanced AI/ML capabilities, providing a robust foundation for research and production adaptive systems.

## 🏗️ System Architecture

POLARIS follows a layered, event-driven architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POLARIS Framework                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  🎯 Control & Reasoning Layer                                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ Adaptive        │ │ Agentic         │ │ Meta            │              │
│  │ Controller      │ │ Reasoner        │ │ Learner         │              │
│  │ • MAPE-K Loop   │ │ • LLM-based     │ │ • Strategy      │              │
│  │ • Strategy      │ │ • Tool Usage    │ │   Learning      │              │
│  │   Selection     │ │ • Autonomous    │ │ • Parameter     │              │
│  │ • Orchestration │ │   Reasoning     │ │   Tuning        │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│  🧠 Digital Twin Layer                                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ World Model     │ │ Knowledge Base  │ │ Learning Engine │              │
│  │ • Bayesian      │ │ • Time Series   │ │ • Pattern       │              │
│  │ • LLM-based     │ │ • Graph DB      │ │   Recognition   │              │
│  │ • Statistical   │ │ • Document      │ │ • Reinforcement │              │
│  │ • Hybrid        │ │   Store         │ │   Learning      │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│  🔌 Adapter Layer                                                          │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ Monitor         │ │ Execution       │ │ Verification    │              │
│  │ Adapter         │ │ Adapter         │ │ Adapter         │              │
│  │ • Telemetry     │ │ • Action        │ │ • Safety        │              │
│  │   Collection    │ │   Execution     │ │   Constraints   │              │
│  │ • Metric        │ │ • Result        │ │ • Policy        │              │
│  │   Processing    │ │   Publishing    │ │   Enforcement   │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│  🏗️ Infrastructure Layer                                                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ Message Bus     │ │ Data Storage    │ │ Observability   │              │
│  │ • NATS          │ │ • Time Series   │ │ • Structured    │              │
│  │ • Event         │ │ • Graph DB      │ │   Logging       │              │
│  │   Streaming     │ │ • Document      │ │ • Metrics       │              │
│  │ • Pub/Sub       │ │   Store         │ │ • Tracing       │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│  🔧 Plugin Interface                                                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │ SWIM Plugin     │ │ SWITCH Plugin   │ │ Custom Plugins  │              │
│  │ • Web Service   │ │ • ML Model      │ │ • Your System   │              │
│  │   Simulation    │ │   Switching     │ │   Integration   │              │
│  │ • Server        │ │ • YOLO          │ │ • HTTP/TCP/     │              │
│  │   Scaling       │ │   Variants      │ │   Custom        │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **NATS Server** (included in `polaris_poc/bin/` or install separately)
- **Virtual Environment** (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd POLARIS

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
cd polaris_poc
pip install -r requirements.txt

# Set up environment variables
export GEMINI_API_KEY="your-gemini-api-key"  # For LLM-based components
export NATS_URL="nats://localhost:4222"
```

### Running the SWIM System (Complete Example)

The SWIM (Simulated Web Infrastructure Manager) system demonstrates POLARIS's full capabilities:

```bash
# Option 1: Automated startup with tmux (Recommended)
./start_polaris_swim_system.sh

# Option 2: Manual component startup
# Terminal 1: Start NATS server
./bin/nats-server --port 4222

# Terminal 2: Start Knowledge Base
python src/scripts/start_component.py knowledge-base

# Terminal 3: Start Digital Twin with Bayesian World Model
python src/scripts/start_component.py digital-twin --world-model bayesian

# Terminal 4: Start Verification Adapter
python src/scripts/start_component.py verification --plugin-dir extern

# Terminal 5: Start Kernel (coordination)
python src/scripts/start_component.py kernel

# Terminal 6: Start Monitor Adapter (SWIM telemetry)
python src/scripts/start_component.py monitor --plugin-dir extern

# Terminal 7: Start Execution Adapter (SWIM actions)
python src/scripts/start_component.py execution --plugin-dir extern

# Terminal 8: Start Agentic Reasoner (AI decision making)
python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model

# Terminal 9: Start Meta Learner (strategy optimization)
python src/scripts/start_component.py meta-learner
```

### Running the SWITCH System (ML Model Adaptation)

The SWITCH system demonstrates ML model adaptation with YOLO variants:

```bash
# Start SWITCH system components
./start_switch_system.sh

# Or manually:
python src/scripts/start_component.py monitor --plugin-dir extern/switch_plugin
python src/scripts/start_component.py execution --plugin-dir extern/switch_plugin
python src/scripts/start_component.py digital-twin --world-model bayesian
python extern/switch_plugin/run_switch_kernel.py
```

### Monitoring System Activity

```bash
# Monitor all POLARIS messages
python src/scripts/nats_spy.py

# Monitor specific message types
python src/scripts/nats_spy.py --preset telemetry
python src/scripts/nats_spy.py --preset execution
python src/scripts/nats_spy.py --subjects "polaris.verification.>"

# Check component health
./start_polaris_swim_system.sh --check-health
```

## 📁 Project Structure

### POLARIS POC (`polaris_poc/`)
The complete, production-ready implementation with all features:

```
polaris_poc/
├── src/polaris/                    # Core framework
│   ├── adapters/                   # System interface adapters
│   │   ├── monitor.py             # Telemetry collection
│   │   ├── execution.py           # Action execution
│   │   └── verification.py        # Safety validation
│   ├── agents/                     # AI/ML reasoning agents
│   │   ├── agentic_reasoner.py    # LLM-based reasoning
│   │   ├── digital_twin_agent.py  # Digital twin management
│   │   └── meta_learner_agent.py  # Strategy learning
│   ├── controllers/                # Control strategies
│   │   ├── fast_controller.py     # Reactive control
│   │   └── slow_controller.py     # Deliberative control
│   ├── kernel/                     # Core coordination
│   │   └── kernel.py              # MAPE-K orchestration
│   ├── models/                     # Data models & world models
│   │   ├── world_model.py         # Abstract world model
│   │   ├── bayesian_world_model.py # Bayesian implementation
│   │   └── gemini_world_model.py  # LLM-based implementation
│   └── services/                   # gRPC services
│       └── digital_twin_service.py # Digital twin API
├── extern/                         # Managed system plugins
│   ├── swim/                      # SWIM exemplar system
│   ├── switch/                    # SWITCH ML system
│   └── switch_plugin/             # SWITCH POLARIS plugin
├── config/                        # System configurations
│   ├── swim_optimized_config.yaml # SWIM-specific config
│   └── switch_optimized_config.yaml # SWITCH-specific config
├── examples/                      # Usage examples & demos
├── tests/                         # Comprehensive test suite
└── docs/                          # Detailed documentation
```

### POLARIS Refactored (`polaris_refactored/`)
Clean architecture implementation following enterprise patterns:

```
polaris_refactored/
├── src/
│   ├── framework/                 # Core framework services
│   │   ├── configuration/         # Config management
│   │   └── plugin_management/     # Plugin system
│   ├── adapters/                  # Adapter implementations
│   │   ├── monitor_adapter/       # Monitoring strategies
│   │   └── execution_adapter/     # Execution pipelines
│   ├── digital_twin/              # Digital twin components
│   │   ├── world_model.py         # World model interface
│   │   ├── knowledge_base.py      # Knowledge management
│   │   └── learning_engine.py     # Learning algorithms
│   ├── control_reasoning/         # Control & reasoning
│   │   ├── adaptive_controller.py # MAPE-K controller
│   │   └── reasoning_engine.py    # Multi-strategy reasoning
│   ├── infrastructure/            # Infrastructure services
│   │   ├── message_bus.py         # Event messaging
│   │   └── data_storage/          # Data persistence
│   └── domain/                    # Domain models
└── plugins/                       # System plugins
    ├── swim/                      # SWIM plugin
    └── switch/                    # SWITCH plugin
```

## 🔧 Core Components

### 1. Monitor Adapter
Collects telemetry from managed systems and publishes to NATS.

**Features:**
- Plugin-driven metric collection
- Batch and streaming telemetry
- Derived metric calculations
- Configurable collection strategies
- Error handling and retry logic

**Usage:**
```bash
python src/scripts/start_component.py monitor --plugin-dir extern --log-level DEBUG
```

### 2. Execution Adapter
Executes control actions on managed systems with safety validation.

**Features:**
- Action validation and precondition checking
- Parameter type and range validation
- Concurrent execution control
- Result publishing and metrics
- Queue management with throttling

**Usage:**
```bash
python src/scripts/start_component.py execution --plugin-dir extern
```

### 3. Verification Adapter
Validates control actions before execution to ensure safety and policy compliance.

**Features:**
- Safety constraint checking
- Organizational policy enforcement
- Digital Twin integration for predictive verification
- Multi-level verification (Basic, Policy, Formal, Comprehensive)
- Comprehensive violation reporting

**Usage:**
```bash
# With plugin constraints
python src/scripts/start_component.py verification --plugin-dir extern

# Standalone with built-in defaults
python src/scripts/start_component.py verification
```

### 4. Digital Twin
Provides intelligent system modeling and predictive capabilities.

**Features:**
- **NATS Ingestion**: Processes telemetry and execution events
- **gRPC Services**: Query, Simulation, Diagnosis, Management APIs
- **World Model**: Pluggable AI/ML implementations
- **Real-time Processing**: Batch processing with configurable timeouts

**World Model Options:**
- `mock`: Simple implementation for testing
- `bayesian`: Deterministic Bayesian/Kalman filter model
- `gemini`: Google Gemini LLM-based model
- `statistical`: Statistical analysis model
- `hybrid`: Combination approach

**Usage:**
```bash
# Start with Bayesian world model (recommended)
python src/scripts/start_component.py digital-twin --world-model bayesian

# Start with Gemini LLM model
python src/scripts/start_component.py digital-twin --world-model gemini

# Health check
python src/scripts/start_component.py digital-twin --health-check
```

### 5. Agentic Reasoner
Advanced LLM-based reasoning agent with autonomous tool usage.

**Features:**
- Improved gRPC client with circuit breaker
- Automatic retry with exponential backoff
- Performance monitoring and metrics
- Autonomous tool usage (Knowledge Base, Digital Twin)
- Bayesian world model integration

**Usage:**
```bash
# Basic startup with improved gRPC
python src/scripts/start_component.py agentic-reasoner

# With Bayesian world model integration
python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model

# With performance monitoring
python src/scripts/start_component.py agentic-reasoner --monitor-performance
```

### 6. Meta Learner
Learns and adapts reasoning strategies over time.

**Features:**
- Strategy learning from adaptation outcomes
- Parameter tuning based on performance
- Pattern recognition in system behavior
- Continuous improvement of adaptation policies

**Usage:**
```bash
python src/scripts/start_component.py meta-learner
```

### 7. Kernel
Central coordination and MAPE-K loop orchestration.

**Features:**
- MAPE-K loop implementation
- Component coordination
- Action routing and verification
- State management

**Usage:**
```bash
python src/scripts/start_component.py kernel
```

## 🔌 Plugin System

POLARIS uses a plugin architecture to integrate with different managed systems. Each plugin implements the `ManagedSystemConnector` interface.

### Creating a New Plugin

1. **Create Plugin Directory:**
```bash
mkdir my_system_plugin
cd my_system_plugin
touch __init__.py
```

2. **Define Configuration (`config.yaml`):**
```yaml
system_name: "my_system"
implementation:
  connector_class: "connector.MySystemConnector"
connection:
  protocol: "http"
  host: "localhost"
  port: 8080
monitoring:
  metrics:
    - name: "status"
      command: "GET /health"
      unit: "boolean"
execution:
  actions:
    - type: "RESTART"
      command: "POST /restart"
```

3. **Implement Connector (`connector.py`):**
```python
from polaris.adapters.core import ManagedSystemConnector

class MySystemConnector(ManagedSystemConnector):
    async def connect(self):
        # Implementation here
        pass
    
    async def execute_command(self, command, params=None):
        # Implementation here
        pass
```

4. **Test Plugin:**
```bash
python src/scripts/start_component.py monitor --plugin-dir my_system_plugin --validate-only
```

### Included Plugins

#### SWIM Plugin (`extern/`)
**System:** Simulated Web Infrastructure Manager
**Purpose:** Web service simulation with server scaling and QoS controls
**Actions:** ADD_SERVER, REMOVE_SERVER, SET_DIMMER
**Metrics:** Response times, throughput, server utilization, arrival rate

#### SWITCH Plugin (`extern/switch_plugin/`)
**System:** ML Model Switching System
**Purpose:** YOLO model adaptation for optimal utility
**Actions:** SWITCH_MODEL (between YOLOv5 variants)
**Metrics:** Processing time, confidence, utility, CPU usage

## 🧪 Examples and Demos

### Verification Demo
```bash
python examples/verification_demo.py
```
Interactive demonstration of the verification system with various constraint scenarios.

### Agentic Reasoner Demo
```bash
python examples/agentic_reasoner_demo.py
```
Shows LLM-based reasoning capabilities with different system scenarios.

### Production Usage Examples
```bash
python examples/production_usage_example.py
```
Complete production deployment examples with monitoring and alerting.

## 📊 Monitoring and Observability

### NATS Message Monitoring
```bash
# Monitor all POLARIS messages
python src/scripts/nats_spy.py

# Monitor specific subjects
python src/scripts/nats_spy.py --subjects "polaris.telemetry.>" "polaris.execution.>"

# Show full message content
python src/scripts/nats_spy.py --show-data

# Use presets
python src/scripts/nats_spy.py --preset telemetry
python src/scripts/nats_spy.py --preset execution
```

### Key NATS Subjects
- `polaris.telemetry.events.stream` - Individual telemetry events
- `polaris.telemetry.events.batch` - Batched telemetry events
- `polaris.execution.actions` - Control actions to execute
- `polaris.execution.results` - Action execution results
- `polaris.verification.requests` - Verification requests
- `polaris.verification.results` - Verification results
- `polaris.digitaltwin.*` - Digital twin communications

### Digital Twin gRPC Services
- **Query Service** (`:50051`): Current and historical system state
- **Simulation Service**: Predictive "what-if" analysis
- **Diagnosis Service**: Root cause analysis
- **Management Service**: Health checks and metrics

### Performance Metrics
- Telemetry processing throughput
- Adaptation decision latency
- Action execution success rates
- Verification approval/rejection rates
- World model prediction accuracy

## 🔧 Configuration

### Framework Configuration
Main configuration in `src/config/polaris_config.yaml`:
- NATS connection settings
- Telemetry batching parameters
- Component timeouts and retries
- Logging configuration

### System-Specific Configurations
- `config/swim_optimized_config.yaml` - SWIM system optimization
- `config/switch_optimized_config.yaml` - SWITCH system optimization
- `config/bayesian_world_model_config.yaml` - Bayesian model parameters

### Plugin Configuration
Each plugin has its own `config.yaml` with:
- System identification and metadata
- Connection parameters
- Metric definitions and collection strategies
- Action definitions and validation rules
- Verification constraints and policies

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python scripts/run_tests.py

# Run only non-async tests (fast)
python scripts/run_tests.py --non-async

# Run specific test file
python -m pytest tests/test_verification_adapter.py -v

# Run with coverage
python -m pytest tests/ --cov=src/polaris --cov-report=html
```

### Integration Testing
```bash
# Validate all configurations
python src/scripts/start_component.py all --plugin-dir extern --validate-only

# Test component startup
python src/scripts/start_component.py digital-twin --dry-run
```

### Performance Testing
```bash
# Analyze performance metrics
python scripts/analyze_performance.py logs/polaris_metrics.log
```

## 🚨 Troubleshooting

### Common Issues

1. **NATS Connection Failures**
   ```bash
   # Check NATS server status
   nc -z localhost 4222
   
   # Start NATS server
   ./bin/nats-server --port 4222
   ```

2. **Plugin Not Found**
   ```bash
   # Validate plugin configuration
   python src/scripts/start_component.py monitor --plugin-dir extern --validate-only
   ```

3. **Digital Twin gRPC Errors**
   ```bash
   # Check Digital Twin health
   python src/scripts/start_component.py digital-twin --health-check
   ```

4. **Gemini API Issues**
   ```bash
   # Verify API key
   echo $GEMINI_API_KEY
   
   # Test API connectivity
   python examples/test_interactive_api_key.py
   ```

### Debug Mode
```bash
# Enable debug logging for any component
python src/scripts/start_component.py <component> --log-level DEBUG
```

### Health Checks
```bash
# Check system health
./start_polaris_swim_system.sh --check-health

# Show component logs
./start_polaris_swim_system.sh --show-logs digital-twin
```

## 📚 Documentation

### Comprehensive Guides
- [`docs/COMPONENT_STARTUP_GUIDE.md`](polaris_poc/docs/COMPONENT_STARTUP_GUIDE.md) - Complete component startup reference
- [`docs/DIGITAL_TWIN_IMPLEMENTATION_SUMMARY.md`](polaris_poc/docs/DIGITAL_TWIN_IMPLEMENTATION_SUMMARY.md) - Digital Twin architecture
- [`docs/VERIFICATION_IMPLEMENTATION_SUMMARY.md`](polaris_poc/docs/VERIFICATION_IMPLEMENTATION_SUMMARY.md) - Verification system details
- [`docs/verification_agent_guide.md`](polaris_poc/docs/verification_agent_guide.md) - Verification usage guide

### API Documentation
- gRPC service definitions in `src/polaris/proto/`
- Comprehensive docstrings throughout codebase
- Configuration schema documentation

### Architecture Documentation
- [`polaris_refactored/doc/design.md`](polaris_refactored/doc/design.md) - System design principles
- [`polaris_refactored/doc/requirements.md`](polaris_refactored/doc/requirements.md) - Detailed requirements

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Run code quality checks
python scripts/validate_configs.py
python scripts/run_tests.py

# Generate protocol buffers
python scripts/generate_proto.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Comprehensive docstrings for all public APIs
- Structured logging with correlation IDs

### Testing Requirements
- Unit tests for all new components
- Integration tests for system interactions
- Performance tests for critical paths
- Configuration validation tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Carnegie Mellon University ABLE Team** - SWIM exemplar system
- **NATS.io** - High-performance messaging system
- **Google Gemini** - LLM capabilities for intelligent reasoning
- **gRPC** - High-performance RPC framework

## 📞 Support

For questions, issues, or contributions:
1. Check the [documentation](polaris_poc/docs/)
2. Review [troubleshooting guide](#-troubleshooting)
3. Run health checks and validation
4. Create an issue with detailed logs and configuration

---

**POLARIS** - Building the future of adaptive systems, one adaptation at a time. 🌟