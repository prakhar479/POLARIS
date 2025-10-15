# POLARIS Component Startup Guide

This guide provides comprehensive instructions for starting and managing POLARIS framework components using the enhanced `start_component.py` script.

## Quick Start

```bash
# Show detailed component information
python start_component.py help

# Start individual components
python start_component.py digital-twin
python start_component.py agentic-reasoner --use-bayesian-world-model
python start_component.py monitor --plugin-dir extern

# Validate configurations
python start_component.py digital-twin --validate-only
python start_component.py agentic-reasoner --validate-only

# Development mode
python start_component.py digital-twin --dry-run
python start_component.py all --dry-run --plugin-dir extern
```

## Component Overview

### 📡 Adapters (System Interface)

#### Monitor Adapter
Collects telemetry data from managed systems.

```bash
# Basic usage
python start_component.py monitor --plugin-dir extern

# With custom configuration
python start_component.py monitor --plugin-dir extern --config custom_config.yaml

# Debug mode
python start_component.py monitor --plugin-dir extern --log-level DEBUG
```

#### Execution Adapter
Executes adaptation actions on managed systems.

```bash
# Basic usage
python start_component.py execution --plugin-dir extern

# With performance monitoring
python start_component.py execution --plugin-dir extern --monitor-performance
```

#### Verification Adapter
Validates system constraints and policies.

```bash
# With plugin directory
python start_component.py verification --plugin-dir extern

# Framework defaults only (no plugin)
python start_component.py verification
```

### 🧠 Core Services (Framework Infrastructure)

#### Digital Twin
Central world model and system state management.

```bash
# Basic startup
python start_component.py digital-twin

# With specific world model
python start_component.py digital-twin --world-model bayesian
python start_component.py digital-twin --world-model gemini

# Health check
python start_component.py digital-twin --health-check

# Configuration validation
python start_component.py digital-twin --validate-only
```

**World Model Options:**
- `mock`: Simple implementation for testing
- `gemini`: Google Gemini LLM-based model (requires API key)
- `bayesian`: Deterministic Bayesian/Kalman filter model
- `statistical`: Statistical analysis model
- `hybrid`: Combination approach

#### Knowledge Base Service
Stores and queries historical system data.

```bash
# Basic startup
python start_component.py knowledge-base

# With custom NATS configuration
python start_component.py knowledge-base --config custom_kb_config.yaml
```

#### Kernel
Central coordination and decision making.

```bash
# Basic startup
python start_component.py kernel

# With debug logging
python start_component.py kernel --log-level DEBUG
```

### 🤖 Reasoning Agents (Decision Making)

#### Basic Reasoner
LLM-based reasoning agent.

```bash
# Basic startup
python start_component.py reasoner

# With custom reasoning mode
python start_component.py reasoner --reasoning-mode hybrid

# With custom prompts
python start_component.py reasoner --prompt-config custom_prompts.yaml
```

**Reasoning Modes:**
- `llm`: Pure LLM-based reasoning
- `hybrid`: Combination of LLM and statistical methods
- `statistical`: Statistical analysis only

#### Agentic Reasoner (Enhanced)
Advanced agent with autonomous tool usage and improved reliability.

```bash
# Basic startup with improved GRPC
python start_component.py agentic-reasoner

# With Bayesian world model
python start_component.py agentic-reasoner --use-bayesian-world-model

# With custom timeout configuration
python start_component.py agentic-reasoner --timeout-config robust

# With performance monitoring
python start_component.py agentic-reasoner --monitor-performance
```

**Timeout Configurations:**
- `default`: Balanced timeouts for most use cases
- `fast`: Quick timeouts for low-latency environments
- `robust`: Extended timeouts for high-reliability scenarios
- `custom`: Predefined custom timeout values

**Key Features:**
- Improved GRPC client with circuit breaker
- Automatic retry with exponential backoff
- Performance monitoring and metrics
- Optional Bayesian world model integration
- Autonomous tool usage (Knowledge Base, Digital Twin)

#### Meta-Learner
Learns and adapts reasoning strategies over time.

```bash
# Basic startup
python start_component.py meta-learner

# With custom update interval
python start_component.py meta-learner --config meta_learner_config.yaml
```

## Advanced Usage

### Complete Framework Startup

```bash
# Start all components (development only)
python start_component.py all --plugin-dir extern

# Exclude specific components
python start_component.py all --plugin-dir extern --exclude-components meta-learner verification

# Custom startup order
python start_component.py all --plugin-dir extern --start-order digital-twin kernel agentic-reasoner monitor

# Validate all configurations
python start_component.py all --plugin-dir extern --validate-only
```

### Configuration Management

```bash
# Use specific configuration file
python start_component.py digital-twin --config config/production_config.yaml

# Validate configuration only
python start_component.py digital-twin --validate-only

# Dry run (initialize but don't start)
python start_component.py digital-twin --dry-run
```

### Performance and Monitoring

```bash
# Enable performance monitoring
python start_component.py agentic-reasoner --monitor-performance

# Enable profiling (development)
python start_component.py agentic-reasoner --enable-profiling

# Debug logging
python start_component.py digital-twin --log-level DEBUG
```

## Production Deployment

### Recommended Startup Order

1. **Infrastructure Services**
   ```bash
   python start_component.py knowledge-base
   python start_component.py digital-twin --world-model bayesian
   python start_component.py kernel
   ```

2. **System Adapters**
   ```bash
   python start_component.py monitor --plugin-dir extern
   python start_component.py execution --plugin-dir extern
   python start_component.py verification --plugin-dir extern
   ```

3. **Reasoning Agents**
   ```bash
   python start_component.py agentic-reasoner --use-bayesian-world-model --timeout-config robust
   python start_component.py meta-learner
   ```

### Environment Setup

#### Required Environment Variables

```bash
# For Gemini-based components
export GEMINI_API_KEY="your-gemini-api-key"

# For NATS connection
export NATS_URL="nats://localhost:4222"

# For database connections
export DB_CONNECTION_STRING="postgresql://localhost:5432/polaris"
```

#### Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Bayesian world model dependencies
pip install -r requirements_bayesian.txt

# Additional packages for specific components
pip install google-genai langchain  # For Gemini world model
```

### Configuration Files

#### Digital Twin with Bayesian World Model
```yaml
# config/bayesian_world_model_config.yaml
world_model:
  implementation: "bayesian"
  config:
    prediction_horizon_minutes: 120
    correlation_threshold: 0.7
    anomaly_threshold: 2.5
    process_noise: 0.01
    measurement_noise: 0.1
```

#### Agentic Reasoner with Improved GRPC
```yaml
# config/agentic_reasoner_config.yaml
reasoner:
  digital_twin_client:
    type: "improved_grpc"
    query_timeout: 30.0
    simulation_timeout: 120.0
    diagnosis_timeout: 60.0
    max_retries: 3
    circuit_breaker_enabled: true
```

## Troubleshooting

### Common Issues

#### GRPC Connection Errors
```bash
# Check if Digital Twin is running
python start_component.py digital-twin --health-check

# Use robust timeout configuration
python start_component.py agentic-reasoner --timeout-config robust
```

#### Environment Validation Failures
```bash
# Validate environment before starting
python start_component.py digital-twin --validate-only

# Check specific component requirements
python start_component.py agentic-reasoner --validate-only
```

#### Performance Issues
```bash
# Enable performance monitoring
python start_component.py agentic-reasoner --monitor-performance

# Use Bayesian world model for faster predictions
python start_component.py agentic-reasoner --use-bayesian-world-model
```

### Health Checks

```bash
# Digital Twin health
python start_component.py digital-twin --health-check

# Component validation
python start_component.py agentic-reasoner --validate-only

# Complete framework validation
python start_component.py all --plugin-dir extern --validate-only
```

### Monitoring Commands

```bash
# Check GRPC metrics (while agentic reasoner is running)
# Metrics are logged every 30 seconds with --monitor-performance

# Check world model health
# Health status is reported in component logs

# System resource usage
htop  # or your preferred system monitor
```

## Best Practices

### Development
- Use `--validate-only` to check configurations before deployment
- Use `--dry-run` to test component initialization
- Enable `--monitor-performance` for debugging
- Use `--log-level DEBUG` for detailed troubleshooting

### Production
- Start components in separate processes/containers
- Use appropriate timeout configurations for your environment
- Monitor component health and performance metrics
- Set up proper logging and alerting
- Use Bayesian world model for deterministic, fast predictions
- Configure circuit breaker thresholds based on SLA requirements

### Configuration Management
- Use environment-specific configuration files
- Validate configurations before deployment
- Keep sensitive data (API keys) in environment variables
- Document configuration changes and their impact

## Examples by Use Case

### High-Performance Setup
```bash
# Fast, deterministic predictions
python start_component.py digital-twin --world-model bayesian
python start_component.py agentic-reasoner --use-bayesian-world-model --timeout-config fast
```

### High-Reliability Setup
```bash
# Robust configuration with extended timeouts
python start_component.py digital-twin --world-model gemini
python start_component.py agentic-reasoner --timeout-config robust --monitor-performance
```

### Development/Testing Setup
```bash
# Mock components for testing
python start_component.py digital-twin --world-model mock
python start_component.py agentic-reasoner --dry-run
python start_component.py all --plugin-dir extern --validate-only
```

### Hybrid Setup
```bash
# Combine LLM creativity with statistical rigor
python start_component.py digital-twin --world-model bayesian
python start_component.py agentic-reasoner --use-improved-grpc --monitor-performance
python start_component.py reasoner --reasoning-mode hybrid
```