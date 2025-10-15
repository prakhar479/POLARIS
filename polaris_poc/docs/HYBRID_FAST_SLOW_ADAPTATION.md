# Hybrid Fast + Slow Controller Adaptation Guide

## 🎯 **Overview**

POLARIS implements a sophisticated hybrid adaptation system with two complementary controllers:

- **Fast Controller**: Reactive, immediate responses to performance issues
- **Slow Controller**: Deliberative, reasoner-based decisions for complex scenarios
- **Controller Strategy**: Intelligent switching between fast and slow based on system conditions

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Telemetry Data                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Controller Strategy                          │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Condition Check │    │    Decision Logic               │ │
│  │ CPU > 80%?      │───▶│ Fast: Immediate reaction        │ │
│  │ Complex issue?  │    │ Slow: Reasoner delegation       │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────┬───────────────────┬───────────────────┘
                      │                   │
                      ▼                   ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│        Fast Controller          │ │        Slow Controller          │
│  • Immediate actions            │ │  • Delegates to Reasoner        │
│  • ADD_SERVER                   │ │  • Complex analysis             │
│  • SET_DIMMER                   │ │  • Long-term optimization       │
│  • Threshold-based              │ │  • AI-powered decisions         │
└─────────────────────────────────┘ └─────────────────────────────────┘
                      │                   │
                      ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 Execution System                            │
│  • Verification Adapter (validates actions)                │
│  • Execution Adapter (applies actions)                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 **How to Run Hybrid Fast + Slow Adaptation**

### **Step 1: Start Core Infrastructure**

#### **Terminal 1: Start Digital Twin (for Slow Controller)**
```bash
python src/scripts/start_component.py digital-twin --world-model bayesian
```

#### **Terminal 2: Start Knowledge Base (for Slow Controller)**
```bash
python src/scripts/start_component.py knowledge-base
```

#### **Terminal 3: Start Agentic Reasoner (Slow Controller Backend)**
```bash
python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model --monitor-performance
```

### **Step 2: Start System Adapters**

#### **Terminal 4: Start Monitor Adapter**
```bash
python src/scripts/start_component.py monitor --plugin-dir extern
```

#### **Terminal 5: Start Execution Adapter**
```bash
python src/scripts/start_component.py execution --plugin-dir extern
```

#### **Terminal 6: Start Verification Adapter**
```bash
python src/scripts/start_component.py verification --plugin-dir extern
```

### **Step 3: Start the Hybrid Kernel**

#### **Terminal 7: Start SWIM Kernel (Hybrid Controller)**
```bash
python src/scripts/start_component.py kernel
```

## 🔄 **How the Hybrid System Works**

### **1. Controller Selection Logic**
```python
# In ControllerStrategy.select_controller()
cpu = telemetry_data.get("cpu", 0)

if cpu > 80:
    # High CPU = Fast reaction needed
    return FastController()
else:
    # Normal conditions = Deliberative approach
    return SlowController()
```

### **2. Fast Controller Path (Immediate Actions)**
```
Telemetry → Fast Controller → Immediate Decision → Verification → Execution
```

**Fast Controller Actions:**
- **High Response Time**: ADD_SERVER or SET_DIMMER
- **Low Utilization**: REMOVE_SERVER or INCREASE_DIMMER
- **Threshold-based**: Simple, fast decisions

**Example Fast Controller Flow:**
```
📊 Telemetry: Response time = 0.85s (threshold: 0.75s)
⚡ Fast Controller: "Response time breach detected"
✅ Decision: ADD_SERVER (immediate)
🔍 Verification: "Action approved - within server limits"
🚀 Execution: New server added to pool
```

### **3. Slow Controller Path (Reasoner Delegation)**
```
Telemetry → Slow Controller → Reasoner → AI Analysis → Complex Decision → Execution
```

**Slow Controller Process:**
- Delegates to Agentic Reasoner via NATS
- Reasoner uses Digital Twin for analysis
- AI-powered decision making
- Complex scenario handling

**Example Slow Controller Flow:**
```
📊 Telemetry: CPU = 65%, Memory trending up, Complex pattern
🧠 Slow Controller: "Delegating to reasoner for analysis"
🤖 Agentic Reasoner: 
   - Queries Digital Twin for predictions
   - Analyzes correlations and trends
   - Simulates multiple action scenarios
✅ Decision: OPTIMIZE_MEMORY_ALLOCATION (complex)
🚀 Execution: Memory optimization applied
```

## 📊 **Controller Selection Criteria**

### **Fast Controller Triggers:**
- **CPU > 80%**: Immediate scaling needed
- **Response time > threshold**: Performance breach
- **Simple threshold violations**: Clear, immediate actions
- **Emergency conditions**: Fast reaction required

### **Slow Controller Triggers:**
- **CPU ≤ 80%**: Normal conditions, time for analysis
- **Complex patterns**: Multi-metric correlations
- **Optimization opportunities**: Long-term improvements
- **Uncertain scenarios**: Need AI analysis

## 🔧 **Configuration Options**

### **Customize Controller Thresholds**
Create `config/hybrid_controller_config.yaml`:
```yaml
controller_strategy:
  fast_controller_triggers:
    cpu_threshold: 80.0
    response_time_threshold: 0.75
    memory_threshold: 85.0
    emergency_mode: true
  
  slow_controller_conditions:
    analysis_timeout: 30.0
    complexity_threshold: 0.7
    optimization_mode: true
  
  switching_logic:
    hysteresis: 5.0  # Prevent rapid switching
    cooldown_period: 60.0  # Seconds between switches

fast_controller:
  thresholds:
    RT_THRESHOLD: 0.75
    DIMMER_STEP: 0.1
    MAX_SERVERS: 10
    MIN_SERVERS: 1
  
  actions:
    server_scaling: true
    dimmer_control: true
    immediate_mode: true

slow_controller:
  reasoner_integration:
    nats_subject: "polaris.reasoner.kernel.requests"
    timeout: 45.0
    fallback_to_fast: true
  
  analysis_depth:
    correlation_analysis: true
    predictive_modeling: true
    optimization_search: true
```

### **Start with Custom Configuration**
```bash
python src/scripts/start_component.py kernel --config config/hybrid_controller_config.yaml
```

## 📈 **Monitoring the Hybrid System**

### **Fast Controller Logs**
```
INFO - Selected controller: FastController
INFO - Fast action generated: ADD_SERVER
INFO - Action sent for verification: server-add-001
```

### **Slow Controller Logs**
```
INFO - Selected controller: SlowController
INFO - Delegated telemetry to Reasoner via polaris.reasoner.kernel.requests
INFO - Reasoner analysis in progress...
```

### **Controller Switching Logs**
```
INFO - Controller switch: FastController → SlowController (CPU: 75%)
INFO - Controller switch: SlowController → FastController (CPU: 85%)
```

## 🎯 **Example Scenarios**

### **Scenario 1: Sudden Load Spike (Fast Controller)**
```
1. Monitor: CPU jumps to 90%
2. Strategy: Selects FastController (CPU > 80%)
3. Fast Controller: Immediate ADD_SERVER decision
4. Verification: Approves action (within limits)
5. Execution: Server added in ~2 seconds
6. Result: CPU drops to 65%
```

### **Scenario 2: Gradual Performance Degradation (Slow Controller)**
```
1. Monitor: CPU at 70%, memory slowly increasing
2. Strategy: Selects SlowController (CPU ≤ 80%)
3. Slow Controller: Delegates to Agentic Reasoner
4. Reasoner: Analyzes patterns, correlations, predictions
5. Decision: OPTIMIZE_CACHE_SETTINGS (complex)
6. Execution: Cache optimization applied
7. Result: Memory stabilized, performance improved
```

### **Scenario 3: Hybrid Switching**
```
1. Normal operation: SlowController active (CPU: 65%)
2. Load spike: CPU → 85%, switches to FastController
3. Fast action: ADD_SERVER executed immediately
4. Load stabilizes: CPU → 70%, switches back to SlowController
5. Optimization: SlowController optimizes resource allocation
```

## 🚨 **Troubleshooting**

### **Controllers Not Switching**
1. **Check telemetry data:**
   ```bash
   # Look for CPU values in kernel logs
   INFO - Selected controller based on CPU: 75%
   ```

2. **Verify strategy logic:**
   ```python
   # In controller_strategy.py
   if cpu > 80:  # Fast controller threshold
       return self.controllers["fast"]
   ```

### **Fast Controller Not Acting**
1. **Check thresholds:**
   ```python
   # In fast_controller.py
   self.RT_THRESHOLD = 0.75  # Response time threshold
   ```

2. **Verify telemetry validation:**
   ```bash
   # Look for validation errors
   WARNING - Telemetry validation failed: missing swim.active.servers
   ```

### **Slow Controller Not Delegating**
1. **Check reasoner connection:**
   ```bash
   # Verify agentic reasoner is running
   ✅ Agentic Reasoner agent started successfully
   ```

2. **Monitor NATS messages:**
   ```bash
   # Look for delegation messages
   INFO - Delegated telemetry to Reasoner via polaris.reasoner.kernel.requests
   ```

## 🎯 **Performance Comparison**

| Aspect | Fast Controller | Slow Controller |
|--------|----------------|-----------------|
| **Response Time** | ~2 seconds | ~30 seconds |
| **Decision Quality** | Good for simple cases | Excellent for complex cases |
| **Resource Usage** | Low | Higher (AI analysis) |
| **Accuracy** | High for thresholds | Very high for patterns |
| **Adaptability** | Limited | Excellent |
| **Use Cases** | Emergency response | Optimization, complex issues |

## 🏁 **Quick Start Script**

Create `start_hybrid_adaptation.sh`:
```bash
#!/bin/bash

echo "🚀 Starting POLARIS Hybrid Fast + Slow Adaptation System"

# Start infrastructure (slow controller backend)
gnome-terminal -- bash -c "python src/scripts/start_component.py digital-twin --world-model bayesian; exec bash"
sleep 5

gnome-terminal -- bash -c "python src/scripts/start_component.py knowledge-base; exec bash"
sleep 3

gnome-terminal -- bash -c "python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model --monitor-performance; exec bash"
sleep 5

# Start system adapters
gnome-terminal -- bash -c "python src/scripts/start_component.py monitor --plugin-dir extern; exec bash"
sleep 3

gnome-terminal -- bash -c "python src/scripts/start_component.py execution --plugin-dir extern; exec bash"
sleep 3

gnome-terminal -- bash -c "python src/scripts/start_component.py verification --plugin-dir extern; exec bash"
sleep 3

# Start hybrid kernel
gnome-terminal -- bash -c "python src/scripts/start_component.py kernel; exec bash"

echo "✅ Hybrid adaptation system started!"
echo "📊 Monitor terminals for fast/slow controller switching"
echo "⚡ Fast controller: Immediate reactions (CPU > 80%)"
echo "🧠 Slow controller: AI-powered analysis (CPU ≤ 80%)"
```

## 🎉 **Benefits of Hybrid Approach**

1. **Best of Both Worlds**: Fast reactions + intelligent analysis
2. **Adaptive Strategy**: Switches based on system conditions
3. **Emergency Handling**: Fast controller for critical situations
4. **Optimization**: Slow controller for complex improvements
5. **Scalable**: Handles both simple and complex scenarios
6. **Reliable**: Verification ensures safe actions

The hybrid fast + slow controller system provides comprehensive adaptation capabilities, combining the speed of reactive control with the intelligence of AI-powered reasoning! 🚀