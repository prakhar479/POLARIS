# Agentic Reasoner Troubleshooting Guide

## 🚨 **Issue: "No implementation for reasoning type: ReasoningType.INFERENCE"**

### **Root Cause**
The agentic reasoner agent is created but the reasoning implementations aren't properly registered for all reasoning types, specifically `INFERENCE` which is used by the slow controller.

### **Why This Happens**
1. The slow controller delegates telemetry to the reasoner via NATS
2. The reasoner creates a `ReasoningContext` with `reasoning_type="inference"` (default)
3. The agentic reasoner doesn't have an implementation registered for `ReasoningType.INFERENCE`

## 🔧 **Quick Fix**

### **Step 1: Restart Agentic Reasoner**
Stop your current agentic reasoner and restart it:

```bash
# Stop current agentic reasoner (Ctrl+C)
# Then restart with:
python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model --monitor-performance
```

### **Step 2: Check Logs for Implementation Registration**
Look for these log messages during startup:
```
INFO - Added reasoning implementation for inference
INFO - Added reasoning implementation for planning  
INFO - Added reasoning implementation for analysis
INFO - Added reasoning implementation for decision
INFO - Added reasoning implementation for prediction
```

If you don't see these messages, the implementations aren't being added.

### **Step 3: Verify Setup**
Run the test script:
```bash
python examples/test_agentic_reasoner_setup.py
```

## 🔍 **Detailed Diagnosis**

### **Check Reasoning Implementations**
The agentic reasoner should register implementations for all reasoning types:

```python
# This should happen during startup
for reasoning_type in ReasoningType:
    agentic_reasoner = AgenticLLMReasoner(...)
    agent.add_reasoning_implementation(reasoning_type, agentic_reasoner)
```

### **Verify ReasoningType Enum**
Make sure there are no conflicting ReasoningType definitions:

```python
# From reasoner_core.py
class ReasoningType(Enum):
    INFERENCE = "inference"
    PLANNING = "planning" 
    ANALYSIS = "analysis"
    DECISION = "decision"
    PREDICTION = "prediction"
```

## 🎯 **Complete Hybrid System Restart**

If the issue persists, restart the entire hybrid system:

### **Step 1: Stop All Components**
Stop all running components (Ctrl+C in each terminal)

### **Step 2: Start in Correct Order**
```bash
# Terminal 1: Digital Twin
python src/scripts/start_component.py digital-twin --world-model bayesian

# Terminal 2: Knowledge Base  
python src/scripts/start_component.py knowledge-base

# Terminal 3: Agentic Reasoner (wait for this to fully start)
python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model --monitor-performance

# Terminal 4: Monitor
python src/scripts/start_component.py monitor --plugin-dir extern

# Terminal 5: Execution
python src/scripts/start_component.py execution --plugin-dir extern

# Terminal 6: Verification
python src/scripts/start_component.py verification --plugin-dir extern

# Terminal 7: Kernel (start this last)
python src/scripts/start_component.py kernel
```

### **Step 3: Verify Agentic Reasoner Startup**
In the agentic reasoner logs, you should see:
```
✅ Agentic Reasoner agent started successfully
🔧 Available tools: Knowledge Base queries, Digital Twin interactions
INFO - Added reasoning implementation for inference
INFO - Added reasoning implementation for planning
INFO - Added reasoning implementation for analysis
INFO - Added reasoning implementation for decision
INFO - Added reasoning implementation for prediction
```

### **Step 4: Test Slow Controller**
Trigger the slow controller by ensuring CPU < 80% in your system, then check logs:

**Kernel logs:**
```
INFO - Selected controller: SlowController
INFO - Delegated telemetry to Reasoner via polaris.reasoner.kernel.requests
```

**Agentic Reasoner logs:**
```
INFO - Starting reasoning session [session-id] with input: [telemetry-data]
INFO - Using inference reasoning implementation
INFO - Reasoning session [session-id] completed with result: [action]
```

## 🚨 **Alternative Workaround**

If the issue persists, you can modify the kernel to specify a different reasoning type:

### **Option 1: Use DECISION Instead of INFERENCE**
Modify the kernel request to use DECISION reasoning:

```python
# In kernel.py, when delegating to reasoner
telemetry_data["reasoning_type"] = "decision"  # Instead of default "inference"
await self.nats_client.publish(
    "polaris.reasoner.kernel.requests", 
    json.dumps(telemetry_data).encode()
)
```

### **Option 2: Use Regular Agentic Reasoner**
Instead of the Bayesian version, use the regular agentic reasoner:

```bash
python src/scripts/start_component.py agentic-reasoner --timeout-config robust --monitor-performance
```

## 📊 **Expected Behavior After Fix**

### **Slow Controller Flow**
```
1. Telemetry: CPU = 70% (< 80%)
2. Kernel: Selects SlowController
3. SlowController: Delegates to agentic reasoner
4. Agentic Reasoner: Uses INFERENCE implementation
5. Reasoning: Analyzes telemetry with AI tools
6. Decision: Generates appropriate action
7. Execution: Action is executed on system
```

### **Success Logs**
```
# Kernel
INFO - Selected controller: SlowController
INFO - Delegated telemetry to Reasoner

# Agentic Reasoner  
INFO - Received kernel telemetry request
INFO - Starting reasoning session with INFERENCE
INFO - Using inference reasoning implementation
INFO - Tool Call: Query Digital Twin
INFO - Tool Call: Query Knowledge Base
INFO - Reasoning completed successfully
INFO - Publishing action to execution layer
```

## 🎯 **Prevention**

To prevent this issue in the future:

1. **Always check startup logs** for reasoning implementation registration
2. **Use the test script** to verify setup before running the hybrid system
3. **Start components in the correct order** (reasoner before kernel)
4. **Monitor logs** for any initialization errors

The fix I've implemented adds logging to show when reasoning implementations are registered, making it easier to diagnose this issue in the future.