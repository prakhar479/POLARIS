# Bayesian Agentic Reasoner Setup Guide

## 🎯 Quick Setup: Agentic Reasoner + Bayesian Digital Twin

This guide shows you how to set up an agentic reasoner that uses a Bayesian/Kalman filter digital twin for deterministic, statistical predictions.

## 🚀 Method 1: Automatic Setup (Recommended)

### **Single Command Setup**
```bash
# This automatically configures everything for you
python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model
```

**What this does:**
- Creates a temporary configuration with Bayesian world model settings
- Starts the agentic reasoner with improved GRPC client
- Connects to a Digital Twin that uses Bayesian/Kalman filtering
- Provides deterministic predictions with confidence intervals

### **With Performance Monitoring**
```bash
# Add performance monitoring for production use
python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model --monitor-performance
```

## 🔧 Method 2: Manual Setup (Full Control)

### **Step 1: Start Digital Twin with Bayesian World Model**
```bash
# Terminal 1: Start Digital Twin
python src/scripts/start_component.py digital-twin --world-model bayesian
```

**Expected output:**
```
✅ Digital Twin agent started successfully
🌐 gRPC service available at 0.0.0.0:50051
📊 World Model Health: Status: healthy
🧮 Using deterministic Bayesian world model for predictions
```

### **Step 2: Start Agentic Reasoner**
```bash
# Terminal 2: Start Agentic Reasoner
python src/scripts/start_component.py agentic-reasoner --timeout-config robust
```

**Expected output:**
```
✅ Agentic Reasoner agent started successfully
🔧 Available tools: Knowledge Base queries, Digital Twin interactions
🔧 Using improved GRPC client with circuit breaker
📡 Agentic Reasoner is running - press Ctrl+C to stop
```

### **Step 3: Optional - Start Monitor for Data**
```bash
# Terminal 3: Feed telemetry data
python src/scripts/start_component.py monitor --plugin-dir extern
```

## 📊 What You Get

### **Bayesian Digital Twin Features:**
- **Deterministic predictions**: Consistent, reproducible results
- **Statistical confidence**: All predictions include uncertainty bounds  
- **Real-time learning**: Adapts based on actual system behavior
- **Correlation discovery**: Automatically finds metric relationships
- **Anomaly detection**: Statistical threshold-based detection
- **10x faster** than LLM-based predictions

### **Agentic Reasoner Features:**
- **Autonomous tool usage**: Automatically queries Digital Twin when needed
- **LLM reasoning**: Creative problem-solving and context understanding
- **Improved reliability**: Circuit breaker and retry logic for GRPC calls
- **Performance monitoring**: Real-time metrics on tool usage

## 🔍 Verification Steps

### **1. Check Digital Twin Health**
```bash
python src/scripts/start_component.py digital-twin --world-model bayesian --health-check
```

**Expected output:**
```
✅ World Model health: healthy
✅ NATS connection is healthy  
✅ Health check passed
```

### **2. Check Agentic Reasoner Connection**
Look for these log messages:
```
✅ Agentic Reasoner agent started successfully
🔧 Digital Twin client: ImprovedGRPCDigitalTwinClient at localhost:50051
📊 GRPC Performance Metrics: Success Rate: 100%
```

### **3. Monitor Performance (if enabled)**
```
📊 GRPC Performance Metrics:
   Success Rate: 98.50%
   Avg Response Time: 0.245s
   Circuit Breaker: closed

📊 World Model Health:
   Status: healthy
   Health Score: 0.95
   Metrics Tracked: 4
   Correlations Found: 2
```

## 🎯 Example Usage Flow

Once both components are running, here's how the agentic reasoner will use the Bayesian digital twin:

### **1. System Analysis**
```
🤖 Agentic Reasoner: "Analyzing system with CPU at 85%"

🔧 Tool Call: Query Digital Twin
   Request: "Get current system state with statistical analysis"
   
📊 Bayesian Response:
   - CPU: 85.2% ± 2.1% (95% confidence)
   - Trend: +1.5%/min
   - Correlation with memory: 0.73
   - Anomaly score: 1.8 (normal)
```

### **2. Predictive Simulation**
```
🔧 Tool Call: Simulate Action
   Request: "Predict impact of adding 1 server"
   
📊 Bayesian Prediction:
   - CPU in 5 min: 68.4% ± 3.2%
   - Memory impact: -12% ± 2%
   - Confidence: 0.89
   - Risk assessment: Low
```

### **3. Decision Making**
```
✅ Decision: ADD_SERVER
   Reasoning: "Statistical analysis shows 89% confidence that adding 
   a server will reduce CPU to 68% within 5 minutes with low risk"
```

## 🛠️ Configuration Options

### **Bayesian World Model Tuning**
Create `config/bayesian_world_model_config.yaml`:
```yaml
world_model:
  implementation: "bayesian"
  config:
    prediction_horizon_minutes: 120    # How far to predict
    correlation_threshold: 0.7         # Correlation significance
    anomaly_threshold: 2.5            # Anomaly detection sensitivity
    process_noise: 0.01               # Kalman filter process noise
    measurement_noise: 0.1            # Kalman filter measurement noise
    learning_rate: 0.05               # Adaptation speed
```

### **GRPC Client Tuning**
```yaml
reasoner:
  digital_twin_client:
    type: "improved_grpc"
    query_timeout: 30.0
    simulation_timeout: 120.0
    diagnosis_timeout: 60.0
    max_retries: 3
    circuit_breaker_enabled: true
    failure_threshold: 5
    recovery_timeout: 60.0
```

## 🚨 Troubleshooting

### **"Failed to create Agentic Reasoner" Error**
This was fixed in the latest version. Make sure you're using the updated code.

### **"Connection refused" Error**
```bash
# Make sure Digital Twin is running first
python src/scripts/start_component.py digital-twin --world-model bayesian --health-check
```

### **"API key required" Error**
```bash
# Set your Gemini API key (for the LLM reasoning part)
export GEMINI_API_KEY="your-api-key-here"
```

### **Performance Issues**
```bash
# Use robust timeout configuration
python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model --timeout-config robust
```

## 📈 Performance Comparison

| Feature | Gemini World Model | Bayesian World Model |
|---------|-------------------|----------------------|
| **Speed** | 2-5 seconds | 0.1-0.5 seconds |
| **Consistency** | Variable | Deterministic |
| **Confidence** | Subjective | Statistical (95% CI) |
| **Learning** | Context-based | Mathematical |
| **Predictions** | Creative | Rigorous |
| **Resource Usage** | High (API calls) | Low (local computation) |

## 🎯 Best Practices

### **Development**
- Use `--use-bayesian-world-model` for quick setup
- Enable `--monitor-performance` for debugging
- Start with default settings, tune later

### **Production**
- Use separate processes for Digital Twin and Agentic Reasoner
- Configure appropriate timeouts for your environment
- Monitor GRPC metrics and world model health
- Set up proper logging and alerting

### **Tuning**
- Adjust `correlation_threshold` based on your metrics
- Tune `anomaly_threshold` to reduce false positives
- Modify `prediction_horizon_minutes` for your use case
- Optimize `learning_rate` for adaptation speed

## 🏁 Summary

The Bayesian Agentic Reasoner setup gives you:

1. **Fast, deterministic predictions** from the Bayesian world model
2. **Creative reasoning** from the LLM-based agentic reasoner  
3. **Reliable communication** via improved GRPC client
4. **Statistical rigor** with confidence intervals and correlation analysis
5. **Real-time adaptation** based on actual system behavior

This combination provides the best of both worlds: mathematical precision and intelligent reasoning.