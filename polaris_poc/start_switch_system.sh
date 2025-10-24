#!/bin/bash

# POLARIS SWITCH System Startup Script
# ====================================
# This script starts all components of the POLARIS framework
# configured for the SWITCH ML-enabled adaptive system

echo "🚀 Starting POLARIS Framework for SWITCH System"
echo "================================================"

# Check if we're in the correct directory
if [ ! -f "src/scripts/start_component.py" ]; then
    echo "❌ Error: Please run this script from the polaris_poc directory"
    exit 1
fi

# Function to start component in background
start_component() {
    local component=$1
    local args="${@:2}"
    echo "🔄 Starting $component..."
    python src/scripts/start_component.py $component $args &
    local pid=$!
    echo "✅ $component started (PID: $pid)"
    sleep 2  # Give component time to initialize
}

# Start components in correct order
echo ""
echo "📊 Starting Monitor Adapter..."
start_component monitor --plugin-dir extern/switch_plugin --log-level=DEBUG

echo ""
echo "⚡ Starting Execution Adapter..."
start_component execution --plugin-dir extern/switch_plugin --log-level=DEBUG

echo ""
echo "🧠 Starting Knowledge Base Service..."
start_component knowledge-base --log-level=DEBUG

echo ""
echo "🔮 Starting Digital Twin with Bayesian World Model..."
start_component digital-twin --world-model bayesian --log-level=DEBUG

echo ""
echo "🎯 Starting SWITCH Kernel..."
python extern/switch_plugin/run_switch_kernel.py &
KERNEL_PID=$!
echo "✅ SWITCH Kernel started (PID: $KERNEL_PID)"
sleep 2

echo ""
echo "🤖 Starting Agentic Reasoner..."
start_component agentic-reasoner --config config/switch_optimized_config.yaml

echo ""
echo "📚 Starting Meta-Learner..."
start_component meta-learner --config config/switch_optimized_config.yaml

echo ""
echo "✅ Starting Verification Adapter..."
start_component verification --config config/switch_verification_config.yaml --plugin-dir extern/switch_plugin

echo ""
echo "🎉 All POLARIS components started successfully!"
echo ""
echo "📋 System Status:"
echo "   - Monitor Adapter: Collecting SWITCH metrics every 5 seconds"
echo "   - Execution Adapter: Ready to execute model switching actions"
echo "   - Knowledge Base: Storing telemetry and adaptation history"
echo "   - Digital Twin: Bayesian world model for system prediction"
echo "   - SWITCH Kernel: Routing telemetry to fast/slow controllers"
echo "   - Agentic Reasoner: LLM-based optimization decisions"
echo "   - Meta-Learner: Learning from adaptation patterns"
echo "   - Verification: Validating actions before execution"
echo ""
echo "🌐 Key Endpoints:"
echo "   - Digital Twin gRPC: localhost:50051"
echo "   - NATS Message Bus: localhost:4222"
echo "   - SWITCH System API: localhost:3001"
echo ""
echo "📝 Logs are available in the logs/ directory"
echo "🛑 Press Ctrl+C to stop all components"

# Wait for interrupt signal
trap 'echo "🛑 Shutting down all components..."; kill $(jobs -p); exit' INT
wait