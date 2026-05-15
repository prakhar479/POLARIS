#!/bin/bash

# SWIM Startup Helper Script
# This script pulls the SWIM Docker image, runs a simulation,
# and automatically extracts metrics to display utility values.
# Usage: ./start_swim.sh

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🐳 SWIM Startup Script"
echo "====================="
echo ""

# Step 1: Pull the Docker image
echo "📦 Pulling SWIM Docker image (vcnk4v/polaris-swim:trimmed-traces)..."
docker pull vcnk4v/polaris-swim:trimmed-traces
echo "✓ Image pulled successfully"
echo ""

# Step 2: Start the container (detached mode for better control)
echo "🚀 Starting SWIM container..."
CONTAINER_NAME="polaris-swim-$$"
docker run -d --name "$CONTAINER_NAME" -p 4242:4242 \
  vcnk4v/polaris-swim:trimmed-traces bash -c "\
    cd ~/seams-swim/swim/simulations/swim/ && \
    echo '✓ Container started. Running simulation...' && \
    echo '' && \
    ./run.sh sim 1
  "

echo "✓ Container started: $CONTAINER_NAME"
echo "  - Port 4242 mapped for POLARIS connection"
echo "  - Logs: docker logs -f $CONTAINER_NAME"
echo ""

# Step 3: Wait for simulation to complete
echo "⏱️  Waiting for SWIM simulation to complete (~10 minutes)..."
echo "   (This is a trimmed trace - simulation runs for approximately 10 minutes)"
echo ""

# Monitor progress
while docker ps | grep -q "$CONTAINER_NAME"; do
    # Show progress indicator
    echo -n "."
    sleep 30

    # Show dimmer changes from logs (if any)
    docker logs "$CONTAINER_NAME" 2>&1 | grep -E "setDimmer|executing" | tail -3 || true
done

echo ""
echo ""

# Step 4: Check if container exited
echo "🔍 Checking simulation status..."
EXIT_CODE=$(docker inspect "$CONTAINER_NAME" --format='{{.State.ExitCode}}' 2>/dev/null || echo "1")

if [ "$EXIT_CODE" -eq 0 ] || [ "$EXIT_CODE" -eq 137 ] || [ "$EXIT_CODE" -eq 143 ]; then
    echo "✓ SWIM simulation completed"
else
    echo "⚠️  SWIM container exited with code $EXIT_CODE"
fi

# Step 5: Cleanup container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "========================================"
echo ""

# Step 6: Extract and display metrics
echo "📊 Extracting metrics from POLARIS logs..."
echo ""

if [ -f "$SCRIPT_DIR/scripts/extract_swim_metrics.py" ]; then
    cd "$SCRIPT_DIR"
    python3 scripts/extract_swim_metrics.py 2>/dev/null || echo "⚠️  Could not extract metrics - ensure POLARIS was running during simulation"
else
    echo "⚠️  Metrics extraction script not found at: $SCRIPT_DIR/scripts/extract_swim_metrics.py"
fi

echo ""
echo "========================================"
echo ""
echo "📁 Output files:"
echo "   - Plot: $SCRIPT_DIR/swim_metrics_plot.png"
echo "   - Summary: $SCRIPT_DIR/swim_experiment_summary.json"
echo "   - POLARIS Logs: $SCRIPT_DIR/logs/swim_polaris_run_*.log"
echo ""
