#!/bin/bash

# SWIM Startup Helper Script
# This script pulls the SWIM Docker image and starts a simulation.
# Usage: ./start_swim.sh

set -e  # Exit on error

echo "🐳 SWIM Startup Script"
echo "====================="
echo ""

# Step 1: Pull the Docker image
echo "📦 Pulling SWIM Docker image (vcnk4v/polaris-swim:trimmed-traces)..."
docker pull vcnk4v/polaris-swim:trimmed-traces
echo "✓ Image pulled successfully"
echo ""

# Step 2: Start the container
echo "🚀 Starting SWIM container..."
docker run -it --rm vcnk4v/polaris-swim:trimmed-traces bash -c "\
  cd ~/seams-swim/swim/simulations/swim/ && \
  echo '✓ Container started. Running simulation...' && \
  echo '' && \
  ./run.sh sim 1
"

echo ""
echo "✓ SWIM simulation complete!"
