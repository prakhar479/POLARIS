#!/bin/bash
# SWIM POLARIS Adaptation System Startup Script

cd "/home/prakhar/dev/prakhar479/POLARIS/polaris_refactored/plugins/swim/polaris_ablation"

echo "Starting SWIM POLARIS Adaptation System..."

# Check if NATS server is running
if ! pgrep -x "nats-server" > /dev/null; then
    echo "Starting NATS server..."
    nats-server &
    sleep 2
fi

# Start the system
python3 scripts/start_system.py "$@"
