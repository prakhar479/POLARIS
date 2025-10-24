#!/bin/bash

# Start Switch Monitor Adapter for POLARIS
# ========================================
# This script starts the Switch system monitor adapter that collects
# comprehensive metrics and publishes them to NATS for POLARIS processing.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/switch_optimized_config.yaml"
PLUGIN_DIR="${SCRIPT_DIR}/extern/switch_plugin"
LOG_LEVEL="INFO"
PYTHON_CMD="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check config file
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Check plugin directory
    if [ ! -d "$PLUGIN_DIR" ]; then
        print_error "Plugin directory not found: $PLUGIN_DIR"
        exit 1
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p "${SCRIPT_DIR}/logs"
    
    print_success "Prerequisites check passed"
}

# Function to check if NATS is running
check_nats() {
    print_status "Checking NATS server..."
    
    if ! nc -z localhost 4222 2>/dev/null; then
        print_warning "NATS server is not running on localhost:4222"
        print_warning "Please start NATS server before running the monitor"
        print_warning "You can start NATS with: nats-server"
        return 1
    fi
    
    print_success "NATS server is running"
    return 0
}

# Function to check if Switch system is running
check_switch_system() {
    print_status "Checking Switch system..."
    
    if ! nc -z localhost 3001 2>/dev/null; then
        print_warning "Switch system is not running on localhost:3001"
        print_warning "Please start the Switch system before running the monitor"
        return 1
    fi
    
    print_success "Switch system is running"
    return 0
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE     Path to POLARIS configuration file (default: $CONFIG_FILE)"
    echo "  --plugin-dir DIR  Path to Switch plugin directory (default: $PLUGIN_DIR)"
    echo "  --log-level LEVEL Logging level: DEBUG, INFO, WARNING, ERROR (default: $LOG_LEVEL)"
    echo "  --status          Show service status and exit"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Start with default settings"
    echo "  $0 --log-level DEBUG                 # Start with debug logging"
    echo "  $0 --config custom_config.yaml       # Use custom configuration"
    echo "  $0 --status                          # Check service status"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --plugin-dir)
            PLUGIN_DIR="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Change to script directory
cd "$SCRIPT_DIR"

# Check prerequisites
check_prerequisites

# If status only, run status check and exit
if [ "$STATUS_ONLY" = true ]; then
    print_status "Checking Switch Monitor Service status..."
    $PYTHON_CMD start_switch_monitor.py --status --config "$CONFIG_FILE" --plugin-dir "$PLUGIN_DIR"
    exit 0
fi

# Check dependencies
if ! check_nats; then
    print_error "NATS server is required but not running"
    exit 1
fi

if ! check_switch_system; then
    print_error "Switch system is required but not running"
    exit 1
fi

# Display startup information
echo ""
print_status "Starting Switch Monitor Adapter"
print_status "================================"
print_status "Configuration: $CONFIG_FILE"
print_status "Plugin Directory: $PLUGIN_DIR"
print_status "Log Level: $LOG_LEVEL"
print_status "Logs Directory: ${SCRIPT_DIR}/logs"
echo ""

# Start the monitor adapter
print_status "Starting monitor adapter..."
exec $PYTHON_CMD start_switch_monitor.py \
    --config "$CONFIG_FILE" \
    --plugin-dir "$PLUGIN_DIR" \
    --log-level "$LOG_LEVEL"