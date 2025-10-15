#!/bin/bash

# POLARIS SWIM System Startup Script
# ===================================
# Starts all 9 POLARIS components in an organized tmux environment
# optimized for SWIM system with Gemini Pro model and Bayesian world model.
#
# Components started in dependency order:
# 1. NATS Server (message bus infrastructure)
# 2. Knowledge Base (data storage and retrieval)
# 3. Digital Twin (core reasoning with Bayesian world model)
# 4. Verification Adapter (safety and policy enforcement)
# 5. Kernel (coordination and action routing)
# 6. Monitor Adapter (SWIM telemetry collection)
# 7. Execution Adapter (SWIM action execution)
# 8. Agentic Reasoner (AI reasoning with Gemini Pro)
# 9. Meta Learner (learning and adaptation optimization)

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
TMUX_SESSION="polaris-swim"
VENV_PATH="$PROJECT_ROOT/../.venv"  # Virtual environment path
PLUGIN_DIR="extern"
NATS_PORT="4222"
STARTUP_DELAY=5          # Seconds between component starts
HEALTH_CHECK_TIMEOUT=45  # Seconds to wait for health checks
SWIM_HOST="localhost"
SWIM_PORT="4242"

# Initialize variables
CHECK_HEALTH=false
SHOW_LOGS=""
VALIDATE_FIRST=false
DRY_RUN=false
KILL_SESSION=false
LIST_COMPONENTS=false

# SWIM-optimized configuration
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"  # Set your Gemini API key

# --- Color output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Helper functions ---
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_component() {
    echo -e "${CYAN}[COMPONENT]${NC} $1"
}

log_step() {
    echo -e "${MAGENTA}[STEP]${NC} $1"
}

log_header() {
    echo -e "${BOLD}${CYAN}$1${NC}"
}

# Configuration paths - use SWIM-optimized config if available
if [[ -f "$PROJECT_ROOT/config/swim_optimized_config.yaml" ]]; then
    export POLARIS_CONFIG_PATH="$PROJECT_ROOT/config/swim_optimized_config.yaml"
    log_info "Using SWIM-optimized configuration"
else
    export POLARIS_CONFIG_PATH="$PROJECT_ROOT/src/config/polaris_config.yaml"
    log_info "Using default configuration"
fi

export WORLD_MODEL_CONFIG="bayesian_world_model_config.yaml"
export PLUGIN_CONFIG_PATH="$PROJECT_ROOT/extern/config.yaml"

# --- Helper function to create error report ---
create_error_report() {
    local report_file="$PROJECT_ROOT/logs/startup_error_report.txt"
    
    log_info "Creating comprehensive error report: $report_file"
    
    # Ensure logs directory exists
    mkdir -p "$PROJECT_ROOT/logs"
    
    {
        echo "POLARIS SWIM System Startup Error Report"
        echo "========================================"
        echo "Generated: $(date)"
        echo "Session: $TMUX_SESSION"
        echo "SWIM: $SWIM_HOST:$SWIM_PORT"
        echo "NATS: localhost:$NATS_PORT"
        echo ""
        
        echo "System Information:"
        echo "-------------------"
        echo "OS: $(uname -a)"
        echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
        echo "Virtual Environment: $VENV_PATH"
        echo "Project Root: $PROJECT_ROOT"
        echo ""
        
        echo "Environment Variables:"
        echo "----------------------"
        echo "GEMINI_API_KEY: ${GEMINI_API_KEY:+SET}" # Don't expose the actual key
        echo "PYTHONPATH: ${PYTHONPATH:-Not set}"
        echo ""
        
        echo "Component Status:"
        echo "-----------------"
        local components=("nats-server" "knowledge-base" "digital-twin" "verification" "kernel" "monitor" "execution" "reasoner" "meta-learner")
        
        for component in "${components[@]}"; do
            echo "=== $component ==="
            local exit_code_file="$PROJECT_ROOT/logs/${component}_exit_code"
            local log_file="$PROJECT_ROOT/logs/${component}.log"
            
            if [[ -f "$exit_code_file" ]]; then
                echo "Exit Code: $(cat "$exit_code_file" 2>/dev/null || echo 'unknown')"
            else
                echo "Exit Code: Not available"
            fi
            
            if [[ -f "$log_file" ]]; then
                echo "Log File Size: $(wc -l < "$log_file" 2>/dev/null || echo 0) lines"
                echo "Last 10 lines:"
                tail -n 10 "$log_file" 2>/dev/null | sed 's/^/  /'
            else
                echo "Log File: Not found"
            fi
            echo ""
        done
        
        echo "Network Connectivity:"
        echo "--------------------"
        echo "NATS Port $NATS_PORT: $(nc -z localhost $NATS_PORT && echo 'Open' || echo 'Closed')"
        echo "Digital Twin Port 50051: $(nc -z localhost 50051 && echo 'Open' || echo 'Closed')"
        echo "SWIM Port $SWIM_PORT: $(nc -z $SWIM_HOST $SWIM_PORT && echo 'Open' || echo 'Closed')"
        echo ""
        
        echo "Tmux Session Status:"
        echo "-------------------"
        if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            tmux list-windows -t "$TMUX_SESSION" -F "#{window_index}: #{window_name} - #{window_active}"
        else
            echo "Session not found"
        fi
        
    } > "$report_file"
    
    log_info "Error report saved to: $report_file"
}

# --- Helper function to show component errors ---
show_component_errors() {
    local component_name="$1"
    local log_file="$PROJECT_ROOT/logs/${component_name}.log"
    local exit_code_file="$PROJECT_ROOT/logs/${component_name}_exit_code"
    
    log_error "=== $component_name Error Details ==="
    
    # Show exit code if available
    if [[ -f "$exit_code_file" ]]; then
        local exit_code=$(cat "$exit_code_file" 2>/dev/null || echo "unknown")
        log_error "Exit code: $exit_code"
    fi
    
    # Show last 20 lines of log file
    if [[ -f "$log_file" ]]; then
        log_error "Last 20 lines of log:"
        echo -e "${RED}----------------------------------------${NC}"
        tail -n 20 "$log_file" | sed "s/^/${RED}| ${NC}/"
        echo -e "${RED}----------------------------------------${NC}"
        log_info "Full log available at: $log_file"
    else
        log_error "No log file found at: $log_file"
    fi
    
    # Show tmux pane content if session exists
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null && tmux list-windows -t "$TMUX_SESSION" | grep -q "$component_name"; then
        log_error "Current tmux pane output:"
        echo -e "${RED}----------------------------------------${NC}"
        tmux capture-pane -t "$TMUX_SESSION:$component_name" -p | tail -n 10 | sed "s/^/${RED}| ${NC}/"
        echo -e "${RED}----------------------------------------${NC}"
    fi
}

# --- Helper function to check all component health ---
check_component_health() {
    log_info "Checking component health..."
    local failed_components=()
    
    # List of components to check
    local components=("nats-server" "knowledge-base" "digital-twin" "verification" "kernel" "monitor" "execution" "reasoner" "meta-learner")
    
    for component in "${components[@]}"; do
        local exit_code_file="$PROJECT_ROOT/logs/${component}_exit_code"
        local log_file="$PROJECT_ROOT/logs/${component}.log"
        
        # Check if component has exited with error
        if [[ -f "$exit_code_file" ]]; then
            local exit_code=$(cat "$exit_code_file" 2>/dev/null || echo "")
            # Only consider it failed if it's a real error exit code
            # 130 = Ctrl+C, 143 = SIGTERM, 148 = SIGTSTP (normal shutdown signals)
            if [[ "$exit_code" != "" && "$exit_code" != "0" && "$exit_code" != "130" && "$exit_code" != "143" && "$exit_code" != "148" ]]; then
                failed_components+=("$component")
                log_error "$component failed with exit code $exit_code"
                continue
            elif [[ "$exit_code" == "130" || "$exit_code" == "143" || "$exit_code" == "148" ]]; then
                log_info "$component stopped by signal (exit code $exit_code) - this is normal"
            elif [[ "$exit_code" == "0" ]]; then
                log_success "$component exited cleanly"
            fi
        fi
        
        # Check for error patterns in logs
        if [[ -f "$log_file" ]]; then
            if grep -qi "error\|exception\|failed\|traceback" "$log_file" | tail -n 50 | grep -qi "error\|exception\|failed"; then
                log_warn "$component may have errors (check logs)"
            fi
        fi
    done
    
    if [[ ${#failed_components[@]} -gt 0 ]]; then
        log_error "Failed components detected: ${failed_components[*]}"
        for component in "${failed_components[@]}"; do
            show_component_errors "$component"
        done
        return 1
    fi
    
    log_success "All components appear healthy"
    return 0
}

# Trap to handle script interruption
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && "$DRY_RUN" != true ]]; then
        log_error "Script interrupted or failed with exit code: $exit_code"
        if [[ -d "$PROJECT_ROOT/logs" ]]; then
            create_error_report
            log_info "Error report created: $PROJECT_ROOT/logs/startup_error_report.txt"
        fi
    fi
}
trap cleanup_on_exit EXIT

# --- Usage function ---
show_usage() {
    cat << EOF
POLARIS SWIM System Startup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -s, --session NAME      Tmux session name (default: polaris-swim)
    -p, --swim-port PORT    SWIM TCP port (default: 4242)
    -n, --nats-port PORT    NATS server port (default: 4222)
    -d, --delay SECONDS     Startup delay between components (default: 5)
    -k, --kill              Kill existing session and exit
    -l, --list              List running components
    --check-health          Check component health and show errors
    --show-logs COMPONENT   Show logs for specific component
    --validate-first        Validate all components before starting
    --dry-run               Show commands without executing

Environment Variables:
    GEMINI_API_KEY          Required: Your Google Gemini API key
    SWIM_HOST               SWIM server host (default: localhost)
    
Examples:
    $0                      Start system with defaults
    $0 -k                   Kill existing session
    $0 -s my-session        Use custom session name
    $0 --dry-run            Preview commands

EOF
}

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -s|--session)
            TMUX_SESSION="$2"
            shift 2
            ;;
        -p|--swim-port)
            SWIM_PORT="$2"
            shift 2
            ;;
        -n|--nats-port)
            NATS_PORT="$2"
            shift 2
            ;;
        -d|--delay)
            STARTUP_DELAY="$2"
            shift 2
            ;;
        -k|--kill)
            KILL_SESSION=true
            shift
            ;;
        -l|--list)
            LIST_COMPONENTS=true
            shift
            ;;
        --check-health)
            CHECK_HEALTH=true
            shift
            ;;
        --show-logs)
            SHOW_LOGS="$2"
            shift 2
            ;;
        --validate-first)
            VALIDATE_FIRST=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# --- Kill existing session ---
if [[ "$KILL_SESSION" == true ]]; then
    log_info "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || log_warn "Session $TMUX_SESSION not found"
    exit 0
fi

# --- List components ---
if [[ "$LIST_COMPONENTS" == true ]]; then
    log_info "Listing components in session: $TMUX_SESSION"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux list-windows -t "$TMUX_SESSION" -F "#{window_index}: #{window_name} - #{window_active}"
    else
        log_warn "Session $TMUX_SESSION not found"
    fi
    exit 0
fi

# --- Check health ---
if [[ "$CHECK_HEALTH" == true ]]; then
    log_info "Checking system health..."
    if [[ ! -d "$PROJECT_ROOT/logs" ]]; then
        log_error "Logs directory not found. System may not have been started yet."
        exit 1
    fi
    check_component_health
    exit $?
fi

# --- Show logs for specific component ---
if [[ -n "$SHOW_LOGS" ]]; then
    log_info "Showing logs for component: $SHOW_LOGS"
    local log_file="$PROJECT_ROOT/logs/${SHOW_LOGS}.log"
    if [[ -f "$log_file" ]]; then
        echo "=== $SHOW_LOGS Log File ==="
        cat "$log_file"
        echo "=== End of Log ==="
    else
        log_error "Log file not found: $log_file"
        log_info "Available log files:"
        ls -la "$PROJECT_ROOT/logs/" 2>/dev/null || log_warn "No logs directory found"
    fi
    exit 0
fi

# --- Pre-flight checks ---
log_header "🚀 POLARIS SWIM System Startup"
log_info "Session: $TMUX_SESSION"
log_info "SWIM: $SWIM_HOST:$SWIM_PORT"
log_info "NATS: localhost:$NATS_PORT"
log_info "Virtual Environment: $VENV_PATH"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"
log_info "Logs directory: $PROJECT_ROOT/logs/"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    log_error "tmux is required but not installed"
    exit 1
fi

# Check virtual environment
if [[ ! -d "$VENV_PATH" ]]; then
    log_error "Virtual environment not found at: $VENV_PATH"
    log_info "Please create virtual environment: python -m venv $VENV_PATH"
    exit 1
fi

# Check Gemini API key
if [[ -z "$GEMINI_API_KEY" ]]; then
    log_error "GEMINI_API_KEY environment variable is required"
    log_info "Please set your Gemini API key: export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

# Check NATS server availability and suggest installation
check_nats_availability() {
    if command -v nats-server &> /dev/null; then
        log_success "nats-server found in PATH"
        return 0
    elif command -v docker &> /dev/null; then
        log_info "nats-server not found, but Docker is available"
        return 0
    else
        log_warn "nats-server not found and Docker not available"
        log_info "To install NATS server:"
        log_info "  Ubuntu/Debian: sudo apt-get install nats-server"
        log_info "  macOS: brew install nats-server"
        log_info "  Or install Docker: https://docs.docker.com/get-docker/"
        log_info "  Or use Go: go install github.com/nats-io/nats-server/v2@latest"
        return 1
    fi
}

check_nats_availability

# Check project structure
required_paths=(
    "$PROJECT_ROOT/src/polaris"
    "$PROJECT_ROOT/src/config"
    "$PROJECT_ROOT/extern/config.yaml"
    "$PROJECT_ROOT/src/scripts/start_component.py"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        log_error "Required path not found: $path"
        exit 1
    fi
done

# Check start_component.py is executable
if [[ ! -x "$PROJECT_ROOT/src/scripts/start_component.py" ]]; then
    log_info "Making start_component.py executable..."
    chmod +x "$PROJECT_ROOT/src/scripts/start_component.py"
fi

# Test start_component.py script
test_start_component() {
    log_info "Testing start_component.py script..."
    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"
    export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
    
    if python src/scripts/start_component.py help &>/dev/null; then
        log_success "start_component.py script is working"
        return 0
    else
        log_error "start_component.py script test failed"
        log_info "Try running: python src/scripts/start_component.py help"
        return 1
    fi
}

if [[ "$DRY_RUN" != true ]]; then
    test_start_component || exit 1
fi

# Validate all components if requested
validate_all_components() {
    log_step "Validating all POLARIS components..."
    local components=("knowledge-base" "digital-twin" "verification" "kernel" "monitor" "execution" "agentic-reasoner" "meta-learner")
    local failed_components=()
    
    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"
    export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
    
    for component in "${components[@]}"; do
        log_info "Validating $component..."
        
        local validation_cmd="python src/scripts/start_component.py $component --validate-only --config '$POLARIS_CONFIG_PATH'"
        
        # Add plugin-dir for adapters
        if [[ "$component" == "monitor" || "$component" == "execution" || "$component" == "verification" ]]; then
            validation_cmd+=" --plugin-dir extern"
        fi
        
        # Add specific options for components
        if [[ "$component" == "digital-twin" ]]; then
            validation_cmd+=" --world-model bayesian"
        elif [[ "$component" == "agentic-reasoner" ]]; then
            validation_cmd+=" --use-improved-grpc"
        fi
        
        if eval "$validation_cmd" &>/dev/null; then
            log_success "$component validation passed"
        else
            log_error "$component validation failed"
            failed_components+=("$component")
        fi
    done
    
    if [[ ${#failed_components[@]} -gt 0 ]]; then
        log_error "Component validation failed for: ${failed_components[*]}"
        return 1
    else
        log_success "All component validations passed"
        return 0
    fi
}

if [[ "$VALIDATE_FIRST" == true && "$DRY_RUN" != true ]]; then
    validate_all_components || exit 1
fi

# --- Kill existing session if it exists ---
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log_warn "Killing existing session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# --- Helper function to run commands ---
run_command() {
    local cmd="$1"
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: $cmd"
    else
        eval "$cmd"
    fi
}

# --- Helper function to create tmux window with error logging ---
create_window() {
    local window_name="$1"
    local command="$2"
    local window_index="$3"
    local log_file="$PROJECT_ROOT/logs/${window_name}.log"
    local error_log="$PROJECT_ROOT/logs/${window_name}_error.log"
    
    log_component "Creating window: $window_name"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: tmux new-window -t $TMUX_SESSION:$window_index -n '$window_name'"
        echo "DRY RUN: tmux send-keys -t $TMUX_SESSION:$window_name '$command' Enter"
        return
    fi
    
    # Ensure logs directory exists
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Wrap command with logging and error capture
    local wrapped_command="$command 2>&1 | tee '$log_file'; echo \$? > '$PROJECT_ROOT/logs/${window_name}_exit_code'"
    
    tmux new-window -t "$TMUX_SESSION:$window_index" -n "$window_name"
    tmux send-keys -t "$TMUX_SESSION:$window_name" "$wrapped_command" Enter
}

# --- Helper function to wait for service with error reporting ---
wait_for_service() {
    local service_name="$1"
    local check_command="$2"
    local timeout="$3"
    local window_name="$4"  # Optional: window name for error log checking
    
    log_info "Waiting for $service_name to be ready..."
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: Would wait for $service_name"
        return 0
    fi
    
    local count=0
    while [[ $count -lt $timeout ]]; do
        if eval "$check_command" &>/dev/null; then
            log_success "$service_name is ready"
            return 0
        fi
        
        # Check for early exit/failure if window name provided
        if [[ -n "$window_name" ]]; then
            local exit_code_file="$PROJECT_ROOT/logs/${window_name}_exit_code"
            if [[ -f "$exit_code_file" ]]; then
                local exit_code=$(cat "$exit_code_file" 2>/dev/null || echo "")
                if [[ "$exit_code" != "" && "$exit_code" != "0" ]]; then
                    log_error "$service_name failed with exit code: $exit_code"
                    show_component_errors "$window_name"
                    return 1
                fi
            fi
        fi
        
        sleep 1
        ((count++))
        if [[ $((count % 10)) -eq 0 ]]; then
            log_info "Still waiting for $service_name... ($count/$timeout)"
        fi
    done
    
    log_error "$service_name failed to start within $timeout seconds"
    if [[ -n "$window_name" ]]; then
        show_component_errors "$window_name"
    fi
    return 1
}

# --- Helper function to show component errors ---
show_component_errors() {
    local component_name="$1"
    local log_file="$PROJECT_ROOT/logs/${component_name}.log"
    local exit_code_file="$PROJECT_ROOT/logs/${component_name}_exit_code"
    
    log_error "=== $component_name Error Details ==="
    
    # Show exit code if available
    if [[ -f "$exit_code_file" ]]; then
        local exit_code=$(cat "$exit_code_file" 2>/dev/null || echo "unknown")
        log_error "Exit code: $exit_code"
    fi
    
    # Show last 20 lines of log file
    if [[ -f "$log_file" ]]; then
        log_error "Last 20 lines of log:"
        echo -e "${RED}----------------------------------------${NC}"
        tail -n 20 "$log_file" | sed "s/^/${RED}| ${NC}/"
        echo -e "${RED}----------------------------------------${NC}"
        log_info "Full log available at: $log_file"
    else
        log_error "No log file found at: $log_file"
    fi
    
    # Show tmux pane content if session exists
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null && tmux list-windows -t "$TMUX_SESSION" | grep -q "$component_name"; then
        log_error "Current tmux pane output:"
        echo -e "${RED}----------------------------------------${NC}"
        tmux capture-pane -t "$TMUX_SESSION:$component_name" -p | tail -n 10 | sed "s/^/${RED}| ${NC}/"
        echo -e "${RED}----------------------------------------${NC}"
    fi
}

# --- Helper function to create error report ---
create_error_report() {
    local report_file="$PROJECT_ROOT/logs/startup_error_report.txt"
    
    log_info "Creating comprehensive error report: $report_file"
    
    {
        echo "POLARIS SWIM System Startup Error Report"
        echo "========================================"
        echo "Generated: $(date)"
        echo "Session: $TMUX_SESSION"
        echo "SWIM: $SWIM_HOST:$SWIM_PORT"
        echo "NATS: localhost:$NATS_PORT"
        echo ""
        
        echo "System Information:"
        echo "-------------------"
        echo "OS: $(uname -a)"
        echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
        echo "Virtual Environment: $VENV_PATH"
        echo "Project Root: $PROJECT_ROOT"
        echo ""
        
        echo "Environment Variables:"
        echo "----------------------"
        echo "GEMINI_API_KEY: ${GEMINI_API_KEY:+SET}" # Don't expose the actual key
        echo "PYTHONPATH: ${PYTHONPATH:-Not set}"
        echo ""
        
        echo "Component Status:"
        echo "-----------------"
        local components=("nats-server" "knowledge-base" "digital-twin" "verification" "kernel" "monitor" "execution" "reasoner" "meta-learner")
        
        for component in "${components[@]}"; do
            echo "=== $component ==="
            local exit_code_file="$PROJECT_ROOT/logs/${component}_exit_code"
            local log_file="$PROJECT_ROOT/logs/${component}.log"
            
            if [[ -f "$exit_code_file" ]]; then
                echo "Exit Code: $(cat "$exit_code_file" 2>/dev/null || echo 'unknown')"
            else
                echo "Exit Code: Not available"
            fi
            
            if [[ -f "$log_file" ]]; then
                echo "Log File Size: $(wc -l < "$log_file" 2>/dev/null || echo 0) lines"
                echo "Last 10 lines:"
                tail -n 10 "$log_file" 2>/dev/null | sed 's/^/  /'
            else
                echo "Log File: Not found"
            fi
            echo ""
        done
        
        echo "Network Connectivity:"
        echo "--------------------"
        echo "NATS Port $NATS_PORT: $(nc -z localhost $NATS_PORT && echo 'Open' || echo 'Closed')"
        echo "Digital Twin Port 50051: $(nc -z localhost 50051 && echo 'Open' || echo 'Closed')"
        echo "SWIM Port $SWIM_PORT: $(nc -z $SWIM_HOST $SWIM_PORT && echo 'Open' || echo 'Closed')"
        echo ""
        
        echo "Tmux Session Status:"
        echo "-------------------"
        if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            tmux list-windows -t "$TMUX_SESSION" -F "#{window_index}: #{window_name} - #{window_active}"
        else
            echo "Session not found"
        fi
        
    } > "$report_file"
    
    log_info "Error report saved to: $report_file"
}

# --- Create new tmux session ---
log_step "Creating tmux session: $TMUX_SESSION"
run_command "tmux new-session -d -s '$TMUX_SESSION' -n 'main'"

# Activate virtual environment in the main window
run_command "tmux send-keys -t '$TMUX_SESSION:main' 'cd $PROJECT_ROOT' Enter"
run_command "tmux send-keys -t '$TMUX_SESSION:main' 'source $VENV_PATH/bin/activate' Enter"
run_command "tmux send-keys -t '$TMUX_SESSION:main' 'echo \"POLARIS SWIM System Control Panel\"' Enter"
run_command "tmux send-keys -t '$TMUX_SESSION:main' 'echo \"Session: $TMUX_SESSION\"' Enter"

# --- Component 1: NATS Server ---
log_step "1/9 Starting NATS Server"
nats_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
nats_command+="echo 'Starting NATS Server on port $NATS_PORT...' && "

# Try different NATS server options in order of preference
nats_command+="if command -v nats-server &> /dev/null; then "
nats_command+="  echo 'Using system nats-server' && "
nats_command+="  nats-server --port $NATS_PORT --http_port 8222 --log_file logs/nats.log; "
nats_command+="elif command -v docker &> /dev/null; then "
nats_command+="  echo 'Using Docker NATS server' && "
nats_command+="  docker run --rm -p $NATS_PORT:4222 -p 8222:8222 --name polaris-nats nats:latest -p 4222 -m 8222; "
nats_command+="elif python -c 'import nats' &> /dev/null; then "
nats_command+="  echo 'Using Python NATS server (asyncio-nats-streaming)' && "
nats_command+="  python -c \"import asyncio; import nats; asyncio.run(nats.run_server(port=$NATS_PORT))\"; "
nats_command+="else "
nats_command+="  echo 'ERROR: No NATS server available. Please install nats-server, Docker, or python nats package' && "
nats_command+="  exit 1; "
nats_command+="fi"

create_window "nats-server" "$nats_command" 1

# Wait for NATS to be ready
if ! wait_for_service "NATS Server" "nc -z localhost $NATS_PORT" 30 "nats-server"; then
    log_error "NATS Server failed to start - this is critical for system operation"
    log_info "Try starting NATS manually: nats-server --port $NATS_PORT"
    create_error_report
    log_info "Error report created: $PROJECT_ROOT/logs/startup_error_report.txt"
    exit 1
fi

sleep $STARTUP_DELAY

# --- Component 2: Knowledge Base ---
log_step "2/9 Starting Knowledge Base"
kb_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
kb_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
kb_command+="echo 'Starting Knowledge Base...' && "
kb_command+="python src/scripts/start_component.py knowledge-base --config '$POLARIS_CONFIG_PATH' --log-level INFO"

create_window "knowledge-base" "$kb_command" 2

sleep $STARTUP_DELAY

# --- Component 3: Digital Twin (with Bayesian World Model) ---
log_step "3/9 Starting Digital Twin with Bayesian World Model"
dt_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
dt_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
dt_command+="export GEMINI_API_KEY='$GEMINI_API_KEY' && "
dt_command+="export WORLD_MODEL_IMPLEMENTATION=bayesian && "
dt_command+="echo 'Starting Digital Twin with Bayesian World Model...' && "
dt_command+="python src/scripts/start_component.py digital-twin --world-model bayesian --config '$POLARIS_CONFIG_PATH' --log-level INFO"

create_window "digital-twin" "$dt_command" 3

# Wait for Digital Twin gRPC service
if ! wait_for_service "Digital Twin" "nc -z localhost 50051" 45 "digital-twin"; then
    log_error "Digital Twin failed to start - this is critical for system operation"
    log_warn "Check if Gemini API key is valid and network connectivity is available"
    # Continue anyway as other components might still be useful
fi

sleep $STARTUP_DELAY

# --- Component 4: Verification Adapter ---
log_step "4/9 Starting Verification Adapter"
verification_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
verification_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
verification_command+="export PLUGIN_CONFIG_PATH='$PLUGIN_CONFIG_PATH' && "
verification_command+="echo 'Starting Verification Adapter...' && "
verification_command+="python src/scripts/start_component.py verification --plugin-dir extern --config '$POLARIS_CONFIG_PATH' --log-level INFO"

create_window "verification" "$verification_command" 4

sleep $STARTUP_DELAY

# --- Component 5: Kernel ---
log_step "5/9 Starting Kernel"
kernel_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
kernel_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
kernel_command+="echo 'Starting Kernel...' && "
kernel_command+="python src/scripts/start_component.py kernel --config '$POLARIS_CONFIG_PATH' --log-level INFO"

create_window "kernel" "$kernel_command" 5

sleep $STARTUP_DELAY

# --- Component 6: Monitor Adapter (SWIM) ---
log_step "6/9 Starting Monitor Adapter for SWIM"
monitor_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
monitor_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
monitor_command+="export PLUGIN_CONFIG_PATH='$PLUGIN_CONFIG_PATH' && "
monitor_command+="export SWIM_HOST='$SWIM_HOST' && "
monitor_command+="export SWIM_PORT='$SWIM_PORT' && "
monitor_command+="echo 'Starting Monitor Adapter for SWIM...' && "
monitor_command+="echo 'Connecting to SWIM at $SWIM_HOST:$SWIM_PORT' && "
monitor_command+="python src/scripts/start_component.py monitor --plugin-dir extern --config '$POLARIS_CONFIG_PATH' --log-level INFO"

create_window "monitor" "$monitor_command" 6

sleep $STARTUP_DELAY

# --- Component 7: Execution Adapter (SWIM) ---
log_step "7/9 Starting Execution Adapter for SWIM"
execution_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
execution_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
execution_command+="export PLUGIN_CONFIG_PATH='$PLUGIN_CONFIG_PATH' && "
execution_command+="export SWIM_HOST='$SWIM_HOST' && "
execution_command+="export SWIM_PORT='$SWIM_PORT' && "
execution_command+="echo 'Starting Execution Adapter for SWIM...' && "
execution_command+="echo 'Connecting to SWIM at $SWIM_HOST:$SWIM_PORT' && "
execution_command+="python src/scripts/start_component.py execution --plugin-dir extern --config '$POLARIS_CONFIG_PATH' --log-level INFO"

create_window "execution" "$execution_command" 7

sleep $STARTUP_DELAY

# --- Component 8: Agentic Reasoner (Gemini Pro) ---
log_step "8/9 Starting Agentic Reasoner with Gemini Pro"
reasoner_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
reasoner_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
reasoner_command+="export GEMINI_API_KEY='$GEMINI_API_KEY' && "
reasoner_command+="export LLM_MODEL='gemini-2.0-flash' && "
reasoner_command+="export LLM_TEMPERATURE=0.3 && "
reasoner_command+="echo 'Starting Agentic Reasoner with Gemini Pro...' && "
reasoner_command+="python src/scripts/start_component.py agentic-reasoner --use-improved-grpc --timeout-config robust --config '$POLARIS_CONFIG_PATH' --log-level INFO --monitor-performance"

create_window "reasoner" "$reasoner_command" 8

sleep $STARTUP_DELAY

# --- Component 9: Meta Learner ---
log_step "9/9 Starting Meta Learner"
meta_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
meta_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
meta_command+="export GEMINI_API_KEY='$GEMINI_API_KEY' && "
meta_command+="echo 'Starting Meta Learner...' && "
meta_command+="python src/scripts/start_component.py meta-learner --config '$POLARIS_CONFIG_PATH' --log-level INFO"

create_window "meta-learner" "$meta_command" 9

# --- Final setup ---
log_step "Setting up system monitoring"

# Create a monitoring window
monitoring_command="cd $PROJECT_ROOT && source $VENV_PATH/bin/activate && "
monitoring_command+="export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-} && "
monitoring_command+="echo 'POLARIS System Monitor' && "
monitoring_command+="echo '=====================' && "
monitoring_command+="echo 'Available commands:' && "
monitoring_command+="echo '  python src/scripts/nats_spy.py - Monitor NATS messages' && "
monitoring_command+="echo '  python src/scripts/kb_query_client.py - Query knowledge base' && "
monitoring_command+="echo '  python src/scripts/digital_twin_probe.py - Test digital twin' && "
monitoring_command+="echo '' && "
monitoring_command+="echo 'SWIM Status:' && "
monitoring_command+="nc -z $SWIM_HOST $SWIM_PORT && echo 'SWIM is reachable' || echo 'SWIM is not reachable'"

create_window "monitor-tools" "$monitoring_command" 10

# Switch to main window
run_command "tmux select-window -t '$TMUX_SESSION:main'"

# --- Final health check and success message ---
if [[ "$DRY_RUN" != true ]]; then
    log_step "Performing final system health check..."
    sleep 5  # Give components a moment to fully initialize
    
    if check_component_health; then
        log_success "POLARIS SWIM System started successfully!"
        echo
        log_info "Tmux session: $TMUX_SESSION"
        log_info "Components:"
        echo "  1. nats-server    - NATS message bus"
        echo "  2. knowledge-base - Data storage"
        echo "  3. digital-twin   - Bayesian world model"
        echo "  4. verification   - Safety enforcement"
        echo "  5. kernel         - Action coordination"
        echo "  6. monitor        - SWIM telemetry"
        echo "  7. execution      - SWIM actions"
        echo "  8. reasoner       - Gemini Pro AI"
        echo "  9. meta-learner   - Learning optimization"
        echo "  10. monitor-tools - System monitoring"
        echo
        log_info "Logs directory: $PROJECT_ROOT/logs/"
        log_info "To attach to session: tmux attach-session -t $TMUX_SESSION"
        log_info "To kill session: $0 -k"
        log_info "To list components: $0 -l"
        log_info "To check errors: $0 --check-health"
        echo
        log_warn "Make sure SWIM is running at $SWIM_HOST:$SWIM_PORT"
        log_info "SWIM Docker command: docker run -d -p 4242:4242 -p 5901:5901 -p 6901:6901 --name swim gabrielmoreno/swim"
        echo
        
        # Attach to the session
        exec tmux attach-session -t "$TMUX_SESSION"
    else
        log_error "System startup completed with errors. Check component logs above."
        create_error_report
        log_info "Logs are available in: $PROJECT_ROOT/logs/"
        log_info "Error report: $PROJECT_ROOT/logs/startup_error_report.txt"
        log_info "To attach to session anyway: tmux attach-session -t $TMUX_SESSION"
        log_info "To kill session: $0 -k"
        log_info "To check health: $0 --check-health"
        exit 1
    fi
fi