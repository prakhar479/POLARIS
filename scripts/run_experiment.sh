#!/bin/bash
# ==============================================================================
# POLARIS Experiment Runner
# Unified orchestration script for autonomous self-adaptation experiments.
#
# Supported Exemplars:
#   - swim      (SWIM Docker container over TCP 4242)
#   - wildfire  (Wildfire UAV simulation over REST 5000)
#
# Usage:
#   ./scripts/run_experiment.sh --exemplar swim --config config/swim.yaml --experiment-name swim_baseline
#   ./scripts/run_experiment.sh --exemplar wildfire --config config/wildfire.yaml --experiment-name wf_baseline
# ==============================================================================

set -euo pipefail

# Default parameters
EXEMPLAR="swim"
CONFIG_FILE="config/swim.yaml"
EXPERIMENT_NAME="exp_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=""
SKIP_SIM_BOOT=false
POLARIS_PID=""
SIM_CONTAINER=""
SIM_PID=""

# Help documentation
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  -e, --exemplar <swim|wildfire>  Target exemplar environment (default: swim)
  -c, --config <path>             Path to YAML configuration (default: config/swim.yaml)
  -n, --experiment-name <name>    Unique experiment identifier (default: exp_<timestamp>)
  -d, --dry-run                   Run Polaris in dry-run mode (assess without mutating system)
  -s, --skip-sim-boot             Skip starting simulation container/process (assumes already running)
  -h, --help                      Show this help message and exit

Examples:
  $0 --exemplar swim --config config/swim.yaml --experiment-name swim_hybrid_run1
  $0 --exemplar wildfire --config config/wildfire.yaml --experiment-name wf_agentic_run1
EOF
    exit 1
}

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--exemplar)
            EXEMPLAR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -n|--experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -s|--skip-sim-boot)
            SKIP_SIM_BOOT=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown argument: $1"
            usage
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Activate virtual environment if present
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

METRICS_DIR="${REPO_ROOT}/metrics/${EXEMPLAR}"
LOGS_DIR="${REPO_ROOT}/logs"
RESULTS_DIR="${REPO_ROOT}/results"

mkdir -p "${METRICS_DIR}" "${LOGS_DIR}" "${RESULTS_DIR}"

echo "============================================================"
echo "🌟 POLARIS Experiment Runner"
echo "   Exemplar:        ${EXEMPLAR}"
echo "   Configuration:   ${CONFIG_FILE}"
echo "   Experiment:      ${EXPERIMENT_NAME}"
echo "   Metrics Dir:     ${METRICS_DIR}"
echo "   Dry Run:         ${DRY_RUN:-false}"
echo "============================================================"
echo ""

# Graceful cleanup handler
cleanup() {
    echo ""
    echo "🧹 Cleaning up experiment processes..."
    if [[ -n "${POLARIS_PID}" ]] && kill -0 "${POLARIS_PID}" 2>/dev/null; then
        echo "   Stopping POLARIS (PID ${POLARIS_PID})..."
        kill -INT "${POLARIS_PID}" 2>/dev/null || true
        wait "${POLARIS_PID}" 2>/dev/null || true
    fi

    if [[ -n "${SIM_PID}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
        echo "   Stopping Simulation Process (PID ${SIM_PID})..."
        kill -TERM "${SIM_PID}" 2>/dev/null || true
    fi

    if [[ -n "${SIM_CONTAINER}" ]]; then
        echo "   Stopping Simulation Container (${SIM_CONTAINER})..."
        docker stop "${SIM_CONTAINER}" >/dev/null 2>&1 || true
        docker rm "${SIM_CONTAINER}" >/dev/null 2>&1 || true
    fi
    echo "✓ Cleanup complete."
}
trap cleanup EXIT INT TERM

# Step 1: Pre-flight check with polaris doctor
echo "🩺 Step 1: Running pre-flight diagnostic check..."
python3 -m polaris.cli.main doctor --config "${CONFIG_FILE}" || {
    echo "❌ Doctor check failed! Please fix issues in configuration or environment before proceeding."
    exit 1
}
echo "✓ Pre-flight check passed."
echo ""

# Step 2: Boot simulation if requested
if [[ "${SKIP_SIM_BOOT}" == "false" ]]; then
    if [[ "${EXEMPLAR}" == "swim" ]]; then
        echo "🐳 Step 2: Booting SWIM Docker container..."
        SIM_CONTAINER="polaris-swim-${EXPERIMENT_NAME}"
        docker run -d --name "${SIM_CONTAINER}" -p 4242:4242 \
            vcnk4v/polaris-swim:trimmed-traces bash -c "\
                cd ~/seams-swim/swim/simulations/swim/ && \
                ./run.sh sim 1
            "
        echo "   Waiting for SWIM socket localhost:4242..."
        for i in {1..30}; do
            if nc -z localhost 4242 2>/dev/null; then
                echo "✓ SWIM container listening on port 4242."
                break
            fi
            sleep 1
        done
    elif [[ "${EXEMPLAR}" == "wildfire" ]]; then
        if [[ -f "wildfire/adapter.py" ]]; then
            echo "🔥 Step 2: Booting Wildfire REST Adapter..."
            python3 wildfire/adapter.py &
            SIM_PID=$!
            echo "   Waiting for Wildfire adapter on localhost:5000..."
            for i in {1..20}; do
                if curl -sf http://localhost:5000/health >/dev/null 2>&1 || nc -z localhost 5000 2>/dev/null; then
                    echo "✓ Wildfire adapter online on port 5000."
                    break
                fi
                sleep 1
            done
        else
            echo "⚠️  wildfire/adapter.py not found, assuming external simulator..."
        fi
    fi
else
    echo "⏭️  Step 2: Skipping simulation boot (running against existing simulation)."
fi
echo ""

# Step 3: Run POLARIS coordinator
LOG_FILE="${LOGS_DIR}/${EXPERIMENT_NAME}.log"
echo "🚀 Step 3: Launching POLARIS coordinator..."
echo "   Logging to: ${LOG_FILE}"

polaris_args=(
    --config "${CONFIG_FILE}"
    --metrics-export "${METRICS_DIR}"
    --metrics-experiment "${EXPERIMENT_NAME}"
    --export-logs "${LOG_FILE}"
    --log-format "structured"
)

if [[ -n "${DRY_RUN}" ]]; then
    polaris_args+=("${DRY_RUN}")
fi

python3 -m polaris.cli.main "${polaris_args[@]}" &
POLARIS_PID=$!

echo "   POLARIS running with PID ${POLARIS_PID}."
echo "   Press Ctrl+C to terminate the experiment gracefully."
echo ""

# Wait for Polaris process to finish (or wait for simulation if attached)
wait "${POLARIS_PID}" || true
POLARIS_PID=""

echo ""
echo "📊 Step 4: Extracting experiment metrics..."
SUMMARY_MD="${RESULTS_DIR}/${EXPERIMENT_NAME}_summary.md"

if [[ -f "scripts/extract_swim_metrics.py" ]] && [[ "${EXEMPLAR}" == "swim" ]]; then
    python3 scripts/extract_swim_metrics.py \
        --metrics-dir "${METRICS_DIR}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --output "${SUMMARY_MD}" || true
    if [[ -f "${SUMMARY_MD}" ]]; then
        echo "✓ Metric summary generated: ${SUMMARY_MD}"
        head -n 25 "${SUMMARY_MD}"
    fi
fi

echo ""
echo "🎉 Experiment '${EXPERIMENT_NAME}' completed successfully."
