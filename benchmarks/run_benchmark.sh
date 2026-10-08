#!/usr/bin/env bash
# Universal benchmark experiment runner for POLARIS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXEMPLAR="${1:-switch}"
SEEDS="${2:-3}"

echo "=========================================================="
echo "POLARIS Universal Benchmark Runner"
echo "Target Exemplar: ${EXEMPLAR} | Seeds: ${SEEDS}"
echo "=========================================================="

case "$EXEMPLAR" in
  switch)
    echo "Running SWITCH model switching benchmark..."
    "${ROOT_DIR}/.venv/bin/polaris" doctor --config "${SCRIPT_DIR}/exemplars/switch/config.yaml"
    "${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/reproduce_paper.py" --exemplar switch --seeds "$SEEDS"
    ;;
  swim)
    echo "Running SWIM web infrastructure benchmark..."
    "${ROOT_DIR}/.venv/bin/polaris" doctor --config "${SCRIPT_DIR}/exemplars/swim/config.yaml"
    "${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/reproduce_paper.py" --exemplar swim --seeds "$SEEDS"
    ;;
  all)
    echo "Running full paper benchmark reproduction suite..."
    "${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/reproduce_paper.py" --exemplar all --seeds "$SEEDS"
    ;;
  *)
    echo "Unknown exemplar: $EXEMPLAR (choose: switch, swim, all)"
    exit 1
    ;;
esac

echo ""
echo "✅ Benchmark execution finished. Artifacts saved in ${SCRIPT_DIR}/results/"
