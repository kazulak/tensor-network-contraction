#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/parallel_contraction_research/gemini_3.5_flash/venv"
PYTHON_BIN="$VENV_DIR/bin/python"

echo "=========================================================================="
echo "  Tensor Network Contraction Research: Master Reproduction Suite"
echo "=========================================================================="

# 1. Check Julia installation
if ! command -v julia &> /dev/null; then
    echo "Error: julia is not installed or not found in PATH."
    exit 1
fi
echo "✓ Julia detected: $(julia --version)"

# 2. Check Python virtual environment
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r "$REPO_DIR/parallel_contraction_research/proper_research/requirements.txt"
fi
echo "✓ Python environment verified."

TARGET="${1:-all}"

case "$TARGET" in
    quantum)
        echo ""
        echo ">>> Reproducing Quantum Circuit Tensor Network Contraction Research..."
        cd "$REPO_DIR/quantum_circuit_research"
        "$PYTHON_BIN" run_advanced_scientific_research.py
        "$PYTHON_BIN" plot_advanced_scientific_research.py
        echo "✓ Quantum research reproduction finished! Results in quantum_circuit_research/results/"
        ;;
    grid)
        echo ""
        echo ">>> Reproducing Grid-based Parallel Contraction & Slicing Research..."
        cd "$REPO_DIR/parallel_contraction_research/proper_research"
        "$PYTHON_BIN" run_advanced_scaling_sweep.py
        echo "✓ Grid scaling reproduction finished! Results in parallel_contraction_research/proper_research/results/"
        ;;
    all)
        echo ""
        echo ">>> [1/2] Reproducing Quantum Circuit Contraction Research (Roofline & Multi-Thread Scaling)..."
        cd "$REPO_DIR/quantum_circuit_research"
        "$PYTHON_BIN" run_advanced_scientific_research.py
        "$PYTHON_BIN" plot_advanced_scientific_research.py
        
        echo ""
        echo ">>> [2/2] Reproducing Grid-based Parallel Contraction & Slicing Research..."
        cd "$REPO_DIR/parallel_contraction_research/proper_research"
        "$PYTHON_BIN" run_advanced_scaling_sweep.py
        
        echo ""
        echo "=========================================================================="
        echo "  MASTER REPRODUCTION COMPLETED SUCCESSFULLY!"
        echo "=========================================================================="
        echo "Outputs available at:"
        echo "  - Quantum Circuit Research: quantum_circuit_research/results/"
        echo "  - Grid Contraction Research: parallel_contraction_research/proper_research/results/"
        ;;
    *)
        echo "Usage: $0 [all|quantum|grid]"
        exit 1
        ;;
esac
