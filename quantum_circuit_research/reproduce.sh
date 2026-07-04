#!/bin/bash
set -e

# Spinoff reproduction entry point
echo "=========================================================================="
echo "Starting Reproduction of Quantum Circuit Contraction Research"
echo "=========================================================================="

# Find local python virtual environment
PYTHON_ENV="../parallel_contraction_research/gemini_3.5_flash/venv/bin/python"

if [ ! -f "$PYTHON_ENV" ]; then
    echo "Error: Python virtual environment not found at $PYTHON_ENV."
    exit 1
fi

echo "1. Running the 7x7 profiling sweep..."
$PYTHON_ENV run_quantum_circuit_sweep.py

echo "2. Generating comparative plots..."
$PYTHON_ENV plot_quantum_circuits.py

echo "=========================================================================="
echo "Reproduction complete! Visual plots saved to results/quantum_cost_progression_v2.png"
echo "=========================================================================="
