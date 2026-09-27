#!/bin/bash
set -e

# Spinoff reproduction entry point
echo "=========================================================================="
echo "Starting Reproduction of Quantum Circuit Contraction Research"
echo "=========================================================================="

# Find local python executable
if [ -f "../parallel_contraction_research/gemini_3.5_flash/venv/bin/python" ]; then
    PYTHON_ENV="../parallel_contraction_research/gemini_3.5_flash/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_ENV="python3"
else
    PYTHON_ENV="python"
fi

echo "1. Running the 7x7 profiling sweep..."
$PYTHON_ENV run_quantum_circuit_sweep.py

echo "2. Generating comparative plots..."
$PYTHON_ENV plot_quantum_circuits.py

echo "=========================================================================="
echo "Reproduction complete! Visual plots saved to results/quantum_cost_progression_v2.png"
echo "=========================================================================="
