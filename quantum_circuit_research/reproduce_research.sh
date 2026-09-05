#!/bin/bash
set -e

echo "=========================================================================="
echo "Starting Reproduction of Quantum Circuit Contraction Parallel Research"
echo "=========================================================================="

PYTHON_ENV="../parallel_contraction_research/gemini_3.5_flash/venv/bin/python"

if [ ! -f "$PYTHON_ENV" ]; then
    echo "Error: Python virtual environment not found at $PYTHON_ENV."
    exit 1
fi

echo "1. Running the 28-configuration benchmark across all 4 contraction methods..."
$PYTHON_ENV run_parallel_research.py

echo "2. Generating 4 publication-quality research plots..."
$PYTHON_ENV plot_parallel_research.py

echo "=========================================================================="
echo "Research reproduction complete! Results and plots saved to results/."
echo "=========================================================================="
