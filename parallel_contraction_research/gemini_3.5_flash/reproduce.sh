#!/bin/bash
# Reproducible script for parallel tensor network contraction benchmark

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================================"
echo "Reproducing Parallel Tensor-Network Contraction Benchmarks"
echo "======================================================================"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate virtual environment and install packages
echo "Installing dependencies..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install numpy scipy quimb cotengra joblib matplotlib autoray opt_einsum dask networkx

# 3. Run benchmarks
echo "Running benchmarks (this might take 1-2 minutes)..."
./venv/bin/python run_benchmarks.py

echo "======================================================================"
echo "Benchmarks finished successfully!"
echo "- Results saved to: benchmark_results.json"
echo "- Formatted table: benchmark_table.md"
echo "- Speedup plot: speedup_plot.png"
echo "======================================================================"
