#!/bin/bash
set -e

# Define paths
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$REPO_DIR/parallel_contraction_research/proper_research"
VENV_DIR="$REPO_DIR/parallel_contraction_research/gemini_3.5_flash/venv"

echo "=========================================================================="
echo "    Tensor Network Contraction: Slicing and Parallel Scaling Reproducer"
echo "=========================================================================="

# 1. Check Julia installation
if ! command -v julia &> /dev/null; then
    echo "Error: julia is not installed or not in PATH."
    exit 1
fi
JULIA_VER=$(julia --version)
echo "Found Julia: $JULIA_VER"

# 2. Check Python virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install requirements
echo "Checking and installing python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r "$RESEARCH_DIR/requirements.txt"

# 4. Run the reproducible scaling sweep
echo "Launching advanced benchmark sweep..."
python "$RESEARCH_DIR/run_advanced_scaling_sweep.py"

echo "=========================================================================="
echo "    REPRODUCTION SUCCESSFUL"
echo "=========================================================================="
echo "Results are compiled and saved to:"
echo "  $RESEARCH_DIR/results/advanced_scaling_report.md"
echo "=========================================================================="
