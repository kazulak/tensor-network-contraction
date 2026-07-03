# Parallel Tensor Network Contraction Benchmark

This repository contains the code and research paper for benchmarking parallel tensor network contraction using the slicing technique.

## Prerequisites
- Python 3.10+
- The required packages can be installed via `pip`.

## Reproduction Instructions

1. **Set up the virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install quimb cotengra opt_einsum networkx matplotlib numpy scipy psutil tqdm autoray "dask[complete]"
   ```
   *(Note: This installs cotengra and quimb along with numerical backends).*

3. **Run the benchmark:**
   The `benchmark.py` script automatically builds a 12x12 2D grid tensor network, finds an optimal contraction path using slicing (with 81 slices), and contracts the slices serially and in parallel using `concurrent.futures.ProcessPoolExecutor`.
   
   ```bash
   python benchmark.py
   ```
   
   The output will display the execution times for serial evaluation and parallel evaluation using 1, 2, 4, and 8 workers. Numerical correctness is verified at each step.
   *Note*: The script inherently locks BLAS threading parameters (such as `OMP_NUM_THREADS=1`) prior to execution to avoid CPU thread oversubscription.

4. **Generate the plot:**
   A script to generate the speedup curve plot is provided (you may edit the hardcoded values inside the script if your benchmark run produced different numbers).
   ```bash
   python plot_results.py
   ```
   This will output `speedup_plot.png`.

## Files
- `benchmark.py`: Main parallel contraction test logic using slicing.
- `plot_results.py`: Script to generate speedup and efficiency curves.
- `research_paper.md`: The research paper discussing the benchmark methodology, results, and limitations.
- `speedup_plot.png`: Performance graph showcasing the sublinear speedup scaling.
