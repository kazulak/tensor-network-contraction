# Parallel Tensor-Network Contraction Benchmarks in Python

This repository contains a reproducible benchmark pipeline that evaluates the performance and speedups achieved by parallelizing tensor-network contractions using **slicing** (bond cutting) in Python. 

The pipeline uses `quimb` to build tensor networks, `cotengra` to plan contractions and find optimal slice indices, and `concurrent.futures.ProcessPoolExecutor` with cached process-level initialization for parallel execution.

## Research Question
*How much speedup can be achieved by parallelizing tensor-network contraction?*

## Core Findings Summary
1. **Parallel Speedup**: Parallelizing sliced contractions yields a speedup of **2.6x to 3.5x** on a 12-core CPU relative to the *sliced serial baseline*. The speedup peaks around 6 to 8 workers and flatlines due to **memory bandwidth saturation** (RAM bus bottlenecks) during large tensor multiplications (GEMM).
2. **FLOP Inflation & Slicing Overhead**: Slicing introduces a massive **8.5x to 58x FLOP inflation** (redundant calculations) because indices that would have been contracted locally must be evaluated repeatedly.
3. **Net Speedup**: Relative to the optimal, unsliced serial contraction, the parallelized sliced contraction is **3x to 50x slower**.
4. **Conclusion**: Slicing is a **memory-scaling technique**, not a speedup technique. It is essential to avoid Out-Of-Memory (OOM) errors for very large networks, but should be avoided when the network fits in memory.

---

## Code Structure

```text
.
├── parallel_contraction/         # Core python package
│   ├── __init__.py
│   ├── network_builder.py        # Generates Random Regular Graphs & 2D Grid Networks
│   ├── parallel_contractor.py    # Implements serial and optimized parallel contractors
│   └── benchmark.py              # Coordinates benchmarks, collects metrics, generates plot
├── run_benchmarks.py             # Script entry point to run benchmarks
├── reproduce.sh                  # Automation script to set up environment and run benchmarks
├── paper.md                      # Short research paper in Markdown format
├── benchmark_results.json        # Raw benchmark results data
├── benchmark_table.md            # Results compiled into a Markdown table
├── speedup_plot.png              # Speedup and efficiency plot image
└── README.md                     # Setup and reproduction instructions (this file)
```

---

## Installation & Reproduction Instructions

To reproduce the benchmarks and generate the results table and plot, run the automated reproduction shell script:

```bash
chmod +x reproduce.sh
./reproduce.sh
```

The script will automatically:
1. Create a Python virtual environment (`venv`).
2. Upgrade `pip` and install all required libraries (`numpy`, `scipy`, `quimb`, `cotengra`, `matplotlib`, `networkx`, `autoray`, `opt_einsum`, `dask`, etc.).
3. Run the complete benchmark suite.
4. Save the raw JSON data, generate a Markdown results table, and plot the performance curves.

### Manual Setup

If you prefer to run the steps manually:

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install --upgrade pip
   pip install numpy scipy quimb cotengra matplotlib networkx autoray opt_einsum dask
   ```

3. Execute the benchmark:
   ```bash
   python run_benchmarks.py
   ```

---

## Deliverables Generated
- **Research Paper**: [paper.md](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/paper.md) (also available as user-facing artifact [research_paper.md](file:///home/tom/.gemini/antigravity-cli/brain/5fa0210a-145b-4237-bf2e-0e50bfd4cba8/research_paper.md))
- **Benchmark Table**: [benchmark_table.md](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/benchmark_table.md)
- **Performance Plot**: [speedup_plot.png](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/speedup_plot.png)
- **Raw Data**: [benchmark_results.json](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/benchmark_results.json)
