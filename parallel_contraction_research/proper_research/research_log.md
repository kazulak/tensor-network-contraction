# Tensor Network Contraction Parallelization: Research Log

This document serves as a persistent, chronological record of the research phases, execution models, and performance results. 

> [!NOTE]
> **Instructions for Future Phases:** When adding a new phase of research, append it to the **top** of this log (just below this instruction block) to keep the latest results immediately visible.

---

## Phase 6: Step-by-Step Cost Accumulation & "99% Root Contraction" Bottleneck
* **Date:** July 2026
* **Objective:** Profile cost progression (time and theoretical FLOPs) along the contraction path for 5 topologies and 3 enlarged size configs, comparing sequential vs. parallel sliced execution.
* **Implementation:**
  - Profiler: [step_profiler.jl](parallel_contraction_research/proper_research/src/step_profiler.jl)
  - Runner: [run_step_profiling_research.py](parallel_contraction_research/proper_research/run_step_profiling_research.py)
  - Visualizer: [plot_step_progression.py](parallel_contraction_research/proper_research/plot_step_progression.py)
* **Results & Key Findings:**
  - **Peak Treewidth Bottleneck Discovered**: The "99% final root step" bottleneck is incorrect for closed networks, as the final step collapses indices to a scalar (taking only **0.49%** of time on the largest 3D PEPS). The true bottleneck lies at the peak-treewidth intermediate steps (consuming **35%** in a single step).
  - **Timing Gap & Speedup Inversion**: For small networks, sequential execution is faster due to task-spawning and serialization overheads. For large networks (e.g. `3x3x3, D=4` PEPS), parallel sliced execution sits significantly below sequential, showing a large timing gap (speedup).
  - **Curve Flattening**: Slicing flattens the cumulative cost progression (%) by shrinking intermediate tensor shapes, distributing cost evenly across steps.
  - Report: [step_profiling_report.md](parallel_contraction_research/proper_research/results/step_profiling_report.md)

---

## Phase 5: 35-Case Master Sweep & Multi-Panel Visualization
* **Date:** July 2026
* **Objective:** Benchmark the 3 multithreaded backends (Pure Tree-Node, Static Slicing, and Active Slicing) across 5 topologies and 7 sizes using JIT-warmed runs to extract clean scaling characteristics.
* **Implementation:**
  - Sweep Script: [run_advanced_scaling_sweep.py](parallel_contraction_research/proper_research/run_advanced_scaling_sweep.py)
  - Visualizer: [plot_results.py](parallel_contraction_research/proper_research/plot_results.py)
* **Results & Key Findings:**
  - **Latency-Crossover Point**: Static Slicing suffers from a constant **~80ms overhead** due to threading initialization and slice copying. Tree-Node and Active Slicing run in sub-milliseconds on small sizes, yielding up to **690x speedup** over Static Slicing.
  - **Slicing Dominance at Extreme Scales**: On the largest 3D PEPS grid (`3x3x3, D=4`), Static Slicing is the superior backend by a wide margin (**0.50s** vs. **3.16s** for Active Slicing and **3.82s** for Tree-Node). Slicing globally reduces the dimension of all intermediate tensors, avoiding memory bus saturation.
  - **Memory Bus Saturation**: Concurrent matrix operations (GEMM) are limited by CPU memory bandwidth rather than core counts when intermediate tensors are large.

---

## Phase 4: Advanced Active Slicing Parallelism
* **Date:** July 2026
* **Objective:** Design and implement a dynamic slicing model that avoids global FLOP inflation by delaying slicing until index boundaries are encountered in the tree.
* **Implementation:** [active_slicing_contractor.jl](parallel_contraction_research/proper_research/src/active_slicing_contractor.jl)
* **Results & Key Findings:**
  - Successfully validated numerical correctness.
  - Achieved **1.48x compute speedup** over Traditional Static Slicing by restricting slice replication to only those nodes containing the cut indices.

---

## Phase 3: Recursive Tree-Node Parallelism
* **Date:** July 2026
* **Objective:** Implement work-stealing task-based parallelism using Julia's thread pool to contract independent branches of the binary contraction tree.
* **Implementation:** [tree_parallel_contractor.jl](parallel_contraction_research/proper_research/src/tree_parallel_contractor.jl)
* **Results & Key Findings:**
  - Scaled by **1.8x on 8 threads** for tree networks with zero slicing.
  - Bottlenecked primarily by the large root-node contractions where fork-join opportunities disappear and execution becomes sequential.

---

## Phase 2: persistent Julia Sockets Daemon & Verification
* **Date:** July 2026
* **Objective:** Eliminate Julia's runtime JIT pre-compilation latency and verify full double-precision numerical equivalence.
* **Implementation:** [hybrid_daemon.jl](parallel_contraction_research/proper_research/src/hybrid_daemon.jl)
* **Results & Key Findings:**
  - Demonstrated a **396x speedup** on warm roundtrips ($0.6\text{ms}$ hot contraction vs. $2.81\text{s}$ cold JIT-compiling contraction) by reusing compilation profiles in a socket server.
  - Verified numerical difference $< 10^{-12}$ against Python's reference contractor.

---

## Phase 1: Python Slicing Benchmarks (GIL Constraints)
* **Date:** July 2026
* **Objective:** Evaluate Python's native parallel capability for sliced contractions using `ProcessPoolExecutor` and `ThreadPoolExecutor`.
* **Results & Key Findings:**
  - **Multithreading** failed to yield any speedup due to Python's Global Interpreter Lock (GIL).
  - **Multiprocessing** achieved speedups of **1.9x to 2.3x** on large workloads but introduced a severe **10x latency regression** on small workloads due to Pickle serialization and IPC overhead.
  - Slicing introduced **FLOP inflation** of **2x to 48x** depending on the treewidth of the topology.
