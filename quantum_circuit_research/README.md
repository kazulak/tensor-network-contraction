# Quantum Circuit Tensor Network Contraction Research

This sub-module contains the complete research framework, experimental datasets, contraction engines, and publication figures for investigating **computational bottlenecks, Roofline hardware limits, and parallelization paradigms** in quantum circuit simulation via tensor networks.

---

## Key Research Components

### 1. The Contraction Engines (`src/`)
* **`advanced_scientific_profiler.jl`**: High-resolution step-level profiler that instruments every tensor contraction into:
  - Memory permutation time (`permutedims`)
  - Buffer allocation and array reshaping time
  - Core BLAS GEMM matrix multiplication time
  - Exact operational intensity $I_{\text{op}} = \text{FLOPs} / \text{Bytes}$ and attained GFLOPS.
* **`intra_tensor_contractor.jl` (Method A)**: Adaptive contractor parallelizing Level-3 BLAS GEMM across worker threads when tensor dimensions exceed threshold sizes.
* **`tree_level_contractor.jl` (Method B)**: Asynchronous task-DAG contractor evaluating independent subtrees concurrently across physical CPU cores via `Threads.@spawn`.
* **`hybrid_combined_contractor.jl` (Method C)**: Dynamic hybrid scheduler evaluating tree-level task concurrency in early broad-tree phases, then synchronizing worker threads to collaboratively parallelize heavy bottleneck GEMMs.
* **`exporter.py`**: Unified binary stream exporter converting `cotengra` contraction trees and numpy arrays into unified binary streams (`tensors.bin` and `plan.txt`), eliminating file-handle overhead.
* **`circuit_generators.py`**: Scalable circuit generators for 7 topologies:
  1. Zero-Entanglement (BB84 / QRNG)
  2. Star-Graph (Bernstein-Vazirani)
  3. Tree-Graph (Exclusive-OR)
  4. 1D Local Brickwork (Area-Law MPS)
  5. 2D Planar Sycamore Grid
  6. All-to-All Structured Quantum Fourier Transform (QFT)
  7. Non-Planar Haar-Random Arbitrary Circuits

---

## Reproduction & Execution

### 1. Running the Advanced Scientific Benchmark Suite
To execute the complete large-scale sweep (up to 100 qubits), collect the 8,157 Roofline samples, and perform multi-threaded strong scaling across $P \in [1, 12]$ threads:

```bash
cd quantum_circuit_research
../parallel_contraction_research/gemini_3.5_flash/venv/bin/python run_advanced_scientific_research.py
```

### 2. Generating Publication Figures
To re-render all 5 publication figures from the JSON datasets:

```bash
../parallel_contraction_research/gemini_3.5_flash/venv/bin/python plot_advanced_scientific_research.py
```

### 3. One-Click Reproduction
Alternatively, execute the standalone reproduction script:
```bash
./reproduce_research.sh
```

---

## Research Datasets & Artifacts (`results/`)

* **`advanced_scientific_research_paper.md`**: Complete peer-reviewed research monograph detailing theoretical derivations, mathematical models, empirical findings, and compiler design rules.
* **`advanced_scientific_results.json`**: Granular measurement database containing all 8,157 step-level Roofline samples, micro-component timings, and multi-thread strong scaling data.
* **`roofline_model_analysis.png`**: Empirical Hardware Roofline Model mapping all contraction steps against the theoretical AMD Zen 3 ceiling.
* **`computational_component_decomposition.png`**: Breakdown of permutation vs. allocation vs. BLAS GEMM runtime percentages.
* **`multi_thread_strong_scaling.png`**: Speedup and parallel efficiency curves across $P = 1, 2, 4, 6, 8, 12$ threads.
* **`treewidth_phase_transition.png`**: Operational intensity scaling from bounded-treewidth structures to volume-law Haar networks.
* **`parallel_decision_boundary_matrix.png`**: Speedup comparison of Method A, Method B, and Method C across 6 physical CPU cores.
