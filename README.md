# Multi-Agent Quantum & Algorithmic Research Repository

This repository serves as a generalized research environment for evaluating Large Language Model (LLM) agents on complex scientific and algorithmic problems. Multiple autonomous agents are deployed in isolated environments to design, implement, and benchmark solutions to selected research questions.

---

## Closed-Network Scalar-Amplitude Scope & Cost Metrics

### Current Research Scope
- **Domain**: Closed tensor networks representing quantum circuits (e.g., BB84, Bernstein–Vazirani, EDC, Hidden Subgroup, QRNG, XOR, Sycamore-like, Random Arbitrary) and condensed matter / graph networks (1D MPS, 2D PEPS grid, 3D PEPS grid, Random Regular Graphs, Binary Trees).
- **Output Target**: Closed-network scalar contraction amplitudes $\langle \text{bra} | \psi \rangle \in \mathbb{R}$.
- **Slicing & Parallel Execution**: Cotengra-guided hyper-optimization producing sliced pairwise contraction trees, executed sequentially or in parallel across CPU threads via Julia backends.

### Key Cost Metrics Collected
1. **FLOPs (Floating-Point Operations)**: Analytical GEMM operational count per step ($2 \times M \times N \times K$).
2. **Step Contraction Time (seconds)**: High-resolution per-step execution wall-clock time.
3. **Root-to-Total Ratio**: Percentage of total contraction time consumed by the final root node contraction.
4. **Parallel Speedup Factor**: Comparative speedup ratio of Active Slicing vs Static Slicing and Pure Tree-Node Parallelism.

### Scope Boundaries (Explicit Non-Claims)
- **Statevector Sampling / Open Networks**: Uncollected for generic tensor networks; specialized computational-basis sampling is supported for Clifford-only stabilizer states via Phase 2 Tableau engine.
- **GPU / Accelerator Hardware**: Uncollected; execution is CPU-only multithreading with BLAS thread clamping (`BLAS.set_num_threads(1)`).
- **Clifford Engine**: Implemented in Phase 2 via Aaronson-Gottesman (2004) binary symplectic stabilizer tableau baseline (`quantum_circuit_research/src/tableau_simulator.py`). Spezialized for outcome probability queries $P(b)$ and sampling; closed-network scalar amplitudes remain on the generic tensor-network path.

---

## Setup & Execution Guide (Fresh Linux Clone)

### 1. Python Environment Setup (Required for Core Generators & Oracle Tests)
To run network generators, quantum circuit generators, serialization export, and exact small-system numerical oracle test suites:
```bash
git clone <repository_url>
cd tensor-network-contraction

# Create and activate Python 3.10+ virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required core dependencies
pip install -r requirements.txt

# Run pytest discovery or fast correctness test suites
pytest tests/
python tests/test_oracle_and_serialization.py
python tests/test_clifford_tableau.py

# Run Phase 2 Clifford Tableau Benchmark (contrasts N=4 to N=1024 qubits against dense limit)
python quantum_circuit_research/benchmark_clifford_tableau.py
```

### 2. Julia Environment Setup (Optional: Required only for Julia Parallel Backends & Sweeps)
To run full Julia parallel contraction backends, active slicing, or scaling sweeps:
```bash
# Verify Julia (>= 1.6) is installed
julia --version

# Instantiate Julia project dependencies (JSON, LinearAlgebra, Sockets)
julia --project=parallel_contraction_research/proper_research/src/ -e 'using Pkg; Pkg.instantiate()'
```

### 3. Running Reproducible Sweeps
- **Master Parallel Scaling Sweep**:
  ```bash
  ./reproduce.sh
  ```
- **Quantum Circuit Profiling Sweep**:
  ```bash
  cd quantum_circuit_research
  ./reproduce.sh
  ```
- **Clifford Stabilizer Tableau Scaling Benchmark**:
  ```bash
  python quantum_circuit_research/benchmark_clifford_tableau.py
  ```

---

## Repository Structure

* **[reproduce.sh](reproduce.sh)**: Master one-command execution script for the hybrid scaling benchmark.
* **[requirements.txt](requirements.txt)**: Core and optional Python dependencies.
* **[tests/test_oracle_and_serialization.py](tests/test_oracle_and_serialization.py)**: Fast correctness test suite and independent exact numerical statevector oracle.
* **[tests/test_clifford_tableau.py](tests/test_clifford_tableau.py)**: Phase 2 Clifford stabilizer tableau test suite comparing against dense statevector oracle.
* **[parallel_contraction_research/](parallel_contraction_research/)**: Workspace root for parallel tensor network contraction.
  * **[proper_research/](parallel_contraction_research/proper_research/)**: Consolidated research code, requirements, and Project.toml.
  * **[hybrid_python_julia_design.md](parallel_contraction_research/hybrid_python_julia_design.md)**: Design blueprint for hybrid Python-Julia execution.
* **[quantum_circuit_research/](quantum_circuit_research/)**: Quantum circuit spinoff research, 7x7 profiling sweep, plotting tools, and Clifford engine.
  * **[src/tableau_simulator.py](quantum_circuit_research/src/tableau_simulator.py)**: Phase 2 binary symplectic stabilizer tableau baseline (Aaronson & Gottesman 2004).
  * **[benchmark_clifford_tableau.py](quantum_circuit_research/benchmark_clifford_tableau.py)**: Minimal reproducible benchmark contrasting tableau behavior across qubit counts.
* **[.github/workflows/](.github/workflows/)**: Continuous integration smoke benchmark workflow.
* **[LICENSE](LICENSE)**: Project license (MIT License).

