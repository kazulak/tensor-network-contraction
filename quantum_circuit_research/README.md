# Spinoff Research: Tensor Network Contraction of Quantum Circuits

This spinoff project investigates the **step-by-step cost accumulation and parallelization limits** of tensor networks representing quantum circuits.

---

## 1. Quantum Circuit Topologies
We evaluate 7 distinct quantum circuit classes:
1. **1D Random Quantum Circuit (MPS-like)**: Nearest-neighbor CNOT and random $O(2)$ rotation layers in 1D. Bounded treewidth.
2. **2D Random Quantum Circuit (PEPS-like)**: Alternating horizontal and vertical nearest-neighbor CZ layers in 2D. Treewidth scales as $O(\sqrt{N})$.
3. **Clifford Circuit**: Composed entirely of Hadamard and CNOT gates. Classic stabilizer state simulation is polynomial, but tensor contraction remains exponentially complex.
4. **Shallow Entanglement**: Product states with only 1 layer of 2-qubit gates, followed by heavy single-qubit rotations. Very low contraction complexity.
5. **Quantum Fourier Transform (QFT)**: Standard QFT circuit. High long-range connectivity causes rapid treewidth growth.
6. **Sycamore-like**: Simulated Google Sycamore patterns in 2D grid, optimized for rapid entanglement.
7. **Random Circuit (Arbitrary Connectivity)**: Random $O(4)$ gates applied to arbitrary random pairs of qubits. Rapidly saturates treewidth.

---

## 2. File Structure
* `src/circuit_generators.py`: Generators for the 7 topologies as closed networks.
* `src/exporter.py`: Cotengra path optimizer and binary data exporter.
* `src/step_profiler.jl`: Thread-safe, JIT-warmed atomic step profiler.
* `run_quantum_circuit_sweep.py`: Orchestrates the 7x7 profiling sweep.
* `plot_quantum_circuits.py`: Generates the 2x7 comparative progression plots.
* `reproduce.sh`: Reproduction entry-point bash script.
* `results/`: contains raw data and the final scientific report.

---

## 3. How to Reproduce
To clean intermediate data, run the 49 profiling configurations, and generate the plots, run:
```bash
./reproduce.sh
```
All visual scaling progression curves will be saved to `results/quantum_cost_progression_v1.png`.
The final scientific analysis is documented in `results/quantum_scaling_report.md`.

---

## 4. Phase 2: Polynomial-Time Stabilizer Tableau Baseline for Clifford Circuits

### Primary Reference & Citation
* **Paper**: Scott Aaronson and Daniel Gottesman, *"Improved Simulation of Stabilizer Circuits"*, Phys. Rev. A 70, 052328 (2004) [[arXiv:quant-ph/0406196](https://arxiv.org/abs/quant-ph/0406196)].

### Algorithm & Data Representation
The Phase 2 engine implements an explicit binary symplectic stabilizer tableau baseline for Clifford-only circuits:
* **Tableau Matrix**: A $(2N) \times (2N + 1)$ binary numpy array (`np.uint8`) representing $N$ destabilizers $R_1 \dots R_N$, $N$ stabilizers $R_{N+1} \dots R_{2N}$, and sign bit vector $r \in \{0, 1\}^{2N}$.
* **Supported Gate IR**: $H, S, S^\dagger, X, Y, Z, CX/CNOT, CZ$.
* **Gate Classification Adapter**: `from_tensor_network_gates` classifies $2 \times 2$ and $4 \times 4$ gate matrices and converts tensor network generator outputs to `CliffordCircuit`. Non-Clifford gates (such as continuous $O(2)$ or $U(4)$ rotations) are explicitly rejected with `NonCliffordGateError`.

### Asymptotic Costs
* **Storage Space**: $O(N^2)$ bits ($\approx (2N) \times (2N+1)$ bits, $< 1 \text{ KB}$ for $N=64$, $\sim 260 \text{ KB}$ for $N=1024$).
* **Gate Simulation Time**: $O(N)$ row operations per 1-qubit or 2-qubit gate (total $O(M \cdot N)$ for $M$ gates).
* **Exact Bitstring Probability $P(b)$**: $O(N^3)$ polynomial time deterministically via conditional measurement row-reduction chain.
* **Seeded Sampling**: $O(M_{\text{shots}} \cdot N^2)$ time.
* **Comparison with Dense Statevector**: Dense statevector requires $O(2^N)$ memory ($16 \text{ GB}$ for $N=30$, $> 1 \text{ TB}$ for $N=36$) and $O(M \cdot 2^N)$ FLOPs, scaling exponentially while the tableau engine remains polynomial.

### Dispatch Boundary & Semantic Constraints
* **Outcome Probabilities vs Closed Contraction**: The tableau baseline is specialized for **state evolution, computational-basis bitstring probabilities $P(b) = |\langle b | \psi \rangle|^2$, and seeded sampling**.
* **Generic Tensor Network Path**: The closed-network scalar amplitude $\langle \text{bra} | \psi \rangle$ remains on the generic tensor network path because tensor networks naturally evaluate closed contractions across arbitrary boundary projections.
* **Safety**: Automatic dispatch is specialized only for outcome probabilities.

### Minimal Reproducible Benchmark
Run the reproducible benchmark contrasting tableau behavior across qubit counts $N \in [4 \dots 1024]$:
```bash
python quantum_circuit_research/benchmark_clifford_tableau.py
```
**Output Summary**:
* $N \le 12$: Validated 100% against dense statevector oracle (0.0000s difference).
* $N = 16 \dots 1024$: Tableau simulation completes in milliseconds ($\sim 0.05 \text{s}$ at $N=1024$), while dense statevector encounters out-of-memory (OOM) limits.

**Limitations**:
* Restricted exclusively to the Clifford group. Non-Clifford gates (e.g. $T$ gate, $R_x(\theta)$) cannot be represented without stabilizer decomposition gadgets.

