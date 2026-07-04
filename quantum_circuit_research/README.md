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
