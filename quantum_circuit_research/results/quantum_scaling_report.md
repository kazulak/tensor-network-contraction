# Spinoff Paper: Tensor Network Contraction Complexity of Quantum Algorithms and Protocols

**Date:** July 2026  
**Author:** Antigravity Spinoff Research Group  

---

## 1. Executive Summary
Tensor network (TN) methods are widely used to simulate quantum circuits. While random quantum circuits are designed to maximize entanglement and treewidth (making classical simulation hard), standard quantum algorithms and communication protocols are highly structured. 

This report presents a systematic 7x7 empirical profiling sweep (7 topologies, 7 sizes per topology) to evaluate the contraction cost progression of **6 standard quantum algorithms/protocols** (BB84, Bernstein-Vazirani, EDC, Hidden Subgroup, QRNG, Exclusive-OR) against a **Google Sycamore grid reference**.

Our findings demonstrate that:
1. **The 99% Root Contraction Bottleneck is Trivial**: Across all algorithmic networks, the final step collapses indices to a scalar and takes under **4.24%** of the budget.
2. **Structural Treewidth Simplicity**: Standard quantum algorithms are characterized by extremely low treewidths ($w \le 2$), rendering their sequential contraction incredibly cheap ($<0.5\text{ms}$) and making parallel slicing inefficient due to thread-spawning overhead.
3. **Entanglement vs. Geometry**: Topologies with high 2D geometric connectivity (Sycamore) are the only networks exhibiting the S-curve timing jump characteristic of heavy tensor bottlenecks.

---

## 2. Quantitative Benchmark Results

The table below summarizes the total sequential contraction time, the time spent on the final step, and the corresponding percentage of total compute time for the largest scale ($N=18$ or $5\times 5$ grid) of each topology:

| Circuit Topology | Largest Size | Total Time (s) | Final Step Time (s) | Final Step % | Peak Intermediate Step % |
|---|---|---|---|---|---|
| **BB84 Protocol (BB_n)** | N=18 | 0.0002s | 0.00000008s | **0.04%** | **5.22%** |
| **Bernstein-Vazirani (BV_n)** | N=18 | 0.0005s | 0.000019s | **3.76%** | **8.12%** |
| **Error Detection Code (EDC_n)** | N=18 | 0.0005s | 0.000017s | **3.40%** | **7.60%** |
| **Hidden Subgroup (HS_2n)** | N=18 | 0.0005s | 0.00000005s | **0.01%** | **4.22%** |
| **Quantum Random Gen (QRNG_n)**| N=18 | 0.0001s | 0.00000005s | **0.05%** | **4.90%** |
| **Exclusive-OR (XOR_n)** | N=18 | 0.0005s | 0.000021s | **4.24%** | **9.12%** |
| **Sycamore-like Reference** | 5x5, D=6 | 0.0019s | 0.000020s | **1.04%** | **14.10%** |

---

## 3. Visualization

The plot below shows:
* **Row 1 (Cumulative Timing in %)**: Directly compares the cumulative sequential (solid lines) vs. sliced parallel (dashed lines) timing shapes for all 7 topologies.
* **Row 2 (Absolute Execution Time in seconds)**: Uses a logarithmic y-axis to show the exact execution gap in seconds between sequential (solid) and parallel (dashed) execution.

![Quantum Circuit Scaling Progression](/home/tom/.gemini/antigravity-cli/brain/b65b070e-93d8-4296-adfa-19d50dd59548/quantum_cost_progression_v2.png)

---

## 4. Key Physical & Graph-Theoretic Insights

### 4.1. Zero-Entanglement Topologies (QRNG and BB84)
* **Physical Insight**: QRNG and BB84 protocols consist entirely of single-qubit gates (superposition state preparation and bases measurements). Since there are $0$ two-qubit gates, there is no entanglement.
* **Graph-Theoretic Impact**: The tensor network splits into $N$ completely independent 1D paths. The treewidth is exactly 1. Contraction time scales strictly linearly with qubit count $N$, taking less than 200 microseconds. Parallel sliced execution (dashed lines) shows a massive performance penalty because thread spawning and slicing coordination overhead dwarf the actual arithmetic.

### 4.2. Star and Tree Topologies (BV, XOR, and EDC)
* **Physical Insight**: 
  * Bernstein-Vazirani uses CNOT oracle queries from $N-1$ qubits targeting 1 central qubit. Structurally, this forms a **star graph** centered on the target qubit.
  * XOR parity check circuits use CNOT trees to sum values into a single register.
* **Graph-Theoretic Impact**: Despite having $O(N)$ two-qubit gates, their bipartite star/tree connectivity keeps the treewidth bounded at $w \le 2$. The contraction optimizer can easily collapse the leaves of the tree sequentially, resulting in flat cumulative cost progression and sub-millisecond execution times.

### 4.3. The 2D Grid Scaling Exception (Sycamore-like)
* **Physical Insight**: The Sycamore-like grid arranges qubits in 2D with nearest-neighbor entangling gates. 
* **Graph-Theoretic Impact**: Unlike the other 6 linear or star-like topologies, 2D grids have a treewidth that grows as $O(\sqrt{N})$. As a result, the Sycamore reference is the only curve that exhibits a distinct S-curve shape in the cumulative plots, with intermediate steps consuming up to **14%** of the entire budget. At larger scales, it is the only topology that benefits from parallel slicing.

---

## 5. Spinoff Conclusions & Future Research Directions
This study establishes that:
* **Quantum algorithms are structurally simple**: Standard algorithms and communication protocols are designed for linear or tree-like information routing, keeping their tensor network representation extremely cheap to contract sequentially ($O(N)$ or $O(N^2)$).
* **Parallelization overhead dominates**: Slicing parallelization is a major regression for these low-treewidth networks due to threading overhead.

### Next Spinoff Research Direction: Graph-Theoretic Adaptive Slicers
Future spinoff work should focus on **adaptive slicing algorithms** that evaluate the graph treewidth before committing to a parallelization strategy. If the predicted treewidth is $w \le 3$, the compiler should disable slicing and execute sequentially, reserving parallel slicing and BLAS multithreading exclusively for high-treewidth geometries like 2D grids and dense random graphs.
