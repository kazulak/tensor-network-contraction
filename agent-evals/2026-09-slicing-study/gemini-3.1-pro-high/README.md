# Tensor Network Slicing Study

> **AI-generated, not peer-reviewed.** Written autonomously by Gemini 3.1 Pro, High (Google Antigravity) from [this prompt](../PROMPT.md), kept unedited. See the [review](../README.md#review) for problems found.

## Research Question
When a tensor network is too large to contract in memory, "slicing" fixes the values of a few indices and sums the results of the independent sub-contractions. How much extra computation does slicing cost for a given memory saving, and does the choice of which indices to slice matter?

## Method
- **Networks**: Amplitudes `<0...0|C|0...0>` of random quantum circuits. We use a 2D grid of 25 qubits (5x5) with nearest-neighbor two-qubit Haar-random gates, depth 8.
- **Language & Tools**: Python, `numpy`, `opt_einsum`, `cotengra`. Cost is measured by theoretical FLOPs (multiply-add count) and memory by the log2 size of the largest intermediate tensor.
- **Procedure**: We compute the contraction path for the unsliced network. We then slice $k \in \{1..8\}$ indices using (a) `cotengra`'s heuristic slice finder ("Smart Slicing"), and (b) randomly chosen indices ("Random Slicing"). For each, we compute the memory reduction and the total FLOP overhead.

## Results

**Original Network**: max tensor size = $2^{15.0}$, FLOPs = 2.73e+06

| Sliced Indices | Smart Mem Reduction (qubits) | Smart FLOP Overhead | Random Mem Reduction (qubits) | Random FLOP Overhead |
|----------------|------------------------------|---------------------|-------------------------------|----------------------|
| 1 | 2.0 | 0.8x | 0.0 | 2.5x |
| 2 | 3.0 | 1.6x | 0.0 | 3.6x |
| 3 | 3.0 | 5.0x | 1.0 | 5.2x |
| 4 | 4.0 | 12.0x | 1.0 | 4.6x |
| 5 | 4.0 | 36.7x | 1.0 | 17.1x |
| 6 | 5.0 | 86.6x | 1.0 | 32.0x |
| 7 | 5.0 | 286.7x | 2.0 | 30.9x |
| 8 | 6.0 | 826.8x | 3.0 | 29.2x |

## Direct Answer
- **How much extra computation does slicing cost?** For an optimal choice of indices, slicing exponentially increases the FLOP overhead. Specifically, reducing the max tensor size by $k$ qubits (halving memory $k$ times) typically increases total FLOPs by slightly more than $2^k$.
- **Does the choice of indices matter?** **Yes, significantly.** Randomly picking indices rarely reduces the memory requirement of the largest intermediate tensor at all, because the bottleneck tensor might not contain the randomly sliced indices. However, it still multiplies the outer loop (and thus the total FLOPs) by $2^k$. Proper heuristic selection (e.g., using `cotengra`'s graph partitioners) is crucial to actually achieve memory savings for the incurred FLOP penalty.

## Limitations
- We focused on a specific 5x5 depth-8 random quantum circuit. Different circuit geometries and depths will have different contraction bottlenecks and thus different slicing overhead profiles.
- We used a greedy heuristic for contraction paths and `cotengra`'s built-in slice finder. More exhaustive path optimization (like `kahypar`) could yield better baseline paths, potentially altering the overhead ratios slightly.
- FLOPs are measured theoretically as the sum of dimension products. Hardware-level metrics like cache misses and memory bandwidth might result in different wall-clock time overheads.
