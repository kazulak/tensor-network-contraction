# Round 1: Mathematician

## Summary
The pedagogical progression across topics 00–03 is conceptually strong, taking the reader from matrix multiplication primitives to circuit networks, contraction complexity, and slicing trade-offs with 100% test pass rates across exact references. However, mathematical precision needs sharpening in several foundational places: Markov & Shi's treewidth theorem is quoted without its essential bounded-degree hypothesis or distinction between circuit graph and line graph treewidth, and key bounds such as the $L \le W \le L+1$ grid contraction width and the $C_s \ge C$ slicing overhead are verified empirically without the short combinatorial and algebraic arguments that explain why they hold. Furthermore, Topic 00 introduces graph colouring without its complexity-theoretic motivation (#P-hardness of tensor contraction) and runs an unnecessary $4^{10}$-state brute-force search that can be replaced with the exact chromatic polynomial. Bridging these gaps and clarifying the thematic pivot in PLAN.md from exact contraction to approximate low-rank truncation (Eckart–Young / Schmidt decomposition) will give students a rigorous, unified foundation in linear algebra and complexity.

## Findings

### M1: Treewidth bound requires bounded degree and line graph formulation
- Topic: 02
- Type: ERROR
- Severity: high
- Evidence: `topics/02-contraction-order-and-cost/README.md:6` quotes "Markov & Shi, arXiv:quant-ph/0511069, Theorem 1.1: the best contraction costs exp(O(treewidth))." and `topics/02-contraction-order-and-cost/order.py:1-5` states "the best achievable cost is set by the graph (treewidth)."
  - In graph theory and complexity, the contraction complexity of an edge-contraction sequence on graph $G$ equals the treewidth of its *line graph* $G^*$ (Proposition 4.2 of Markov & Shi: $cc(G) = \mathrm{tw}(G^*)$), which is equivalent to the branch-width / carving-width of $G$.
  - Markov & Shi Lemma 4.4 and Theorem 4.5 show that $(\mathrm{tw}(G) - 1)/2 \le cc(G) = \mathrm{tw}(G^*) \le \Delta(G)(\mathrm{tw}(G) + 1) - 1$. The bound $\exp(O(\mathrm{tw}(G)))$ holds *if and only if* the maximum vertex degree $\Delta(G)$ is bounded by a constant.
  - Counterexample: For a star graph $K_{1, m}$ (representing an $m$-qubit gate or high-degree copy tensor), the graph treewidth is $\mathrm{tw}(K_{1, m}) = 1$, yet contracting the network forces a rank-$m$ intermediate tensor with cost $2^m$. Its line graph is the complete graph $K_m$, with $\mathrm{tw}(K_m) = m - 1$.
- Proposal: Clarify the definitions in `README.md` and `order.py`:
  1. State that optimal contraction width $W$ corresponds to the branch-width of the tensor network graph $G$, which equals the treewidth of its line graph $G^*$.
  2. State explicitly that Markov & Shi's Theorem 1.1 bounds contraction cost by $\exp(O(\mathrm{tw}(G)))$ under the vital hypothesis that gate arity is bounded ($\Delta(G) = O(1)$), which holds for 1- and 2-qubit circuits ($\Delta \le 4$).

### M2: Prove the grid width invariant $L \le W_{\text{opt}} \le W_{\text{row}} \le L+1$ via isoperimetry and row-frontier invariants
- Topic: 02
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/02-contraction-order-and-cost/test_order.py:46-52` asserts `assert L <= W_opt <= W_row <= L + 1` for $L \in \{2, 3, 4, 5\}$, and `topics/02-contraction-order-and-cost/README.md:37-38` notes that width satisfies $L \le W \le L+1$. The output of `python topics/02-contraction-order-and-cost/order.py` confirms this for $L=2\dots 6$:
  - $L=2 \implies W_{\text{opt}}=2, W_{\text{row}}=2$
  - $L=3 \implies W_{\text{opt}}=4, W_{\text{row}}=4$
  - $L=4 \implies W_{\text{opt}}=4, W_{\text{row}}=5$
  - $L=5 \implies W_{\text{opt}}=6, W_{\text{row}}=6$
  - $L=6 \implies W_{\text{opt}}=6, W_{\text{row}}=7$
  No explanation is provided for why this tight interval holds.
- Proposal: Add a short, elegant combinatorial explanation to `README.md`:
  1. **Lower bound ($W_{\text{opt}} \ge L$):** Any contraction tree defines a sequence of vertex cuts. At the median contraction step where $L^2 / 3 \le |S| \le 2L^2 / 3$, the edge isoperimetric theorem on the 2D grid graph guarantees that the boundary $|\delta(S)| \ge L$. Hence any binary contraction tree must produce an intermediate tensor with at least $L$ open legs.
  2. **Upper bound ($W_{\text{row}} \le L+1$):** In row-by-row bubbling, when contracting vertex $k \in \{1, \dots, L-1\}$ in row $r$, the active frontier consists of $L-k$ downward edges from row $r-1$, $k$ downward edges to row $r+1$, and at most 1 horizontal edge in row $r$. The active frontier size is identically $(L-k) + k + 1 = L+1$ (and exactly $L$ at row boundaries).
  This proves $W_{\text{row}} \le L+1$ for all $L$, demonstrating that simple row-by-row bubbling is within an additive constant $+1$ of the NP-hard optimal contraction width.

### M3: Formulate the exact per-step slicing cost formula to explain $C_s \ge C$ and overhead explosion
- Topic: 03
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/03-slicing/README.md:8-9` states "each costs *at least* C/d_sliced, so the total work C_s ≥ C" without showing where this bound comes from. `topics/03-slicing/slicing.py:59-68` computes $C_s = 2^{|s|} \sum_{v} 2^{|(s_l(v) \cup s_r(v)) \setminus s|}$.
- Proposal: Include the exact algebraic breakdown in `README.md`:
  For any pairwise contraction step $v$ in tree $B$ with unsliced cost $C(v) = 2^{|s_l(v) \cup s_r(v)|}$, slicing an index set $s$ results in total work across all $d_{\text{sliced}} = 2^{|s|}$ slices:
  $$C_s(v) = 2^{|s \setminus (s_l(v) \cup s_r(v))|} C(v)$$
  - Steps containing all sliced indices break even ($2^0 C(v) = C(v)$).
  - Steps containing none of the sliced indices are duplicated $2^{|s|}$ times ($2^{|s|} C(v)$).
  Summing over all steps, $C_s = \sum_v C_s(v) \ge \sum_v C(v) = C$ follows immediately because every multiplier $2^{|s \setminus (s_l \cup s_r)|} \ge 1$.
  This formulation directly explains why greedy slicing on bottleneck edges achieves an $8\times$ memory reduction with only a $14\%$ FLOP increase (bottleneck steps break even), while random slicing produces a catastrophic $10^{25}\times$ blowup (unrelated slices double the entire circuit's work).

### M4: Motivate graph colouring via #P-hardness and replace Petersen brute force with chromatic polynomial
- Topic: 00
- Type: SIMPLIFY
- Severity: medium
- Evidence: `topics/00-tensor-network-basics/README.md:17-18,28-30` introduces the graph-colouring network without explaining why graph colouring belongs in a tensor network tutorial. `topics/00-tensor-network-basics/basics.py:120-124` runs `count_colourings_brute(PETERSEN, 10, q)` for $q \in (2, 3, 4)$. For $q=4$, it evaluates $4^{10} = 1,048,576$ states in pure Python, causing `python basics.py` to freeze for ~7 seconds. Consequently, `topics/00-tensor-network-basics/test_basics.py:47-51` omits $q=4$ because brute force is too slow for tests. Bridgeman & Chubb (arXiv:1603.03039) §1.5 explicitly uses graph colouring to prove that exact tensor network contraction is #P-hard.
- Proposal:
  1. Add 2 sentences to `README.md` explaining the complexity connection: evaluating a tensor network with copy tensors ($e$) on vertices and constraint tensors ($n = \mathbf{1} - I$) on edges solves counting CSPs (#3-SAT, #$q$-colouring for $q \ge 3$), proving exact tensor network contraction is #P-hard.
  2. Replace `count_colourings_brute` on the Petersen graph with its known chromatic polynomial:
     $$P(\text{Petersen}, q) = q(q-1)(q-2)(q^7 - 12q^6 + 67q^5 - 230q^4 + 529q^3 - 814q^2 + 775q - 352)$$
     Evaluating this polynomial takes $< 1\ \mu\text{s}$, eliminates the 7-second runtime pause in `basics.py`, and allows `test_basics.py` to test $q=4$ (12960) and $q=5$ (117120) with exact mathematical verification.

### M5: Derive contraction cost from matrix multiplication linear algebra and accumulate cost in bubbling
- Topic: 00
- Type: MISSING
- Severity: low
- Evidence: `topics/00-tensor-network-basics/README.md:6-8` states: "Every pairwise contraction is a transpose, a reshape and a matrix product. Its cost is the product of the dimensions of all legs involved." `topics/00-tensor-network-basics/basics.py:30-40` defines `bubble`, which only tracks `max_rank` and never calculates or returns the contraction cost. `topics/00-tensor-network-basics/test_basics.py:19-25` tests `pair_cost` for matrix chains, but `basics.py` never outputs operation counts for the ladder network.
- Proposal:
  1. In `README.md`, provide the standard linear algebra derivation: partitioning legs into free $I$, contracted $K$, and free $J$ reshapes $A$ to $(d_I \times d_K)$ and $B$ to $(d_K \times d_J)$. Standard matrix multiplication requires $d_I \cdot d_K \cdot d_J = \prod_{l \in a \cup b} d_l$ scalar operations, storing an intermediate of size $d_I \cdot d_J = \prod_{l \in a \oplus b} d_l$.
  2. Update `bubble` in `basics.py` to accumulate total multiply-adds alongside `max_rank`, and display both rank and cost in the ladder table. For length $n$, the "along" order incurs $O(2^n)$ cost while the rung-by-rung order incurs $O(n)$ cost, demonstrating the time scaling promised in the README.

### M6: Demarcate the pivot from exact contraction (00-04) to approximate methods (05-07) in the roadmap
- Topic: roadmap
- Type: SEQUENCE
- Severity: medium
- Evidence: `PLAN.md:15-28` lists topics 00–04 (exact contraction, tree width, slicing, parallelism) followed immediately by 05–07 (MPS, SVD truncation, DMRG) without framing the conceptual discontinuity.
- Proposal: Explicitly structure the roadmap in `PLAN.md` into two distinct mathematical regimes:
  - **Part I: Exact Contraction of Tensor Networks (00–04).** Bounded by graph treewidth; slicing trades space for time, but total FLOPs obey $C_s \ge C$. Exact simulation of deep 2D circuits hits the exponential wall.
  - **Part II: Controlled Approximation and Matrix Product States (05–07).** When exact contraction is computationally intractable, classical simulation relies on low-rank matrix approximation via truncated SVD (the Eckart–Young–Mirsky theorem). The state is projected onto an MPS whose bond dimension $\chi$ is bounded by entanglement entropy (area law).
  Adding this bridge in `PLAN.md` explains to the newcomer why the mathematical framework fundamentally shifts from tree decompositions to SVD truncation.

## Keep as is
- **Topic 00 (`basics.py`):** The pairwise contraction engine `contract_pair` using `transpose`, `reshape`, and `@` is minimal, clear, and perfectly matches `np.einsum` on complex tensors. The matrix-chain associativity test (`test_matrix_chain_cost`) grounds the lesson directly in standard dynamic programming.
- **Topic 01 (`circuits.py`):** The Haar-unitary generation via QR decomposition with diagonal phase normalization ($Q \operatorname{diag}(R/|R|)$) correctly implements Mezzadri's theorem (arXiv:math-ph/0609050). The dual validation of closed networks (individual amplitudes) vs open networks (full statevector) against the analytic DFT formula ($F_{x, k} = \frac{1}{\sqrt{N}} e^{2\pi i x k / N}$) is textbook-grade pedagogy.
- **Topic 02 (`order.py`):** The direct implementation of edge congestion $W$ (max intermediate size) and vertex congestion $C$ (total multiply-adds) following Gray & Kourtis Eqs. (2)–(5) using `opt_einsum` contraction paths. The ring benchmark demonstrating invariant width $W=2$ across all $n$ gives students an immediate reference point before facing the grid.
- **Topic 03 (`slicing.py`):** The greedy slicing implementation on the 2D grid circuit, reproducing the empirical observation from Gray & Kourtis Fig. 10 that the first few slices yield an $8\times$ memory reduction with only a $14\%$ FLOP penalty. The exact check `contract_sliced` summing over all $2^{|s|}$ sub-networks against `statevector` provides rigorous proof that slicing preserves the quantum amplitude.

## Suggested learning path
- 00: Tensor network basics — tensor legs as vector spaces; pairwise contraction as transpose-reshape-GEMM; order sets intermediate rank and cost; exact contraction is #P-hard via graph colouring.
- 01: Circuits as tensor networks — quantum gates as tensors; wires as shared indices; closed network yields one amplitude $\langle x|C|0\dots 0\rangle$ while open network yields the full state vector.
- 02: Contraction order and cost — contraction trees; width $W$ (peak memory) and cost $C$ (FLOPs); graph treewidth sets the lower bound ($W \ge L$ on an $L \times L$ grid).
- 03: Slicing — projecting index values to trade memory for FLOPs ($W_s < W$ at cost $C_s \ge C$); greedy selection targets bottleneck edges.
- 04: Parallel contraction — distributing independent slices across hardware; communication-free scaling and Amdahl limits.
- 05: MPS and SVD truncation — low-rank matrix approximation via truncated SVD (Eckart–Young–Mirsky); bipartite Schmidt decomposition and discarded weight error.
- 06: MPS circuit simulation — 1D circuit simulation with bond dimension $\chi$; entanglement entropy growth vs truncation error (Vidal 2003).
- 07: DMRG — variational optimization of 1D tensor networks; sweeping local eigensolvers to find ground states (Schollwöck 2011).
