# Round 2: Senior Software Engineer

## Responses

- P1 (Ladder network not physically identified as MPS norm and transfer operator): AGREE. In [00-tensor-network-basics/basics.py](topics/00-tensor-network-basics/basics.py#L51-L75), the ladder is constructed and contracted purely as an abstract graph exercise. Bridgeman & Chubb §1.4 introduce this network immediately after discussing Matrix Product States (MPS). Adding a concise 3-sentence note in [00-tensor-network-basics/README.md](topics/00-tensor-network-basics/README.md#L17-L18) identifying the rungs as physical qubit legs, rails as virtual bonds, rung-by-rung bubbling as the local transfer-matrix / environment contraction, and top-row bubbling as dense state-vector generation establishes critical architectural intuition for Topic 05 without modifying code.

- P2 (Graph colouring lacks context: Potts model partition function and #P-hardness): AGREE. From an educational and engineering lens, diving into 50 lines of graph theory in Topic 00 creates cognitive friction for readers expecting quantum circuit simulation. Bridgeman & Chubb §1.5 introduced the copy tensor [`e`](topics/00-tensor-network-basics/basics.py#L90-L93) and inequality tensor [`n`](topics/00-tensor-network-basics/basics.py#L85) under "Computational Complexity" specifically to prove that contracting an arbitrary tensor network is #P-hard. Stating this complexity result in [00-tensor-network-basics/README.md](topics/00-tensor-network-basics/README.md#L17-L18) justifies the presence of the colouring network—demonstrating why classical circuit simulation faces an exponential barrier—without touching code.

- P3 (Missing state norm conservation check and physical observable expectation values): REFINE. We agree with part 1: adding an explicit check [`np.isclose(np.linalg.norm(psi), 1.0)`](topics/01-circuits-as-tensor-networks/test_circuits.py#L42) in [`test_circuits.py`](topics/01-circuits-as-tensor-networks/test_circuits.py) is a 3-line addition that enforces Rule 4 of [GUIDELINES.md](GUIDELINES.md#L25-L26) ("norms or fidelities reported") and guards against normalization bugs.
  However, we push back on implementing folded networks $\langle 0 | C^\dagger O C | 0 \rangle$ for observable expectation values inside [circuits.py](topics/01-circuits-as-tensor-networks/circuits.py). Topic 01's stated idea in [PLAN.md](PLAN.md#L20) is establishing the mapping from unitary gates to open and closed tensor networks. Implementing adjoint circuit construction, operator insertion, and conjugate leg wiring would double the complexity of [circuits.py](topics/01-circuits-as-tensor-networks/circuits.py), violating Rule 2's under-200-lines simplicity directive. Explaining expectation values conceptually in [01-circuits-as-tensor-networks/README.md](topics/01-circuits-as-tensor-networks/README.md) as an extension to Markov & Shi §3 is the right boundary.

- P4 (Physical discrepancy between Haar-random $U(4)$ gates and Sycamore architecture): AGREE. In [03-slicing/README.md](topics/03-slicing/README.md#L15-L16) and [slicing.py](topics/03-slicing/slicing.py#L21), the circuit is labelled "Sycamore-like". Physical Google Sycamore circuits employ cycles of single-qubit rotations and entangling fSim gates whose operator Schmidt rank is 2 (Gray & Kourtis §4.6.2), which allows an SVD rank-2 split that dramatically reduces contraction width. Dense Haar $U(4)$ gates generically have full operator Schmidt rank 4. Adding a short note in [03-slicing/README.md](topics/03-slicing/README.md) explaining that "Sycamore-like" refers strictly to the 2D grid ABCD coupler geometry, and that dense Haar unitaries represent a pessimistic worst-case benchmark without low-rank structure, prevents misconceptions without altering code.

- P5 (Clarify the origin of the slicing work overhead ratio $C_s / C$): AGREE. In [03-slicing/README.md](topics/03-slicing/README.md#L8-L9), stating that each slice costs at least $C / d_{\text{sliced}}$ leaves readers baffled as to why slicing $2^{20} \approx 10^6$ slices only inflates total work by $C_s / C = 1345\times$. Explaining that fixing 20 bottleneck indices shrinks intermediate tensors throughout the tree so that each individual slice costs only $\sim 0.13\%$ of the unsliced contraction ($C_{\text{slice}} \approx 1345 \cdot C / 2^{20}$) resolves the apparent paradox. This high-level intuition works hand-in-hand with M3's per-step algebraic formula.

- P6 (Missing conceptual bridge between exact contraction and approximate MPS): AGREE. Unanimous across all three panellists (P6, M6, E8). The repository pivots abruptly at Topic 05 from exact spacetime contraction to approximate 1D state evolution. Bounding the two regimes in [PLAN.md](PLAN.md#L17-L28)—Part I (00–04) bounded by graph treewidth and cut boundaries; Part II (05–07) bounded by entanglement entropy and Eckart–Young rank truncation—gives newcomers a coherent architectural roadmap.

- P7 (Prerequisite for Topic 05: Gauge freedom and canonical form): AGREE. In numerical tensor network algorithms, local SVD truncation of an MPS is optimal under the Eckart–Young–Mirsky theorem if and only if the MPS is in canonical (left- or right-orthogonal) form ($A^\dagger A = I$) (Schollwöck §4.2–§4.5). Truncating an un-gauged MPS throws away significant fidelity and corrupts the state norm. Specifying canonical form via QR sweeps as an explicit prerequisite in [PLAN.md](PLAN.md#L24) prevents a critical algorithmic flaw when implementing Topic 05.

- P8 (Disambiguate Topic 06 as TEBD (Vidal 2003)): AGREE. Applying 2-qubit gates to an MPS followed by SVD truncation is the discrete-time Time-Evolving Block Decimation (TEBD) algorithm (Vidal 2003, cited in Topic 06). Having Topic 08 list "TEBD" as an unassigned optional idea creates duplicate and conflicting terminology. Renaming Topic 06 to "MPS circuit simulation (TEBD)" in [PLAN.md](PLAN.md#L25) clarifies the topic and frees Topic 08 for genuinely distinct methods (e.g. PEPS boundary contraction or Clifford tableaus).

- P9 (Remove $4^{10}$ brute-force loop in `basics.py` to prevent script freeze): REFINE. Rerunning `python topics/00-tensor-network-basics/basics.py` hangs for 28.06 seconds, of which ~27.5 seconds are spent in [`count_colourings_brute(PETERSEN, 10, 4)`](topics/00-tensor-network-basics/basics.py#L97-L99) evaluating $4^{10} = 1,048,576$ states in pure Python.
  However, we push back against M4's proposal to delete [`count_colourings_brute`](topics/00-tensor-network-basics/basics.py#L97-L99) in favor of the 7th-degree chromatic polynomial. Rule 3 of [GUIDELINES.md](GUIDELINES.md#L23-L24) explicitly requires an exact brute-force reference at small sizes: [`count_colourings_brute`](topics/00-tensor-network-basics/basics.py#L97-L99) is a transparent 3-line check directly reflecting the definition of proper vertex colouring. A 7th-degree polynomial is an opaque external formula that cannot be validated from first principles by a newcomer.
  The best engineering fix is P9's first proposal: restrict the Petersen check in [basics.py:121](topics/00-tensor-network-basics/basics.py#L121) to `q in (2, 3)` (matching [`test_colourings_petersen`](topics/00-tensor-network-basics/test_basics.py#L47-L51), which completes in 0.57s), while retaining `q in (2, 3, 4)` for triangle and C6. If $q=4$ is displayed for Petersen, its known value 12960 can be printed with an explanatory note referencing the chromatic polynomial, preserving both interactive responsiveness and first-principles brute-force verification.

- P10 (Mitigate excessive contraction overhead in `test_slicing.py`): REFINE. P10's diagnosis of the root cause in [`test_slicing.py`](topics/03-slicing/test_slicing.py#L6-L18) is accurate and corrected our own misdiagnosis in E4: on the 9-qubit circuit ($W_0=6$), target width $W_0 - 3 = 3$ forces the greedy slicer to slice 12 indices ($2^{12} = 4096$ slices). Contracting 4096 slices across 3 bitstrings executes 12,288 contractions, consuming 4.67s (80% of total pytest time).
  However, P10's proposed fix—setting the target to `W0 - 1` and checking a single bitstring—has two flaws:
  1. In [test_slicing.py:15](topics/03-slicing/test_slicing.py#L15), the test asserts `assert len(sliced) >= 3`. Setting target to `W0 - 1` slices only 1 index, which causes this assertion to immediately fail.
  2. Testing only a single slice on a single bitstring fails to adequately test multi-index slicing across diverse computational states.
  The superior engineering fix is setting the target width to `W0 - 2`. We verified empirically that [`slice_greedy`](topics/03-slicing/slicing.py#L70-L79) with `target_W = W0 - 2` selects exactly 3 indices ($2^3 = 8$ slices). This satisfies `assert len(sliced) >= 3`, thoroughly exercises multi-index slicing across all 3 bitstrings (0, 5, 300), and executes in 0.02s instead of 4.67s, achieving a $230\times$ speedup without weakening test assertions or dropping bitstrings.

- M1 (Treewidth bound requires bounded degree and line graph formulation): AGREE. In graph theory and tensor network complexity, contraction complexity $cc(G)$ is identically the treewidth of the line graph $G^*$, i.e. $cc(G) = \mathrm{tw}(G^*)$ (Markov & Shi Prop 4.2). Markov & Shi Lemma 4.4 and Thm 4.5 establish that contraction cost is $\exp(O(\mathrm{tw}(G)))$ if and only if maximum vertex degree $\Delta(G)$ is bounded by a constant (which holds for 1- and 2-qubit quantum circuits where $\Delta \le 4$, but fails for high-degree copy tensors or star graphs $K_{1, m}$ where $\mathrm{tw}(G)=1$ yet contraction width is $m$). Adding this bounded-degree qualification and line graph definition to [02-contraction-order-and-cost/README.md](topics/02-contraction-order-and-cost/README.md#L6) ensures mathematical precision without adding code complexity.

- M2 (Prove the grid width invariant $L \le W_{\text{opt}} \le W_{\text{row}} \le L+1$ via isoperimetry and row-frontier invariants): AGREE. In [test_order.py:52](topics/02-contraction-order-and-cost/test_order.py#L52), the assertion `assert L <= W_opt <= W_row <= L + 1` appears as an unexplained empirical observation. M2 provides the exact combinatorial justification: the lower bound $W \ge L$ follows from the edge isoperimetric theorem on 2D grids at balanced cuts ($L^2/3 \le |S| \le 2L^2/3$), while the upper bound $W_{\text{row}} \le L+1$ is an exact invariant of the row-by-row bubbling frontier ($(L-k) + k + 1 = L+1$). Documenting this in [02-contraction-order-and-cost/README.md](topics/02-contraction-order-and-cost/README.md#L37-L38) explains why the test passes and demonstrates why naive row bubbling is within an additive $+1$ of NP-hard optimal contraction width.

- M3 (Formulate the exact per-step slicing cost formula to explain $C_s \ge C$ and overhead explosion): AGREE. The per-step formula $C_s(v) = 2^{|s \setminus (s_l(v) \cup s_r(v))|} C(v)$ maps directly to the implementation in [slicing.py:66-67](topics/03-slicing/slicing.py#L66-L67):
  ```python
  W, C = max(W, len(a ^ b)), C + 2 ** len(a | b)
  return W, C * 2 ** len(sliced)
  ```
  Inside the loop, $C$ accumulates the cost of a single sliced tensor network $\sum_v 2^{|(a \cup b) \setminus s|}$. When scaled by $2^{|s|}$, each contraction step $v$ incurs $2^{|s \setminus (a \cup b)|} 2^{|a \cup b|}$. Because $|s \setminus (a \cup b)| \ge 0$, every multiplier is $\ge 1$, proving $C_s \ge C$. Furthermore, it explains the 25-order-of-magnitude gap between greedy and random slicing: greedy slicing targets bottleneck edges where $|s \setminus (a \cup b)| = 0$ (multiplier 1), whereas random slicing cuts non-bottleneck edges where $|s \setminus (a \cup b)| = |s|$, duplicating work by $2^{|s|}$. Including this formula in [03-slicing/README.md](topics/03-slicing/README.md) provides an exact algebraic explanation of the code.

- M4 (Motivate graph colouring via #P-hardness and replace Petersen brute force with chromatic polynomial): REFINE. We agree with motivating graph colouring via #P-hardness in `README.md` (unanimous with P2) and eliminating the 28s freeze in `basics.py`.
  However, we disagree with replacing [`count_colourings_brute`](topics/00-tensor-network-basics/basics.py#L97-L99) with the 7th-degree chromatic polynomial in [test_basics.py](topics/00-tensor-network-basics/test_basics.py#L47-L51). As argued under P9, [`count_colourings_brute`](topics/00-tensor-network-basics/basics.py#L97-L99) is the first-principles brute-force check required by Rule 3 of [GUIDELINES.md](GUIDELINES.md#L23-L24). For $q=2$ and $q=3$, [`count_colourings_brute(PETERSEN, 10, q)`](topics/00-tensor-network-basics/basics.py#L97-L99) runs in <0.6s and directly validates the tensor network from the definition of proper vertex colouring. The chromatic polynomial formula for Petersen is a 7th-degree polynomial that cannot be verified by inspection. The better fix is to keep [`count_colourings_brute`](topics/00-tensor-network-basics/basics.py#L97-L99) in [test_basics.py](topics/00-tensor-network-basics/test_basics.py), restrict the interactive demo loop in [basics.py:121](topics/00-tensor-network-basics/basics.py#L121) to $q \in (2, 3)$, and optionally mention the chromatic polynomial evaluation $P(4) = 12960$ as an analytical reference.

- M5 (Derive contraction cost from matrix multiplication linear algebra and accumulate cost in bubbling): REFINE. We agree with deriving contraction cost from matrix multiplication in [00-tensor-network-basics/README.md](topics/00-tensor-network-basics/README.md#L7-L8): in [`contract_pair`](topics/00-tensor-network-basics/basics.py#L11-L23), the reshaped matrices have dimensions $(d_{\text{free\_a}} \times d_{\text{shared}})$ and $(d_{\text{shared}} \times d_{\text{free\_b}})$, so matrix multiplication explicitly requires $d_{\text{free\_a}} \cdot d_{\text{shared}} \cdot d_{\text{free\_b}} = \prod_{l \in a \cup b} d_l$ scalar multiply-adds, producing an intermediate of size $\prod_{l \in a \oplus b} d_l$.
  However, we disagree with modifying [`bubble`](topics/00-tensor-network-basics/basics.py#L30-L40) in [basics.py](topics/00-tensor-network-basics/basics.py) to accumulate and report cost. In Topic 00, Bridgeman & Chubb §1.4 specifically focuses on the *rank* (memory footprint) of the stored intermediate tensor (Fig. 1.15). Contraction cost $C$ (FLOPs) and contraction width $W$ are the dedicated subject of Topic 02 (Gray & Kourtis §2). Adding cost tracking to [`bubble`](topics/00-tensor-network-basics/basics.py#L30-L40) in Topic 00 blurs the boundary between Topic 00 (mechanics and memory rank) and Topic 02 (algorithmic cost $C$ and width $W$). Instead, the dead code [`pair_cost`](topics/00-tensor-network-basics/basics.py#L25-L27) in [basics.py:25-27](topics/00-tensor-network-basics/basics.py#L25-L27) should either be used in a simple matrix-chain cost example or removed, keeping Topic 00 strictly focused on rank and pairwise mechanics.

- M6 (Demarcate the pivot from exact contraction (00-04) to approximate methods (05-07) in the roadmap): AGREE. Unanimous agreement with P6, M6, and our own E8. Structuring [PLAN.md](PLAN.md#L17-L28) into Part I (Exact Contraction) and Part II (Controlled Approximation & MPS) establishes clear architectural boundaries and motivates why the mathematical paradigm shifts from tree decompositions to low-rank SVD truncation.

---

## Changes to my own findings

- **E4 (Refined root cause and proposal for slicing test latency):** In our round 1 finding E4, we attributed the 4.33s runtime in [`test_sum_over_slices_equals_amplitude`](topics/03-slicing/test_slicing.py#L6-L18) to repeated calls to [`find_path`](topics/03-slicing/slicing.py#L53-L57) and [`slice_greedy`](topics/03-slicing/slicing.py#L70-L79).
  Profiler verification demonstrates that [`find_path`](topics/03-slicing/slicing.py#L53-L57) takes only 0.002s and [`slice_greedy`](topics/03-slicing/slicing.py#L70-L79) takes 0.069s. As Physicist P10 observed, the real bottleneck is in [`contract_sliced`](topics/03-slicing/slicing.py#L89-L106): on the 9-qubit circuit ($W_0=6$), requesting target width $W_0 - 3 = 3$ forces [`slice_greedy`](topics/03-slicing/slicing.py#L70-L79) to slice 12 indices ($2^{12} = 4096$ slices), taking 1.575s per bitstring ($3 \times 4096 = 12,288$ contractions along the path).
  We refine our proposal: instead of P10's proposal (`target_W = W0 - 1`), which slices only 1 index and fails `assert len(sliced) >= 3`, we set target width to `W0 - 2`. Slicing to `W0 - 2` selects exactly 3 indices ($2^3 = 8$ slices). This satisfies `assert len(sliced) >= 3`, exercises multi-index slicing across all 3 bitstrings, and slashes test runtime from 4.67s to 0.02s ($230\times$ speedup). (The precomputed index sets optimization for `slicing.py` remains valid to accelerate the 16s demo script).

- **E3 (Refined proposal regarding chromatic polynomial):** In round 1, we proposed either restricting Petersen to $q \in (2, 3)$ or replacing brute-force evaluation with the chromatic polynomial. Following our debate on M4 and P9, we explicitly withdraw the option of replacing [`count_colourings_brute`](topics/00-tensor-network-basics/basics.py#L97-L99) in [`test_basics.py`](topics/00-tensor-network-basics/test_basics.py#L47-L51) with the 7th-degree polynomial: [`count_colourings_brute`](topics/00-tensor-network-basics/basics.py#L97-L99) is the first-principles brute-force check demanded by Rule 3 of [GUIDELINES.md](GUIDELINES.md#L23-L24). We restrict our recommendation strictly to limiting Petersen in [basics.py:121](topics/00-tensor-network-basics/basics.py#L121) to $q \in (2, 3)$.

- **E1, E2, E5, E6, E7, E8 (Reaffirmed):**
  - E1 (irrational float log2 in cost calculation): stands as an essential numerical bugfix.
  - E2 (QFT tensor network path untested against DFT): confirmed as a critical blind spot that neither peer noticed.
  - E5 (missing termination guards in slicers): stands as an important defensive engineering guard against empty sequences.
  - E6 (data representation divergence and dead code `I2`, `X`, `pair_cost`): stands.
  - E7 (disconnect between Topic 00 GEMM and `opt_einsum`/`np.einsum`): stands.
  - E8 (architectural bridge in PLAN.md): confirmed by unanimous panel consensus.

---

## New findings

### E9: `slice_random` in Topic 03 is completely untested in the test suite
- Topic: 03
- Type: MISSING
- Severity: medium
- Evidence: In [slicing.py:81-86](topics/03-slicing/slicing.py#L81-L86), [`slice_random`](topics/03-slicing/slicing.py#L81-L86) is defined and showcased in the interactive demo and README table (occupying half the columns of the main results table in [03-slicing/README.md](topics/03-slicing/README.md#L26-L34)).
  However, in [test_slicing.py:1-28](topics/03-slicing/test_slicing.py#L1-L28), [`slice_random`](topics/03-slicing/slicing.py#L81-L86) is **not imported, not called, and not tested**. The two existing tests only test [`slice_greedy`](topics/03-slicing/slicing.py#L70-L79). If [`slice_random`](topics/03-slicing/slicing.py#L81-L86) had an off-by-one bug, failed to reach `target_W`, mutated `candidates` improperly, or raised an exception, `pytest` would still report 100% passing tests.
- Proposal: Add a unit test in [`test_slicing.py`](topics/03-slicing/test_slicing.py) verifying that [`slice_random`](topics/03-slicing/slicing.py#L81-L86) reaches the target width and satisfies the theoretical work lower bound $C_s \ge C_0$:
  ```python
  def test_slice_random_reaches_target():
      circ = grid_circuit(3, 3, 6, np.random.default_rng(3))
      _, labels = amplitude_network(circ, 9, "0" * 9)
      path = find_path(labels)
      W0, C0 = width_cost(labels, path)
      target = W0 - 1
      sliced = slice_random(labels, path, target, np.random.default_rng(4))
      W, C = width_cost(labels, path, sliced)
      assert W <= target and C >= C0
  ```

### E10: Silent trace / marginalization on bitstring length mismatch in circuit network builders
- Topic: 01 | 03
- Type: ERROR
- Severity: medium
- Evidence: In [circuits.py:91-93](topics/01-circuits-as-tensor-networks/circuits.py#L91-L93):
  ```python
  for q, b in enumerate(bitstring):
      tensors.append((np.eye(2, dtype=complex)[int(b)], (wire[q],)))
  return tensors, ()
  ```
  and identically in [slicing.py:47-50](topics/03-slicing/slicing.py#L47-L50):
  ```python
  for q, bit in enumerate(bitstring):
      tensors.append(np.eye(2, dtype=complex)[int(bit)])
      labels.append((wire[q],))
  ```
  If `bitstring` has length $m < n$ (e.g. `amplitude(ghz(3), 3, "00")`), the loop projects only the first $m$ qubit wires. The remaining $n - m$ wires remain open in the tensor network, but [`to_network`](topics/01-circuits-as-tensor-networks/circuits.py#L72-L94) returns `()` for the output legs.
  When [`contract`](topics/01-circuits-as-tensor-networks/circuits.py#L96-L100) is invoked, `oe.contract` treats the unspecified open wire legs as implicit contraction indices, silently summing over the unprojected qubits!
  For example, `amplitude(ghz(3), 3, "00")` silently computes $\sum_{b \in \{0, 1\}} \langle 00b | \text{GHZ} \rangle = 0.707107$ without raising any warning or error, returning a mathematically incorrect value for a partial state projection instead of failing fast.
- Proposal: Add an explicit precondition check in [`to_network`](topics/01-circuits-as-tensor-networks/circuits.py#L72-L94) and [`amplitude_network`](topics/03-slicing/slicing.py#L38-L51):
  ```python
  if bitstring is not None and len(bitstring) != n:
      raise ValueError(f"Bitstring length ({len(bitstring)}) must equal qubit count ({n})")
  ```

### E11: Redundant restart from empty slice list in interactive slicing demo script
- Topic: 03
- Type: SIMPLIFY
- Severity: low
- Evidence: In [slicing.py:118-124](topics/03-slicing/slicing.py#L118-L124):
  ```python
  for target in range(W0 - 1, W0 - 9, -1):
      g = slice_greedy(labels, path, target)
      rand = [slice_random(labels, path, target, rng) for _ in range(20)]
  ```
  The loop steps `target` down from $W_0 - 1$ to $W_0 - 8$. At every step, [`slice_greedy`](topics/03-slicing/slicing.py#L70-L79) is called with `sliced = []`. Because greedy index selection is strictly sequential and prefix-stable (empirically verified: the 5 indices chosen for target 12 are the exact first 5 indices chosen for target 8: `[46, 102, 44, 42, 103]`), calling [`slice_greedy`](topics/03-slicing/slicing.py#L70-L79) 8 times redundantly recalculates the candidate rankings for slices 1 through 5 over and over.
  Furthermore, [`slice_random`](topics/03-slicing/slicing.py#L81-L86) is rerun 20 times from scratch for each of the 8 target widths (160 full randomized search iterations), causing `python slicing.py` to spend 16.3 seconds in demonstration loops.
- Proposal: Allow [`slice_greedy`](topics/03-slicing/slicing.py#L70-L79) to accept an existing `sliced` list (defaulting to `None`):
  ```python
  def slice_greedy(labels, path, target_W, sliced=None):
      sliced = list(sliced) if sliced is not None else []
      ...
  ```
  In [slicing.py](topics/03-slicing/slicing.py), maintain `g` across the target loop: `g = slice_greedy(labels, path, target, sliced=g)`. This computes each greedy slice exactly once, eliminating ~70% of greedy search evaluations while keeping the code clean and transparent.

### E12: Fragile stochastic threshold in `test_haar_unitary_is_complex`
- Topic: 01
- Type: ERROR
- Severity: low
- Evidence: In [test_circuits.py:18-21](topics/01-circuits-as-tensor-networks/test_circuits.py#L18-L21):
  ```python
  def test_haar_unitary_is_complex():
      # A real orthogonal matrix is not a Haar U(4) sample, which is complex.
      assert np.abs(haar_unitary(4, rng).imag).max() > 0.1
  ```
  This test checks that a single random draw of [`haar_unitary(4, rng)`](topics/01-circuits-as-tensor-networks/circuits.py#L25-L30) has at least one entry with an imaginary component greater than 0.1.
  While this passes for the hardcoded seed `rng = np.random.default_rng(2)` at module level, relying on a magic threshold `> 0.1` on a single stochastic sample is a test anti-pattern: if the test suite is run with randomized seeds, or if the test execution order changes in parallel runners, there is a non-zero probability of false failure.
  More fundamentally, testing whether a distribution is complex should verify that the complex phase distribution is non-trivial, or assert `np.any(U.imag != 0)` over a small deterministic check.
- Proposal: Replace the arbitrary scalar threshold on a single draw with a mathematically robust check that does not depend on a specific RNG sequence:
  ```python
  def test_haar_unitary_is_complex():
      U = haar_unitary(4, np.random.default_rng(42))
      assert np.linalg.norm(U.imag) > 1e-3
  ```
  or draw multiple independent unitaries to guarantee statistical certainty across arbitrary seeds.
