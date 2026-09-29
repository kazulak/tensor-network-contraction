# Round 2: Lead Physicist

## Responses

- E1 (Exact operation count $C$ computed via floating-point logarithms and exponentiation): AGREE. In `topics/02-contraction-order-and-cost/order.py:35-42`, defining scalar multiply-add cost via `C += 2 ** size(a | b)` with `size = lambda s: sum(math.log2(dims[l]) for l in s)` is mathematically backwards for exact integer counting. For non-power-of-2 dimensions (such as qutrits $q=3$ in Topic 00, or planned arbitrary bond dimensions $\chi$ in MPS/DMRG), summing irrational floating-point logarithms accumulates rounding errors (e.g. `2 ** (math.log2(3) + math.log2(5)) == 14.999999999999998`). Gray & Kourtis Eq. (5) defines vertex congestion directly as the integer product of dimensions $C(B) = \sum_{v \in V_B} \prod_{e \in v} d_e$. The engineer's proposal to use `math.prod(dims[l] for l in (a | b))` is exact, avoids transcendental calls in the contraction loop, and matches standard linear algebra and tensor contraction conventions.

- E2 (QFT tensor network converter and contractor are untested against DFT reference): REFINE. The engineer is completely right that `test_circuits.py` never calls `to_network` or `contract` on the QFT circuit, leaving the tensor network representation of controlled-phase and SWAP gates entirely untested. However, the engineer's proposed test—evaluating `contract(to_network(qft(n), n))` against `statevector(qft(n), n, 0)`—is physically and mathematically vacuous! On the input state $|0\dots 0\rangle$, controlled-phase gates do not activate at all because every control qubit is 0 ($C\text{-Phase}(\phi)|00\rangle = |00\rangle$). The output is identically the uniform superposition $2^{-n/2}\sum |k\rangle$, so a broken controlled-phase gate or swapped control/target legs would still pass the engineer's test. To genuinely test QFT phase kickback and entanglement, the network must be evaluated on non-zero basis states $|x\rangle$ (e.g. $x=13$ as in `circuits.py:122` or arbitrary $x$) or closed-network transition amplitudes $\langle y | \text{QFT} | x \rangle = \frac{1}{\sqrt{N}} e^{2\pi i x y / N}$. As detailed in new finding P11, this requires `to_network` to accept an `initial` bitstring parameter rather than hardcoding $|0\dots 0\rangle$.

- E3 (Unvectorized brute-force enumeration causes a 30-second stall in Topic 00 demo script): REFINE. We verified experimentally that `python topics/00-tensor-network-basics/basics.py` takes 29.35 seconds, with 28.5 seconds spent inside `count_colourings_brute(PETERSEN, 10, 4)` ($4^{10} = 1,048,576$ iterations in Python). All three panellists flagged this freeze (E3, M4, P9). However, between the engineer's two options (restricting to $q \in (2, 3)$ vs. using the chromatic polynomial), the best pedagogical fix is a hybrid: retain `count_colourings_brute` for $q \in (2, 3)$ on Petersen (which finishes in 30 ms and verifies 0 and 120 against `einsum_reference`), and cite the known chromatic polynomial value $P(4) = 12960$ for $q=4$. Discarding brute force entirely obscures Guideline 3's principle of hands-on verification against an independent reference, while restricting $q$ keeps the interactive script instantaneous.

- E4 (Full path re-simulation and $O(N^2)$ list re-allocation in greedy slicer and tests): REFINE. The engineer's profiling of `test_slicing.py` is factually incorrect. The engineer asserts that `find_path` and `slice_greedy` consumed 4.33s of the 4.60s runtime in `test_sum_over_slices_equals_amplitude` and proposes moving them outside the bitstring loop. We benchmarked this directly: `find_path` takes 0.002s and `slice_greedy` takes 0.057s (total 0.06s). Moving them outside the loop still results in a 4.42s runtime! The actual bottleneck is `contract_sliced`, which takes 1.45s per bitstring because requesting target width $W_0 - 3$ forces the greedy slicer to slice 12 indices, executing $3 \times 2^{12} = 12,288$ full tensor network contractions. The real fix for the test duration is our finding P10 (requesting target width $W_0 - 1$ or $W_0 - 2$), which drops the test runtime from 4.5s to 0.02s. However, the engineer's second proposal—precomputing intermediate index sets along `path` in `slicing.py`—is a sound algorithmic improvement for the greedy slicer in `slicing.py`, reducing candidate evaluation cost from $O(S \cdot E \cdot N^2)$ to $O(S \cdot E \cdot N)$.

- E5 (Missing termination guards in greedy and random slicers): DISAGREE. Slicing all candidate indices in `slicing.py` trivially reduces contraction width to 0. An empty candidate pool only occurs if a user requests an unphysical negative target width ($W < 0$). In minimal educational scripts constrained to under 150 lines (Guideline 2), adding defensive exception handling and boilerplate checks (`if not remaining: raise ValueError(...)`) clutters the code without teaching any physics or tensor network concepts. The existing 4-line `while` loop is concise, idiomatic, and correct for all valid physical parameters ($0 \le \text{target\_W} \le W_0$).

- E6 (Inconsistent data representations and code duplication across topics): REFINE. We agree with the engineer that Topic 03's split into two parallel lists `(tensors, labels)` is an unnecessary divergence from Topic 00 and 01, and should be unified to a list of `(tensor, legs)` tuples. However, we strongly push back against the engineer's recommendations on code duplication and dead code:
  1. *Duplication:* Duplicating `haar_u4` and `statevector` across topics is an intentional requirement of Guideline 2 ("Each folder is self-contained, so a Julia, C, Fortran or Haskell version can sit next to the Python one"). Cross-topic imports would destroy folder modularity.
  2. *Decoupled labels and dims in Topic 02:* Topic 02 studies contraction path optimization and graph congestion ($W$ and $C$) symbolically. Decoupling topology from tensor data allows students to explore large lattices ($6 \times 6$ grids, $n=64$ rings) interactively without allocating gigabytes of memory.
  3. *Dead code:* The engineer labels gate `X` in Topic 01 and `pair_cost` in Topic 00 as dead code. This is incorrect. Gate `X` is the fundamental Pauli bit-flip operator required to initialize non-zero basis states to test QFT and general transition amplitudes (P11). And `pair_cost` should be integrated into `bubble` to accumulate contraction FLOPs (agreeing with M5). Neither should be deleted.

- E7 (Disconnect between Topic 00 execution model and Topics 01–03): AGREE. In computational quantum physics, students often struggle to connect abstract tensor diagrams to high-performance BLAS routines. Clarifying in the READMEs of Topics 01 and 02 that `opt_einsum` and numpy compile high-level contraction expressions into the exact sequence of `transpose -> reshape -> zgemm` demonstrated in Topic 00 provides vital educational continuity.

- E8 (Missing architectural bridge from exact contraction to approximate MPS): AGREE. Consensus across all three panellists (E8, M6, P6). `PLAN.md` must clearly articulate the physical boundary: Topics 00–04 cover exact spacetime tensor network contraction (governed by graph treewidth and slicing), while Topics 05–07 transition to variational wavefunction simulation (governed by entanglement entropy and bond dimension $\chi$).

- M1 (Treewidth bound requires bounded degree and line graph formulation): AGREE. The mathematician's precision is welcome. Markov & Shi's Theorem 1.1 ($\exp(O(\mathrm{tw}(G)))$) strictly requires bounded vertex degree ($\Delta(G) = O(1)$), which holds for 1- and 2-qubit quantum gates ($\Delta \le 4$). For an $m$-qubit gate or $m$-ary copy tensor, the graph treewidth is 1, yet the contraction cost is $2^m$. Furthermore, contraction width $W$ is precisely the branch-width of the tensor network graph $G$, which equals the treewidth of its line graph $G^*$. Adding this clarification to Topic 02's README and docstring prevents students from confusing the physical interaction graph with the line graph of tensor contractions.

- M2 (Prove the grid width invariant $L \le W_{\text{opt}} \le W_{\text{row}} \le L+1$ via isoperimetry and row-frontier invariants): REFINE. The mathematician's combinatorial bounds (isoperimetric cut $|\delta(S)| \ge L$ and active frontier $(L-k) + k + 1 = L+1$) are mathematically elegant. However, for a physics student, these bounds must be explicitly identified as the **entanglement area law**: any cut bipartitioning an $L \times L$ grid has a boundary length of at least $L$, requiring an intermediate tensor rank of at least $L$ to capture bipartite entanglement. Furthermore, the row-by-row bubbling upper bound $W \le L+1$ is the foundation of 2D PEPS boundary contraction (evolving a 1D boundary MPS across the 2D lattice). Framing M2 through the area law and boundary MPS bridges graph theory directly to quantum many-body physics.

- M3 (Formulate the exact per-step slicing cost formula to explain $C_s \ge C$ and overhead explosion): AGREE. The mathematician's per-step work formula, $C_s(v) = 2^{|s \setminus (a \cup b)|} C(v)$, is brilliant. It provides the exact algebraic mechanism for our finding P5: contraction steps containing all sliced indices break even ($2^0 C(v) = C(v)$), while steps disjoint from the sliced indices are duplicated $2^{|s|}$ times ($2^{|s|} C(v)$). This directly explains why greedy slicing on bottleneck edges achieves an $8\times$ memory reduction with only a $14\%$ FLOP penalty, whereas random slicing duplicates the entire circuit, triggering a catastrophic $10^{25}\times$ blowup.

- M4 (Motivate graph colouring via #P-hardness and replace Petersen brute force with chromatic polynomial): REFINE. We agree with the #P-hardness motivation, but it must be paired with the physical lattice statistical mechanics mapping (P2): the tensor network is identically the zero-temperature partition function of the anti-ferromagnetic Potts model ($Z = \sum_{\{s\}} \prod_{\langle u, v \rangle} (1 - \delta_{s_u, s_v})$). Regarding code, as discussed in E3 and P9, we should keep the brute-force check for $q \in (2, 3)$ on Petersen to preserve Guideline 3's hands-on verification, and cite the chromatic polynomial value $P(4) = 12960$ for $q=4$.

- M5 (Derive contraction cost from matrix multiplication linear algebra and accumulate cost in bubbling): AGREE. Deriving the contraction FLOP cost $d_I d_J d_K$ from partitioning tensor legs into free and contracted sets is standard physics/linear algebra. Furthermore, updating `bubble` in `basics.py` to accumulate multiply-adds puts the already-defined `pair_cost` function to use and demonstrates that rung-by-rung bubbling is $O(n)$ in work while along-the-rails bubbling is $O(2^n)$, directly satisfying Guideline 5.

- M6 (Demarcate the pivot from exact contraction (00-04) to approximate methods (05-07) in the roadmap): AGREE. Fully concordant with P6 and E8. The roadmap in `PLAN.md` should formally separate Part I (Exact Contraction, treewidth-limited) from Part II (Low-Rank MPS Approximations, entanglement-entropy-limited).

---

## Changes to my own findings

- **P1 (Ladder network not physically identified as MPS norm and transfer operator):** Refined. While the ladder network topology corresponds to an MPS inner product, in `basics.py` the top and bottom rows are generated with independent random Gaussian tensors. This evaluates the overlap $\langle \phi | \psi \rangle$ between two distinct states, producing an arbitrary complex scalar (e.g. $59.7 + 258.2i$). To represent a physical state norm $\langle \psi | \psi \rangle$ and positive transfer operator ($T = \sum_s A^s \otimes (A^s)^*$), the bottom tensors must be complex conjugates ($B = A^*$). We clarify that the current code demonstrates a general transition matrix element $\langle \phi | \psi \rangle$, and explain the conjugate requirement for the MPS norm $\|\psi\|^2$.
- **P5 (Clarify the origin of the slicing work overhead ratio $C_s / C$):** Refined. We adopt Mathematician M3's exact per-step work formula $C_s(v) = 2^{|s \setminus (a \cup b)|} C(v)$ into P5. Combining this algebraic formula with our physical interpretation ($C_{slice} \ll C$) provides the complete explanation for why $C_s / C \ll d_{sliced}$.
- **P9 (Remove $4^{10}$ brute-force loop in `basics.py` to prevent script freeze):** Refined. In light of M4 and E3, we recommend retaining `count_colourings_brute` for $q \in (2, 3)$ on Petersen (running in 30 ms) to preserve Guideline 3's brute-force verification, while replacing the 29-second $q=4$ evaluation with the known chromatic polynomial value $P(4) = 12960$.
- **P10 (Mitigate excessive contraction overhead in `test_slicing.py`):** Maintained firmly with experimental refutation of E4. Profiling proved that moving `find_path` and `slice_greedy` outside the loop leaves runtime unchanged at 4.42s because `contract_sliced` executes $3 \times 4096 = 12,288$ contractions. Reducing the target width offset from $-3$ to $-1$ or $-2$ is the only change that resolves the 4.5s test stall.

---

## New findings

### P11: `to_network` hardcodes input state $|0\dots 0\rangle$, preventing arbitrary transition amplitudes $\langle y | C | x \rangle$ and general QFT simulation
- Topic: 01
- Type: MISSING
- Severity: high
- Evidence: In `topics/01-circuits-as-tensor-networks/circuits.py:81`:
  ```python
  tensors = [(np.array([1, 0], dtype=complex), (q,)) for q in range(n)]
  ```
  `to_network` unconditionally initializes every input qubit wire in state $|0\rangle$. While `statevector(circ, n, x=0)` accepts any initial computational basis state $x \in \{0, \dots, 2^n - 1\}$, `to_network` only permits projecting the *output* with `bitstring`.
  Consequently:
  1. The tensor network representation cannot compute general transition amplitudes $\langle y | C | x \rangle$ or unitary matrix elements $U_{yx}$.
  2. This design limitation is the direct root cause of finding E2: the author could not run `to_network` on `qft(n)` with non-zero inputs (such as $x=13$ used in `circuits.py:122`). On input $|0\dots 0\rangle$, all controlled-phase gates act trivially as identity, rendering any test on $|0\dots 0\rangle$ incapable of verifying phase kickback or gate orientation.
- Proposal:
  Add an `initial` bitstring parameter to `to_network(circ, n, bitstring=None, initial=None)`:
  ```python
  def to_network(circ, n, bitstring=None, initial=None):
      wire = list(range(n))
      next_label = n
      init_bits = initial if initial is not None else "0" * n
      tensors = [(np.eye(2, dtype=complex)[int(b)], (q,)) for q, b in enumerate(init_bits)]
      for U, qs in circ:
          ...
  ```
  This restores full parity with `statevector`, allows computing arbitrary transition amplitudes $\langle y | C | x \rangle$, and enables genuine verification of QFT on non-trivial basis states.

### P12: Slicing is the hybrid Schrödinger–Feynman simulation algorithm: bridging tensor networks to path integrals
- Topic: 03
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/03-slicing/README.md:6-9` introduces slicing purely as an abstract graph-cutting trick.
  In quantum computational physics (Markov & Shi 2008 §4, Boixo et al. Nature Phys. 2018, Arute et al. Nature 2019, Chen et al. 2018), slicing is known as the **Schrödinger–Feynman algorithm**.
  Fixing an internal bond or wire index $e$ to values $s \in \{0, \dots, d-1\}$ corresponds to inserting a resolution of the identity $I = \sum_{s=0}^{d-1} |s\rangle\langle s|$ across a spacetime cut.
  - When no indices are sliced, contraction corresponds to full Schrödinger wavefunction evolution (peak memory $2^W$).
  - When all internal indices are sliced, the contraction reduces to the Feynman sum over all spacetime configurations/histories ($O(1)$ memory, but exponential paths).
  - Slicing bottleneck cut edges is the hybrid regime that interpolates between Schrödinger memory and Feynman path counts.
- Proposal:
  Add a physical note in `03-slicing/README.md` identifying slicing as the Schrödinger–Feynman hybrid algorithm:
  > **Physical connection (Schrödinger vs. Feynman):** Slicing is the hybrid Schrödinger–Feynman simulation method used in quantum supremacy experiments (Markov & Shi 2008, Arute et al. 2019). Fixing an index inserts a resolution of the identity $I = \sum_s |s\rangle\langle s|$ across an internal bond. Contracting an unsliced network is pure Schrödinger simulation (keeping the full state, requiring $2^W$ memory). Slicing all spacetime edges yields the pure Feynman path integral (summing over paths, $O(1)$ memory, but $2^M$ terms). Slicing bottleneck cut edges interpolates smoothly between the two, slashing memory while keeping the path count manageable.

### P13: Connection between 2D circuit treewidth ($W \ge L$) and 2D MPS simulation limits (Zhou et al. 2020)
- Topic: roadmap / 02 / 06
- Type: SEQUENCE
- Severity: medium
- Evidence: `PLAN.md:21, 25` and `topics/02-contraction-order-and-cost/README.md:37-38`.
  Topic 02 establishes that contracting an $L \times L$ grid requires width $W \ge L$. Topic 06 cites Zhou, Stoudenmire, & Waintal (PRX 2020) for MPS circuit simulation.
  A physics student needs to know how Topic 02's grid width $W \ge L$ and Topic 06's MPS bond dimension $\chi$ are related. When a 2D $L \times L$ grid circuit is simulated using a 1D MPS via a snake-like path, vertical gates become long-range gates spanning $L$ physical sites. The bipartite entanglement entropy across the middle of the snake scales as the boundary cut length $S \sim L$, requiring bond dimension $\chi \sim 2^S \sim 2^L$.
  Thus, the $W \ge L$ lower bound in exact contraction and the $\chi \sim 2^L$ barrier in MPS simulation are two manifestations of the exact same physical law: the **entanglement area law in 2D**.
- Proposal:
  Add a short forward-reference in Topic 02's README and in `PLAN.md` connecting the 2D grid width $W \ge L$ to the 2D-to-1D snake mapping in Zhou et al. (PRX 2020) in Topic 06.
