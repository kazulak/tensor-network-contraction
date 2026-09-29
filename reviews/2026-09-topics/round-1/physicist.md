# Round 1: Lead Physicist

## Summary
The introductory topics (00–03) establish the fundamentals of exact tensor network contraction with admirable code clarity, strictly minimal dependencies, and robust numerical verification against brute-force references. The mathematical mechanics of pairwise contraction, path optimization, and greedy slicing are correctly executed and well-aligned with the cited computer science literature (Bridgeman & Chubb, Markov & Shi, Gray & Kourtis). However, from the perspective of computational quantum physics, the sequence currently treats quantum states almost entirely as abstract algebraic arrays, leaving out critical physical intuition: the canonical ladder network is not identified as the MPS norm/transfer operator, statevector norm conservation and physical observables ($\langle \psi | O | \psi \rangle$) are omitted in favor of isolated amplitudes, and the fundamental dichotomy between exact spacetime contraction (Topics 00–04) and approximate wavefunction simulation (Topics 05–07) is unmotivated. Addressing these missing conceptual bridges will transform an already solid contraction library into an outstanding educational foundation for graduate students in quantum physics.

## Findings

### P1: Ladder network not physically identified as MPS norm and transfer operator
- Topic: 00
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/00-tensor-network-basics/README.md:17-18`, `topics/00-tensor-network-basics/basics.py:51-75`, and Bridgeman & Chubb §1.4, pp. 7–8 (Eqs. 1.14–1.17).
  In `basics.py`, the ladder network is constructed and contracted either along the top or rung by rung. In `README.md`, this is presented as an abstract graph-theoretic exercise. To a physics graduate student, however, this ladder is the single most important diagram in 1D tensor networks: the inner product $\langle \psi | \psi \rangle$ of a Matrix Product State (MPS), where the rungs are physical qubit degrees of freedom and the horizontal rails are virtual bonds. Rung-by-rung contraction is precisely the transfer matrix / environment operator method used throughout DMRG and MPS algorithms, keeping an environment tensor of rank 2 (or 3 during contraction), while contracting along the top first corresponds to generating the exponentially large $2^n$ state vector $|\psi\rangle$ before taking the inner product. Bridgeman & Chubb explicitly introduce this example immediately following their discussion of low-entanglement states and MPS (Aside 1).
- Proposal:
  Add an explanation to `00-tensor-network-basics/README.md` highlighting the physical meaning of the ladder:
  > **Physical context:** The ladder network is the tensor network representation of an MPS inner product $\langle \psi | \psi \rangle$. The rungs are physical indices (qubits) contracted between bra and ket; the rails are virtual bonds. Rung-by-rung bubbling is the transfer-matrix method, keeping a small $\chi \times \chi$ environment tensor at each step. Contracting along the top first amounts to computing the full $2^n$ dense state vector $|\psi\rangle$ before taking the inner product—demonstrating why MPS algorithms contract locally rather than forming the global wavefunction.

### P2: Graph colouring lacks context: Potts model partition function and #P-hardness
- Topic: 00
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/00-tensor-network-basics/README.md:17-18, 28-30`, `topics/00-tensor-network-basics/basics.py:80-94`, and Bridgeman & Chubb §1.5, pp. 8–9 (Eq. 1.19).
  Bridgeman & Chubb introduce the vertex-copy tensor $e$ and adjacent-inequality tensor $n$ in Section 1.5 under the heading "Computational Complexity" to prove that contracting an arbitrary tensor network is #P-hard (since counting proper graph colourings is #P-complete). Furthermore, in lattice statistical mechanics, this tensor network is identically the zero-temperature partition function of the anti-ferromagnetic $q$-state Potts model ($Z = \sum_{\{s\}} \prod_{\langle u, v \rangle} (1 - \delta_{s_u, s_v})$). Without this context, a physics student moving from tensor basics to quantum circuits will find the sudden detour into graph theory and the Petersen graph arbitrary and unmotivated.
- Proposal:
  Clarify in `00-tensor-network-basics/README.md` why graph colouring is included:
  1. State the complexity result: counting 3-colourings is #P-complete, which establishes that exact contraction of an arbitrary tensor network is #P-hard in general. This proves why classical simulation of quantum circuits cannot succeed in general without structural advantages (low treewidth) or approximations (MPS).
  2. Note the physical mapping: this network is the partition function of the zero-temperature anti-ferromagnetic Potts model on the graph.

### P3: Missing state norm conservation check and physical observable expectation values
- Topic: 01
- Type: MISSING
- Severity: high
- Evidence: `GUIDELINES.md:25-26` ("4. Physics correct by default. Complex numbers, unitary gates (tested), and norms or fidelities reported.") versus `topics/01-circuits-as-tensor-networks/circuits.py:106-126` and `topics/01-circuits-as-tensor-networks/test_circuits.py:1-60`.
  While gate unitarity is thoroughly tested (`test_gates_are_unitary`), the state vector itself is never checked for norm conservation ($\|\psi\|^2 = \langle \psi | \psi \rangle = 1$), which is the most fundamental sanity check in quantum mechanics.
  Furthermore, Topic 01 focuses exclusively on single transition amplitudes $\langle x | C | 0\dots 0 \rangle$ and full $2^n$ state vectors. In quantum physics and experimental circuit simulation (e.g. VQE, QAOA, condensed matter), single amplitudes of $n \ge 30$ circuits are exponentially small ($\sim 2^{-n/2} \le 10^{-5}$) and cannot be directly measured. The quantities of experimental interest are expectation values of observables $\langle \psi | O | \psi \rangle$ (such as Pauli strings $Z_i$ or correlators $Z_i Z_j$). As formulated by Markov & Shi §3, this is computed via the folded/doubled network $\langle 0 | C^\dagger O C | 0 \rangle$ without ever generating the state vector.
- Proposal:
  1. Add an explicit check for state norm conservation in `test_circuits.py`:
     ```python
     def test_statevector_norm_is_conserved():
         circ = random_circuit(6, 6, rng)
         psi = statevector(circ, 6)
         assert np.isclose(np.linalg.norm(psi), 1.0)
     ```
  2. In `circuits.py` and `README.md`, explain how expectation values $\langle \psi | O | \psi \rangle$ are represented as closed tensor networks by sandwiching an observable between $C$ and $C^\dagger$, connecting directly to Markov & Shi §3.

### P4: Physical discrepancy between Haar-random $U(4)$ gates and Sycamore architecture
- Topic: 03
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/03-slicing/README.md:15-16`, `topics/03-slicing/slicing.py:20-27`, compared to Gray & Kourtis §4.6.2 & §4.7.1, and Arute et al. (Nature 574, 505–510, 2019).
  `slicing.py` labels the circuit "Sycamore-like: each layer couples nearest neighbours in one of four patterns ... Haar U(4) gates."
  In actual physical implementations (such as Google's Sycamore experiment), circuits do not use dense Haar-random $U(4)$ gates; they consist of cycles of random single-qubit rotations ($\sqrt{X}, \sqrt{Y}, \sqrt{W}$) followed by fixed entangling gates ($\text{fSim}(\theta, \phi) = \text{iSWAP}^\dagger \cdot \text{CPHASE}(\phi)$).
  Crucially, Gray & Kourtis demonstrate in Section 4.6.2 and Table 2 that the physical Sycamore gate has an operator Schmidt rank of 2 (or two dominant singular values under SVD), which is the entire basis for their `Sycamore-53*` decomposition that drastically lowers contraction width. Haar-random $U(4)$ gates generically have full operator Schmidt rank 4 and no low-rank structure.
- Proposal:
  Clarify in `03-slicing/README.md` that while the 2D grid connectivity and alternating 4-layer coupling layout mirror Sycamore's ABCD coupler cycles, the gates are Haar-random $U(4)$ unitaries rather than physical fSim gates. Explain that dense Haar gates lack the low-rank Schmidt decomposition of physical entangling gates, representing a pessimistic worst-case benchmark for tensor contraction.

### P5: Clarify the origin of the slicing work overhead ratio $C_s / C$
- Topic: 03
- Type: EXPLAIN
- Severity: low
- Evidence: `topics/03-slicing/README.md:26-38`, `topics/03-slicing/slicing.py:117-124`.
  In the results table, slicing 20 indices yields $d_{sliced} = 2^{20} \approx 1.05 \times 10^6$ independent tensor networks, but the reported total work overhead is $C_s / C = 1345\times$.
  A student reading line 8 ("each costs at least $C / d_{sliced}$, so total work $C_s \ge C$") can easily be baffled as to why 1 million slices only increase total work by $1345\times$.
  The physical/computational reason is that fixing 20 indices removes them from the intermediate tensors, reducing their dimensions and making each individual sliced contraction much cheaper than the unsliced one ($C_{slice} \approx 1345 \cdot C / 2^{20} \approx 0.0013 \cdot C$).
- Proposal:
  Add an explanatory note below the table in `03-slicing/README.md`:
  > **Why $C_s / C \ll d_{sliced}$:** Although slicing 20 indices creates $2^{20} \approx 10^6$ independent sub-networks, fixing those indices strips dimensions from intermediate tensors throughout the tree. Each sliced network costs only $\sim 0.13\%$ of the original unsliced contraction ($C_{slice} \approx 1345 \cdot C / 2^{20}$), leading to a net computational overhead of $1345\times$ across all $10^6$ slices.

### P6: Missing conceptual bridge between exact contraction and approximate MPS
- Topic: roadmap
- Type: SEQUENCE
- Severity: high
- Evidence: `PLAN.md:17-27` and `topics/README.md:6-16`.
  Topics 00–04 focus entirely on **exact** contraction of arbitrary spacetime tensor networks, where all indices are contracted without loss of information. Topics 05–07 abruptly transition to **approximate** 1D tensor networks (MPS, SVD truncation, DMRG).
  A newcomer will not understand why this paradigm shift occurs. The bridge is the exponential scaling of treewidth: for circuits beyond $\sim 50$ qubits and depth $> 30$, the contraction width $W$ exceeds 40, requiring $> 10^{20}$ FLOPs even with slicing and exascale supercomputers. When exact contraction hits this hard physical limit, one must switch to controlled variational approximations (MPS/TEBD), where truncation error is governed by entanglement entropy rather than graph treewidth.
- Proposal:
  Add a bridging section in `PLAN.md` (and introduce it at the end of Topic 04) that frames the two distinct tensor network paradigms:
  1. **Exact spacetime contraction (Topics 00–04):** Simulates arbitrary 2D/3D circuits exactly; complexity is governed by graph treewidth and cut boundaries; memory managed via slicing.
  2. **Approximate state evolution (Topics 05–07):** Simulates 1D/quasi-1D states variationally; complexity is governed by bipartite entanglement entropy $S$ and bond dimension $\chi \sim e^S$; managed via SVD truncation.

### P7: Prerequisite for Topic 05: Gauge freedom and canonical form
- Topic: roadmap
- Type: MISSING
- Severity: medium
- Evidence: `PLAN.md:24` ("05 | MPS and SVD truncation | Schmidt decomposition; error = discarded weights | Orús; Schollwöck §4; Bridgeman & Chubb §1.2 | dense state, Eckart–Young bound | planned").
  The plan notes that truncation error equals the discarded singular values by the Eckart–Young theorem. In tensor-network quantum physics, however, local SVD truncation of an MPS is optimal under the Eckart–Young bound **if and only if the MPS is in canonical (orthogonal/isometric) form** (Schollwöck §4.2–§4.5, Orús §3.2). In an un-gauged MPS, the singular values of an individual tensor do not correspond to the Schmidt coefficients of the global state, and naive truncation severely corrupts the state norm and fidelity.
- Proposal:
  Update the Topic 05 entry in `PLAN.md` to explicitly include gauge freedom and canonical form:
  `MPS, canonical form and SVD truncation | Left/right isometries ($A^\dagger A = I$); Schmidt decomposition; Eckart–Young bound`

### P8: Disambiguate Topic 06 as TEBD (Vidal 2003)
- Topic: roadmap
- Type: SEQUENCE
- Severity: low
- Evidence: `PLAN.md:25, 27`. Topic 06 cites Vidal (PRL 91, 147902, 2003), which is the foundational paper that introduced Time-Evolving Block Decimation (TEBD). Simulating a quantum circuit by applying two-qubit gates to an MPS and performing SVD truncation *is* the TEBD algorithm in discrete time. Yet Topic 08 in `PLAN.md` lists "TEBD" as an unassigned optional idea.
- Proposal:
  Explicitly title Topic 06 as "MPS circuit simulation (TEBD)" in `PLAN.md` and reserve Topic 08 for genuinely distinct methods (such as PEPS boundary contraction or stabilizer/Clifford tableau methods).

### P9: Remove $4^{10}$ brute-force loop in `basics.py` to prevent script freeze
- Topic: 00
- Type: SIMPLIFY
- Severity: low
- Evidence: `topics/00-tensor-network-basics/basics.py:121` (`for q in (2, 3, 4):` for the Petersen graph).
  Executing `python topics/00-tensor-network-basics/basics.py` hangs for ~25–30 seconds. This delay is entirely caused by `count_colourings_brute(PETERSEN, 10, 4)`, which iterates over $4^{10} = 1,048,576$ states in pure Python.
  In contrast, `test_basics.py:48` only tests $q \in (2, 3)$, which executes in milliseconds.
  For the Petersen graph, $q=3$ (yielding 120 colourings) is the famous non-trivial case.
- Proposal:
  In `basics.py:121`, restrict the Petersen check to `q in (2, 3)` (matching `test_basics.py`), or compare $q=4$ against the known Petersen chromatic polynomial value $P(4) = 12960$ rather than evaluating 1 million combinations in an interactive script.

### P10: Mitigate excessive contraction overhead in `test_slicing.py`
- Topic: 03
- Type: SIMPLIFY
- Severity: low
- Evidence: `topics/03-slicing/test_slicing.py:14`:
  `sliced = slice_greedy(labels, path, width_cost(labels, path)[0] - 3)`
  On the 9-qubit depth-6 circuit ($W_0=6$), demanding a target width of $W_0 - 3 = 3$ forces the greedy slicer to slice 12 indices. This creates $2^{12} = 4096$ slices, which are contracted across three separate bitstrings ($x \in \{0, 5, 300\}$), executing $3 \times 4096 = 12,288$ contractions along the path.
  This single test takes ~7.0 seconds of the 7.9-second `pytest` runtime.
- Proposal:
  In `test_slicing.py:14`, set the target to `width_cost(labels, path)[0] - 1` (which slices 1–2 indices, contracting 2–4 slices) and check a single bitstring. This verifies the mathematical exactness of `contract_sliced` just as thoroughly while reducing total test suite runtime from ~8 seconds to under 1 second.

## Keep as is
- **Topic 01 pedagogical structure:** The clean separation between the reference state-vector simulator (`statevector`) and the tensor network formulation (`to_network` + `oe.contract`) is exceptionally clear. Demonstrating both closed networks (amplitudes) and open networks (state vectors) alongside the Nielsen & Chuang QFT check is an exemplary educational pattern.
- **Mezzadri Haar unitary implementation (`circuits.py:25-30`):** Properly generates genuine Haar-random complex unitaries via QR decomposition with diagonal phase regularization ($Q \cdot \text{diag}(R)/\lvert\text{diag}(R)\rvert$), avoiding the common beginner mistake of sampling real orthogonal matrices.
- **Symbolic cost and width tracking (`order.py:33-46`):** Computing contraction width $W$ and FLOP cost $C$ directly from graph incidence sets without allocating tensor memory allows students to explore large lattices ($6 \times 6$ grids, $n=64$ rings) interactively without memory overflow.
- **Greedy vs. random slicing demonstration (`slicing.py:117-124`):** The side-by-side comparison between greedy and random slicing provides undeniable proof that slicing must target bottleneck cut edges, brilliantly demonstrating why the first few slices are virtually free ($8\times$ memory reduction for only $14\%$ FLOP overhead).
- **Minimal dependencies and code brevity:** Keeping all scripts under 150 lines with plain NumPy and `opt_einsum` avoids cognitive overhead from heavyweight frameworks, adhering strictly to the ethos of establishing one idea per folder.

## Suggested learning path
- **Topic 00 (Tensor Network Basics):** Contraction is reshape + matmul; order decides memory; the ladder network represents an MPS norm contracted via transfer operators; general contraction is #P-hard (Potts model partition function).
- **Topic 01 (Circuits as Tensor Networks):** Quantum circuits are spacetime networks; closed networks evaluate transition amplitudes and folded networks evaluate observable expectation values without forming $2^n$ state vectors.
- **Topic 02 (Contraction Order and Cost):** The graph's treewidth and entanglement area law dictate the minimum memory width $W$; 1D circuits have constant $W$ while 2D/3D circuits have $W \sim L$, requiring heuristic contraction paths.
- **Topic 03 (Slicing):** When optimal contraction width $W$ exceeds physical hardware memory, slicing bottleneck indices trades a modest increase in total FLOPs ($C_s \ge C$) for an exponential reduction in required RAM ($2^W \to 2^{W_s}$).
- **Topic 04 (Parallel Contraction):** Independent sliced sub-networks enable embarrassingly parallel computation across distributed GPU/CPU nodes, bounded by communication and Amdahl limits.
- **Topic 05 (MPS, Canonical Form & SVD Truncation):** When exact contraction exceeds computational limits for deep circuits, we must approximate; putting 1D tensor networks into canonical form enables optimal state truncation via the Eckart–Young theorem.
- **Topic 06 (MPS Circuit Simulation / TEBD):** Propagating an MPS through a circuit gate-by-gate tracks entanglement growth; bond dimension $\chi$ truncates area-law states accurately but degrades fidelity under volume-law entanglement.
- **Topic 07 (DMRG):** Transitioning from time-dependent circuits to static ground states, sweeping local variational updates on an MPS solves for the ground state of 1D quantum spin chains (TFIM, Heisenberg) with machine precision.
