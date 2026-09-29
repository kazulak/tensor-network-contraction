# Round 2: Mathematician

## Responses

- P1 (Ladder network not physically identified as MPS norm and transfer operator): AGREE. In linear algebra and tensor network theory, the ladder network in [basics.py](topics/00-tensor-network-basics/basics.py#L51-L75) and Bridgeman & Chubb §1.4 (Eqs. 1.14–1.17) is indeed the canonical inner product $\langle \psi | \psi \rangle$ of a Matrix Product State (MPS). Framing the two bubblings algebraically clarifies the entire lesson: rung-by-rung contraction iteratively applies the transfer operator $\mathbb{E} = \sum_i A^i \otimes (A^i)^*$, preserving a small rank $\le 3$ environment matrix, whereas along-the-top contraction constructs the global rank-$n$ tensor product state $|\psi\rangle$ before taking the inner product. Adding this two-sentence bridge to `00-tensor-network-basics/README.md` creates an immediate forward link to Topic 05 without complicating the code.

- P2 (Graph colouring lacks context: Potts model partition function and #P-hardness): AGREE. This reinforces M4. In complexity theory and statistical mechanics, a tensor network with copy tensors $e$ on vertices and constraint matrices $n = \mathbf{1}\mathbf{1}^T - I$ on edges is identically the zero-temperature partition function of the anti-ferromagnetic $q$-state Potts model ($Z = \sum_{\{s\}} \prod_{\langle u, v \rangle} (1 - \delta_{s_u, s_v})$), and evaluating it counts proper $q$-colourings. Because counting 3-colourings (#3-COL) is #P-complete (Garey & Johnson 1979), exact tensor network contraction is #P-hard in general. Bridgeman & Chubb §1.5 explicitly introduce colouring to establish this computational hardness. Stating this in `00-tensor-network-basics/README.md` provides essential motivation for why classical simulation hits hard complexity barriers.

- P3 (Missing state norm conservation check and physical observable expectation values): REFINE.
  1. *Norm conservation:* In linear algebra, norm conservation ($\|\psi\|_2 = 1$) is an immediate mathematical consequence of gate unitarity: each gate $U_k$ is unitary, hence an $L_2$ isometry, so $\|U_m \cdots U_1 |0\dots 0\rangle\|_2 = \||0\dots 0\rangle\|_2 = 1$ by induction. Gate unitarity is already tested in `test_gates_are_unitary`. Adding `assert np.isclose(np.linalg.norm(psi), 1.0)` is a harmless one-line test, but the README should note that norm preservation follows from unitarity.
  2. *Expectation values:* Evaluating an observable $\langle \psi | O | \psi \rangle = \langle 0 | C^\dagger O C | 0 \rangle$ produces a doubled/folded network (Markov & Shi §3). However, naively contracting a doubled network doubles the contraction width ($W \to 2W$) unless one exploits unitarity: gates outside the past causal cone of the support of $O$ cancel ($U^\dagger U = I$). The proposal should be refined: explain the algebraic formulation of expectation values as quadratic forms and mention causal light-cone cancellation in `01-circuits-as-tensor-networks/README.md`, but do not bloat `circuits.py` with doubled-network contraction code that would violate the 200-line guideline.

- P4 (Physical discrepancy between Haar-random $U(4)$ gates and Sycamore architecture): AGREE. In matrix analysis, an entangling two-qubit gate $U \in U(4) \subset \mathbb{C}^{4 \times 4}$ has an operator Schmidt rank across the bipartite cut $\mathcal{H}_A \otimes \mathcal{H}_B \to \mathcal{H}_A \otimes \mathcal{H}_B$, defined as the rank of $U$ reshaped to $\mathbb{C}^{(d_A d_A) \times (d_B d_B)} = \mathbb{C}^{4 \times 4}$. Physical Sycamore gates ($\text{fSim}$) have operator Schmidt rank 2, decomposing as $U = \sum_{r=1}^2 A_r \otimes B_r$, which replaces an edge of dimension 4 with an internal bond of dimension 2 (Gray & Kourtis §4.6.2, Table 2). Generic Haar-random $U(4)$ unitaries have full operator Schmidt rank 4 almost surely. Clarifying in `03-slicing/README.md` that Haar unitaries represent an adversarial, full-rank worst-case benchmark prevents confusion when students consult Gray & Kourtis Fig. 10.

- P5 (Clarify the origin of the slicing work overhead ratio $C_s / C$): REFINE. The physicist correctly notes that slicing 20 indices yields $d_{\text{sliced}} = 2^{20} \approx 10^6$ slices while total work increases by only $C_s / C = 1345 \ll 2^{20}$. However, explaining this as merely "each sliced network costs $0.13\%$ of the original" is an arithmetic tautology ($C_{\text{slice}} = C_s / d_{\text{sliced}}$). The structural explanation is given by the exact per-step formula from M3:
  $$C_s(v) = 2^{|s \setminus (s_l(v) \cup s_r(v))|} C(v)$$
  Bottleneck steps $v$ containing all sliced indices ($s \subseteq s_l(v) \cup s_r(v)$) break even with zero overhead ($C_s(v) = C(v)$). Because greedy slicing specifically targets the widest intermediate tensors (which carry the most legs), the bottleneck steps dominating total cost suffer minimal duplication, while only non-bottleneck steps incur the full $2^{|s|}$ multiplier. Refining P5 with M3's algebraic formula explains *why* the average cost per slice drops.

- P6 (Missing conceptual bridge between exact contraction and approximate MPS): AGREE. This directly aligns with M6 and E8. Exact contraction (Topics 00–04) computes tensor contractions without error, but is bounded by graph treewidth: 2D circuits have treewidth $\Omega(L)$, requiring exponential time $\exp(\Omega(L))$ even with slicing. Approximate simulation (Topics 05–07) bypasses this barrier by projecting onto a low-rank matrix manifold via truncated SVD (the Eckart–Young–Mirsky theorem), where runtime is polynomial in bond dimension $\chi$ and error is governed by bipartite entanglement entropy. Adding this bridge to `PLAN.md` provides essential mathematical motivation.

- P7 (Prerequisite for Topic 05: Gauge freedom and canonical form): AGREE. From linear algebra, the singular values of an uncontracted 3-index tensor in an MPS have no invariant meaning due to internal gauge freedom: for any invertible matrix $X$, the transformation $A^{[k]} \to A^{[k]} X$ and $A^{[k+1]} \to X^{-1} A^{[k+1]}$ leaves the global state $|\psi\rangle$ invariant. Local SVD truncation minimizes global Frobenius error under the Eckart–Young–Mirsky theorem if and only if the MPS is in canonical form (left/right isometries $\sum_s (A^s)^\dagger A^s = I$). Without canonical form, naive local SVD destroys the state norm and fidelity. This is a vital linear-algebraic theorem that must be stated as an explicit prerequisite in `PLAN.md` for Topic 05.

- P8 (Disambiguate Topic 06 as TEBD (Vidal 2003)): AGREE. Vidal (PRL 91, 147902, 2003) introduced Time-Evolving Block Decimation (TEBD) by applying local gates to an MPS in canonical form followed by SVD truncation. Discrete circuit simulation with MPS *is* TEBD. Retitling Topic 06 as "MPS circuit simulation (TEBD)" in `PLAN.md` avoids redundancy and frees Topic 08 for genuinely distinct methods, such as stabilizer / Clifford tableau simulation (Aaronson & Gottesman 2004; polynomial time via linear algebra over $\mathbb{F}_2$) or 2D PEPS boundary contraction.

- P9 (Remove $4^{10}$ brute-force loop in `basics.py` to prevent script freeze): REFINE. P9 proposes restricting Petersen to $q \in (2, 3)$ or hardcoding $P(4) = 12960$. As noted in M4 and E3, hardcoding a magic number is poor pedagogy, and restricting to $q \in (2, 3)$ drops the generic non-trivial colour count ($q=2$ is 0, $q=3$ is 120). Defining the exact chromatic polynomial function for the Petersen graph:
  $$P(\text{Petersen}, q) = q(q-1)(q-2)(q^7 - 12q^6 + 67q^5 - 230q^4 + 529q^3 - 814q^2 + 775q - 352)$$
  evaluates instantaneously for all $q$, demonstrates algebraic graph theory, removes the 28-second freeze, and enables exact automated testing for $q=4$ and $q=5$ in `test_basics.py`.

- P10 (Mitigate excessive contraction overhead in `test_slicing.py`): REFINE. P10 proposes changing the target width from `W0 - 3` to `W0 - 1` and testing only 1 bitstring. However, for a $3 \times 3$ grid with $W_0=6$, `W0 - 1 = 5` slices only a single index ($2^1 = 2$ slices), which is trivial. On the other hand, `W0 - 3 = 3` forces 12 slices ($2^{12} = 4096$ slices contracted 3 times = 12,288 contractions). The sweet spot is `W0 - 2 = 4`, which greedily slices exactly 3 indices ($2^3 = 8$ slices). Combined with Engineer E4 (moving `find_path` and `slice_greedy` outside the bitstring loop), contracting 8 slices across 3 bitstrings takes $<0.05$ seconds while testing genuine multi-edge slicing rigorously.

- E1 (Exact operation count $C$ computed via floating-point logarithms and exponentiation): AGREE. In [order.py:35, 42](topics/02-contraction-order-and-cost/order.py#L35-L42), contraction cost $C$ is accumulated via `size = lambda s: sum(math.log2(dims[l]) for l in s)` and `C += 2 ** size(a | b)`. Mathematically, $C = \sum_{v} \prod_{l \in a \cup b} d_l$ is an integer sum of integer products (Gray & Kourtis Eq. 5). Computing integer products via $\exp_2(\sum \log_2 d)$ is numerically unsound and fails for non-power-of-2 dimensions (e.g. $q=3$ from Topic 00, or arbitrary bond dimension $\chi$). Computing $C$ with `math.prod(dims[l] for l in (a | b))` is exact, avoids transcendental functions, and eliminates precision loss.

- E2 (QFT tensor network converter and contractor are untested against DFT reference): AGREE. In [test_circuits.py:53-60](topics/01-circuits-as-tensor-networks/test_circuits.py#L53-L60), `test_qft_matches_dft` tests only `statevector(qft(n), n, x)` against the analytical DFT matrix, and never calls `to_network` or `contract`. Because all other tests use only 2-qubit Haar gates or $H$/$CNOT$, controlled-phase and SWAP gates are completely unexercised in tensor network form. Adding `test_qft_network_matches_dft` closes this test gap.

- E3 (Unvectorized brute-force enumeration causes a 30-second stall in Topic 00 demo script): REFINE. E3 correctly diagnoses that `count_colourings_brute(PETERSEN, 10, 4)` consumes 28 seconds over $4^{10} = 1,048,576$ iterations. However, E3 proposes either restricting to $q \in (2, 3)$ or using the chromatic polynomial. As noted in M4 and P9, restricting to $q \in (2, 3)$ weakens the check. The closed-form chromatic polynomial is strictly superior, running in $O(1)$ arithmetic operations, demonstrating the exact algebraic solution, and allowing tests of $q=4$ and $q=5$.

- E4 (Full path re-simulation and $O(N^2)$ list re-allocation in greedy slicer and tests): REFINE.
  1. *Test refactor:* Strong AGREE. Computing `find_path` and `slice_greedy` once outside the bitstring loop in `test_slicing.py` removes redundant optimization across bitstrings, cutting test runtime significantly.
  2. *Precomputing intermediate sets:* REFINE. In `slicing.py`, precomputing static tree intermediate bags $s_v$ is mathematically elegant (since slicing only computes $s_v \setminus \text{sliced}$). However, Rule 2 of GUIDELINES.md requires code to remain under 200 lines and readable to beginners. Re-implementing a tree data structure with explicit node pointers adds cognitive overhead. The primary runtime bottleneck in `slicing.py` (~20s) is running 20 repeats of `slice_random` across 8 width targets (160 randomized search runs). Reducing the number of random targets or candidate search space achieves a sub-second runtime with simpler code.

- E5 (Missing termination guards in greedy and random slicers): REFINE. Slicing internal edges cannot reduce the width below the number of open boundary legs ($W_{\min} = |s_{\text{open}}|$). If a user requests `target_W < |s_{\text{open}}|`, the target is topologically unreachable. Adding a simple guard `remaining = [l for l in candidates if l not in sliced]; if not remaining: break` prevents crashes cleanly without over-engineering exception handling.

- E6 (Inconsistent data representations and code duplication across topics): REFINE.
  1. *Code duplication:* Disagree with unifying code across folders. Rule 2 of GUIDELINES.md explicitly mandates that each folder is self-contained so that a Julia, C, or Haskell version can sit alongside it. Sharing helper functions across folders violates self-containment.
  2. *Data structure drift:* In Topic 02, decoupling `labels` and `dims` from tensor arrays is a vital mathematical concept: contraction path optimization (finding min $W$ and min $C$) is a purely combinatorial problem on hypergraphs and does not require allocating tensor memory. In contrast, Topic 01 and 03 perform numerical contraction. Conflating these representations would obscure this key theoretical distinction.
  3. However, dead code (`I2`, `X` in `circuits.py`) and the unused `pair_cost` in Topic 00 should be cleaned up.

- E7 (Disconnect between Topic 00 execution model and Topics 01–03): AGREE. In numerical linear algebra and BLAS, tensor contraction is implemented via `transpose -> reshape -> GEMM`. Stating explicitly in the READMEs of Topics 01 and 02 that `np.einsum`, `opt_einsum`, and high-performance libraries (e.g. cuTENSOR) compile multi-index contractions into the exact sequence implemented in Topic 00 (`contract_pair`) resolves student confusion and validates Topic 00 as the foundational computational engine.

- E8 (Missing architectural bridge from exact contraction to approximate MPS): AGREE. Agrees with M6 and P6. Connecting the exponential treewidth wall of exact 2D circuit simulation to the low-rank SVD truncation of 1D states (Eckart–Young–Mirsky) provides the essential mathematical motivation for the transition in `PLAN.md`.

## Changes to my own findings

- **M4 (Motivate graph colouring via #P-hardness and chromatic polynomial):** Refined. In light of P2, P9, and E3, keep `count_colourings_brute` for small graphs ($K_3$ and $C_6$) to demonstrate the exact numerical equivalence between einsum and state enumeration, but replace the Petersen brute force entirely with the closed-form chromatic polynomial $P(\text{Petersen}, q)$. This eliminates the 28-second stall while preserving the pedagogical value of brute force on small graphs.
- **M5 (Contraction cost linear algebra and accumulation):** Refined. In light of E1, combine the derivation of contraction cost ($d_I d_K d_J$) with exact integer arithmetic (`math.prod`) rather than floating-point logarithms, ensuring integer precision across both Topic 00 and Topic 02.
- **M6 (Roadmap pivot from exact contraction to approximate MPS):** Refined. In light of P6, P7, P8, and E8, expand the roadmap revision to incorporate:
  1. The canonical form / isometric gauge constraint ($A^\dagger A = I$) as a prerequisite for Eckart–Young optimality in Topic 05 (P7).
  2. The explicit naming of Topic 06 as "MPS circuit simulation (TEBD)" after Vidal (2003) (P8).
  3. Reallocating Topic 08 to stabilizer / Clifford tableau simulation (Aaronson & Gottesman 2004) and 2D PEPS boundary contraction.

## New findings

### M7: Slicing as resolution of the identity $I = \sum |k\rangle\langle k|$ and strict work inequality $C_s > C$
- Topic: 03
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/03-slicing/README.md:6-9` and `topics/03-slicing/test_slicing.py:27` assert `C >= C0` and describe slicing as "fix some indices... sum". Neither the topic nor the panellists explain the underlying linear algebra: slicing an internal bond $e$ of dimension $d$ is mathematically inserting the resolution of the identity:
  $$I_d = \sum_{k=0}^{d-1} |k\rangle \langle k|$$
  Fixing index $e=k$ attaches the rank-1 boundary vectors $|k\rangle$ and $\langle k|$ to the two incident tensors, decoupling the bond into two independent sub-problems.
  Furthermore, all three panellists noted $C_s \ge C$, but missed that for any connected tensor network with $|V| \ge 3$ and any non-empty sliced set $s \ne \emptyset$, **$C_s > C$ is strictly greater**.
  Proof: In M3, total work across all $d_{\text{sliced}} = 2^{|s|}$ slices is:
  $$C_s = \sum_{v \in V_B} 2^{|s \setminus (s_l(v) \cup s_r(v))|} C(v)$$
  Since $|s \setminus (s_l(v) \cup s_r(v))| \ge 0$, every multiplier is $\ge 1$. Equality $C_s = C$ holds if and only if $|s \setminus (s_l(v) \cup s_r(v))| = 0$ for every contraction step $v$, meaning *every* step in the tree must contain all sliced indices. For any tree with $|V| \ge 3$, at least two disjoint leaf subtrees contract without involving edge $e$, incurring a strict multiplier $2^{|s|} > 1$. Hence $C_s > C$ is strict for all non-trivial networks.
- Proposal: Include the resolution-of-the-identity formulation and the strict inequality $C_s > C$ in `03-slicing/README.md`. A minimal 3-tensor chain example explains the redundant work duplication immediately:
  $$(A B) C = \sum_k (A |k\rangle)(\langle k| B) C$$
  Multiplying $A |k\rangle$ and $\langle k| B$ breaks even, but the subsequent contraction with $C$ is repeated $d$ times, proving $C_s > C$.

### M8: Distinguish Contraction Cost $C$ (Complex MACs) from IEEE 754 FLOPs for Roofline Modeling
- Topic: 02 | 03 | roadmap
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/02-contraction-order-and-cost/README.md:11` defines $C$ as "total multiply-add count (time)", while `topics/03-slicing/README.md:35` and `PLAN.md:27` refer to $C$ as "FLOPs" or "FLOP increase". In Gray & Kourtis page 4, $C$ counts abstract scalar multiply-accumulate operations (MACs). For complex tensors (as used throughout Topics 01–03), each MAC $(a+ib)(c+id) + (e+if)$ requires 6 real multiplications and 2 real additions = 8 real IEEE 754 floating-point operations (FLOPs). For real tensors, 1 MAC = 2 FLOPs.
  In planned Topic 04 ("Parallel contraction" citing Williams et al. 2009 Roofline model), performance is benchmarked by operational intensity: $\text{FLOPs} / \text{Byte}$. Conflating complex MACs with real FLOPs produces an immediate $8\times$ error in hardware roofline arithmetic intensity.
- Proposal: Clarify in `02-contraction-order-and-cost/README.md` and `03-slicing/README.md` that $C$ is the count of complex multiply-accumulate operations (MACs), and state the exact conversion factor:
  $$\text{FLOPs}_{\mathbb{C}} = 8 C$$
  This prepares students for the Roofline model in Topic 04 and ensures consistency with scientific computing standards.

### M9: Unify Transfer Matrix Traces Across Topics 00 and 01: $\operatorname{Tr}(T^n)$ Chromatic Polynomials and MPS Norms
- Topic: 00 | 01
- Type: EXPLAIN
- Severity: medium
- Evidence: `topics/00-tensor-network-basics/test_basics.py:41` asserts the cycle chromatic polynomial $P(C_n, q) = (q-1)^n + (-1)^n(q-1)$ without explanation. Topic 00 presents the ladder network and cycle colouring as disconnected exercises. In fact, both are exact instances of the 1D transfer matrix method: contracting a 1D periodic ring of tensors is computing the matrix trace $\operatorname{Tr}(T^n) = \sum_i \lambda_i^n$.
  For graph colouring on $C_n$, the edge constraint matrix $T = \mathbf{1}\mathbf{1}^T - I$ has spectrum:
  - $\lambda_1 = q-1$ (eigenvector $\mathbf{1}$, multiplicity 1)
  - $\lambda_2 = -1$ (orthogonal complement $\mathbf{1}^\perp$, multiplicity $q-1$)
  The number of proper $q$-colourings is identically:
  $$P(C_n, q) = \operatorname{Tr}(T^n) = 1 \cdot (q-1)^n + (q-1) \cdot (-1)^n$$
  For the ladder network, rung-by-rung contraction is iterating the transfer operator $\mathbb{E} = \sum_i A^i \otimes (A^i)^*$.
- Proposal: Add a short explanatory note in `00-tensor-network-basics/README.md` showing how contracting a cycle network corresponds to $\operatorname{Tr}(T^n)$. This demonstrates the power of linear algebra, unites graph colouring with MPS transfer matrices, and provides an elegant one-line spectral derivation of the cycle chromatic polynomial.

### M10: Contraction Treewidth of All-to-All Circuits (QFT) vs Local 1D/2D Architectures
- Topic: 01 | 02
- Type: EXPLAIN
- Severity: low
- Evidence: `topics/01-circuits-as-tensor-networks/circuits.py:38-46` implements QFT, and `topics/02-contraction-order-and-cost/README.md:13-14` states contraction width is "constant for a chain or ring, growing with the side $L$ on an $L \times L$ grid." The topics never explain the contraction complexity of non-local circuits like QFT.
  Because QFT applies two-qubit controlled-phase gates between every pair of qubits $(j, c)$ with $j < c$, its circuit interaction graph contains the complete graph $K_n$ as a minor. The treewidth of its interaction graph is $\Theta(n)$, forcing contraction width $W = \Theta(n)$ and contraction cost $\exp(\Omega(n))$ for any contraction tree.
- Proposal: Add a brief note in `01-circuits-as-tensor-networks/README.md` and `02-contraction-order-and-cost/README.md` contrasting local architectures (1D line: $W = O(1)$; 2D grid: $W = O(L)$) with all-to-all architectures (QFT: $W = \Theta(n)$). This explains why exact tensor network contraction offers quadratic/exponential speedups for local shallow circuits, but cannot beat statevector FFT ($O(n 2^n)$) for all-to-all circuits like QFT.
