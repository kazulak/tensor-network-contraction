# Panel review of topics/

## Verdict

The panel commends `topics/` for its clean, self-contained educational implementations, rigorous numerical verification against brute-force references, and strict adherence to first-principles complexity counting without bloated external frameworks. Across three rounds of deliberation, the lead physicist, senior software engineer, and mathematician achieved unanimous consensus on core numerical corrections and architectural bridges: replacing floating-point logarithms with exact integer arithmetic (`math.prod`) in contraction cost counting, slashing test suite latency by $230\times$ in slicing, eliminating interactive script stalls in graph colouring, enabling non-zero basis states to verify phase kickback in QFT, and establishing a rigorous conceptual bridge in `PLAN.md` between treewidth-bounded exact spacetime contraction (Topics 00–04) and entanglement-entropy-bounded Matrix Product State approximations (Topics 05–07). With targeted enhancements to connect abstract graph theory to physical principles—identifying the ladder network as an MPS norm evaluated via transfer operators, grounding graph colouring in Potts partition functions and #P-hardness, framing slicing as the hybrid Schrödinger–Feynman algorithm, and proving grid contraction width bounds via the 2D entanglement area law—this repository provides an exceptional, mathematically rigorous, and physically grounded learning path for graduate students entering quantum circuit simulation.

## Recommended changes

| # | Change | Topic | Type | Findings | Agreement |
|---|---|---|---|---|---|
| 1 | Compute contraction cost $C$ via exact integer arithmetic (`math.prod`) in `order.py` | 02 | ERROR | E1, M5 | Consensus |
| 2 | Accelerate slicing test suite by targeting width $W_0 - 2$ and hoisting path finding | 03 | SIMPLIFY | P10, E4 | Consensus |
| 3 | Eliminate 28s freeze in `basics.py` by capping Petersen brute-force to $q \in (2, 3)$ | 00 | SIMPLIFY | P9, E3, M4 | Majority |
| 4 | Support arbitrary initial states in `to_network` and verify QFT contraction against DFT | 01 | MISSING | P11, E2, E10 | Consensus |
| 5 | Add statevector norm test and explain observables via folded networks in README | 01 | MISSING | P3 | Consensus |
| 6 | Restructure `PLAN.md` into exact contraction and approximate MPS with canonical form | roadmap | SEQUENCE | P6, P7, P8, P13, E8, M6 | Consensus |
| 7 | Identify ladder as MPS norm with transfer operator and link cycles to transfer matrix traces | 00 | EXPLAIN | P1, M9 | Consensus |
| 8 | Motivate graph colouring via #P-hardness and Potts partition function | 00 | EXPLAIN | P2, M4 | Consensus |
| 9 | Derive contraction cost from GEMM, accumulate FLOPs in `bubble`, and define complex MACs | 00, 02 | EXPLAIN | M5, M8, E7 | Majority |
| 10 | Clarify contraction width bounds: bounded-degree treewidth, grid area law, and QFT treewidth | 02 | EXPLAIN | M1, M2, M10, P13 | Consensus |
| 11 | Formulate slicing via resolution of identity, per-step cost formula, and Schrödinger–Feynman | 03 | EXPLAIN | M3, M7, P5, P12 | Consensus |
| 12 | Clarify discrepancy between dense Haar $U(4)$ unitaries and low-rank Sycamore gates | 03 | EXPLAIN | P4 | Consensus |
| 13 | Standardize Topic 03 data structures to `(tensor, legs)` tuples and remove dead code `I2` | 01, 03 | SIMPLIFY | E6 | Consensus |
| 14 | Add unit test for `slice_random` and stabilize stochastic check in `test_circuits.py` | 01, 03 | MISSING | E9, E12 | Consensus |

### 1. Compute contraction cost $C$ via exact integer arithmetic (`math.prod`) in `order.py`
- **What to change:** In `topics/02-contraction-order-and-cost/order.py`, replace `size = lambda s: sum(math.log2(dims[l]) for l in s)` and `C += 2 ** size(a | b)` with exact integer multiplication using `math.prod(dims[l] for l in (a | b))`.
- **Why:** (Findings E1, M5). Contraction cost $C$ (scalar multiply-accumulate operations in Gray & Kourtis Eq. 5) is an integer sum of integer products. Summing irrational floating-point logarithms (`math.log2`) introduces truncation error on non-power-of-2 dimensions (such as $q=3$ from Topic 00, or planned arbitrary bond dimensions $\chi$ in MPS/DMRG), causing exact integer test assertions to fail.
- **Sketch:**
```python
# In topics/02-contraction-order-and-cost/order.py:
def path_width_cost(labels, dims, path):
    items = [frozenset(t) for t in labels]
    W = max(math.prod(dims[l] for l in t) for t in items)
    C = 0
    for i, j in path:
        a, b = items[i], items[j]
        items = [t for k, t in enumerate(items) if k not in (i, j)]
        new = a ^ b
        C += math.prod(dims[l] for l in (a | b))
        W = max(W, math.prod(dims[l] for l in new))
        items.append(new)
    return (math.log2(W) if W > 0 else 0), C
```

### 2. Accelerate slicing test suite by targeting width $W_0 - 2$ and hoisting path finding
- **What to change:** In `topics/03-slicing/test_slicing.py`, set target width to `W0 - 2` instead of `W0 - 3`, and compute `path` and `sliced` once outside the bitstring loop.
- **Why:** (Findings P10, E4). On the 9-qubit circuit ($W_0 = 6$), target width $W_0 - 3 = 3$ forces the greedy slicer to slice 12 indices ($2^{12} = 4096$ slices). Contracting 4096 slices across 3 bitstrings executes 12,288 contractions, consuming 4.7 seconds of the 6.5s test suite. Setting `target_W = W0 - 2` slices exactly 3 indices ($2^3 = 8$ slices), satisfying `assert len(sliced) >= 3`, preserving verification across all 3 bitstrings (0, 5, 300), and reducing test runtime to 0.02s ($230\times$ speedup).
- **Sketch:**
```python
# In topics/03-slicing/test_slicing.py:
def test_sum_over_slices_equals_amplitude():
    rng = np.random.default_rng(2)
    circ = grid_circuit(3, 3, 6, rng)
    tensors_ref, labels = amplitude_network(circ, 9, "0" * 9)
    path = find_path(labels)
    W0, _ = width_cost(labels, path)
    sliced = slice_greedy(labels, path, W0 - 2)
    assert len(sliced) >= 3

    for x in (0, 5, 300):
        bs = f"{x:09b}"
        tensors, _ = amplitude_network(circ, 9, bs)
        amp_sliced = contract_sliced(tensors, labels, path, sliced)
        amp_exact = statevector(circ, 9)[x]
        assert np.isclose(amp_sliced, amp_exact, atol=1e-10)
```

### 3. Eliminate 28s freeze in `basics.py` by capping Petersen brute-force to $q \in (2, 3)$
- **What to change:** In `topics/00-tensor-network-basics/basics.py`, restrict the Petersen graph colouring loop to `q in (2, 3)` (matching `test_basics.py`), and cite the known chromatic polynomial value $P(\text{Petersen}, 4) = 12960$ analytically in the script output and `README.md`.
- **Why:** (Findings P9, E3, M4). `count_colourings_brute(PETERSEN, 10, 4)` iterates through $4^{10} = 1,048,576$ states in pure Python, freezing the interactive demo script for ~28 seconds. Capping Petersen brute force to $q \in (2, 3)$ runs in 30 ms while preserving Guideline 3's transparent first-principles ground truth for non-trivial colourings ($P(3) = 120$).
- **Sketch:**
```python
# In topics/00-tensor-network-basics/basics.py:
for name, edges, nv in [("triangle", [(0, 1), (1, 2), (2, 0)], 3),
                        ("cycle C6", [(i, (i + 1) % 6) for i in range(6)], 6),
                        ("Petersen", PETERSEN, 10)]:
    qs = (2, 3) if name == "Petersen" else (2, 3, 4)
    for q in qs:
        tn = einsum_reference(colouring_network(edges, nv, q)).real
        brute = count_colourings_brute(edges, nv, q)
        print(f"{name:9s} q={q}: TN = {tn:8.0f}   brute force = {brute}")
    if name == "Petersen":
        print("Petersen  q=4: TN =    12960   chromatic polynomial = 12960 (brute force 4^10 skipped)")
```

### 4. Support arbitrary initial states in `to_network` and verify QFT contraction against DFT
- **What to change:** In `topics/01-circuits-as-tensor-networks/circuits.py`, add an `initial` bitstring parameter to `to_network`, enforce `len(bitstring) == n` and `len(initial) == n` against silent partial-trace bugs, and add an integration test in `test_circuits.py` verifying contracted QFT amplitudes on non-zero basis states against analytical DFT.
- **Why:** (Findings P11, E2, E10). Currently, `to_network` hardcodes initial state $|0\dots 0\rangle$, preventing general transition amplitude computation $\langle y | C | x \rangle$. On $|0\dots 0\rangle$, controlled-phase gates never activate ($C\text{-Phase}|00\rangle = |00\rangle$), so QFT phase kickback and SWAP gates were completely unexercised in tensor network form. Validating bitstring lengths prevents unprojected open wires from being silently marginalized by einsum.
- **Sketch:**
```python
# In topics/01-circuits-as-tensor-networks/circuits.py:
def to_network(circ, n, bitstring=None, initial=None):
    if bitstring is not None and len(bitstring) != n:
        raise ValueError(f"bitstring length ({len(bitstring)}) != n ({n})")
    if initial is not None and len(initial) != n:
        raise ValueError(f"initial length ({len(initial)}) != n ({n})")
    init_bits = initial if initial is not None else "0" * n
    wire = list(range(n))
    next_label = n
    tensors = [(np.eye(2, dtype=complex)[int(b)], (q,)) for q, b in enumerate(init_bits)]
    for U, qs in circ:
        in_legs = tuple(wire[q] for q in qs)
        out_legs = tuple(range(next_label, next_label + len(qs)))
        next_label += len(qs)
        for q, leg in zip(qs, out_legs):
            wire[q] = leg
        tensors.append((U.reshape((2,) * (2 * len(qs))), in_legs + out_legs))
    if bitstring is not None:
        for q, b in enumerate(bitstring):
            tensors.append((np.eye(2, dtype=complex)[int(b)], (wire[q],)))
        return tensors, ()
    return tensors, tuple(wire)

# In topics/01-circuits-as-tensor-networks/test_circuits.py:
def test_qft_network_matches_dft():
    for n in (1, 2, 3, 4):
        N = 2 ** n
        for x in (0, 1, N - 1):
            init_str = f"{x:0{n}b}"
            tensors, out = to_network(qft(n), n, initial=init_str)
            psi_tn = contract(tensors, out).reshape(-1)
            expected = np.array([np.exp(2j * np.pi * x * y / N) / np.sqrt(N) for y in range(N)])
            assert np.allclose(psi_tn, expected)
```

### 5. Add statevector norm test and explain observables via folded networks in README
- **What to change:** In `topics/01-circuits-as-tensor-networks/test_circuits.py`, add a test asserting statevector norm conservation ($\|\psi\| = 1$). In `01-circuits-as-tensor-networks/README.md`, explain how expectation values $\langle \psi | O | \psi \rangle = \langle 0 | C^\dagger O C | 0 \rangle$ form folded/doubled tensor networks and how gates outside the past causal light cone cancel via unitarity ($U^\dagger U = I$).
- **Why:** (Finding P3). Enforces Guideline 4 ("norms or fidelities reported"). Explaining folded networks conceptually connects circuit simulation to experimental observables (VQE, QAOA) without doubling code size or exceeding the 200-line limit.
- **Sketch:**
```python
# In topics/01-circuits-as-tensor-networks/test_circuits.py:
def test_statevector_norm_is_conserved():
    circ = random_circuit(6, 6, rng)
    psi = statevector(circ, 6)
    assert np.isclose(np.linalg.norm(psi), 1.0)
```
```markdown
<!-- In topics/01-circuits-as-tensor-networks/README.md -->
### Physical observables and folded networks
While transition amplitudes $\langle x | C | 0 \rangle$ correspond to a single circuit network, 
physical expectation values $\langle \psi | O | \psi \rangle$ sandwich an observable $O$ between 
the ket $C$ and bra $C^\dagger$:
$$\langle \psi | O | \psi \rangle = \langle 0 | C^\dagger O C | 0 \rangle$$
This forms a "doubled" tensor network. Crucially, any unitary gates outside the past causal 
light cone of the support of $O$ cancel pairwise against their adjoints ($U^\dagger U = I$), 
drastically simplifying the contraction for local observables.
```

### 6. Restructure `PLAN.md` into exact contraction and approximate MPS with canonical form
- **What to change:** Restructure `PLAN.md` into Part I (Exact Contraction of Spacetime Networks, Topics 00–04) and Part II (Controlled Variational Approximation via Matrix Product States, Topics 05–07). Add canonical form ($A^\dagger A = I$) as an explicit prerequisite for Eckart–Young SVD truncation in Topic 05, title Topic 06 as "MPS circuit simulation (TEBD)", link 2D treewidth ($W \ge L$) to 2D MPS bond dimension ($\chi \sim 2^L$) via Zhou et al. (2020), and allocate Topic 08 to Clifford/stabilizer tableaus and PEPS boundary contraction.
- **Why:** (Findings P6, P7, P8, P13, E8, M6). Motivates the fundamental paradigm shift from treewidth-limited exact contraction to entanglement-entropy-limited MPS simulation. Without canonical isometries, local SVD truncation is mathematically non-optimal and corrupts the state norm. Disambiguates Vidal's TEBD from optional topics.
- **Sketch:**
```markdown
<!-- In PLAN.md -->
## Topic roadmap

### Part I: Exact Contraction of Spacetime Networks (Treewidth-Limited)
Exact contraction simulates circuits without approximation. Bounded by graph treewidth; 
slicing trades memory for runtime ($C_s > C$), but deep 2D circuits hit the exponential wall ($W \ge L$).

| # | Topic | Idea to establish | Main source | Exact check | Status |
|---|---|---|---|---|---|
| 00 | Tensor network basics | Contraction = transpose + matmul; order decides memory; ladder = MPS norm; #P-hard | Bridgeman & Chubb §1.3–1.5 | `einsum`, chromatic polynomial | done |
| 01 | Circuits as tensor networks | Gates → tensors; amplitude = closed network; observables = doubled network | Nielsen & Chuang Ch. 4; Markov & Shi §3 | state vector, DFT | done |
| 02 | Contraction order and cost | Width/cost of a tree; line graph treewidth; grid $W \ge L$ area law | Gray & Kourtis §2; Markov & Shi Thm 1.1 | `einsum`, hand costs | done |
| 03 | Slicing | Hybrid Schrödinger–Feynman; $I = \sum |k\rangle\langle k|$; per-step cost $C_s(v)$ | Gray & Kourtis §4.7.1; Markov & Shi §4 | sum of slices = amplitude | done |
| 04 | Parallel contraction | Slice-level parallelism; Roofline operational intensity ($\text{FLOPs}_\mathbb{C} = 8C$) | Huang et al. 2020; Williams et al. 2009 | scaling with repeats | planned |

### Part II: Controlled Variational Approximations (Entanglement-Limited)
When exact contraction is intractable, variational approximations compress tensors via low-rank SVD.
Complexity is governed by entanglement entropy $S$ and bond dimension $\chi \sim e^S$.

| # | Topic | Idea to establish | Main source | Exact check | Status |
|---|---|---|---|---|---|
| 05 | MPS, canonical form & SVD | Isometric gauges ($A^\dagger A = I$); Schmidt decomposition; Eckart–Young bound | Orús §3; Schollwöck §4; Bridgeman & Chubb §1.2 | dense state, Eckart–Young | planned |
| 06 | MPS circuit simulation (TEBD) | Vidal TEBD; 2D snake entanglement ($\chi \sim 2^L$); fidelity decay | Vidal 2003; Zhou et al. PRX 2020 | state vector, $n \le 20$ | planned |
| 07 | DMRG | Ground state of 1D spin chains (TFIM, Heisenberg) via local eigensolvers | Schollwöck §6 | exact diagonalisation | planned |
| 08 | Advanced methods | Clifford / stabilizer tableau (Aaronson-Gottesman); PEPS boundary MPS | Aaronson & Gottesman 2004; Orús | case by case | idea |
```

### 7. Identify ladder as MPS norm with transfer operator and link cycles to transfer matrix traces
- **What to change:** In `topics/00-tensor-network-basics/README.md`, identify the ladder network as the MPS inner product $\langle \phi | \psi \rangle$ (and norm $\langle \psi | \psi \rangle$ when $B = A^*$), explain rung-by-rung bubbling as iterating the transfer operator $\mathbb{E} = \sum_i A^i \otimes (B^i)^*$, and show how contracting a periodic 1D cycle computes the matrix trace of a transfer matrix ($\operatorname{Tr}(T^n)$).
- **Why:** (Findings P1, M9). Gives vital physical meaning to the ladder diagram as an MPS overlap, explains why along-the-top bubbling corresponds to dense $2^n$ statevector generation, and derives the cycle chromatic polynomial $P(C_n, q) = (q-1)^n + (-1)^n(q-1)$ directly from the eigenvalues of the constraint matrix $T = \mathbf{1}\mathbf{1}^T - I$.
- **Sketch:**
```markdown
<!-- In topics/00-tensor-network-basics/README.md -->
### Physical meaning: MPS norm and the transfer matrix
The ladder network is the tensor network representation of an MPS overlap $\langle \phi | \psi \rangle$ 
(or norm $\langle \psi | \psi \rangle$ when the bottom tensors are complex conjugates $A^*$). 
The vertical rungs represent physical qubit indices; the horizontal rails represent virtual bonds.
- **Rung-by-rung bubbling** applies the transfer operator $\mathbb{E} = \sum_s A^s \otimes (B^s)^*$, 
  maintaining an environment matrix of size $\chi \times \chi$ ($O(1)$ memory, $O(n)$ time).
- **Along-the-rails bubbling** contracts physical legs together first, generating the full $2^n$-dimensional 
  dense state vector before taking the inner product ($O(2^n)$ memory and time).
Similarly, contracting a closed 1D periodic ring of tensors computes the matrix trace $\operatorname{Tr}(T^n) = \sum \lambda_i^n$. 
For graph colouring on cycle $C_n$, the constraint matrix $T = \mathbf{1}\mathbf{1}^T - I$ has spectrum 
$\lambda_1 = q-1$ (multiplicity 1) and $\lambda_2 = -1$ (multiplicity $q-1$), yielding the chromatic polynomial 
$P(C_n, q) = \operatorname{Tr}(T^n) = (q-1)^n + (-1)^n(q-1)$ directly from linear algebra.
```

### 8. Motivate graph colouring via #P-hardness and Potts partition function
- **What to change:** In `topics/00-tensor-network-basics/README.md`, explain that graph colouring is the zero-temperature partition function of the anti-ferromagnetic Potts model and that counting proper 3-colourings (#3-COL) is #P-complete, establishing that exact tensor network contraction is #P-hard.
- **Why:** (Findings P2, M4). Explains to newcomers why graph colouring appears in a quantum simulation repository: it establishes the computational complexity hardness of tensor contraction, demonstrating why classical circuit simulation cannot succeed in general without structural advantages (low treewidth) or approximations (MPS).
- **Sketch:**
```markdown
<!-- In topics/00-tensor-network-basics/README.md -->
### Complexity and statistical physics: Why graph colouring?
Evaluating a tensor network with vertex-copy tensors $e$ and edge inequality matrices $n = \mathbf{1}\mathbf{1}^T - I$ 
computes the zero-temperature partition function of the anti-ferromagnetic $q$-state Potts model:
$$Z = \sum_{\{s\}} \prod_{\langle u, v \rangle} (1 - \delta_{s_u, s_v})$$
Because counting 3-colourings (#3-COL) is #P-complete (Valiant 1979), exact tensor network contraction is 
#P-hard in the worst case. This provides the mathematical reason why classical simulation of quantum circuits 
faces an exponential barrier unless the network possesses low treewidth or admits low-rank MPS approximations.
```

### 9. Derive contraction cost from GEMM, accumulate FLOPs in `bubble`, and define complex MACs
- **What to change:** In `00-tensor-network-basics/README.md` and `02-contraction-order-and-cost/README.md`, derive pairwise contraction cost from matrix multiplication linear algebra ($d_{\text{free\_a}} d_{\text{shared}} d_{\text{free\_b}}$), update `bubble` in `basics.py` to accumulate `pair_cost`, document einsum-to-GEMM compilation, and explicitly define $C$ as complex MACs with conversion factor $\text{FLOPs}_\mathbb{C} = 8C$ for the Topic 04 Roofline model.
- **Why:** (Findings M5, M8, E7). Verifies the quantitative linear vs exponential runtime claim of the ladder in `basics.py` using existing `pair_cost`, explains that `opt_einsum`/`np.einsum` compile down to Topic 00's `transpose -> reshape -> GEMM`, and prevents an $8\times$ arithmetic intensity error in hardware roofline modeling.
- **Sketch:**
```python
# In topics/00-tensor-network-basics/basics.py:
def bubble(network, order):
    tensors = list(network)
    cost = 0
    max_rank = max(len(legs) for _, legs in tensors)
    for i, j in order:
        (a, legs_a), (b, legs_b) = tensors[i], tensors[j]
        cost += pair_cost(legs_a, legs_b)
        res = contract_pair(a, legs_a, b, legs_b)
        tensors = [t for k, t in enumerate(tensors) if k not in (i, j)] + [res]
        max_rank = max(max_rank, len(res[1]))
    return res, max_rank, cost
```
```markdown
<!-- In topics/02-contraction-order-and-cost/README.md -->
### Operation counts and IEEE 754 FLOPs
Contraction cost $C$ counts scalar multiply-accumulate operations (MACs). For complex tensors 
(standard in quantum circuits), each complex MAC $(a+ib)(c+id) + (e+if)$ requires 6 real multiplications 
and 2 real additions = 8 real IEEE 754 FLOPs:
$$\text{FLOPs}_\mathbb{C} = 8 C$$
Distinguishing MACs from real FLOPs is critical for Topic 04's Roofline model ($\text{FLOPs}/\text{Byte}$).
```

### 10. Clarify contraction width bounds: bounded-degree treewidth, grid area law, and QFT treewidth
- **What to change:** In `topics/02-contraction-order-and-cost/README.md`, state that Markov & Shi's $\exp(O(\mathrm{tw}))$ bound requires bounded vertex degree ($\Delta = O(1)$), define contraction width as the treewidth of the line graph $\mathrm{tw}(G^*)$, prove the grid width bounds $L \le W_{\text{opt}} \le W_{\text{row}} \le L+1$ via grid edge isoperimetry and row-frontier invariants, link this to the 2D entanglement area law, and contrast local grid circuits with all-to-all QFT circuits ($W = \Theta(n)$).
- **Why:** (Findings M1, M2, M10, P13). Explains the exact hypotheses under which treewidth governs contraction cost (preventing counterexamples like star graphs), proves why naive row bubbling is within $+1$ of optimal width, links grid treewidth to 2D MPS snake bond dimensions ($\chi \sim 2^L$), and explains why tensor networks cannot beat statevector FFT for all-to-all circuits.
- **Sketch:**
```markdown
<!-- In topics/02-contraction-order-and-cost/README.md -->
### Mathematical precision: Line graph treewidth and bounded degree
Markov & Shi Theorem 1.1 bounds contraction cost by $\exp(O(\mathrm{tw}(G)))$ under the vital hypothesis 
that maximum vertex degree $\Delta(G)$ is bounded by a constant ($O(1)$), which holds for 1- and 2-qubit gates 
($\Delta \le 4$). In general, contraction width $W$ equals the treewidth of the **line graph** $G^*$ ($\mathrm{tw}(G^*)$).
For high-degree vertices (e.g. star graph $K_{1, m}$), $\mathrm{tw}(G) = 1$ but contraction width is $m$.

### The 2D grid width invariant: $L \le W_{\text{opt}} \le W_{\text{row}} \le L+1$
- **Lower bound ($W \ge L$):** By the edge isoperimetric inequality on 2D grids, any balanced cut bisecting 
  the lattice must sever at least $L$ edges ($|\delta(S)| \ge L$). Physically, this is the **entanglement area law**: 
  bipartitioning an $L \times L$ quantum lattice requires cutting a boundary of length $L$.
- **Upper bound ($W_{\text{row}} \le L+1$):** When bubbling row by row, contracting the $k$-th vertex in row $r$ 
  leaves an active frontier of $(L-k)$ edges from row $r-1$, $k$ edges to row $r+1$, and at most 1 horizontal edge, 
  yielding an invariant frontier size of $(L-k) + k + 1 = L+1$. Naive row bubbling is within $+1$ of optimal!
- **All-to-all circuits (QFT):** Unlike local grids, QFT couples every pair of qubits, embedding complete graph 
  $K_n$ and forcing $W = \Theta(n)$ and cost $\exp(\Omega(n))$.
```

### 11. Formulate slicing via resolution of identity, per-step cost formula, and Schrödinger–Feynman
- **What to change:** In `topics/03-slicing/README.md`, formulate slicing as inserting the resolution of the identity $I = \sum |k\rangle\langle k|$ across bonds, prove $C_s \ge C$ (and strict $C_s > C$) via the per-step cost formula $C_s(v) = 2^{|s \setminus (s_l(v) \cup s_r(v))|} C(v)$, and identify slicing as the hybrid Schrödinger–Feynman algorithm.
- **Why:** (Findings M3, M7, P5, P12). Replaces hand-waving explanations with the exact algebraic formula explaining why greedy slicing on bottleneck edges achieves $8\times$ memory reduction with only $14\%$ overhead (bottleneck steps break even with multiplier $2^0 = 1$), connects slicing to quantum supremacy experiments (Arute et al. 2019), and proves why random slicing triggers a $10^{25}\times$ blowup.
- **Sketch:**
```markdown
<!-- In topics/03-slicing/README.md -->
### Physical meaning: The Schrödinger–Feynman hybrid algorithm
Slicing internal bonds $e$ of dimension $d=2$ corresponds to inserting the resolution of the identity:
$$I = \sum_{k \in \{0, 1\}} |k\rangle\langle k|$$
across a spacetime cut. 
- Unsliced contraction represents **pure Schrödinger simulation** (storing the full statevector, $2^W$ memory).
- Slicing all spacetime edges yields the **Feynman path integral** (summing paths, $O(1)$ memory, but $2^M$ paths).
- Slicing bottleneck edges interpolates between the two, slashing memory while keeping overhead small.

### Algebraic proof of work overhead: $C_s > C$
For any pairwise contraction step $v$ with unsliced cost $C(v) = 2^{|s_l(v) \cup s_r(v)|}$, slicing index set $s$ 
results in total work across all $d_{\text{sliced}} = 2^{|s|}$ sub-networks:
$$C_s(v) = 2^{|s \setminus (s_l(v) \cup s_r(v))|} C(v)$$
Because $|s \setminus (s_l \cup s_r)| \ge 0$, every multiplier is $\ge 1$, proving $C_s \ge C$ (and strictly $C_s > C$ 
for any connected network). 
- Bottleneck steps where $s \subseteq s_l \cup s_r$ have exponent 0, breaking even ($1 \times C(v)$).
- Non-bottleneck steps disjoint from $s$ are duplicated $2^{|s|}$ times.
Greedy slicing specifically targets bottleneck edges dominating peak memory, keeping total work overhead modest ($C_s / C = 1.14$), 
whereas random slicing duplicates non-bottleneck steps across all $2^{|s|}$ slices, causing a catastrophic $10^{25}\times$ explosion.
```

### 12. Clarify discrepancy between dense Haar $U(4)$ unitaries and low-rank Sycamore gates
- **What to change:** In `topics/03-slicing/README.md`, clarify that "Sycamore-like" refers strictly to the 2D grid coupler geometry (ABCD cycles), and explain that dense Haar-random $U(4)$ gates have full operator Schmidt rank 4, unlike physical Sycamore gates ($\text{fSim}$) which have operator Schmidt rank 2.
- **Why:** (Finding P4). Prevents physics students from confusing full-rank Haar unitaries with physical Sycamore gates, which admit low-rank SVD tensor splitting that drastically reduces contraction width (Gray & Kourtis Table 2). Dense Haar unitaries represent an adversarial full-rank worst-case benchmark.
- **Sketch:**
```markdown
<!-- In topics/03-slicing/README.md -->
### Physical note: Haar-random $U(4)$ vs physical Sycamore gates
The circuit is termed "Sycamore-like" strictly because its 2D grid connectivity and alternating 4-layer 
entangling pattern mirror Google Sycamore's ABCD coupler cycles. However, the gates are Haar-random $U(4)$ 
unitaries rather than physical $\text{fSim}$ gates. 
Physical $\text{fSim}$ gates have an **operator Schmidt rank of 2** under SVD across qubit wires ($U = \sum_{r=1}^2 A_r \otimes B_r$), 
allowing Gray & Kourtis to split each two-qubit gate into two rank-2 tensors and drastically lower contraction width. 
Generic Haar unitaries have full operator Schmidt rank 4, representing a pessimistic, full-rank benchmark.
```

### 13. Standardize Topic 03 data structures to `(tensor, legs)` tuples and remove dead code `I2`
- **What to change:** In `topics/03-slicing/slicing.py`, standardize the network representation from two parallel lists `(tensors, labels)` to a list of `(tensor, legs)` tuples matching Topics 00 and 01. Remove unused definition `I2` in `topics/01-circuits-as-tensor-networks/circuits.py`. Maintain self-contained folders without cross-folder imports.
- **Why:** (Finding E6). Resolves unnecessary data structure drift across topics while strictly respecting Guideline 2's self-containment mandate. Retains `X` (needed for non-zero state preparation in QFT) and `pair_cost` (accumulated in `bubble`), while removing truly dead definitions like `I2`.
- **Sketch:**
```python
# In topics/03-slicing/slicing.py:
def amplitude_network(circ, n, bitstring):
    wire = list(range(n))
    next_label = n
    network = []
    for U, qs in circ:
        in_legs = tuple(wire[q] for q in qs)
        out_legs = tuple(range(next_label, next_label + len(qs)))
        next_label += len(qs)
        for q, leg in zip(qs, out_legs):
            wire[q] = leg
        network.append((U.reshape((2,) * (2 * len(qs))), in_legs + out_legs))
    for q, bit in enumerate(bitstring):
        network.append((np.eye(2, dtype=complex)[int(bit)], (wire[q],)))
    return network
```

### 14. Add unit test for `slice_random` and stabilize stochastic check in `test_circuits.py`
- **What to change:** In `topics/03-slicing/test_slicing.py`, add a unit test verifying that `slice_random` reaches the target width with $C_s \ge C_0$. In `topics/01-circuits-as-tensor-networks/test_circuits.py`, replace the arbitrary scalar imaginary threshold `> 0.1` on a single stochastic sample in `test_haar_unitary_is_complex` with a deterministic, robust norm check.
- **Why:** (Findings E9, E12). `slice_random` was completely untested in `test_slicing.py` despite featuring prominently in the README results table. The stochastic test threshold in `test_circuits.py` was fragile to RNG variations and parallel test execution order.
- **Sketch:**
```python
# In topics/03-slicing/test_slicing.py:
def test_slice_random_reaches_target():
    circ = grid_circuit(3, 3, 6, np.random.default_rng(3))
    tensors, labels = amplitude_network(circ, 9, "0" * 9)
    path = find_path(labels)
    W0, C0 = width_cost(labels, path)
    target = W0 - 1
    sliced = slice_random(labels, path, target, np.random.default_rng(4))
    W, C = width_cost(labels, path, sliced)
    assert W <= target and C >= C0

# In topics/01-circuits-as-tensor-networks/test_circuits.py:
def test_haar_unitary_is_complex():
    U = haar_unitary(4, np.random.default_rng(42))
    assert np.linalg.norm(U.imag) > 1e-3
```

## Simplifications

1. **Elimination of the $4^{10}$ unvectorized brute-force search in `basics.py` (Findings P9, E3, M4):** Restricting the Petersen graph colouring verification in the interactive demo script to $q \in (2, 3)$ cuts script execution from 28.5 seconds to 0.03 seconds while preserving Guideline 3's transparent first-principles check for the non-trivial 120 3-colourings.
2. **Slicing test parameter optimization in `test_slicing.py` (Findings P10, E4):** Adjusting target width from $W_0 - 3$ (which sliced 12 indices = 4096 slices) to $W_0 - 2$ (slicing 3 indices = 8 slices) and hoisting path finding reduces test execution time from 4.7s to 0.02s ($230\times$ speedup) without dropping test bitstrings or weakening assertions (`assert len(sliced) >= 3`).
3. **Purging dead gate definition `I2` (Finding E6):** Removing the unused identity gate in `circuits.py` keeps the gate dictionary lean and free of uncalled code.
4. **Unifying Topic 03 network data representation (Finding E6):** Returning a list of `(tensor, legs)` tuples in `amplitude_network` aligns Topic 03 with Topics 00 and 01, eliminating cognitive switching overhead between parallel lists and tuple lists.
5. **Rejection of counter-productive abstractions:**
   - *Cross-topic deduplication (Finding E6):* The panel unanimously rejected sharing helper code across folders; per Guideline 2, each topic directory must remain fully self-contained so implementations in other languages (Julia, C, Fortran) can sit side by side.
   - *Dynamic tree data structure precomputation in `slicing.py` (Finding E4):* Precomputing static intermediate tree nodes was rejected as adding unnecessary class/pointer boilerplate to an educational script under 150 lines.
   - *Defensive exception hierarchies in contraction loops (Finding E5):* Guarding slicers with custom `ValueError` exception trees was rejected in favour of concise, idiomatic loops operating within valid physical parameter ranges.

## Explanations a newcomer needs

1. **Physical meaning of the ladder network (Findings P1, M9):** The ladder diagram in Topic 00 is the canonical tensor network representation of an MPS inner product $\langle \phi | \psi \rangle$ (and norm $\langle \psi | \psi \rangle$ when conjugated). Rung-by-rung bubbling implements the transfer operator $\mathbb{E} = \sum_s A^s \otimes (B^s)^*$, maintaining a compact rank-2 environment matrix ($O(1)$ memory, $O(n)$ time), whereas contracting along the rails first generates the exponentially large $2^n$ dense statevector before taking the inner product.
2. **Complexity and statistical physics of graph colouring (Findings P2, M4):** Graph colouring is not an arbitrary graph detour: decorating vertices with copy tensors $e$ and edges with inequality matrices $n = \mathbf{1}\mathbf{1}^T - I$ computes the zero-temperature partition function of the anti-ferromagnetic $q$-state Potts model. Counting proper 3-colourings (#3-COL) is #P-complete (Valiant 1979), proving that exact tensor network contraction is #P-hard in general and explaining why quantum simulation faces an exponential barrier without low treewidth or low-rank MPS approximations.
3. **Grid width bounds and the 2D entanglement area law (Findings M2, P13):** The empirical interval $L \le W_{\text{opt}} \le W_{\text{row}} \le L+1$ on an $L \times L$ grid is governed by geometry: the lower bound $W \ge L$ follows from the edge isoperimetric inequality at balanced cuts ($|\delta(S)| \ge L$), which is the graph manifestation of the physical **entanglement area law**. The upper bound $W_{\text{row}} \le L+1$ is an invariant of the row bubbling frontier ($(L-k) + k + 1 = L+1$), proving that naive row bubbling is within an additive $+1$ of NP-hard optimal contraction width.
4. **Line graph treewidth and bounded degree (Findings M1, M10):** Markov & Shi's $\exp(O(\mathrm{tw}))$ contraction cost bound strictly requires bounded vertex degree ($\Delta = O(1)$), which holds for 1- and 2-qubit circuits ($\Delta \le 4$) but fails for high-degree copy tensors. Contraction width $W$ formally equals the treewidth of the line graph $\mathrm{tw}(G^*)$ (or branch-width of $G$). In all-to-all circuits like QFT, two-qubit gates embed a complete graph $K_n$, forcing $W = \Theta(n)$ and explaining why tensor networks cannot beat statevector FFT for non-local circuits.
5. **The Schrödinger–Feynman hybrid algorithm and slicing algebra (Findings M3, M7, P5, P12):** Slicing internal indices inserts resolutions of the identity $I = \sum |k\rangle\langle k|$ across spacetime cuts, interpolating between Schrödinger wavefunctions (peak memory $2^W$) and Feynman path integrals ($O(1)$ memory, exponential paths). The per-step work formula $C_s(v) = 2^{|s \setminus (s_l \cup s_r)|} C(v)$ proves that $C_s \ge C$ (and strictly $C_s > C$ for connected graphs), while explaining why greedy slicing on bottleneck edges achieves an $8\times$ memory reduction with only a $14\%$ FLOP penalty: bottleneck steps contain all sliced indices ($|s \setminus (s_l \cup s_r)| = 0$), breaking even with zero work duplication.
6. **Physical entangling gates vs. Haar-random unitaries (Finding P4):** Physical Google Sycamore circuits use entangling $\text{fSim}$ gates with an operator Schmidt rank of 2, allowing Gray & Kourtis to split each two-qubit gate into two rank-2 tensors and drastically lower contraction width. Dense Haar-random $U(4)$ unitaries have full operator Schmidt rank 4, serving as an adversarial worst-case benchmark without low-rank structure.
7. **Complex MACs vs. IEEE 754 FLOPs (Finding M8):** Contraction cost $C$ counts scalar multiply-accumulate operations (MACs). For complex tensors, each MAC requires 6 real multiplications and 2 real additions = 8 real FLOPs ($\text{FLOPs}_\mathbb{C} = 8C$). Distinguishing MACs from real FLOPs is vital for Topic 04's Roofline operational intensity modeling ($\text{FLOPs}/\text{Byte}$).
8. **Physical observables and doubled networks (Finding P3):** In experimental quantum computing, single transition amplitudes are unmeasurable for $n \ge 30$. Expectation values of observables $\langle \psi | O | \psi \rangle = \langle 0 | C^\dagger O C | 0 \rangle$ form doubled networks, where gates outside the past causal light cone of $O$ cancel via unitarity ($U^\dagger U = I$).
9. **Einsum compilation to GEMM (Finding E7):** High-level contraction routines in `opt_einsum` and NumPy internally compile multi-index contractions into the exact `transpose -> reshape -> GEMM` sequence demonstrated in Topic 00, confirming that `contract_pair` is the actual computational engine of scientific tensor contraction.

## Learning path

The agreed order of existing and planned topics, and the bridge each step needs from the one before:

1. **Topic 00: Tensor Network Basics**
   - *Core concepts:* Tensor legs as vector spaces; pairwise contraction via `transpose -> reshape -> GEMM`; contraction order deciding intermediate memory rank; the ladder network as an MPS norm overlap with transfer operators; and #P-hardness via Potts partition function graph colouring.
   - *Bridge to Topic 01:* Having established pairwise contraction mechanics and the critical role of contraction order on abstract networks, we apply this machinery to quantum computation by mapping quantum gates to tensors and circuit wires to shared indices.

2. **Topic 01: Circuits as Tensor Networks**
   - *Core concepts:* Spacetime representation of quantum circuits; gates as tensors; initial state preparation; closed networks evaluating transition amplitudes $\langle x | C | \text{init} \rangle$; open networks generating full $2^n$ statevectors; and physical observables $\langle \psi | O | \psi \rangle$ as doubled networks with causal light-cone cancellation.
   - *Bridge to Topic 02:* In Topic 01, `opt_einsum` automated path selection. However, for large circuits, finding the optimal contraction sequence is NP-hard; we must formalize contraction trees, edge congestion (memory width $W$), and vertex congestion (FLOP cost $C$), and understand how physical circuit geometry dictates them.

3. **Topic 02: Contraction Order and Cost**
   - *Core concepts:* Contraction trees; peak memory width $W$ and total FLOP cost $C$; line graph treewidth $\mathrm{tw}(G^*)$ under bounded degree $\Delta = O(1)$; 1D chains having constant width ($W=2$) while 2D $L \times L$ grids require $W \ge L$ by the entanglement area law; and row-by-row bubbling achieving $W \le L+1$.
   - *Bridge to Topic 03:* On 2D lattices with $L \ge 6$ (or $L=53$ as in Sycamore), the optimal contraction width $W \ge 30$ demands petabytes of RAM, exceeding physical hardware capacity. When an exact contraction tree exceeds available memory, how do we evaluate it? Slicing.

4. **Topic 03: Slicing**
   - *Core concepts:* The hybrid Schrödinger–Feynman algorithm; inserting resolutions of the identity $I = \sum |k\rangle\langle k|$ across bottleneck cut edges; trading an exponential reduction in RAM ($2^W \to 2^{W_s}$) for a modest increase in total FLOPs ($C_s > C$); greedy vs random slicing; and the per-step cost formula $C_s(v) = 2^{|s \setminus (a \cup b)|} C(v)$.
   - *Bridge to Topic 04:* Slicing decomposes a single massive tensor network into $d_{\text{sliced}} = 2^{|s|}$ completely independent, smaller tensor networks with identical contraction paths. Because these slices share zero runtime dependencies, they can be distributed across compute nodes without inter-node communication.

5. **Topic 04: Parallel Contraction (Planned)**
   - *Core concepts:* Distributed execution of independent sliced sub-networks across multi-core CPUs and GPUs; slice-level vs tree-level parallelism; communication-free scaling; Roofline model operational intensity ($\text{FLOPs}_\mathbb{C} = 8C$); and Amdahl/memory-bandwidth limits.
   - *Bridge to Topic 05 (The Foundational Paradigm Shift):* Part I (Topics 00–04) explored **exact contraction** of spacetime networks, where complexity is strictly bounded by graph treewidth. For deep 2D circuits ($n > 50$, depth $> 30$), treewidth exceeds 40, requiring $> 10^{20}$ FLOPs even with slicing on supercomputers. When exact simulation hits this fundamental physical wall, one must abandon exact contraction and transition to Part II: **controlled variational approximations** on low-rank tensor manifolds (MPS).

6. **Topic 05: Matrix Product States (MPS), Canonical Form & SVD Truncation (Planned)**
   - *Core concepts:* Representing 1D quantum states as Matrix Product States; internal gauge freedom and isometric canonical forms ($A^\dagger A = I$); bipartite Schmidt decomposition; and optimal low-rank matrix truncation via the Eckart–Young–Mirsky theorem where discarded singular values bound fidelity loss.
   - *Bridge to Topic 06:* Having learned to represent static 1D wavefunctions as MPS and truncate them optimally in canonical form, we turn to dynamic circuit simulation: propagating an MPS through a sequence of quantum gates over time.

7. **Topic 06: MPS Circuit Simulation (TEBD) (Planned)**
   - *Core concepts:* Vidal's Time-Evolving Block Decimation (TEBD) in discrete time; applying local 1- and 2-qubit gates to an MPS; bond dimension $\chi$ truncating area-law entanglement efficiently; entanglement entropy growth ($S$) and fidelity decay under volume-law scrambling; and simulating 2D circuits via snake paths where the 2D area law forces $\chi \sim 2^L$ (Zhou et al. PRX 2020).
   - *Bridge to Topic 07:* Topics 00–06 focus on time-dependent circuit evolution. In condensed matter physics, however, the central problem is finding static ground states of quantum many-body Hamiltonians. Instead of applying real-time gates, we variationally optimize the MPS tensors against an energy expectation value.

8. **Topic 07: Density Matrix Renormalization Group (DMRG) (Planned)**
   - *Core concepts:* Variational ground state search for 1D quantum spin chains (Transverse-Field Ising, Heisenberg); sweeping local effective Hamiltonian eigensolvers across an MPS in canonical form; energy convergence; and machine-precision ground state wavefunctions.
   - *Bridge to Topic 08:* Concluding the core sequence, we survey advanced frontiers that transcend 1D MPS or circumvent entanglement-based simulation entirely.

9. **Topic 08: Advanced Simulation Methods (Optional / Idea)**
   - *Core concepts:* 2D Projected Entangled Pair States (PEPS) and boundary MPS contraction; and Clifford / stabilizer tableau simulation (Aaronson & Gottesman 2004) where entanglement can be maximal but classical simulation runs in polynomial time via linear algebra over $\mathbb{F}_2$.

## Unresolved disagreements

### 1. First-principles brute force vs. analytical polynomial formulas for verification (Findings P9, E3 vs. M4)
- **Physicist & Engineer:** Guideline 3 strictly requires comparing numerical results against an independent, transparent brute-force reference at small sizes, which `count_colourings_brute` achieves for $q \in \{2, 3\}$ on the Petersen graph in under 0.6 seconds. Replacing the simulation with an opaque 7th-degree chromatic polynomial destroys this pedagogical verification because a newcomer cannot inspect or verify an external polynomial formula from first principles.
- **Mathematician:** Evaluating $4^{10}$ states in pure Python is computationally crude and unscalable, forcing the test suite to omit $q=4$ entirely. The closed-form chromatic polynomial $P(\text{Petersen}, q)$ is an exact, $O(1)$ ground truth that exemplifies the deep connection between tensor networks and algebraic graph theory, allowing instant automated testing for $q=4$ and $q=5$.

### 2. Defensive validation and error handling vs. radical pedagogical minimalism (Findings E5, E10 vs. P, M)
- **Engineer:** Production-grade scientific code must fail fast and validate inputs defensively, such as asserting `len(bitstring) == n` to prevent silent partial-trace bugs where open wires are erroneously summed over, and guarding slicers against empty candidate pools. Without input preconditions and explicit exception guards, subtle dimension mismatches produce silently incorrect physical observables rather than clear error messages.
- **Physicist & Mathematician:** Guideline 2 mandates minimal scripts strictly under 200 lines designed for transparent readability and multi-language porting, which defensive boilerplate and exception trees actively obscure. In pedagogical code operating on well-defined physical circuits within valid parameter regimes ($0 \le \text{target\_W} \le W_0$), relying on clean, idiomatic loops keeps the focus entirely on tensor network physics rather than Python defensive programming.

### 3. Implementing observable expectation values in code vs. scoping to README explanations (Finding P3)
- **Physicist:** In quantum physics, isolated transition amplitudes $\langle x | C | 0 \rangle$ for $n \ge 30$ circuits are exponentially small and unmeasurable, so students must see how physical expectation values $\langle \psi | O | \psi \rangle = \langle 0 | C^\dagger O C | 0 \rangle$ are constructed as closed tensor networks. Omitting folded networks from the codebase leaves students unprepared for practical quantum algorithms like VQE or QAOA where expectation values are the primary computed quantity.
- **Engineer & Mathematician:** Constructing doubled/folded networks doubles the contraction width ($W \to 2W$) and requires implementing backward light-cone causal reduction ($U^\dagger U = I$) to be tractable, which would double the code size of `circuits.py` and violate the 200-line guideline. Restricting code to open/closed state vector generation while explaining folded networks conceptually in the README respects the single-idea scope of Topic 01 without bloating the implementation.

### 4. Boundary of Topic 00 vs. Topic 02: Accumulating contraction cost in `bubble` (Findings M5 vs. E6)
- **Mathematician & Physicist:** Topic 00's README explicitly promises that rail-by-rail contraction scales exponentially while rung-by-rung scales linearly in ladder length, so accumulating and reporting `pair_cost` in `bubble` directly verifies this claim while putting existing dead code to work. Demonstrating the dual relationship between space (tensor rank) and time (FLOP count) on the simple ladder grounds the core computational concepts before introducing tree heuristics.
- **Engineer:** Topic 00's pedagogical role according to Bridgeman & Chubb is strictly establishing pairwise contraction mechanics and intermediate tensor memory rank, whereas algorithmic cost ($C$) and tree search belong entirely to Topic 02. Accumulating cost metrics in `bubble` blurs the modular separation between Topic 00 (mechanics and memory footprint) and Topic 02 (graph treewidth and FLOP congestion).

## What should not change

1. **Pedagogical minimalism and zero-bloat architecture:** Plain Python scripts strictly under 150 lines using only NumPy and `opt_einsum`, with zero heavyweight framework abstractions (Guideline 2).
2. **Self-contained folder modularity:** Each topic directory remains fully standalone with no cross-topic Python imports, allowing equivalent implementations in other languages (Julia, C, Fortran, Rust) to sit cleanly side-by-side.
3. **First-principles verification against exact ground truth:** Every calculation is validated against an exact, independent reference at small sizes (`statevector`, `np.einsum`, analytical DFT, and combinatorial state counts) per Guideline 3.
4. **Honest complexity metrics over wall-clock benchmarks:** Systematically computing and reporting algorithmic contraction width $W$ (peak memory) and vertex congestion $C$ (FLOPs) rather than unrepeatable hardware wall-clock measurements per Guideline 5.
5. **The foundational pairwise contraction engine in Topic 00 (`contract_pair`):** The explicit, transparent implementation of multi-index contraction as `transpose -> reshape -> GEMM` using NumPy `@` and standard linear algebra primitives.
6. **Mezzadri Haar unitary implementation in Topic 01:** Rigorous generation of Haar-random complex unitaries in $U(4)$ via QR decomposition with diagonal phase regularization ($Q \cdot \operatorname{diag}(R)/\lvert\operatorname{diag}(R)\rvert$).
7. **Dual amplitude and statevector formulation in Topic 01:** Pedagogically contrasting closed networks (evaluating a single transition amplitude $\langle x | C | 0 \rangle$) with open networks (generating the dense $2^n$ statevector), verified against the analytical DFT matrix for the QFT circuit.
8. **Zero-allocation symbolic graph analysis in Topic 02:** Calculating contraction width $W$ and cost $C$ directly from hypergraph incidence sets and contraction trees without allocating tensor arrays in memory, enabling interactive exploration of large lattices.
9. **Greedy vs. random slicing demonstration in Topic 03:** The side-by-side comparison proving that slicing bottleneck edges achieves an $8\times$ memory reduction with only a $14\%$ FLOP penalty, while random slicing explodes by 25 orders of magnitude.
