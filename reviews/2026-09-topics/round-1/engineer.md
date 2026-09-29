# Round 1: Senior Software Engineer

## Summary
The topics sequence provides an exceptionally clean, well-scoped foundation for tensor network simulation that adheres to first principles, avoids external framework bloat, and correctly prioritizes counted cost over noisy wall-clock metrics.
However, several numerical and software engineering flaws require remediation: exact integer operation counts are calculated using irrational floating-point logarithms, the QFT circuit tensor network implementation lacks end-to-end integration test coverage, and unvectorized brute-force loops cause severe 30-second execution stalls in interactive demonstration scripts.
Furthermore, subtle data structure divergence across topics (switching between tuples of arrays, decoupled label dicts, and parallel lists) together with duplicated contraction routines introduces unnecessary cognitive load for newcomers.
Addressing these issues through unified conventions, robust test assertions, and an explicit bridge in the roadmap connecting exact graph contraction to approximate Matrix Product State (MPS) truncation will elevate the repository to professional scientific computing standards.

## Findings

### E1: Exact operation count $C$ computed via floating-point logarithms and exponentiation
- Topic: 02
- Type: ERROR
- Severity: medium
- Evidence: In [order.py](topics/02-contraction-order-and-cost/order.py#L35-L42), the contraction cost $C$ (defined as scalar multiply-adds in Gray & Kourtis Eq. 5) is accumulated using:
  ```python
  size = lambda s: sum(math.log2(dims[l]) for l in s)
  ...
  C += 2 ** size(a | b)
  ```
  In [test_order.py](topics/02-contraction-order-and-cost/test_order.py#L28), the test asserts exact integer equality: `assert C == 2 * 32 * 4 + 2 * 4 * 64`. For bond dimensions that are powers of 2, $\log_2(d)$ evaluates to an exact float, but for general bond dimensions (such as $q=3$ from Topic 00, or planned arbitrary bond dimensions $\chi$ in MPS/DMRG), `math.log2` produces irrational floats. Summing irrational floats and exponentiating introduces floating-point truncation error (e.g. `2 ** (math.log2(3) + math.log2(5)) == 14.999999999999998 != 15`). In contrast, Topic 00's `pair_cost` correctly used products of integer dimensions.
- Proposal: Compute $C$ directly with exact integer multiplication using `math.prod`:
  ```python
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
  This eliminates transcendental functions from the inner contraction loop, prevents precision loss, and works for any integer bond dimension.

### E2: QFT tensor network converter and contractor are untested against DFT reference
- Topic: 01
- Type: MISSING
- Severity: high
- Evidence: In [test_circuits.py](topics/01-circuits-as-tensor-networks/test_circuits.py#L53-L60), `test_qft_matches_dft` is implemented as:
  ```python
  def test_qft_matches_dft():
      for n in (1, 2, 3, 4, 5):
          N = 2 ** n
          F = np.array([statevector(qft(n), n, x) for x in range(N)]).T
          expected = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
          assert np.allclose(F, expected)
  ```
  And [README.md](topics/01-circuits-as-tensor-networks/README.md#L34) claims:
  "- **QFT = DFT:** the circuit's matrix equals 2^(−n/2)·exp(2πi·xk/2ⁿ) for n = 1…5."
  This test checks the reference `statevector` simulator against DFT, but **never invokes `to_network` or `contract` on the QFT circuit**. Furthermore, `random_circuit` uses only 2-qubit Haar gates, and `ghz` contains only $H$ and $CNOT$. Consequently, neither `controlled_phase` nor `SWAP` gates are ever converted or contracted through the tensor network path in any test. If `to_network` or `contract` had a bug specifically affecting controlled-phase or SWAP tensors, all 7 tests in `test_circuits.py` would still pass.
- Proposal: Add an explicit tensor network check in `test_circuits.py` verifying that the contracted tensor network reproduces the QFT state vector:
  ```python
  def test_qft_network_matches_dft():
      for n in (1, 2, 3, 4):
          tensors, out = to_network(qft(n), n)
          psi_tn = contract(tensors, out).reshape(-1)
          assert np.allclose(psi_tn, statevector(qft(n), n, 0))
  ```

### E3: Unvectorized brute-force enumeration causes a 30-second stall in Topic 00 demo script
- Topic: 00
- Type: SIMPLIFY
- Severity: medium
- Evidence: In [basics.py](topics/00-tensor-network-basics/basics.py#L120-L123):
  ```python
  for name, edges, nv in [("triangle", [(0, 1), (1, 2), (2, 0)], 3),
                          ("cycle C6", [(i, (i + 1) % 6) for i in range(6)], 6),
                          ("Petersen", PETERSEN, 10)]:
      for q in (2, 3, 4):
          tn = einsum_reference(colouring_network(edges, nv, q)).real
          print(f"{name:9s} q={q}: TN = {tn:8.0f}   brute force = {count_colourings_brute(edges, nv, q)}")
  ```
  Running `python topics/00-tensor-network-basics/basics.py` takes ~30 seconds, of which ~28 seconds are spent blocked inside `count_colourings_brute(PETERSEN, 10, 4)`. That function executes $4^{10} = 1,048,576$ iterations of `itertools.product` in pure Python, each checking 15 edges.
  In [test_basics.py](topics/00-tensor-network-basics/test_basics.py#L48-L50), the author specifically avoided this freeze by testing only `for q in (2, 3):`. Yet a newcomer following the README instructions (`python basics.py`) is greeted with an unresponsive terminal on the very first lesson.
- Proposal: Either restrict the Petersen demo in `basics.py` to $q \in (2, 3)$ (which finishes in <0.05 seconds while fully demonstrating the non-trivial 120 3-colourings), or compare against the known closed-form chromatic polynomial for the Petersen graph:
  $$P(G, q) = q(q-1)(q-2)(q^7 - 12q^6 + 67q^5 - 230q^4 + 529q^3 - 814q^2 + 775q - 352)$$
  For $q=4$, $P(G, 4) = 12960$, which evaluates in $O(1)$ time.

### E4: Full path re-simulation and $O(N^2)$ list re-allocation in greedy slicer and tests
- Topic: 03
- Type: SIMPLIFY
- Severity: medium
- Evidence: In [slicing.py](topics/03-slicing/slicing.py#L61-L66,L74-L77), `slice_greedy` evaluates every candidate index $l$ by calling `width_cost(labels, path, sliced + [l])`.
  Inside `width_cost`, the entire contraction sequence is re-simulated from scratch:
  ```python
  items = [frozenset(t) - set(sliced) for t in labels]
  for i, j in path:
      a, b = items[i], items[j]
      items = [t for k, t in enumerate(items) if k not in (i, j)] + [a ^ b]
  ```
  Because `items` is reconstructed at every step of `path` ($P = N - 1$ steps) for every candidate edge $E$ and every slice step $S$, the algorithm performs $O(S \cdot E \cdot N^2)$ operations. For $N=104$, this requires millions of set allocations, causing `python slicing.py` to take ~18–20 seconds.
  Furthermore, in [test_slicing.py](topics/03-slicing/test_slicing.py#L6-L18), `test_sum_over_slices_equals_amplitude` runs `find_path(labels)` and `slice_greedy(labels, ...)` 3 times in a loop over bitstrings `x in (0, 5, 300)`. Because the circuit structure and labels are identical across bitstrings, repeating the greedy path optimization consumes 4.33 seconds of the 4.60-second test run (and ~65% of the total repository pytest duration).
- Proposal:
  1. In `test_slicing.py`, compute `path` and `sliced` once outside the bitstring loop. This reduces test execution from 4.33s to 1.4s with zero reduction in coverage.
  2. In `slicing.py`, precompute the intermediate tensor index sets along `path` once. Because `path` is fixed, slicing an index simply removes it from the precomputed sets ($s_v \setminus \text{sliced}$), reducing cost evaluation to $O(|V_B|)$ without re-allocating dynamic lists. Additionally, candidates can be pruned to only those indices belonging to tensors whose rank exceeds `target_W`.

### E5: Missing termination guards in greedy and random slicers
- Topic: 03
- Type: ERROR
- Severity: low
- Evidence: In [slicing.py](topics/03-slicing/slicing.py#L74-L77,L84-L86):
  ```python
  # slice_greedy:
  while width_cost(labels, path, sliced)[0] > target_W:
      best = min((l for l in candidates if l not in sliced),
                 key=lambda l: width_cost(labels, path, sliced + [l]))
      sliced.append(best)

  # slice_random:
  while width_cost(labels, path, sliced)[0] > target_W:
      sliced.append(candidates.pop())
  ```
  If an unreachable `target_W` is requested (e.g. `target_W < 0`, or an invalid target after all candidates are exhausted), `(l for l in candidates if l not in sliced)` becomes empty and `min()` crashes with `ValueError: min() arg is an empty sequence`. In `slice_random`, exhausting `candidates` crashes with `IndexError: pop from empty list`.
- Proposal: Add explicit termination checks guarding against empty candidate pools:
  ```python
  remaining = [l for l in candidates if l not in sliced]
  if not remaining:
      raise ValueError(f"Cannot achieve target width {target_W}; all candidates exhausted.")
  ```

### E6: Inconsistent data representations and code duplication across topics
- Topic: all
- Type: SIMPLIFY
- Severity: medium
- Evidence:
  1. **Data structure drift:** Each topic invents a slightly different format for representing a tensor network in memory:
     - Topic 00: a network is a list of `(array, tuple_of_labels)` ([basics.py:67](topics/00-tensor-network-basics/basics.py#L67)).
     - Topic 01: `to_network` returns `(tensors, open_labels)` where `tensors` is a list of `(array, tuple_of_labels)` ([circuits.py:90](topics/01-circuits-as-tensor-networks/circuits.py#L90)).
     - Topic 02: a network is decoupled into `labels` (list of label tuples) and `dims` (dict), with tensor arrays passed separately ([order.py:33, 50](topics/02-contraction-order-and-cost/order.py#L33)).
     - Topic 03: `amplitude_network` returns `(tensors, labels)` as two separate parallel lists ([slicing.py:50](topics/03-slicing/slicing.py#L50)).
  2. **Duplicated code:**
     - The step-by-step pairwise einsum contraction loop in `order.py:48-58` (`contract_along`) is duplicated in `slicing.py:98-106` (`contract_sliced`).
     - `haar_u4(rng)` in [slicing.py:14-17](topics/03-slicing/slicing.py#L14-L17) is an exact copy of `haar_unitary(4, rng)` from [circuits.py:25-29](topics/01-circuits-as-tensor-networks/circuits.py#L25-L29).
     - `statevector` in [slicing.py:30-35](topics/03-slicing/slicing.py#L30-L35) duplicates `circuits.py:57-67` restricted to 2-qubit gates.
  3. **Dead code:**
     - `I2` and `X` gates are defined in [circuits.py:12, 14](topics/01-circuits-as-tensor-networks/circuits.py#L12) but never used or tested anywhere.
     - `pair_cost` is defined in [basics.py:25-27](topics/00-tensor-network-basics/basics.py#L25-L27), is never called within `basics.py`, and is re-implemented in `order.py`.
- Proposal: Standardize the tensor network representation across all topics as `(tensors, labels)` or list of `(tensor, legs)` tuples. Maintain self-contained folders per Rule 2, but enforce consistent naming and parameter signatures. Purge unused global definitions (`I2`, `X`).

### E7: Disconnect between Topic 00 execution model and Topics 01–03
- Topic: 00 | 01 | 02
- Type: EXPLAIN
- Severity: low
- Evidence: In Topic 00, the foundational concept taught is that pairwise contraction is implemented via `transpose -> reshape -> matmul` ([basics.py:11-23](topics/00-tensor-network-basics/basics.py#L11-L23)).
  In Topic 01, this execution model is completely replaced by `opt_einsum.contract` ([circuits.py:99](topics/01-circuits-as-tensor-networks/circuits.py#L99)). In Topic 02, neither `contract_pair` nor `opt_einsum.contract` is used; instead, an ad-hoc loop invokes `np.einsum` at each tree step ([order.py:56](topics/02-contraction-order-and-cost/order.py#L56)).
  A newcomer reading sequentially is left confused about whether `contract_pair` was an educational toy or the real computational engine behind tensor network contraction.
- Proposal: Add a short explanatory note in the READMEs of Topics 01 and 02 clarifying that `opt_einsum` and numpy internally compile pairwise contractions into the exact transpose + GEMM sequence implemented in Topic 00, and that `contract_along` in Topic 02 steps through the tree manually to demonstrate intermediate state sizes.

### E8: Missing architectural bridge from exact contraction to approximate MPS
- Topic: roadmap
- Type: SEQUENCE
- Severity: medium
- Evidence: In [PLAN.md](PLAN.md#L17-L28), Topics 00–04 focus exclusively on **exact contraction** of arbitrary/circuit tensor networks (pairwise mechanics, circuit mappings, tree width bounds, index slicing, and parallel distribution).
  Topic 05 abruptly pivots to **MPS and SVD truncation**.
  The motivation for why a practitioner moves from contracting a circuit network to an MPS representation is absent. Even with optimal contraction trees and index slicing, 2D circuits and deep architectures have treewidth scaling as $\Omega(L)$, where the cost $C_s \ge C$ remains exponential in treewidth. When exact contraction is physically intractable, one must transition to controlled low-rank approximations (Eckart–Young theorem).
- Proposal: In `PLAN.md`, explicitly articulate the transition between Topic 04 and Topic 05: Topic 04 defines the practical performance frontier of exact contraction, while Topic 05 introduces the paradigm shift to approximate tensor networks (1D Matrix Product States with bounded bond dimension $\chi$) to circumvent the exponential treewidth barrier.

## Keep as is
- **First-principles minimalism:** Clean, plain Python scripts under 130 lines with zero heavy framework dependencies (only NumPy and `opt_einsum`).
- **Exact ground-truth validation:** Comparing every calculation against independent references (einsum, state-vector simulation, and analytical DFT/chromatic formulas).
- **Honest complexity metrics:** Systematically reporting counted operations ($W$ and $C$) rather than unrepeatable wall-clock times, perfectly fulfilling Guideline 5.
- **Direct literature reproductions:** Faithfully capturing key textbook and paper results, including Bridgeman & Chubb's ladder ranks and colouring counts, Mezzadri's Haar unitaries, and Gray & Kourtis's treewidth and slicing trade-off tables.
- **Fast automated test suite:** Pytest runs cleanly in under 7 seconds across the entire repository.

## Suggested learning path
- **Topic 00:** Pairwise tensor contraction is fundamentally transpose + reshape + GEMM; contraction order dictates intermediate tensor memory footprint.
- **Topic 01:** Quantum gates map to tensors and circuit amplitudes map to closed networks contracted along qubit wires.
- **Topic 02:** Contraction tree width ($W$) bounds peak memory and cost ($C$) bounds total work; the underlying graph's treewidth sets the minimum achievable width.
- **Topic 03:** Slicing indices trades an exponential increase in FLOPs for an exponential reduction in memory width, producing embarrassingly parallel sub-networks.
- **Topic 04 (Roadmap):** Parallel contraction of independent slices across multi-core/GPU workers yields near-linear speedup up to the Amdahl and memory-bandwidth limits.
- **Topic 05 (Roadmap):** When treewidth makes exact contraction intractable, SVD truncation (Eckart–Young) enables controlled approximation using 1D Matrix Product States (MPS).
- **Topic 06 (Roadmap):** Simulating quantum circuits with MPS applies local gates and truncates bond dimension $\chi$, trading simulation fidelity for polynomial runtime.
- **Topic 07 (Roadmap):** Density Matrix Renormalization Group (DMRG) variationally optimizes MPS tensors via local effective Hamiltonian sweeps to determine ground states of 1D quantum many-body systems.
