# Scientific Audit: Study A

> **AI-generated audit, not peer-reviewed.** Written by Gemini 3.8 Flash (High) as a blind auditor; its findings were checked by Claude Code, see [the run README](../README.md#audit-by-gemini-38-flash-how-good-is-an-ai-auditor).

## 1. Executive Summary

Study A is a rigorous, high-quality investigation of tensor network slicing on 1D brickwork and 2D Sycamore-like quantum circuits. It features correct mathematical foundations, genuine Haar-random $U(4)$ gate generation, machine-precision 3-way numerical equivalence between state-vector simulation, unsliced tensor networks, and sliced contractions, and flawless operation counting using `cotengra`. Its empirical numbers reproduce almost exactly, with tiny discrepancies at two points traced to dictionary/set iteration ordering in Python when tie-breaking greedy slice candidates. Its direct answers are clear, quantitative, and insightful. Its only notable defect is a single hallucinated citation (attributing a string theory arXiv ID to a quantum circuit simulator paper).

---

## 2. Detailed Audit Criteria

### 2.1 Runs as Delivered (Score: 2/2)
- **Pytest Execution**: Running `pytest -v` in [`work/study-A/`](../gemini-3.7-flash-high/) ran 9 test cases in [`test_correctness.py`](../gemini-3.7-flash-high/test_correctness.py) and completed with **9 passed in 0.83s** (0 failures, 0 warnings).
- **Main Script Execution**: Running `python slicing_study.py` in [`work/study-A/`](../gemini-3.7-flash-high/) executed without errors in ~2.1 seconds, printed full Markdown summary tables to stdout, and generated [`slicing_study.png`](../gemini-3.7-flash-high/slicing_study.png).

### 2.2 Reproducibility / Honesty (Score: 2/2)
Comparing the numbers in [`README.md`](../gemini-3.7-flash-high/README.md) lines 54–80 against the stdout output of `python slicing_study.py`:

- **1D Brickwork ($N=12, D=10$)**:
  - Baseline: FLOPs $C_0 = 51,320$, $M_0 = 512$ elements ($2^9$). **Exact match** (line 55).
  - $k=1$: Greedy Size 512, FLOPs 70,608 (1.376x); Random Size 512.0, OH 1.90x; Boundary OH 2.00x. **Exact match** (line 59).
  - $k=2$: README reports Greedy FLOPs 104,688 (2.040x) (line 60). Rerun produced 104,752 (2.041x) (difference of +64 FLOPs, < 0.06%).
  - $k=3$: Greedy Size 256, FLOPs 161,504 (3.147x); Random OH 7.11x; Boundary OH 8.00x. **Exact match** (line 61).
  - $k=4$: Greedy Size 256, FLOPs 250,112 (4.874x); Random OH 13.15x; Boundary OH 15.99x. **Exact match** (line 62).
  - $k=5$: Greedy Size 256, FLOPs 395,392 (7.704x); Random OH 23.93x; Boundary OH 31.98x. **Exact match** (line 63).
  - $k=6$: Greedy Size 256, FLOPs 649,472 (12.655x); Random OH 50.01x; Boundary OH 63.94x. **Exact match** (line 64).
  - $k=7$: Greedy Size 128, FLOPs 1,069,568 (20.841x); Random OH 92.31x; Boundary OH 127.86x. **Exact match** (line 65).
  - $k=8$: Greedy Size 64, FLOPs 1,836,032 (35.776x); Random OH 150.80x; Boundary OH 255.68x. **Exact match** (line 66).

- **2D Sycamore Grid ($4 \times 4, D=8$)**:
  - Baseline: FLOPs $C_0 = 255,968$, $M_0 = 4,096$ elements ($2^{12}$). **Exact match** (line 69).
  - $k=1$: Greedy Size 2,048, FLOPs 271,296 (1.060x). **Exact match** (line 73).
  - $k=2$: Greedy Size 1,024, FLOPs 301,376 (1.177x). **Exact match** (line 74).
  - $k=3$: Greedy Size 1,024, FLOPs 355,328 (1.388x). **Exact match** (line 75).
  - $k=4$: README reports Greedy Size 1,024, FLOPs 451,072 (1.762x) (line 76). Rerun produced Greedy Size 512 or 1,024, FLOPs 446,720 (1.745x).
  - $k=5$: Greedy Size 512, FLOPs 593,920 (2.320x). **Exact match** (line 77).
  - $k=6$: Greedy Size 256, FLOPs 866,816 (3.386x). **Exact match** (line 78).
  - $k=7$: Greedy Size 256, FLOPs 1,354,752 (5.293x). **Exact match** (line 79).
  - $k=8$: README reports Greedy Size 256, FLOPs 2,039,808 (7.969x) (line 80). Rerun produced Greedy Size 128 or 256, FLOPs 2,041,856 (7.977x).

- **Root Cause of Slight Variance**: In [`slicing_study.py`](../gemini-3.7-flash-high/slicing_study.py), `size_dict` is constructed via `{ix: 2 for ix in set(sum(inputs, []))}`. Because Python randomizes `hash()` seeds across processes unless `PYTHONHASHSEED` is fixed, iterating over `set` yields slightly different tie-breaking orders for candidate edges with identical scores in `cotengra.SliceFinder`. Testing with `PYTHONHASHSEED=0` reproduces the exact delivered 1D $k=2$ value (`104,688` FLOPs and `2.040x` overhead) byte-for-byte. All numbers are genuine and unmanipulated.

### 2.3 Correctness (Score: 2/2)
1. **Haar Unitary Generation**: In [`slicing_study.py`](../gemini-3.7-flash-high/slicing_study.py), `haar_u4` draws a complex Gaussian Ginibre matrix $Z \in \mathbb{C}^{4 \times 4}$, computes its QR decomposition $Z = Q R$, extracts diagonal phases $d / |d|$, and returns $Q \cdot \operatorname{diag}(d / |d|)$ in `np.complex128`, exactly following Mezzadri (2007). Unitarity $\|U^\dagger U - I\| < 10^{-14}$ is confirmed in [`test_correctness.py`](../gemini-3.7-flash-high/test_correctness.py).
2. **State-Vector Simulation**: In [`slicing_study.py`](../gemini-3.7-flash-high/slicing_study.py), [`simulate_sv`](../gemini-3.7-flash-high/slicing_study.py) implements exact gate application via `np.tensordot` and `np.moveaxis`. In [`test_correctness.py`](../gemini-3.7-flash-high/test_correctness.py), all 3-way checks between state-vector simulation, unsliced contraction, and sliced contraction pass with absolute errors $< 10^{-14}$ (exceeding the required $\sim 10^{-10}$).
3. **Slicing and Cost Accounting**:
   - Unsliced baseline: Generated with `ctg.GreedyOptimizer()`, yielding a genuine contraction tree.
   - Slicing FLOP accounting: When an index `ix` is removed with `t_g.remove_ind_(ix)`, `cotengra` increments `nslices` ($2^k$) and updates the tree's internal multiplicity. Calling `t_g.total_flops()` sums all operations across all $2^k$ slices. FLOPs are counted correctly and are NOT mistakenly counted only once or double-multiplied.
   - Peak intermediate size: `t_g.max_size()` accurately tracks the largest intermediate tensor produced during sub-contraction.
4. **Circuit Geometries**: Correctly implements 1D brickwork and 2D Sycamore ABCD coupler layers ([`slicing_study.py`](../gemini-3.7-flash-high/slicing_study.py)).

### 2.4 Answers Follow from Data (Score: 2/2)
In [`README.md`](../gemini-3.7-flash-high/README.md), the conclusions directly mirror the empirical tables:
- **Cost of memory reduction**: Highlights that 2D Sycamore allows a $2\times$ memory reduction for only $6.0\%$ extra FLOPs ($k=1$), $4\times$ reduction for $17.7\%$ extra FLOPs ($k=2$), and $16\times$ reduction for $3.39\times$ overhead ($k=6$), demonstrating sub-exponential scaling ($\sim 2^{0.35 k}$). For 1D, explains that treewidth limits make deep slicing much more expensive ($35.8\times$ for $8\times$ memory reduction).
- **Index choice**: Concretely proves that random slicing fails to relieve bottlenecks for $k \le 3$ ($0\%$ memory reduction), while incurring exponential penalties ($1.9\times$ to $7.1\times$). Slicing boundary indices incurs theoretical worst-case $2^k$ overhead with zero memory reduction.
All claims strictly follow from the data.

### 2.5 Sources (Score: 1/2)
The study lists 6 references in [`README.md`](../gemini-3.7-flash-high/README.md):
1. `quant-ph/0511069`: **Valid**. Markov & Shi (2008), *Simulating quantum computation by contracting tensor networks*.
2. `2002.01935`: **Valid**. Gray & Kourtis (2021), *Hyper-optimized tensor network contraction*.
3. `1608.00263`: **Valid**. Boixo et al. (2018), *Characterizing Quantum Supremacy in Near-Term Devices*.
4. `1904.01976`: **INVALID / HALLUCINATED**. Cited as: *Villalonga, B., et al. (2020). Flexible resource allocator for tensor network contraction. npj Quantum Information, 6, 43*. On arXiv, `1904.01976` is actually *A torsion-free background solution of the string theory* by Shingo Suzuki. The actual qFlex paper is arXiv:1811.09599 (*A flexible high-performance simulator for verifying and benchmarking quantum circuits implemented on real hardware*).
5. `math-ph/0609050`: **Valid**. Mezzadri (2007), *How to generate random matrices from the classical compact groups*.
6. `1910.11333`: **Valid**. Arute et al. (2019), *Supplementary information for "Quantum supremacy using a programmable superconducting processor"*.

5 of 6 citations are genuine and accurate; 1 is an invented/hallucinated arXiv ID.

### 2.6 Simplicity (Score: 2/2)
- **Line Counts**:
  - [`slicing_study.py`](../gemini-3.7-flash-high/slicing_study.py): 201 lines.
  - [`test_correctness.py`](../gemini-3.7-flash-high/test_correctness.py): 83 lines.
  - Total Python code: **284 lines** (complies with the $\sim 300$ line limit).
- Code is clean, well-commented, modular, and does not import unnecessary frameworks.

---

## 3. Scorecard

| Metric | Score (0–2) | Evidence |
| :--- | :---: | :--- |
| **Runs** | **2** | Pytest passes 9/9 in 0.83s; main script runs cleanly in 2.1s generating stdout tables and plot. |
| **Correct** | **2** | Haar U(4) QR with phase normalization; state-vector matches to $<10^{-14}$; FLOP accounting properly tracks all slices via `cotengra`. |
| **Honest** | **2** | All numbers match rerun output; slight differences at 1D $k=2$ and 2D $k=4,8$ are due to Python hash seed tie-breaking in `SliceFinder`. |
| **Answer** | **2** | Thoroughly answers overhead scaling vs memory reduction and contrasts greedy vs random vs boundary selection. |
| **Sources** | **1** | 5 of 6 citations are genuine; arXiv:1904.01976 is hallucinated (string theory paper cited for qFlex). |
| **Simple** | **2** | 284 total lines of Python; highly readable, robust, and concise. |
| **Total** | **11 / 12** | |
