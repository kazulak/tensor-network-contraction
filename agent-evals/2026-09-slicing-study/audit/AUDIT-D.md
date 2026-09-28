# Scientific Audit: Study D

> **AI-generated audit, not peer-reviewed.** Written by Gemini 3.8 Flash (High) as a blind auditor; its findings were checked by Claude Code, see [the run README](../README.md#audit-by-gemini-38-flash-how-good-is-an-ai-auditor).

## 1. Executive Summary

Study D presents an ostensibly clean and deterministic study whose numbers reproduce byte-for-byte. However, rigorous code inspection reveals a critical implementation bug in its baseline comparison: in its random slicing loop, it invokes `t_copy.remove_ind(ix)` without assigning the return value. In `cotengra`, `remove_ind` defaults to `inplace=False`, returning a new tree while leaving the target unchanged. As a result, no indices were ever sliced in the random trials; the code repeatedly evaluated the unsliced tree and multiplied its FLOPs by $2^k$. The entire Random Slicing column (and the author's primary comparative conclusion) is therefore completely fabricated by a silent code bug. Additionally, 2 of its 3 citations contain severe hallucinations (one references an unrelated scene text recognition paper, and another fabricates co-authors, article title, and journal name).

---

## 2. Detailed Audit Criteria

### 2.1 Runs as Delivered (Score: 2/2)
- **Pytest Execution**: Running `pytest -v` in [`work/study-D/`](../gemini-3.6-flash-high/) ran 3 test cases in [`test_slicing.py`](../gemini-3.6-flash-high/test_slicing.py) and completed with **3 passed in 2.72s**.
- **Main Script Execution**: Running `python slicing_study.py` executed in ~3.3 seconds without warnings or errors, writing [`results.json`](../gemini-3.6-flash-high/results.json) and [`flops_overhead_vs_memory.png`](../gemini-3.6-flash-high/flops_overhead_vs_memory.png).

### 2.2 Reproducibility / Honesty (Score: 2/2)
- **Perfect Determinism**: The author explicitly set `np.random.seed(42)` and `random.seed(42)` at lines 165–166 in [`slicing_study.py`](../gemini-3.6-flash-high/slicing_study.py).
- Every single number in [`README.md`](../gemini-3.6-flash-high/README.md) lines 40–67 reproduced byte-for-byte upon re-running [`slicing_study.py`](../gemini-3.6-flash-high/slicing_study.py), exactly matching [`results.json`](../gemini-3.6-flash-high/results.json):
  - 2D Unsliced: $M_{\text{max}} = 268,435,456$, FLOPs $F = 1.3057 \times 10^{10}$. **Exact match**.
  - 2D Optimal $k=2$: Size $67,108,864$, $4.0\times$ red, FLOPs $1.3058 \times 10^{10}$ ($1.0001\times$ OH). **Exact match**.
  - 2D Optimal $k=12$: Size $1,048,576$, $256.0\times$ red, FLOPs $1.8271 \times 10^{11}$ ($13.99\times$ OH). **Exact match**.
  - 1D Unsliced: $M_{\text{max}} = 32,768$, FLOPs $F = 1.7573 \times 10^7$. **Exact match**.
  - 1D Optimal $k=8$: Size $16,384$, $2.0\times$ red, FLOPs $6.3857 \times 10^8$ ($36.34\times$ OH). **Exact match**.

### 2.3 Correctness (Score: 0/2)
1. **Fatal Implementation Bug in Random Slicing**:
   In [`slicing_study.py`](../gemini-3.6-flash-high/slicing_study.py):
   ```python
   for seed_idx in range(5):
       random.seed(1000 + seed_idx)
       sample_inds = random.sample(internal_inds, k)
       t_copy = tree.copy()
       for ix in sample_inds:
           t_copy.remove_ind(ix)  # <--- CRITICAL BUG: RETURN VALUE DISCARDED!
       rnd_sizes.append(float(t_copy.max_size()))
       rnd_flops_list.append(float(t_copy.total_flops() * (2**k)))
   ```
   In `cotengra.ContractionTree`, `remove_ind(ix, inplace=False)` returns a copy with index `ix` removed, leaving the caller untouched unless `inplace=True` or `t_copy.remove_ind_(ix)` is used.
   Because `t_copy.remove_ind(ix)` was discarded, **`t_copy` was never modified**.
   Consequently:
   - `t_copy.max_size()` returned the original, unsliced maximum size on every single random trial.
   - `t_copy.total_flops()` returned the original, unsliced FLOP count.
   - `t_copy.total_flops() * (2**k)` literally evaluated $\text{unsliced\_flops} \times 2^k$.
   - This explains why every single entry in the Random Slicing table had `rnd_mem_red = 1.0x` and `rnd_overhead = 2**k` ($4.00x, 16.00x, 64.00x, \dots, 16,384.00x$).
   The study never sliced a single random index. The entire Random Slicing column is completely invalid.
2. **Suboptimal Unsliced Baseline Path**:
   In [`slicing_study.py`](../gemini-3.6-flash-high/slicing_study.py), the unsliced contraction path is obtained via `oe.contract_path`, which defaults to a naive greedy search. For a small 25-qubit depth-6 circuit, this resulted in an unsliced peak tensor size of $268,435,456$ elements (4.29 GB for a single tensor) and $1.3 \times 10^{10}$ FLOPs, when standard `cotengra` optimizers easily find paths with peak sizes under $2^{15}$ (32,768 elements) and $< 3 \times 10^6$ FLOPs.
3. **Haar Unitarity & State-Vector Validation**:
   [`generate_haar_unitary`](../gemini-3.6-flash-high/slicing_study.py) generates valid complex128 unitaries via Mezzadri QR. [`state_vector_simulate`](../gemini-3.6-flash-high/slicing_study.py) is mathematically sound and passes the 3-way test suite in [`test_slicing.py`](../gemini-3.6-flash-high/test_slicing.py) ($< 10^{-10}$).

### 2.4 Answers Follow from Data (Score: 1/2)
- The answer regarding **Optimal Slicing** correctly notes that slicing along contraction bottlenecks achieves sub-exponential overhead in 2D networks (e.g. $256\times$ memory reduction for $13.99\times$ FLOP overhead).
- However, the answer regarding **Random Slicing** in [`README.md`](../gemini-3.6-flash-high/README.md):
  > *"Random Slicing is completely ineffective: Selecting random internal indices yields 1.0x memory reduction (peak intermediate tensor size remains entirely unchanged because unsliced graph bottlenecks persist elsewhere in the network) while incurring the full exponential 2^k FLOP overhead penalty"*
  is completely spurious. The empirical data supporting this claim was not generated by random slicing at all, but by the silent `t_copy.remove_ind` no-op bug.

### 2.5 Sources (Score: 0/2)
Study D provides 3 citations in [`README.md`](../gemini-3.6-flash-high/README.md). **2 out of 3 citations contain severe hallucinations:**
1. `quant-ph/0511069`: Cited as *Markov and Shi, Simulating Quantum Circuits Using Tree Decompositions, SIAM J. Comput.* Valid arXiv ID and authors (actual title is *Simulating Quantum Computation by Contracting Tensor Networks*).
2. `1806.07374`: **HALLUCINATED**. Cited as: *B. Villalonga et al., A Flexible High-Performance Simulator for Verifiable Quantum Circuits, NPJ Quantum Inf. 5, 86 (2019)*. On arXiv, `1806.07374` is *Multilingual Scene Character Recognition System using Sparse Auto-Encoder for Efficient Local Features Representation in Bag of Features* by a computer vision research team.
3. `2002.01935`: **FABRICATED METADATA**. Cited as: *J. Gray and S. Kiffner, Hyper-optimized tensor network contraction sequences via low-rank representations and graph partitioning, Phys. Rev. E 104, 015309 (2021)*. The actual paper at `2002.01935` is by Johnnie Gray and **Stefanos Kourtis** (not S. Kiffner), titled ***Hyper-optimized tensor network contraction***, published in ***Quantum*** (not Phys. Rev. E).

### 2.6 Simplicity (Score: 2/2)
- **Line Counts**:
  - [`slicing_study.py`](../gemini-3.6-flash-high/slicing_study.py): 201 lines.
  - [`test_slicing.py`](../gemini-3.6-flash-high/test_slicing.py): 103 lines.
  - Total Python code: **304 lines** (meets the $\sim 300$ line guideline).
- Code is well-structured and readable, despite the critical method call bug.

---

## 3. Scorecard

| Metric | Score (0–2) | Evidence |
| :--- | :---: | :--- |
| **Runs** | **2** | Pytest passes 3/3 in 2.72s; main script runs in 3.3s generating results.json and figure. |
| **Correct** | **0** | Fatal bug in Random Slicing: `t_copy.remove_ind` discarded return value, causing random slicing to be a pure no-op; baseline tree had 268M elements due to unoptimized path. |
| **Honest** | **2** | Strictly seeded; every number in `results.json` and `README.md` reproduced byte-for-byte. |
| **Answer** | **1** | Optimal slicing claims follow from cotengra, but random slicing claims are based entirely on data fabricated by a silent code bug. |
| **Sources** | **0** | 2 of 3 citations severely hallucinated (arXiv:1806.07374 is scene character recognition; arXiv:2002.01935 has fabricated co-author, title, and journal). |
| **Simple** | **2** | 304 lines of Python; clean and concise. |
| **Total** | **7 / 12** | |
