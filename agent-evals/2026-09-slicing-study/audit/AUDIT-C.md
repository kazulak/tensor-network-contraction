# Scientific Audit: Study C

> **AI-generated audit, not peer-reviewed.** Written by Gemini 3.8 Flash (High) as a blind auditor; its findings were checked by Claude Code, see [the run README](../README.md#audit-by-gemini-38-flash-how-good-is-an-ai-auditor).

## 1. Executive Summary

Study C is an exceptionally well-crafted, rigorous study. Its mathematical formulation, state-vector simulation, tensor network construction, and cost accounting are completely correct and elegant. It correctly implements both 1D brickwork and 2D Sycamore (ABCD) circuits, demonstrates machine-precision agreement ($< 10^{-16}$) across state-vector, unsliced, and sliced tensor networks, accurately accounts for FLOPs across all slices, and provides a flawless set of 5 genuine arXiv citations. Its sole limitation is that `cotengra.AutoOptimizer` was instantiated without a fixed seed, causing stochastic path searches to produce different contraction trees upon re-running from scratch.

---

## 2. Detailed Audit Criteria

### 2.1 Runs as Delivered (Score: 2/2)
- **Pytest Execution**: Running `pytest -v` in [`work/study-C/`](../gemini-3.8-flash-high/) ran 3 test cases in [`test_slicing.py`](../gemini-3.8-flash-high/test_slicing.py) and completed with **3 passed in 0.36s** (0 failures, 1 standard cotengra user warning regarding optional hyperopt backends).
- **Main Script Execution**: Running `python experiment.py` executes in ~3.2 seconds without errors, generating an updated [`results.json`](../gemini-3.8-flash-high/results.json) and publication figure [`slicing_study.png`](../gemini-3.8-flash-high/slicing_study.png).

### 2.2 Reproducibility / Honesty (Score: 1/2)
- **Delivered Artifact Alignment**: Every single number in [`README.md`](../gemini-3.8-flash-high/README.md) lines 43–86 matched the pre-delivered [`results.json`](../gemini-3.8-flash-high/results.json) byte-for-byte.
- **Rerun from Scratch**: In [`experiment.py`](../gemini-3.8-flash-high/experiment.py), `opt = ctg.AutoOptimizer(progbar=False)` was left unseeded. Because `AutoOptimizer` relies on random sampling across path-finding heuristics when libraries like `optuna` are absent, re-running `experiment.py` discovers a different contraction tree for the circuits:
  - 1D Brickwork baseline FLOPs: Delivered $66,552$ (line 43); Rerun produced $62,240$.
  - 2D Sycamore baseline: Delivered Peak Size $65,536$ ($2^{16}$), FLOPs $3,378,800$ (line 64); Rerun produced Peak Size $16,384$ ($2^{14}$), FLOPs $914,916$.
  - Because the new baseline peak size was $16,384$, reaching target size $8,192$ required only $k=1$ slice ($S=2$) instead of $k=3$ slices ($S=8$).
- While the numbers were genuinely produced by the code and the underlying methodology is 100% sound, failure to seed the hyper-optimizer prevents byte-for-byte reproducibility of the baseline contraction tree from scratch.

### 2.3 Correctness (Score: 2/2)
1. **Haar Unitary Generation**: In [`circuits.py`](../gemini-3.8-flash-high/circuits.py), [`haar_unitary`](../gemini-3.8-flash-high/circuits.py) implements Mezzadri's QR algorithm with diagonal phase normalization in `np.complex128`. Unitarity error $\|U^\dagger U - I\|_\infty < 10^{-14}$ is confirmed in [`test_slicing.py`](../gemini-3.8-flash-high/test_slicing.py).
2. **State-Vector Simulation**: In [`circuits.py`](../gemini-3.8-flash-high/circuits.py), [`simulate_statevector`](../gemini-3.8-flash-high/circuits.py) uses `np.einsum` to track the state vector and maps the target bitstring to its exact lexicographical index. In [`test_slicing.py`](../gemini-3.8-flash-high/test_slicing.py), 3-way validation matches state-vector, unsliced, and sliced tensor networks to $< 10^{-16}$.
3. **Slicing and Cost Accounting**:
   - Uses `tsl = tree.slice(target_size=target, seed=42)`.
   - FLOPs are tracked using `tsl.total_flops()`. In `cotengra`, `total_flops()` sums all operations and multiplies by the tree's multiplicity ($2^k$). Study C does NOT commit the double-counting mistake of Study B.
   - For Random and Suboptimal slicing, `remove_ind_` properly updates tree multiplicity, so `trand.total_flops()` and `tbad.total_flops()` correctly measure total FLOPs across all $2^k$ slices.
4. **Architectures**: Accurately implements 1D brickwork and 2D Sycamore ABCD layers ([`circuits.py`](../gemini-3.8-flash-high/circuits.py)).

### 2.4 Answers Follow from Data (Score: 2/2)
In [`README.md`](../gemini-3.8-flash-high/README.md), the answers directly reflect the data:
- **Cost of Memory Reduction**: Accurately reports that 2D Sycamore grids permit an $8\times$ memory reduction with only $1.05\times$ FLOP overhead, $64\times$ memory reduction with $1.27\times$ overhead, and $256\times$ reduction with $2.72\times$ overhead across 512 slices. Provides the insightful mathematical mechanism: slicing bottleneck bonds cuts the rank of peak intermediate contractions, dropping per-slice FLOPs from $3.38 \times 10^6$ down to $17,948$ ($188\times$ faster per slice).
- **Index Choice**: Details why index choice is decisive: in 2D at $k=9$, guided slicing achieves $256\times$ memory reduction at $2.72\times$ overhead, whereas random slicing achieves only $1.83\times$ reduction while inflating FLOPs by $207.81\times$. Suboptimal peripheral slicing yields $0\times$ memory reduction while incurring the full $2^k$ penalty.

### 2.5 Sources (Score: 2/2)
Study C cites 5 references in [`README.md`](../gemini-3.8-flash-high/README.md). **All 5 citations are genuine, verified, and correctly matched to their arXiv IDs:**
1. `quant-ph/0511069`: **Valid**. Markov & Shi (2008), *Simulating quantum circuits by contracting tensor networks*.
2. `1712.05384`: **Valid**. Boixo et al. (2017), *Simulation of low-depth quantum circuits as complex undirected graphical models*.
3. `1805.01450`: **Valid**. Chen, Zhang, Huang, Newman, & Shi (2018), *Classical Simulation of Intermediate-Size Quantum Circuits*.
4. `2002.01935`: **Valid**. Gray & Kourtis (2021), *Hyper-optimized tensor network contraction*.
5. `2111.03011`: **Valid**. Pan, Chen, & Zhang (2022), *Solving the Sampling Problem of the Sycamore Quantum Circuits*, Phys. Rev. Lett. 129, 090502.

### 2.6 Simplicity (Score: 2/2)
- **Line Counts**:
  - [`circuits.py`](../gemini-3.8-flash-high/circuits.py): 81 lines.
  - [`experiment.py`](../gemini-3.8-flash-high/experiment.py): 131 lines.
  - [`test_slicing.py`](../gemini-3.8-flash-high/test_slicing.py): 62 lines.
  - Total Python code: **274 lines** (well within the $\sim 300$ line limit).
- Exemplary clarity, modularity, and readability.

---

## 3. Scorecard

| Metric | Score (0–2) | Evidence |
| :--- | :---: | :--- |
| **Runs** | **2** | Pytest passes 3/3 in 0.36s; main experiment script runs in 3.2s without error. |
| **Correct** | **2** | Haar U(4) QR with phase correction; state-vector matches to $<10^{-16}$; FLOP accounting across all slices is 100% correct. |
| **Honest** | **1** | Delivered `results.json` matches README, but unseeded `AutoOptimizer` prevents exact rerun reproduction from scratch. |
| **Answer** | **2** | Lucid, thorough answer explaining the structural mechanism of sub-exponential slicing overhead and comparing 3 strategies. |
| **Sources** | **2** | 5 out of 5 citations are genuine, correctly formatted, and verified on arXiv. |
| **Simple** | **2** | 274 total lines of Python; highly modular, clean, and elegant. |
| **Total** | **11 / 12** | |
