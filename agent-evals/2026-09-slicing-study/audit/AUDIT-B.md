# Scientific Audit: Study B

> **AI-generated audit, not peer-reviewed.** Written by Gemini 3.8 Flash (High) as a blind auditor; its findings were checked by Claude Code, see [the run README](../README.md#audit-by-gemini-38-flash-how-good-is-an-ai-auditor).

## 1. Executive Summary

Study B exhibits fatal conceptual and implementation defects that invalidate its quantitative findings. Most critically, it double-counts slice multipliers for heuristic ("smart") slicing by multiplying `tree_sliced.contraction_cost()` (which in `cotengra` already includes the slice count) by `tree_sliced.nslices` a second time, resulting in an erroneous $2^{2k} = 4^k$ overhead scaling. Concurrently, it evaluates random slicing with an asymmetric method that multiplies by $2^k$ only once, causing the study's own table to falsely report that heuristic slicing is up to $28\times$ more computationally expensive than random slicing (826.8x vs. 29.2x at $k=8$). Furthermore, the optimizer is completely unseeded, causing every single number in the table to mismatch on rerun; the script overwrites `README.md` dynamically; the required 1D brickwork geometry is omitted from the study; and zero sources or arXiv citations are provided.

---

## 2. Detailed Audit Criteria

### 2.1 Runs as Delivered (Score: 1/2)
- **Pytest Execution**: Running `pytest -v` in [`work/study-B/`](../gemini-3.1-pro-high/) passed 3 tests in [`test_circuit.py`](../gemini-3.1-pro-high/test_circuit.py) in **0.24s**.
- **Main Script Execution**: Running `python study.py` executes in ~9 seconds, but:
  1. Emits multiple `SyntaxWarning: "\i" is an invalid escape sequence` on Python 3.12+ (due to raw LaTeX escapes like `\i` inside [`study.py`](../gemini-3.1-pro-high/study.py)).
  2. Directly overwrites [`README.md`](../gemini-3.1-pro-high/README.md) on line 126 (`with open('README.md', 'w') as f:`), modifying workspace deliverables during ordinary execution.

### 2.2 Reproducibility / Honesty (Score: 0/2)
**Not a single number in the delivered README reproduced upon rerunning [`study.py`](../gemini-3.1-pro-high/study.py).**
Because `opt = ctg.ReusableHyperOptimizer(methods=['greedy'], max_repeats=32, progbar=False)` on line 16 has no random seed specified, `cotengra`'s random sampling produces different contraction trees on every execution:

| Parameter / Metric | Delivered in README (lines 13–24) | Re-run Produced | Match? |
| :--- | :---: | :---: | :---: |
| **Original FLOPs** | $2.73 \times 10^6$ | $2.58 \times 10^6$ | **Mismatch** |
| **Smart Mem Red ($k=1$)** | 2.0 qubits | 1.0 qubit | **Mismatch** |
| **Smart FLOP OH ($k=1$)** | 0.8x | 1.61x | **Mismatch** |
| **Smart Mem Red ($k=2$)** | 3.0 qubits | 2.0 qubits | **Mismatch** |
| **Smart FLOP OH ($k=2$)** | 1.6x | 3.21x | **Mismatch** |
| **Smart FLOP OH ($k=4$)** | 12.0x | 16.46x | **Mismatch** |
| **Smart FLOP OH ($k=6$)** | 86.6x | 119.01x | **Mismatch** |
| **Smart FLOP OH ($k=8$)** | 826.8x | 964.24x | **Mismatch** |
| **Random Mem Red ($k=1$)** | 0.0 qubits | 1.0 qubit | **Mismatch** |
| **Random FLOP OH ($k=1$)** | 2.5x | 1.22x | **Mismatch** |
| **Random FLOP OH ($k=6$)** | 32.0x | 9.72x | **Mismatch** |
| **Random FLOP OH ($k=8$)** | 29.2x | 67.30x | **Mismatch** |

Every single entry in the table mismatched, and the script silently rewrote the delivered file.

### 2.3 Correctness (Score: 0/2)
1. **Fatal Slicing Accounting Bug (Double Multiplier)**:
   In [`study.py`](../gemini-3.1-pro-high/study.py):
   ```python
   tree_sliced = tree.slice(target_slices=n_slices)
   ...
   flops = tree_sliced.contraction_cost() * tree_sliced.nslices
   ```
   In `cotengra`, `tree.contraction_cost()` (which delegates to `tree.total_flops()`) sums operations over all nodes and multiplies by `self.multiplicity`. When a tree is sliced, `self.multiplicity` is set to `tree.nslices` ($2^k$). Therefore, `tree_sliced.contraction_cost()` **already is the total FLOP count across all slices**.
   By multiplying by `tree_sliced.nslices` again, Study B multiplied the total FLOPs by $2^k$ twice, scaling as $2^{2k} = 4^k$!
2. **Fatal Asymmetry in Methodology**:
   For Random Slicing ([`study.py`](../gemini-3.1-pro-high/study.py)), Study B did not call `tree.slice()`. Instead, it created `new_size_dict[ind] = 1`, called `opt.search(..., new_size_dict)`, and computed `flops = tree_rand.contraction_cost() * (2**k)`.
   Because `tree_rand` was not sliced via `cotengra`, its `nslices` remained 1, meaning Random Slicing was multiplied by $2^k$ once.
   Consequently, Smart Slicing was artificially penalized by $(2^k)^2$, whereas Random Slicing was penalized by $2^k$. This made Smart Slicing look catastrophically worse than Random Slicing (e.g. 826.8x vs. 29.2x at $k=8$).
3. **Geometry Omission**:
   The prompt specifically mandated:
   > *(a) 1D brickwork of nearest-neighbour two-qubit gates, (b) 2D grid of qubits with nearest-neighbour two-qubit gates (Sycamore-like layers).*
   Study B implemented a 1D function in [`circuit.py`](../gemini-3.1-pro-high/circuit.py), but completely omitted 1D circuits from [`study.py`](../gemini-3.1-pro-high/study.py) and [`README.md`](../gemini-3.1-pro-high/README.md).

### 2.4 Answers Follow from Data (Score: 0/2)
- In [`README.md`](../gemini-3.1-pro-high/README.md), the author writes:
  > *"For an optimal choice of indices, slicing exponentially increases the FLOP overhead. Specifically, reducing the max tensor size by $k$ qubits (halving memory $k$ times) typically increases total FLOPs by slightly more than $2^k$."*
  This conclusion is false and is purely the result of the double-counting bug. In reality, optimal slicing on 2D grids scales well below $2^k$.
- In [`README.md`](../gemini-3.1-pro-high/README.md), the author claims:
  > *"Proper heuristic selection (e.g., using cotengra's graph partitioners) is crucial to actually achieve memory savings for the incurred FLOP penalty."*
  This conclusion contradicts the numbers in the study's own table, where Random Slicing is reported to have drastically lower FLOP overhead than Smart Slicing (29.2x vs. 826.8x at $k=8$).

### 2.5 Sources (Score: 0/2)
[`README.md`](../gemini-3.1-pro-high/README.md) contains **zero citations**, no references section, and zero arXiv IDs, directly violating Requirement 4 of the research prompt.

### 2.6 Simplicity (Score: 1/2)
- Total Python lines: 286 lines ([`circuit.py`](../gemini-3.1-pro-high/circuit.py): 75, [`study.py`](../gemini-3.1-pro-high/study.py): 131, [`test_circuit.py`](../gemini-3.1-pro-high/test_circuit.py): 80).
- While concise, it contains unhandled `SyntaxWarning` escape sequences, unseeded random state, destructive file-overwrite side effects, and defective logic.

---

## 3. Scorecard

| Metric | Score (0–2) | Evidence |
| :--- | :---: | :--- |
| **Runs** | **1** | Pytest passes 3/3; main script emits `SyntaxWarning` and destructively overwrites `README.md`. |
| **Correct** | **0** | Double-counts slice multipliers for smart slicing ($4^k$); asymmetric treatment makes smart slicing appear 28x worse than random; omits 1D geometry. |
| **Honest** | **0** | Unseeded hyper-optimizer causes every single number in the table to mismatch on rerun. |
| **Answer** | **0** | Concludes that optimal slicing scales as $>2^k$ (due to double-counting bug); claims contradict the study's own table. |
| **Sources** | **0** | Zero references or arXiv IDs provided anywhere in the study. |
| **Simple** | **1** | 286 lines, but suffers from severe logical flaws, warnings, and unseeded execution. |
| **Total** | **2 / 12** | |
