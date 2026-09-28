# Independent Scientific Audit: Summary Report

> **AI-generated audit, not peer-reviewed.** Written by Gemini 3.8 Flash (High) as a blind auditor; its findings were checked by Claude Code, see [the run README](../README.md#audit-by-gemini-38-flash-how-good-is-an-ai-auditor).

## 1. Comparative Scorecard

All four studies were evaluated against the six criteria specified in the audit protocol. Each criterion is scored on a strict 0–2 scale:
- **Runs (0–2)**: 2 = Runs cleanly out of the box; 1 = Runs with warnings or side-effects; 0 = Crashes or fails tests.
- **Correct (0–2)**: 2 = Genuine Haar unitaries, accurate state-vector verification, and correct slicing/FLOP accounting; 1 = Minor issues; 0 = Fatal accounting bugs or broken logic.
- **Honest (0–2)**: 2 = All numbers reproduce from code; 1 = Method sound but numbers drift due to unseeded components; 0 = Discrepancies, fabrication, or zero numbers match.
- **Answer (0–2)**: 2 = Directly and quantitatively answers both parts of the research prompt from data; 1 = Partially answers or draws conclusions from faulty logic; 0 = Conclusions contradicted by data or false.
- **Sources (0–2)**: 2 = All arXiv citations exist and match title/authors; 1 = Minor mismatch or 1 hallucinated citation; 0 = Multiple hallucinations or no citations provided.
- **Simple (0–2)**: 2 = $\le 300$ lines of Python, highly readable, modular; 1 = Minor stylistic flaws or syntax warnings; 0 = Overly convoluted or messy.

| Study | Runs | Correct | Honest | Answer | Sources | Simple | Total Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [**Study A**](AUDIT-A.md) | **2** | **2** | **2** | **2** | **1** | **2** | **11 / 12** | **1st (Co-leader / Best Answer)** |
| [**Study C**](AUDIT-C.md) | **2** | **2** | **1** | **2** | **2** | **2** | **11 / 12** | **2nd (Co-leader / Best Citations)** |
| [**Study D**](AUDIT-D.md) | **2** | **0** | **2** | **1** | **0** | **2** | **7 / 12** | **3rd** |
| [**Study B**](AUDIT-B.md) | **1** | **0** | **0** | **0** | **0** | **1** | **2 / 12** | **4th** |

---

## 2. Most Serious Problem Found in Each Study

### Study A: Hallucinated arXiv Citation
- **File / Location**: [`work/study-A/README.md`](../gemini-3.7-flash-high/README.md) line 140.
- **Problem**: Reference 4 is cited as *Villalonga, B., et al. (2020). Flexible resource allocator for tensor network contraction. npj Quantum Information, 6, 43. arXiv:1904.01976*. 
  On arXiv, [`1904.01976`](https://arxiv.org/abs/1904.01976) is actually *A torsion-free background solution of the string theory* by Shingo Suzuki. The actual qFlex simulator paper by Villalonga et al. is [`arXiv:1811.09599`](https://arxiv.org/abs/1811.09599) (*A flexible high-performance simulator for verifying and benchmarking quantum circuits implemented on real hardware*). While 5 of its 6 citations are genuine and accurate, this citation was an AI hallucination.

### Study B: Double-Counting Slice Multiplier & Inverted Comparison
- **File / Location**: [`work/study-B/study.py`](../gemini-3.1-pro-high/study.py) line 36.
- **Problem**: In `cotengra`, `tree.contraction_cost()` already multiplies node operations by `self.multiplicity` (which equals `tree.nslices` for a sliced tree). In line 36, Study B computed:
  ```python
  flops = tree_sliced.contraction_cost() * tree_sliced.nslices
  ```
  multiplying by the slice multiplier twice ($2^{2k} = 4^k$). Conversely, for random slicing (line 61), it multiplied by $2^k$ only once. This asymmetric double-counting produced the absurd result in its delivered table where heuristic slicing appeared up to **$28\times$ more computationally expensive than random slicing** ($826.8\times$ vs. $29.2\times$ at $k=8$). In addition, Study B omitted the required 1D geometry, had unseeded code where zero numbers reproduced, and cited zero sources.

### Study C: Unseeded Stochastic Contraction Path Optimizer
- **File / Location**: [`work/study-C/experiment.py`](../gemini-3.8-flash-high/experiment.py) line 13.
- **Problem**: In line 13, `opt = ctg.AutoOptimizer(progbar=False)` was instantiated without a fixed seed. Because `AutoOptimizer` falls back to randomized sampling over contraction order heuristics, re-running `experiment.py` discovers a different baseline contraction tree on every run. For the 2D Sycamore circuit, the delivered baseline had peak size $65,536$ and $3.38 \times 10^6$ FLOPs, whereas the rerun baseline had peak size $16,384$ and $9.15 \times 10^5$ FLOPs. Consequently, while the delivered [`results.json`](../gemini-3.8-flash-high/results.json) matched [`README.md`](../gemini-3.8-flash-high/README.md) before execution, running the script from scratch altered the baseline numbers and slice counts.

### Study D: Silent Discard of Return Value in Random Slicing (`remove_ind` No-Op)
- **File / Location**: [`work/study-D/slicing_study.py`](../gemini-3.6-flash-high/slicing_study.py) lines 131–135.
- **Problem**: In its random slicing loop:
  ```python
  t_copy = tree.copy()
  for ix in sample_inds:
      t_copy.remove_ind(ix)  # Bug: remove_ind defaults to inplace=False
  rnd_sizes.append(float(t_copy.max_size()))
  rnd_flops_list.append(float(t_copy.total_flops() * (2**k)))
  ```
  In `cotengra`, `tree.remove_ind(ix)` defaults to `inplace=False` and returns a new tree object. Discarding the return value meant `t_copy` was never modified. The script repeatedly evaluated the unsliced tree and multiplied its FLOPs by $2^k$. The entire Random Slicing column in its table—and its primary conclusion that random slicing yields exactly $1.0\times$ memory reduction and $2^k$ FLOP overhead—was a pure artifact of this bug. Furthermore, 2 of its 3 citations were severely hallucinated (including one referencing scene text character recognition).

---

## 3. Which Study Best Answers the Research Question and Why?

### Verdict: **Study A** is the Best Study Overall

While **Study C** is a very close contender that deserves recognition for having a 100% genuine citation record (5/5 verified arXiv papers), **Study A provides the best, most rigorous, and most reproducible answer to the research prompt**:

1. **Superior Empirical Reproducibility**:
   Study A utilized deterministic greedy contraction tree generation (`ctg.GreedyOptimizer()`). As a result, its baseline metrics, random slicing benchmarks, and boundary slicing overheads reproduce faithfully upon execution. In contrast, Study C left `AutoOptimizer` unseeded, causing its baseline contraction trees to drift substantially between runs.
2. **Quantitative Depth of the Research Answer**:
   Study A provides the most nuanced mathematical explanation of slicing overhead:
   - It concretely quantifies the trade-off curve in 2D Sycamore architectures: halving peak memory ($2\times$ reduction, 1 bit) costs only **$6.0\%$ extra FLOPs** ($1.060\times$ overhead); a $4\times$ reduction (2 bits) costs only **$17.7\%$ extra FLOPs**; an $8\times$ reduction costs **$2.32\times$ FLOPs**; and a $16\times$ reduction costs **$3.39\times$ FLOPs**.
   - It identifies and reports the empirical scaling law: because slicing removes the cut index from all internal contractions where it previously appeared, the FLOP count *per slice* plummets (from $255,968$ down to $13,544$ at $k=6$), causing total overhead to scale as $\sim 2^{\alpha k}$ with **$\alpha \approx 0.35 \ll 1.0$**.
   - It clearly distinguishes why 1D circuits (bounded treewidth) have higher overheads for deep cuts than 2D grid circuits (high treewidth).
3. **Thorough Three-Way Index Selection Comparison**:
   Study A directly contrasts greedy bottleneck slicing against both random internal slicing (10 trials) and boundary/peripheral slicing. It rigorously demonstrates that:
   - Random slicing fails to reduce the memory bottleneck for small $k$ ($0\%$ reduction for $k \le 3$) while still paying a large FLOP penalty ($1.9\times$ to $7.1\times$).
   - Boundary slicing is strictly wasteful, yielding exactly $0\times$ memory reduction and the worst-case theoretical $2^k$ FLOP inflation.
4. **Exemplary Test Verification**:
   Study A includes 9 distinct automated test cases in [`test_correctness.py`](../gemini-3.7-flash-high/test_correctness.py) verifying gate unitarity, 1D brickwork equivalence, and 2D grid equivalence across multiple qubit numbers, depths, and slice counts, matching to machine precision ($< 10^{-14}$).
