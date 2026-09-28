An independent, rigorous audit of the four studies in [`work/`](work/) has been completed. The comprehensive audit reports and executive summary have been generated in the current directory:

- [**`AUDIT-A.md`**](AUDIT-A.md): Detailed audit report for Study A
- [**`AUDIT-B.md`**](AUDIT-B.md): Detailed audit report for Study B
- [**`AUDIT-C.md`**](AUDIT-C.md): Detailed audit report for Study C
- [**`AUDIT-D.md`**](AUDIT-D.md): Detailed audit report for Study D
- [**`SUMMARY.md`**](SUMMARY.md): Comparative scorecard, critical failure analysis, and best-study verdict

---

### Score Summary

Each study was evaluated on a strict 0–2 scale across all 6 criteria:

| Study | Runs | Correct | Honest (reproduces) | Answer | Sources | Simple | Total Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [**Study A**](AUDIT-A.md) | **2** | **2** | **2** | **2** | **1** | **2** | **11 / 12** | **1st (Best Answer)** |
| [**Study C**](AUDIT-C.md) | **2** | **2** | **1** | **2** | **2** | **2** | **11 / 12** | **2nd (Best Citations)** |
| [**Study D**](AUDIT-D.md) | **2** | **0** | **2** | **1** | **0** | **2** | **7 / 12** | **3rd** |
| [**Study B**](AUDIT-B.md) | **1** | **0** | **0** | **0** | **0** | **1** | **2 / 12** | **4th** |

---

### Key Findings & Most Serious Problem Found in Each Study

1. **Study A (Score: 11/12)**:
   - *Strengths*: Flawless mathematical formulation; state-vector simulation matches unsliced and sliced tensor networks to $< 10^{-14}$; correctly counts FLOPs across all slices via `cotengra`; evaluates 1D brickwork and 2D Sycamore lattices; empirically quantifies the sub-exponential scaling exponent ($\alpha \approx 0.35$ for $2^{\alpha k}$ overhead).
   - *Most Serious Problem*: In [`README.md`](../gemini-3.7-flash-high/README.md), Reference 4 cites `arXiv:1904.01976` for Villalonga et al. (`qFlex`). On arXiv, `1904.01976` is actually a string theory paper by Shingo Suzuki. The remaining 5 citations are genuine and accurate.

2. **Study B (Score: 2/12)**:
   - *Strengths*: Pytest passes in 0.24s; compact codebase.
   - *Most Serious Problem*: Double-counts slice multipliers for smart slicing in [`study.py`](../gemini-3.1-pro-high/study.py) (`flops = tree_sliced.contraction_cost() * tree_sliced.nslices`, where `contraction_cost` already includes `nslices`), artificially inflating FLOP overhead as $2^{2k} = 4^k$. For random slicing, it multiplied by $2^k$ only once. This caused smart slicing to be reported as up to **$28\times$ more expensive than random slicing** ($826.8\times$ vs. $29.2\times$ at $k=8$). Furthermore, it omitted 1D geometry, had unseeded executions where zero numbers reproduced, emitted syntax warnings, and cited zero sources.

3. **Study C (Score: 11/12)**:
   - *Strengths*: Flawless citation record (all 5 citations exist, match titles/authors, and are verified on arXiv); exact state-vector check agrees to machine precision ($< 10^{-16}$); FLOP accounting across all slices is 100% correct; exceptionally clean code (274 lines).
   - *Most Serious Problem*: In [`experiment.py`](../gemini-3.8-flash-high/experiment.py), `opt = ctg.AutoOptimizer(progbar=False)` was unseeded. Because `AutoOptimizer` uses random sampling, re-running the script from scratch discovers a different baseline contraction tree (2D baseline FLOPs differed: $3.38 \times 10^6$ vs. $9.15 \times 10^5$), causing table numbers to shift on rerun.

4. **Study D (Score: 7/12)**:
   - *Strengths*: Strictly seeded; every number in `README.md` and `results.json` reproduced byte-for-byte.
   - *Most Serious Problem*: In [`slicing_study.py`](../gemini-3.6-flash-high/slicing_study.py), the random slicing loop called `t_copy.remove_ind(ix)` without assigning the return value. In `cotengra`, `remove_ind` defaults to `inplace=False`. As a result, **no random indices were ever sliced**; the code simply evaluated the unsliced tree and multiplied its FLOPs by $2^k$. The entire Random Slicing column was an illusion created by a silent code bug. Additionally, 2 of its 3 citations were severely hallucinated (one cited a computer vision paper on scene text recognition, and another fabricated co-authors, title, and journal).

---

### Which Study Best Answers the Research Question?

**Study A** is judged the **best study overall**. While Study C has a perfect citation record, Study A provides the most robust empirical reproducibility (deterministic contraction trees matching rerun output), the deepest quantitative answer (halving 2D peak memory costs only $6\%$ extra FLOPs; overhead scales as $\sim 2^{0.35 k}$), a rigorous 3-way index selection comparison (greedy vs. random vs. boundary), and a 9-test automated validation suite.
