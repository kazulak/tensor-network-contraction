I have conducted a research study on **Tensor Network Slicing** to evaluate computational FLOP overhead versus memory reduction across different quantum circuit geometries and index selection choices.

---

### Key Findings & Summary of Deliverables

#### 1. Direct Answer to Research Question
* **FLOP Cost vs. Memory Saving**: Optimal slicing achieves massive memory savings with **sub-exponential FLOP overhead** relative to the slice count $N_{\text{slices}} = 2^k$. For example, in a 25-qubit 2D grid circuit, slicing $k=12$ indices reduces peak intermediate tensor memory by **$256\times$** (from $268.4\times 10^6$ elements down to $1.05\times 10^6$ elements) at a FLOP overhead factor of only **$13.99\times$** ($0.34\%$ of the naive $4,096\times$ multiplier). Individual sliced sub-contractions require fewer operations because fixing sliced legs to size 1 shrinks tensor ranks along active sub-contraction branches.
* **Choice of Sliced Indices**: **CRITICAL.** Random index slicing produces **$1.0\times$ memory reduction** (zero peak memory savings because unsliced graph bottlenecks persist elsewhere in the network) while incurring the full exponential $2^k$ FLOP penalty ($16\times, 256\times, 4,096\times, 16,384\times$). Targeted slicing along contraction tree bottlenecks (min-cut / treewidth cuts) is strictly mandatory to achieve memory savings efficiently.

---

#### 2. Created Deliverables (in CWD)

1. [README.md](README.md):
   * Comprehensive research report including the research question, method, empirical results tables, embedded plot, direct answer, limitations, and arXiv citations.
2. [slicing_study.py](slicing_study.py):
   * Python script for circuit construction (1D brickwork and 2D grid), Haar-random $U(4)$ gate generation, contraction path analysis, optimal vs. random index slicing evaluation, data export ([results.json](results.json)), and plot generation.
3. [test_slicing.py](test_slicing.py):
   * `pytest` test suite verifying:
     - 2-qubit gate unitarity ($U^\dagger U = I$) to double precision ($< 10^{-14}$).
     - Numerical agreement between exact State-Vector simulation, Unsliced Tensor Network contraction, and Sliced Tensor Network sum ($\le 10^{-10}$ diff).
4. [flops_overhead_vs_memory.png](flops_overhead_vs_memory.png):
   * PNG figure comparing FLOP overhead scaling across 1D and 2D circuit geometries for Optimal vs. Random slicing.

---

#### 3. Verification Results

All tests executed cleanly via `pytest`:
```bash
pytest test_slicing.py
============================== 3 passed in 5.26s ==============================
```
