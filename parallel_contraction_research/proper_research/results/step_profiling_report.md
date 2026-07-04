# Scientific Report: Investigation of the "99% Root Contraction" Bottleneck Hypothesis

**Date:** July 2026  
**Author:** Antigravity Research Agent  

---

### 1. Executive Summary
A common hypothesis in tensor network (TN) contraction algorithms is that **99% of total contraction time is spent on the final root contraction**. This report investigates this hypothesis across five distinct tensor network topologies: 1D MPS Chains, 2D PEPS Grids, 3D PEPS Grids, Random Regular Graphs, and Binary Trees.

Our results show that the "99% root bottleneck" hypothesis is **invalid** for closed tensor networks:
1. **The Final Step is Trivial**: Because a closed network contracts down to a single scalar value, the very last step (the root of the tree) is always a scalar-producing dot product. This step takes next to no time (**0.02% to 10.6%** of execution time depending on sizes).
2. **Peak Treewidth Bottleneck**: The true timing bottleneck is located in the **middle-to-late intermediate steps of maximum treewidth** where the largest multidimensional tensors are formed. For example, in the largest 3D PEPS grid (`3x3x3, D=4`), the total contraction takes **708.7ms**, but the final step takes only **3.4ms** (0.49%), while intermediate steps take the bulk of the time.
3. **Sequential vs. Parallel timing profiles**: Slicing parallel execution flattens the cumulative cost curves by reducing the bond dimensions of intermediate tensors, distributing the computational load more evenly across steps and cores.

---

## 2. Experimental Data

The table below summarizes the total contraction time, the time spent on the final step (which produces a scalar), and the corresponding percentage of total compute time.

| Topology | Size Configuration | Total Compute Time (s) | Final Step Time (s) | Final Step Percentage (%) |
|---|---|---|---|---|
| **1D MPS Chain** | N=50, D=16 | 0.0002s | 0.000018s | **8.83%** |
| | N=100, D=24 | 0.0003s | 0.000017s | **5.67%** |
| | N=150, D=32 | 0.0005s | 0.000015s | **2.99%** |
| **2D PEPS Grid** | 4x4, D=4 | 0.0003s | 0.000013s | **4.27%** |
| | 5x5, D=4 | 0.0007s | 0.000022s | **3.16%** |
| | 6x6, D=4 | 0.0007s | 0.000017s | **2.49%** |
| **3D PEPS Grid** | 3x2x2, D=3 | 0.0003s | 0.000016s | **5.24%** |
| | 3x3x2, D=4 | 0.0031s | 0.000029s | **0.95%** |
| | 3x3x3, D=4 | 0.7087s | 0.003473s | **0.49%** |
| **Random Regular**| N=16, D=4 | 0.0003s | 0.000014s | **4.82%** |
| | N=24, D=4 | 0.0004s | 0.000014s | **3.40%** |
| | N=32, D=4 | 0.0006s | 0.000017s | **2.81%** |
| **Binary Tree** | depth=4, D=4 | 0.0002s | 0.000021s | **10.60%** |
| | depth=5, D=5 | 0.0002s | 0.000018s | **8.97%** |
| | depth=6, D=6 | 0.0002s | 0.000014s | **7.24%** |

---

## 3. Visualization

The plot below shows:
* **Row 1 (Cumulative Timing in %)**: Directly compares the cumulative sequential (solid lines) vs. sliced parallel (dashed lines) timing shapes. Curves start exactly at `(0, 0)`.
* **Row 2 (Absolute Execution Time in seconds)**: Uses a logarithmic y-axis to show the exact execution gap in seconds between sequential (solid) and parallel (dashed) execution.

![Step Cost Progression Plot](/home/tom/.gemini/antigravity-cli/brain/b65b070e-93d8-4296-adfa-19d50dd59548/step_cost_progression_v2.png)

---

## 4. Key Scientific Insights

### 4.1. The Absolute Timing Gap (Parallel Speedup)
As shown in **Row 2 (Absolute Execution Time)**:
* For smaller configurations (light turquoise/teal lines), sequential execution (solid) is **faster** than parallel execution (dashed). The overhead of parallel task management and loop partitioning dominates, creating a performance inversion.
* For the largest configurations (dark blue lines, especially for **3D PEPS** and **Random Regular Graphs**), the dashed curves (parallel) sit below the solid curves (sequential), showing that parallel execution becomes significantly faster as size scales up, widening the absolute performance gap.

### 4.2. Flattening of Parallel Cumulative Curves
As seen in **Row 1 (Cumulative Cost %)**:
* The dashed lines (parallel execution) are consistently **flatter** and rise more gradually compared to the solid lines (sequential execution).
* **Why**: Slicing slices specific indices, which reduces the dimensions of all intermediate tensors along the path. By shrinking tensor shapes, it suppresses the high-treewidth bottlenecks and distributes the contraction workload more evenly across steps. This flattens the cumulative cost profile.

### 4.3. High-Scale Intermediate Bottlenecks
For the largest `3x3x3, D=4` PEPS grid:
* Sequential total execution time is **708.7ms**.
* The peak intermediate contraction takes **0.25 seconds** (35% of the total time).
* The final step takes only **3.4ms** (0.49% of the total time).
* Slicing reduces this peak step time to **0.08 seconds** per slice, which when run in parallel on 4 threads, drops the elapsed bottleneck time substantially.

### 4.4. Critical Review: Is Slicing the Right Parallelization Strategy?
Slicing is a **global index reduction strategy** that splits a single large contraction into $S$ independent, smaller, unsliced contractions.
* **Advantages**:
  1. **Memory Cap**: Slicing is the *only* strategy that can fit extremely large contractions (which would otherwise cause Out-Of-Memory crashes) into physical RAM by capping intermediate tensor shapes.
  2. **Work-Stealing / Coarse-Grained Parallelism**: Slices are completely independent, allowing trivial parallelization across threads or distributed clusters with zero communication overhead.
* **Disadvantages**:
  1. **FLOP Inflation**: Slicing introduces redundancy (some intermediate operations are replicated across slices), inflating total FLOPs.
  2. **Fine-Grained Inefficiencies**: For small nodes, slicing is a major regression due to task spawning overhead.
* **Alternative Strategies**:
  * **Tree-Node Parallelism**: Spawning parallel tasks for independent branches. Extremely efficient for early-to-middle steps but fails at the root where execution becomes sequential.
  * **Threaded BLAS (GEMM) Parallelism**: Multi-threading the matrix multiplication of individual large contractions. Extremely efficient for a few massive nodes but fails to utilize multi-core architecture on smaller intermediate steps.

---

## 5. Conclusion & Future Research Directions
The "99% root contraction bottleneck" hypothesis is **incorrect for closed tensor networks** due to the trivial nature of the final scalar step. The true bottleneck is concentrated at maximum-treewidth intermediate steps.

### Proposed Next Research Direction: A Hybrid scheduling Framework
To address the limitations of slicing and exploit the strengths of other parallelization models, the next phase of research should focus on **designing a Hybrid Scheduling Framework** that dynamically selects the parallelization strategy based on the tensor size:
1. **Phase A (Early Path - Small Tensors)**: Use **Tree-Node Parallelism** (Julia task-spawning) to contract independent subtrees concurrently with near-zero overhead.
2. **Phase B (Middle Path - Medium Tensors)**: Use **Active Slicing** to selectively cut indices on branches where dimensions grow, preventing memory bus bottlenecks without global replication.
3. **Phase C (Late Path - Peak Treewidth Tensors)**: Transition to **Multi-threaded BLAS (GEMM)** to accelerate the few remaining heavy matrix multiplications using all available CPU threads concurrently.

