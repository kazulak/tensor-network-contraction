# Parallel Tensor Network Contraction Research

This repository contains the research framework, workspaces, and findings for parallelizing tensor network contraction in Python. Three Large Language Model (LLM) agents were deployed as autonomous researchers to address a central research question.

### Central Research Question
> **How much speedup can be achieved by parallelizing tensor-network contraction in a Python-based implementation?**

The focus is strictly on parallelization techniques (e.g., multi-threading, multi-processing, GPU acceleration, distributed computing) and their scaling limits in Python (handling the GIL, overheads, and data sharing).

---

## Executive Summary & Synthesized Findings

Parallelizing tensor network contractions using **slicing** (bond cutting) is a valuable memory-saving technique but exhibits severe scaling limitations on shared-memory multi-core CPUs:
1. **CPU Speedup Bottlenecks**: Both active agents achieved sublinear speedups of **1.9x to 3.5x** relative to the sliced serial baseline. Scaling peaks early (at 4 to 8 workers on a 12-core CPU) due to **memory bandwidth saturation (RAM bus)** and shared cache thrashing when processes concurrently perform large GEMM operations.
2. **FLOP Inflation & Slicing Overhead**: Slicing introduces a massive computational penalty. Cutting bonds leads to redundant evaluation of local sub-trees, causing an **8.5x to 58.0x increase in FLOP count**.
3. **Real-world Recommendation**: Unless constrained by memory limitations (OOM errors), **unsliced serial contraction remains 3x to 50x faster** than sliced parallel contraction because of FLOP inflation. Slicing should be treated as a memory-scaling strategy, not a performance-optimization tool.

---

## Workspace Directory Structure

* **[parallel_contraction_research/](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/)**: Workspaces containing the research outputs of each agent.
  * **[gemini_3.5_flash/](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/)**: Google Gemini 3.5 Flash workspace.
    * Contains [paper.md](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/paper.md) and [speedup_plot.png](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/speedup_plot.png).
  * **[gemini_pro_3.1/](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_pro_3.1/)**: Gemini Pro 3.1 workspace.
    * Contains [research_paper.md](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_pro_3.1/research_paper.md) and [speedup_plot.png](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_pro_3.1/speedup_plot.png).
  * **[gpt_oss_120b/](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gpt_oss_120b/)**: GPT OSS 120b workspace (contains no files).
* **[LICENSE](file:///home/tom/repos/tensor-network-contraction/LICENSE)**: MIT License.
* **[README.md](file:///home/tom/repos/tensor-network-contraction/README.md)**: Main repository documentation (this file).

---

## Agent Performance & Work Review

### 1. Google Gemini 3.5 Flash (High) - *Status: Completed*
* **Implementation Strategy**: Used `quimb` and `cotengra` to build tensor networks and select optimal slice indices. Executed slices using `concurrent.futures.ProcessPoolExecutor`.
* **Optimization Highlight**: To prevent IPC (Inter-Process Communication) and serialization overheads, Gemini 3.5 Flash used worker pool `initializers` to cache the global contraction tree and tensor arrays in the worker processes' memory exactly once.
* **Key Discoveries**:
  - Explicitly computed **FLOP inflation** (8.5x for 3-regular graph $n=40$ and 58.0x for $L=5\times 5$ grid).
  - Proved that due to FLOP inflation, parallelized sliced contractions are **3x to 50x slower** than the optimal unsliced serial baseline.
* **Results Table**:
  - Peak speedup vs sliced serial: **3.45x** (at 8 workers for Random Regular $n=44$, $D=8$).
  - Speedup degrades or flatlines beyond 8 workers.
* **Deliverables**: [paper.md](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/paper.md) | [speedup_plot.png](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/speedup_plot.png) | [benchmark_results.json](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_3.5_flash/benchmark_results.json)

### 2. Gemini Pro 3.1 (High) - *Status: Completed*
* **Implementation Strategy**: Set up a 12x12 2D grid network ($D=3$, 81 slices) using `quimb` and `cotengra`. Used `ProcessPoolExecutor` for multiprocessing with environment variables (`OMP_NUM_THREADS=1`, etc.) set to lock BLAS cores.
* **Key Discoveries**:
  - Found that runtime decreased from 18.1s (serial) to 9.4s (4 workers) but regressed to 10.2s at 8 workers.
  - Attributed scaling limits to the high frequency of allocating large temporary arrays in memory, saturating L3 cache and memory bus bandwidth.
* **Results Table**:
  - Peak speedup vs sliced serial: **1.92x** (at 4 workers).
  - Net efficiency at 8 workers dropped to 0.22.
* **Deliverables**: [research_paper.md](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_pro_3.1/research_paper.md) | [speedup_plot.png](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_pro_3.1/speedup_plot.png) | [benchmark.py](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gemini_pro_3.1/benchmark.py)

### 3. GPT OSS 120b (Medium) - *Status: Failed*
* **Review**: The workspace [gpt_oss_120b/](file:///home/tom/repos/tensor-network-contraction/parallel_contraction_research/gpt_oss_120b/) is completely empty. This agent failed to initialize any code, execute benchmarks, or write research files.

---

## Comparison Matrix

| Metric | Google Gemini 3.5 Flash (High) | Gemini Pro 3.1 (High) | GPT OSS 120b (Medium) |
|---|---|---|---|
| **Test Cases** | 3-Regular Graphs ($n=40,44$), 2D Grids ($L=5,6$) | 2D Grid ($12\times 12$, $D=3$) | None |
| **Max Workers Tested** | 12 | 8 | None |
| **Peak Speedup** | **3.45x** (at 8 workers) | **1.92x** (at 4 workers) | N/A |
| **Core Multiprocessing Cache**| Cached using initializer args | Initialized shared globals | None |
| **FLOP Inflation Analysis** | Yes (Identified 8.5x–58x penalty) | No | No |
| **GIL / BLAS Thread Locks** | Yes (`OMP_NUM_THREADS=1`, etc.) | Yes (`OMP_NUM_THREADS=1`, etc.) | No |
| **Main Bottleneck Identified**| RAM bandwidth saturation (GEMMs) | Memory allocation overhead & cache thrashing | N/A |

---

## Agent System Prompt

Each agent started with this prompt:

```text
You are an autonomous research agent. Your task is to do a small research project on **parallel tensor-network contraction using Python**.

## Research question

**How much speedup can be achieved by parallelizing tensor-network contraction?**

Focus only on parallelization. Do not study the sign problem. Do not invent a new contraction-ordering algorithm. You may use existing contraction-ordering tools only as part of the implementation.

## Requirements

Use **Python** as the main language.

Do not use naive Python loops, raw `numpy.einsum`, or Python threading as the main method. Python should be used with serious tensor-network and parallel-computing tools.

Recommended libraries to consider:

* `quimb` for building tensor networks,
* `cotengra` for contraction planning, slicing, and optimized contraction trees,
* NumPy/SciPy, CuPy, PyTorch, JAX, or cuTensorNet as numerical backends,
* multiprocessing, MPI, Dask, Ray, or joblib for parallel execution.

A strong solution will likely use **slicing**: split a tensor-network contraction into many independent slice contractions, run them in parallel, and combine the results.

## What to build

Create a small reproducible benchmark pipeline that:

1. Builds one or more tensor-network examples.
2. Computes a contraction plan using existing libraries.
3. Runs a serial baseline.
4. Runs a parallel version.
5. Checks that serial and parallel results agree.
6. Measures speedup.

Use at least one realistic benchmark family, such as:

* random tensor networks,
* 2D grid tensor networks,
* quantum-circuit tensor networks,
* Ising-model tensor networks,
* PEPS-like tensor networks.

Use more than one benchmark family if feasible.

## Metrics to report

Report:

* serial runtime,
* parallel runtime,
* number of workers,
* speedup,
* parallel efficiency,
* tensor-network size,
* number of slices or tasks,
* numerical error versus baseline.

Use:

```text
speedup = serial_time / parallel_time
parallel_efficiency = speedup / number_of_workers
```

Control obvious timing issues. In particular, explain how you avoid CPU oversubscription from BLAS or nested parallelism.

## Deliverables

Produce:

1. A short research paper in Markdown or Typst.
2. A small Python code structure or implementation.
3. Reproduction instructions.
4. A benchmark table.
5. At least one plot, preferably speedup versus number of workers.
6. A clear answer to the research question.

## Suggested paper structure

Use this structure:

1. Title
2. Abstract
3. Introduction
4. Background: tensor-network contraction and slicing
5. Python libraries and implementation choices
6. Parallelization method
7. Experiments
8. Results
9. Discussion
10. Limitations
11. Future work
12. Conclusion
13. References

## Important guidance

Be honest. Do not fabricate benchmark results. If full implementation is not possible, provide a clear experimental plan, code skeleton, and expected bottlenecks.

The final paper should clearly answer:

* Did parallelization help?
* How much speedup was achieved or expected?
* What limited the speedup?
* What would be the next improvement?

```
