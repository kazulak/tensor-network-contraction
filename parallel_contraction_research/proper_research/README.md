# Parallel Tensor Network Contraction Research

This workspace contains the research framework, workspaces, and findings for parallelizing tensor network contraction in Python. Three Large Language Model (LLM) agents were deployed as autonomous researchers to address a central research question.

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

* **[parallel_contraction_research/](..)**: Workspaces containing the research outputs of each agent.
  * **[gemini_3.5_flash/](../gemini_3.5_flash/)**: Google Gemini 3.5 Flash workspace.
    * Contains [paper.md](../gemini_3.5_flash/paper.md) and [speedup_plot.png](../gemini_3.5_flash/speedup_plot.png).
  * **[gemini_pro_3.1/](../gemini_pro_3.1/)**: Gemini Pro 3.1 workspace.
    * Contains [research_paper.md](../gemini_pro_3.1/research_paper.md) and [speedup_plot.png](../gemini_pro_3.1/speedup_plot.png).
  * **[gpt_oss_120b/](../gpt_oss_120b/)**: GPT OSS 120b workspace (contains no files).
* **[LICENSE](../../LICENSE)**: MIT License.
* **[README.md](../../README.md)**: Main repository documentation.

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
* **Deliverables**: [paper.md](../gemini_3.5_flash/paper.md) | [speedup_plot.png](../gemini_3.5_flash/speedup_plot.png) | [benchmark_results.json](../gemini_3.5_flash/benchmark_results.json)

### 2. Gemini Pro 3.1 (High) - *Status: Completed*
* **Implementation Strategy**: Set up a 12x12 2D grid network ($D=3$, 81 slices) using `quimb` and `cotengra`. Used `ProcessPoolExecutor` for multiprocessing with environment variables (`OMP_NUM_THREADS=1`, etc.) set to lock BLAS cores.
* **Key Discoveries**:
  - Found that runtime decreased from 18.1s (serial) to 9.4s (4 workers) but regressed to 10.2s at 8 workers.
  - Attributed scaling limits to the high frequency of allocating large temporary arrays in memory, saturating L3 cache and memory bus bandwidth.
* **Results Table**:
  - Peak speedup vs sliced serial: **1.92x** (at 4 workers).
  - Net efficiency at 8 workers dropped to 0.22.
* **Deliverables**: [research_paper.md](../gemini_pro_3.1/research_paper.md) | [speedup_plot.png](../gemini_pro_3.1/speedup_plot.png) | [benchmark.py](../gemini_pro_3.1/benchmark.py)

### 3. GPT OSS 120b (Medium) - *Status: Failed*
* **Review**: The workspace [gpt_oss_120b/](../gpt_oss_120b/) is completely empty. This agent failed to initialize any code, execute benchmarks, or write research files.

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
You are an autonomous research agent. Your task is to design and, where feasible, implement a small research project on parallel tensor-network contraction in Python.

Research question

The central research question is:

How much speedup can be achieved by parallelizing tensor-network contraction in a Python-based implementation?

The focus is parallelization. Do not study the sign problem. Do not attempt to invent a new optimal contraction-ordering algorithm. Contraction ordering may be discussed only as a practical dependency needed to build a realistic contraction pipeline.
```
