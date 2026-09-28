# 2026-07 - Agent comparison: parallel slicing in Python

**Question:** How much speedup can parallelising tensor network contraction give in Python? A
second question sits behind it: how do different AI agents handle the same research prompt?

**Agents:** Gemini 3.5 Flash (High), Gemini Pro 3.1 (High), GPT-OSS-120b (Medium), all run in
Google Antigravity in July 2026. **Prompt:** [PROMPT.md](PROMPT.md), the same for every agent.

## What was produced

| Agent | Output | Peak speedup (vs sliced serial) |
|---|---|---|
| Gemini 3.5 Flash | [gemini-3.5-flash/](gemini-3.5-flash/): `quimb` + `cotengra` package, [paper](gemini-3.5-flash/paper.md), [table](gemini-3.5-flash/benchmark_table.md) | 3.45× at 8 workers |
| Gemini Pro 3.1 | [gemini-pro-3.1/](gemini-pro-3.1/): benchmark script, [paper](gemini-pro-3.1/research_paper.md) | 1.92× at 4 workers |
| GPT-OSS-120b | nothing (empty workspace, not kept) | n/a |

## Results

- Both working agents used `ProcessPoolExecutor` over `cotengra` slices, with BLAS pinned to one
  thread per worker. Speedup peaked at 4–8 workers on a 12-thread CPU and then flattened.
- Gemini 3.5 Flash was the only agent to measure **FLOP inflation** from slicing (8.5–58×).
  Because of it, the parallel sliced runs were still 3–50× *slower* than unsliced serial
  contraction.
- On the meta question, the faster model did the more complete study. One of the three agents
  produced nothing.

## Known issues

- Speedups are relative to the *sliced* serial baseline. Compared with the unsliced baseline,
  parallel slicing is a slowdown.
- Each agent ran one benchmark configuration with few repeats. The results show a trend, not
  precise measurements.
- A leftover debugging file from the agent (`scratch_test.py`) was removed in the cleanup.

## Reproduce

```bash
cd gemini-3.5-flash && pip install quimb cotengra && python run_benchmarks.py
```
