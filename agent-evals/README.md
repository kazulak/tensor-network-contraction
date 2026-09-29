# Agent evals

**This track is for fun, not research.** Each run is an evaluation of AI agents: I fix a setup
and a prompt, give it to several models, and score what comes back. The agents' output is the
thing being evaluated, not a result in its own right.

- Everything in a model folder was written by the agent and is kept as-is (only broken paths and
  links were fixed, and a disclosure banner was added to each paper).
- Runs are **frozen**: not tested in CI and not fixed. The run's README holds the evaluation.

| Run | Agents | Prompt |
|---|---|---|
| [2026-07-parallel-slicing](2026-07-parallel-slicing/) | Gemini 3.5 Flash, Gemini Pro 3.1, GPT-OSS-120b (Antigravity) | [PROMPT.md](2026-07-parallel-slicing/PROMPT.md) |
| [2026-09-slicing-study](2026-09-slicing-study/) | Gemini 3.8 / 3.7 / 3.6 Flash, Gemini 3.1 Pro, all High (Antigravity); Claude Sonnet 5.5, xhigh (Claude Code) | [PROMPT.md](2026-09-slicing-study/PROMPT.md) |

## Adding a run

Create `YYYY-MM-short-name/` with:

- `PROMPT.md`: verbatim, including follow-up instructions.
- A launcher script, so the setup can be rerun.
- One subfolder per model.
- A `README.md` recording the tool, model and version, and date, with the scored comparison and
  known issues.
- An `audit/` folder: a blind review by an auditor agent (see `2026-09-slicing-study/audit/`
  for the launcher and prompt), checked afterwards by Claude Code.
