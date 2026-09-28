# Guidelines

**AI may do much of the work. The repository always says what was done by AI and what by me.**

## The two tracks

| | topics/ | agent-evals/ |
|---|---|---|
| Goal | Establish one idea correctly | Evaluate AI agents on a fixed setup and prompt |
| Rigour | Exact checks, cited sources | None promised |
| Who writes | Me, or AI assistant under my direction | AI agent, largely autonomously |
| After writing | Reviewed and kept correct | Frozen; issues listed, not fixed |

Reproductions of specific papers and my thesis work go into **separate repositories**.

## Topics

1. **Source first.** The README names the book chapter or paper section/equation, and the one
   idea being established.
2. **Minimal code.** Plain scripts plus a README. Python with NumPy (`opt_einsum` for paths).
   No packages or frameworks; aim for under 200 lines. Each folder is self-contained, so a
   Julia, C, Fortran or Haskell version can sit next to the Python one.
3. **Exact check.** Every result is compared with a brute-force reference at small sizes: the
   state vector, exact diagonalisation or `einsum`. The check is in `test_*.py`.
4. **Physics correct by default.** Complex numbers, unitary gates (tested), and norms or
   fidelities reported.
5. **Honest measurements.** Prefer counted cost (FLOPs, memory) to wall-clock time. If timing
   matters, state the hardware, use at least 5 repeats, and report the median with min/max.
6. **Reproduce something.** Where possible, reproduce a figure, number or claim from the source,
   and say how closely it matches.
7. **Authorship line** at the end of each README: who wrote the code, and
   `Human review: pending | done (date)`.

## Agent evals

- One dated folder `YYYY-MM-name/` with a verbatim `PROMPT.md` and one subfolder per model.
- The README records the tool, model and version, date, follow-up instructions, a short
  comparison, and known issues.
- Every AI-written paper starts with the banner
  `> **AI-generated, not peer-reviewed.** …`
- No invented authors ("Research Group", "Team"). No "proves", "peer-reviewed" or
  "monograph" unless literally true.
- **Review loop:** an auditor agent (currently Gemini 3.8 Flash, High) reviews the studies
  first, blind: studies anonymised as `study-A…D`, rerun in scratch copies, scored on a fixed
  rubric. Claude Code then checks the auditor's report and spot-checks its key findings. I read
  the final summary.

## Everywhere

- No local logs, virtualenvs, absolute paths (`/home/...`, `file:///`) or personal data in commits.
- `pytest` (topics only) must pass before pushing.
