# 2026-09 - Slicing study: four Gemini models on one prompt

**Question:** How much extra work does slicing cost for a given memory saving, and does the
choice of sliced indices matter? The checked reference answer is
[topic 03](../../topics/03-slicing/).

**Agents:** Gemini 3.8 Flash, 3.7 Flash, 3.6 Flash and 3.1 Pro, all at High, run through the
Antigravity CLI (`agy -p`, headless, one folder each) on 2026-09-27. All four exited
cleanly. **Prompt:** [PROMPT.md](PROMPT.md), the same for all, no follow-ups.
**Launcher:** [run.sh](run.sh).

Each model folder is the agent's untouched output. Only absolute local paths in links were
removed. `transcript.md` is the agent's final message.

## Review

Reviewed by Claude Code (Claude Opus 5.5). For each agent it reran the tests and the study in a
scratch copy, compared every table against the rerun, and checked each citation on arXiv.

| Criterion (0–2) | 3.1 Pro | 3.6 Flash | 3.7 Flash | 3.8 Flash |
|---|---|---|---|---|
| Runs: code and tests run as delivered | 2 | 2 | 2 | 2 |
| Correct: amplitudes, Haar U(4), slicing method | 0 | 1 | 2 | 2 |
| Honest: numbers reproduce on rerun | 1 | 2 | 2 | 1 |
| Answer: conclusions follow from the data | 0 | 1 | 2 | 1 |
| Sources: citations exist and are right | 1 | 0 | 0 | 1 |
| Simple: ≤ ~300 lines, readable | 2 | 2 | 2 | 2 |
| **Total (of 12)** | **6** | **8** | **10** | **9** |

All four got the basics right: complex Haar-random U(4) gates, unitarity tests, and
sliced = unsliced = state-vector checks that pass. Earlier, less constrained agent runs got these wrong.
The differences are in the analysis.

### Gemini 3.1 Pro: 6/12
- **Counts the slices twice** (found by the auditor, confirmed by rerun). In cotengra,
  `contraction_cost()` of a sliced tree already includes every slice, and the agent multiplies by
  `nslices` again. So every "smart" overhead is inflated by 2^k: at k=8, "827×" is really about
  3×. Random slicing is counted correctly, which makes the comparison between the two meaningless.
- Studies only the 2D grid. The requested 1D brickwork appears only in the tests.
- The circuit is unseeded, so the numbers change on every run. They stay in the same range
  (at k=8: 827× in the README, 657× and 445× in two reruns).
- **The conclusions contradict its own (already wrong) table.** It says "overhead slightly more than 2^k", but
  6 qubits of memory saved costs 827× (2^6 = 64). It says random slicing "always pays 2^k and saves
  nothing", but the table shows 29× at k=8 and 3 qubits saved.
- It reports an overhead of 0.8×, which is impossible (C_s ≥ C). The "smart" tree was
  re-optimised after slicing, so the unsliced baseline was a poor tree. This goes unremarked.
- No citations.

### Gemini 3.6 Flash: 8/12
- Clean setup: seeded, both geometries, and a fair comparison on one fixed tree. Its "optimal
  slicing" results reproduce exactly (14 of 14 rows).
- **Bug:** random slicing calls cotengra's `tree.remove_ind(ix)` and ignores the return value,
  since it is not in-place. The "random" column is therefore exactly 2^k with 1.00× memory
  reduction in every row. **Its headline conclusion ("random slicing is completely
  ineffective") comes from this bug.**
- All three citations are wrong. The Markov & Shi title is wrong. Gray & Kourtis becomes "Gray &
  Kiffner" with an invented title and journal. The arXiv ID given for Villalonga et al. is a
  computer-vision paper.

### Gemini 3.7 Flash: 10/12 (best)
- Seeded, both geometries, correct in-place slicing, and a fair comparison on one fixed tree. It
  added a third baseline of its own: slicing the circuit's input legs.
- 13 of 16 table rows reproduce exactly. The other three differ slightly (e.g. 1.762× vs 1.745×),
  due to randomness inside cotengra's slice finder.
- Conclusions match the data. On the 2D grid: 2× less memory for +6% work, 16× less for 3.4×. On
  the 1D brickwork: 8× less memory for 36×.
- Minor overstatements: "sub-exponential" (it is ~2^(0.35k)), and "agreement < 10⁻¹⁴" when the
  tests assert < 10⁻¹⁰.
- **One invented citation:** the arXiv ID given for Villalonga et al. is a string-theory paper. Two
  other titles are garbled.

### Gemini 3.8 Flash: 9/12
- Clean code: seeded, both geometries, 25 random trials, and a "peripheral index" baseline. Best
  citation record: all five IDs are real and match, with two slightly wrong titles.
- **Not reproducible.** cotengra's `AutoOptimizer` is stochastic, so the baseline tree differs
  between runs. Its headline was "256× less memory for 2.7× more work" on the 2D grid. On rerun, a
  4× better baseline tree gave **64× less memory for 28× more work**. The cheap slicing came from
  a poor starting tree, and the README does not say so.
- Its tests sum the slices with cotengra itself, rather than with an independent loop.

### What this run shows
- **Better baseline, costlier slicing.** Overhead is measured against the unsliced tree. The
  most striking results in 3.1 Pro and 3.8 Flash (0.8×, "256× for 2.7×") came from a weak
  baseline, and neither agent noticed. [Topic 03](../../topics/03-slicing/) fixes the tree first.
- **Agents trust their own tables less than their narrative.** 3.1 Pro wrote conclusions its data
  contradicts. 3.6 Flash built its main claim on an obviously suspicious column (exactly 2^k,
  every row).
- **Citations are still unreliable.** 2 of 4 agents gave an arXiv ID that points to an
  unrelated paper, despite the explicit instruction.
- **Tests passing ≠ study correct.** All 4 test suites passed, and 3 of the 4 studies still
  contain a bug or a misleading headline.
- The newer Flash models (3.7, 3.8) beat 3.1 Pro on this task.

## Audit by Gemini 3.8 Flash: how good is an AI auditor?

Following the [review loop](../../GUIDELINES.md), Gemini 3.8 Flash (High) audited all four studies
**blind**. It saw them as `study-A…D` in shuffled order ([key](audit/KEY.md)), including its own,
and did not see the review above. It reran everything and scored each study on the same rubric.
Prompt: [audit/AUDIT_PROMPT.md](audit/AUDIT_PROMPT.md). Output: [audit/SUMMARY.md](audit/SUMMARY.md)
and one `AUDIT-*.md` per study.

| | 3.1 Pro | 3.6 Flash | 3.7 Flash | 3.8 Flash |
|---|---|---|---|---|
| Auditor score (of 12) | 2 | 7 | 11 | 11 |
| Claude Code score (of 12) | 6 | 8 | 10 | 9 |

**Found (and correct):** the auditor found every major problem found above:
- the `remove_ind` bug in 3.6 Flash (exact line),
- 3.7 Flash's invented citation,
- 3.8 Flash's drifting baseline.

It also found **one bug the review above had missed:** 3.1 Pro's double-counted slices. That bug
was confirmed by rerun and added above.

**Missed or wrong:**
- **Lenient on its own work** (blind, but still). It scored 3.8 Flash 11/12 and treated the
  drifting baseline only as a reproducibility issue. It missed that the headline "256× less
  memory for 2.7×" depends on a poor baseline tree. It also gave full marks for sources despite
  two wrong titles.
- **Invented a detail itself:** it names the author of the unrelated arXiv paper as "Shingo
  Suzuki". The actual authors are Park & Kwon.
- **Repeated an overclaim unchecked:** 3.7 Flash's "agreement < 10⁻¹⁴", which its tests do not
  assert (they assert < 10⁻¹⁰).
- **Harsh scoring by its own rules:** 0/2 for "no citations" and "numbers don't reproduce" on
  3.1 Pro, where the numbers drift but stay in range.

**Verdict:** as a first-pass reviewer it is worth it. It found every major bug, plus one more,
at a fraction of the effort. Its scores and citation checks still need a human-level spot check,
especially of its own model's work. From now on: auditor first, then Claude Code checks the
auditor's key claims.

