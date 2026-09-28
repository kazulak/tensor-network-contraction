# Auditor prompt

Given verbatim to the auditor (Gemini 3.8 Flash, High) as a single non-interactive prompt. The
four studies are anonymised as `work/study-A` … `work/study-D`. The mapping is in
[KEY.md](KEY.md), which the auditor never saw.

````text
You are an independent scientific auditor. Work ONLY inside the current directory; do not read or
modify anything outside it.

The folder ./work/ contains four independent studies, study-A to study-D. Each was produced by an
AI agent answering the research prompt in ./RESEARCH_PROMPT.md. Audit each study in turn, as a
strict but fair referee. Assume nothing is correct until you have checked it yourself.

For EACH study:
1. Run its pytest file and its main script (inside its own folder under ./work/). Record whether
   they run as delivered.
2. Reproducibility: compare every number in its README against what the rerun produces. List any
   mismatch.
3. Correctness: read the code. Check the gates are Haar-random complex U(4), the sliced/unsliced/
   state-vector checks are genuine, and the slicing and cost accounting are right (for example:
   are library calls used correctly, is the unsliced baseline a sensible contraction tree, are
   FLOPs counted once per slice?). Look for bugs that would make a table column meaningless.
4. Claims: does each conclusion in the README actually follow from the study's own numbers?
5. Sources: for each citation, check that the arXiv ID exists and matches the stated title and
   authors (you may open https://arxiv.org/abs/<id>). Flag any that do not.
6. Simplicity: roughly how many lines, is it readable.

Score each study 0-2 on: Runs, Correct, Honest (numbers reproduce), Answer (conclusions follow
from data), Sources, Simple. Be concrete: every deduction must cite a file, a line or a number.

Deliverables in the current directory:
- AUDIT-A.md ... AUDIT-D.md: one report per study, with the evidence for each score.
- SUMMARY.md: a score table for the four studies, the most serious problem found in each, and
  which study best answers the research question and why.
Do not modify the studies' README or code except to run them.
````
