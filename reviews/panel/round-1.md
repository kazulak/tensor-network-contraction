Work ONLY inside the current directory; do not read or modify anything outside it. You may run
any code here and edit files to experiment, but your edits are thrown away: only OUTPUT.md is kept.

## What you are reviewing

./topics/ is a sequence of small, self-contained lessons on tensor networks for quantum circuit
simulation (00, 01, 02, ...). Each takes one idea from a named textbook or paper, implements it
minimally in Python and checks it against an exact reference. The rules are in ./GUIDELINES.md,
the roadmap for the next topics in ./PLAN.md, and the sources in ./references/
(./references/fetch.sh downloads the arXiv PDFs into ./references/local/).

The intended reader is someone getting into the field. They should be able to go through the
topics in order, each building on the last, and understand why each next step is needed.

## Round 1: independent review

Review every topic in ./topics/ with your own expertise. Run `pytest` and each topic's script;
read the code, the README and the cited source sections. Look for:

1. ERROR: something wrong (physics, maths, code), or a claim its own numbers do not support.
2. MISSING: something needed that is not there (a check, a caveat, a concept a later topic
   relies on).
3. SIMPLIFY: something that could be shorter, clearer or more direct without losing anything:
   code, a test, an explanation, a whole section.
4. EXPLAIN: a place where a newcomer would get lost: an undefined term, a jump in reasoning, a
   result stated without the intuition behind it, a missing small worked example or picture.
5. SEQUENCE: the path between topics, and from the last topic into the planned topics in
   PLAN.md. Is each step motivated by the one before? What bridge is missing? Should the order
   change?

Rules:
- Every finding must cite evidence: file:line, a command and its output, or a section or equation
  of a source. If you did not check something, do not claim it.
- Prefer a few important findings over many small ones. Say plainly what is good and should stay.
- Stay in your role, but flag anything outside it that looks wrong.

Write OUTPUT.md in exactly this format:

# Round 1: <your role>
## Summary
3 to 5 sentences.
## Findings
### <ID>: <one-line title>
- Topic: 00 | 01 | ... | all | roadmap
- Type: ERROR | MISSING | SIMPLIFY | EXPLAIN | SEQUENCE
- Severity: high | medium | low
- Evidence: ...
- Proposal: the concrete change (a sketch of the text or code is welcome)
## Keep as is
## Suggested learning path
One line per step: topic, and what the reader must take from it into the next one.
