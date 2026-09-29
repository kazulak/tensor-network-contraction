# 00 - Tensor network basics

**Source:** Bridgeman & Chubb, *Hand-waving and interpretive dance: an introductory course on
tensor networks*, arXiv:1603.03039, Sections 1.3–1.5.

**Idea:** A tensor network has one value, but *how* you contract it decides the cost. Every
pairwise contraction is a transpose, a reshape and a matrix product. An (m×k)·(k×n) product
takes m·k·n multiply-adds, so a contraction's cost is the product of the dimensions of all legs
involved.

## What is here

`basics.py` (~130 lines):

- `contract_pair`: a pairwise contraction written out as transpose → reshape → matmul.
- `bubble`: contract a network one tensor at a time in a given order (a "bubbling", §1.4). It
  tracks the largest stored tensor and counts the multiply-adds.
- A **ladder** network with two bubblings (Eqs. 1.14–1.17), and the **graph-colouring** network
  of §1.5 (Eq. 1.19). Its value equals the number of proper q-colourings of the graph.

## Why these two examples

- **The ladder** is the shape of an overlap ⟨φ|ψ⟩ of two matrix product states. The rungs are
  the physical legs and the rails are the bonds ("we'll see a few of these in the following
  lectures", §1.4).
  - Rung by rung, each step applies one site's transfer matrix (like E in §3.3.1) to a
    bond×bond environment. Memory stays constant and time grows linearly.
  - Along the top, the stored tensor is the whole top state, with dⁿ entries.
- **A cycle Cₙ** has vertex tensors with two legs, which are identity matrices. The network is
  therefore Tr(Tⁿ) with T = J − I, the q×q all-ones matrix minus the identity. T has eigenvalue
  q−1 once and −1 (q−1) times, which gives the chromatic polynomial (q−1)ⁿ + (−1)ⁿ(q−1) checked
  below.
- **Why colourings at all:** deciding whether a q-colouring exists is NP-complete, and counting
  them is #P-complete. So contracting these networks is #P-complete, and no contraction order
  makes every network cheap (§1.5 and its refs [8], [9]).
  - In physics terms, the count is the zero-temperature partition function of the
    antiferromagnetic q-state Potts model (Sokal, arXiv:math/0503607, §2.2).

## Checks

`pytest` (5 tests), all exact:

- `contract_pair` equals `np.einsum` on random complex tensors.
- The matrix-chain costs match the hand-computed values.
- **Reproduced (§1.4):** along the top of the ladder, the stored tensor reaches rank *n* and the
  cost grows at least like 2ⁿ. Rung by rung, it never exceeds rank 3 and each extra rung adds the
  same cost. Both give the same value as `opt_einsum`.
- **Reproduced (§1.5):** the network counts colourings exactly. For cycles it matches the
  chromatic polynomial (q−1)ⁿ + (−1)ⁿ(q−1). For the Petersen graph it matches brute force (120
  three-colourings; `basics.py` also prints 12960 four-colourings).

```bash
python basics.py     # prints the ladder table and colouring counts (~1 s)
pytest
```

**Author:** code by Claude Code (Claude Opus 5.5), directed by T. Kazulak. Human review: pending.
