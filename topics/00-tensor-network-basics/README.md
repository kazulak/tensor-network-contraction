# 00 - Tensor network basics

**Source:** Bridgeman & Chubb, *Hand-waving and interpretive dance: an introductory course on
tensor networks*, arXiv:1603.03039, Sections 1.3–1.5.

**Idea:** A tensor network has one value, but *how* you contract it decides the cost. Every
pairwise contraction is a transpose, a reshape and a matrix product. Its cost is the product of
the dimensions of all legs involved.

## What is here

`basics.py` (~130 lines):

- `contract_pair`: a pairwise contraction written out as transpose → reshape → matmul.
- `bubble`: contract a network one tensor at a time in a given order (a "bubbling", §1.4) and
  track the largest stored tensor.
- A **ladder** network with two bubblings (Eqs. 1.14–1.17), and the **graph-colouring** network
  of §1.5 (Eq. 1.19). Its value equals the number of proper q-colourings of the graph.

## Checks

`pytest` (5 tests), all exact:

- `contract_pair` equals `np.einsum` on random complex tensors.
- The matrix-chain costs match the hand-computed values.
- **Reproduced (§1.4):** along the top of the ladder, the stored tensor reaches rank *n*. Rung
  by rung, it never exceeds rank 3. Both give the same value as `einsum`.
- **Reproduced (§1.5):** the network counts colourings exactly. For cycles it matches the
  chromatic polynomial (q−1)ⁿ + (−1)ⁿ(q−1). For the Petersen graph it matches brute force (120
  three-colourings).

```bash
python basics.py     # prints the ladder table and colouring counts
pytest
```

**Author:** code by Claude Code (Claude Opus 5.5), directed by T. Kazulak. Human review: pending.
