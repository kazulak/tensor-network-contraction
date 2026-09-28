# 02 - Contraction order and cost

**Sources:**
- Gray & Kourtis, *Hyper-optimized tensor network contraction*, arXiv:2002.01935, Sec. 2.
  Eqs. (2)–(3) define the contraction width W and Eq. (5) the contraction cost C.
- Markov & Shi, arXiv:quant-ph/0511069, Theorem 1.1: the best contraction costs exp(O(treewidth)).
- Bridgeman & Chubb, arXiv:1603.03039, Sec. 1.4: a 2D grid forces a boundary of about √n legs.

**Idea:**
- **W** is log₂ of the largest intermediate tensor (memory).
- **C** is the total multiply-add count (time).

Both depend hugely on the order. The *best* order is limited by the graph: constant for a
chain or ring, growing with the side L on an L×L grid.

## What is here

`order.py` (~120 lines): ring and grid networks, and W and C computed straight from the
definitions. Four orders are compared: random bubbling, row-by-row bubbling, `opt_einsum`
greedy, and `opt_einsum` dynamic programming minimising W. Output with bond dimension 2:

| network | random bubbling | row-by-row | greedy | optimal (min W) |
|---|---|---|---|---|
| ring, n=64 | W=36 | W=2 | W=2 | W=2 |
| grid 4×4 | W=12 | W=5 | W=5 | W=4 |
| grid 5×5 | W=20 | W=6 | W=6 | W=6 |
| grid 6×6 | W=37 | W=7 | W=8 | W=6 |

The random column changes with the seed. The others are deterministic.

## Checks

`pytest` (6 tests):

- Every order gives the same value as `np.einsum` (random complex tensors, ring and 3×3 grid).
- W and C of a matrix chain match the hand-computed values.
- **Reproduced:** the optimal width on a ring is W=2 for every n. On an L×L grid it satisfies
  L ≤ W ≤ L+1 for L = 2…5, i.e. it grows with the boundary, not the number of tensors.

```bash
python order.py
pytest
```

**Author:** code by Claude Code (Claude Opus 5.5), directed by T. Kazulak. Human review: pending.
