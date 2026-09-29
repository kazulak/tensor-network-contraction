# 02 - Contraction order and cost

**Sources:**
- Gray & Kourtis, *Hyper-optimized tensor network contraction*, arXiv:2002.01935, Sec. 2.
  Eqs. (2)–(3) define the contraction width W and Eq. (5) the contraction cost C.
- Markov & Shi, arXiv:quant-ph/0511069, Theorem 1.1: the best contraction costs exp(O(treewidth)).
- Bridgeman & Chubb, arXiv:1603.03039, Sec. 1.4: a 2D grid forces a boundary of about √n legs.

**Idea:**
- **W** is log₂ of the largest intermediate tensor (memory).
- **C** is the total multiply-add count (time), counted in exact integers. For complex tensors
  each multiply-add is 8 real FLOPs: 6 for the multiplication and 2 for the addition (Gray &
  Kourtis, Sec. 2, below Eq. 5). Topic 04 will need this conversion.

Both depend hugely on the order. The *best* order is limited by the graph: constant for a
chain or ring, growing with the side L on an L×L grid.

## What is here

`order.py` (~120 lines): ring and grid networks as lists of (tensor, legs), as in topics 00–01,
and W and C computed straight from the definitions. W and C only read legs and shapes, so no
intermediate tensor is built. Four orders are compared: random bubbling, row-by-row bubbling,
`opt_einsum` greedy, and `opt_einsum` dynamic programming minimising W. Output with bond
dimension 2:

| network | random bubbling | row-by-row | greedy | optimal (min W) |
|---|---|---|---|---|
| ring, n=64 | W=36 | W=2 | W=2 | W=2 |
| grid 4×4 | W=12 | W=5 | W=5 | W=4 |
| grid 5×5 | W=20 | W=6 | W=6 | W=6 |
| grid 6×6 | W=37 | W=7 | W=8 | W=6 |

The random column changes with the seed. The others are deterministic.

## Which graph quantity, precisely

- **Markov & Shi** state the result for the line graph G* (tensors' legs become vertices, joined
  when they meet at a tensor). The best one-edge-at-a-time contraction has width tw(G*)
  (Prop. 4.2). That is within a factor of the maximum degree Δ of tw(G):
  (tw(G) − 1)/2 ≤ tw(G*) ≤ Δ(tw(G) + 1) − 1 (Lemma 4.4).
  - So "cost exp(O(treewidth))" needs bounded degree. Circuits of one- and two-qubit gates have
    it.
  - Without it the bound fails: an m-ary tree has treewidth 1 but contraction complexity m
    (M&S, after Lemma 4.4). Any tensor with m legs already has W ≥ m.
  - Their width counts one edge at a time. The pairwise contractions here differ from it by at
    most a factor of 2 (M&S §4), so the statement holds here up to constants.
- **Why L ≤ W ≤ L+1 on the L×L grid:**
  - *Lower bound.* Follow any contraction path. Take the first stored tensor that covers at
    least L²/4 sites. Both of its parts (if it has any) covered fewer, so it covers fewer than L²/2 (the argument
    of M&S Lemma 5.2). Its legs are the grid edges leaving that set of sites. Any set of between
    L²/4 and 3L²/4 sites has at least L such edges (Bollobás & Leader, *Edge-isoperimetric
    inequalities in the grid*, Combinatorica 11, 299 (1991)). So W ≥ L.
  - *Upper bound.* Row by row, when the first k sites of a row have been absorbed, the stored
    tensor has L−k legs down from the row above, k legs down to the next row, and one leg to
    the next site. That is at most L+1.
  - B&C §1.4 give the same picture informally: the contracted region must at some point have a
    long perimeter.

## Checks

`pytest` (7 tests):

- Every order gives the same value as `np.einsum` (random complex tensors, ring and 3×3 grid).
- W and C of a matrix chain match the hand-computed values.
- C is an exact integer for dimensions that are not powers of 2 (summing log₂ of the
  dimensions gave 53.999… instead of 54).
- **Reproduced:** the optimal width on a ring is W=2 for every n. On an L×L grid it satisfies
  L ≤ W ≤ L+1 for L = 2…5, i.e. it grows with the boundary, not the number of tensors.

```bash
python order.py
pytest
```

**Author:** code by Claude Code (Claude Opus 5.5), directed by T. Kazulak. Human review: pending.
