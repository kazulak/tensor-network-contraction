# Plan

Roadmap for this repository. Rules are in [GUIDELINES.md](GUIDELINES.md) and sources in
[references/](references/).

## Identity

Two tracks:

- **topics/** is small, correct, source-based, and checked.
- **agent-evals/** is for fun: evaluations of AI agents on a fixed setup and prompt.

Paper reproductions and the PIM thesis work live in separate repositories.

## Topic roadmap

### Part I: exact contraction

The answer is exact. The cost is set by the contraction width, which the graph's treewidth
bounds from below (topic 02). Slicing trades memory for work but cannot remove that limit.

| # | Topic | Idea to establish | Main source | Exact check | Status |
|---|---|---|---|---|---|
| 00 | Tensor network basics | Contraction = transpose + matmul; order decides memory and time; counting colourings is #P-complete | Bridgeman & Chubb §1.3–1.5 | `einsum`, chromatic polynomial | done |
| 01 | Circuits as tensor networks | Gates → tensors; amplitude = closed network; observables = doubled network | Nielsen & Chuang Ch. 4, §5.1; Markov & Shi §3 | state vector, DFT | done |
| 02 | Contraction order and cost | Width/cost of a tree; best width set by the graph (line-graph treewidth; L ≤ W ≤ L+1 on a grid) | Gray & Kourtis §2; Markov & Shi Prop. 4.2, Lemma 4.4 | `einsum`, hand-computed costs | done |
| 03 | Slicing | Memory ↔ FLOP trade-off, measured by counting; per-step overhead 2^\|s∖s_v\| | Gray & Kourtis §4.7.1 | sum of slices = amplitude | done |
| 04 | Parallel contraction | Slice-level vs tree-level parallelism; Amdahl limits; roofline with FLOPs = 8·C for complex | Huang et al. 2020; Williams et al. 2009; Gray & Kourtis §2 | as 03, plus scaling with repeats | planned |

### Part II: controlled approximation

When the width is too large, give up exactness. Compress states into matrix product states,
where the cost is set by the bond dimension, i.e. by how much entanglement is kept, and the
error is the discarded weight.

| # | Topic | Idea to establish | Main source | Exact check | Status |
|---|---|---|---|---|---|
| 05 | MPS, canonical form and SVD truncation | Schmidt decomposition; canonical (isometric) form, which makes a local SVD cut optimal; error = discarded weights | Orús; Schollwöck §4; Bridgeman & Chubb §1.2, §3.3.2 (Eq. 3.45) | dense state, Eckart–Young bound | planned |
| 06 | MPS circuit simulation (TEBD) | Bond dimension vs entanglement; fidelity under truncation | Vidal 2003; Zhou et al. 2020 | state vector, n ≤ 20 | planned |
| 07 | DMRG | Ground state of a transverse-field Ising / Heisenberg chain | Schollwöck §6 | exact diagonalisation | planned |
| 08 | Optional | PEPS/boundary MPS, Clifford tableau | Orús; Aaronson & Gottesman | case by case | idea |

## Steps

1. ✔ Restructure into `topics/` and `agent-evals/`, freeze the agent evals, point CI at topics only,
   and add `references/`.
2. ✔ Foundations: topics 00–02.
3. Slicing (✔ topic 03) and parallel contraction (topic 04).
4. MPS track: topics 05–06.
5. DMRG: topic 07.
6. Spin-offs: when a topic turns into reproducing a paper, create a standalone repository and
   link it from the README.

## Open items

- Review topics 00–03 and update their `Human review:` lines. The 2026-09 agent panel review
  has been applied; see [reviews/2026-09-topics/APPLIED.md](reviews/2026-09-topics/APPLIED.md).
- Rename the GitHub repository to something broader than "contraction".
