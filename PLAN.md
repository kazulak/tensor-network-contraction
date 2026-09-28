# Plan

Roadmap for this repository. Rules are in [GUIDELINES.md](GUIDELINES.md) and sources in
[references/](references/).

## Identity

Two tracks:

- **topics/** is small, correct, source-based, and checked.
- **agent-evals/** is for fun: evaluations of AI agents on a fixed setup and prompt.

Paper reproductions and the PIM thesis work live in separate repositories.

## Topic roadmap

| # | Topic | Idea to establish | Main source | Exact check | Status |
|---|---|---|---|---|---|
| 00 | Tensor network basics | Contraction = transpose + matmul; order decides memory | Bridgeman & Chubb §1.3–1.5 | `einsum`, chromatic polynomial | done |
| 01 | Circuits as tensor networks | Gates → tensors; amplitude = closed network | Nielsen & Chuang Ch. 4, §5.1; Markov & Shi §3 | state vector, DFT | done |
| 02 | Contraction order and cost | Width/cost of a tree; best width set by the graph | Gray & Kourtis §2; Markov & Shi Thm 1.1 | `einsum`, hand-computed costs | done |
| 03 | Slicing | Memory ↔ FLOP trade-off, measured by counting | Gray & Kourtis §4.7.1 | sum of slices = amplitude | done |
| 04 | Parallel contraction | Slice-level vs tree-level parallelism; Amdahl limits | Huang et al. 2020; Williams et al. 2009 | as 03, plus scaling with repeats | planned |
| 05 | MPS and SVD truncation | Schmidt decomposition; error = discarded weights | Orús; Schollwöck §4; Bridgeman & Chubb §1.2 | dense state, Eckart–Young bound | planned |
| 06 | MPS circuit simulation | Bond dimension vs entanglement; fidelity under truncation | Vidal 2003; Zhou et al. 2020 | state vector, n ≤ 20 | planned |
| 07 | DMRG | Ground state of a transverse-field Ising / Heisenberg chain | Schollwöck §6 | exact diagonalisation | planned |
| 08 | Optional | TEBD, PEPS/boundary MPS, Clifford tableau | Orús; Aaronson & Gottesman | case by case | idea |

## Steps

1. ✔ Restructure into `topics/` and `agent-evals/`, freeze the agent evals, point CI at topics only,
   and add `references/`.
2. ✔ Foundations: topics 00–02.
3. Slicing and parallel contraction: topics 03–04.
4. MPS track: topics 05–06.
5. DMRG: topic 07.
6. Spin-offs: when a topic turns into reproducing a paper, create a standalone repository and
   link it from the README.

## Open items

- Review topics 00–02 and update their `Human review:` lines.
- Rename the GitHub repository to something broader than "contraction".
