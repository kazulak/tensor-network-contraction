# What was applied

Claude Code (Claude Opus 5.5) applied [REPORT.md](REPORT.md) on 2026-09-29, filtered by
[CHECK.md](CHECK.md). None of the REPORT code sketches were pasted. The sketches for items 4 and
13 write gate legs as (in, out) while the code reshapes gates as (out, in), which transposes every
gate. Every new README sentence cites a section or equation, or is derived from the code.
`pytest`: 23 tests pass.

| # | Item | Outcome |
|---|---|---|
| 1 | Exact integer cost in `order.py` | Applied. `math.prod` for sizes; new test gives C = 54 for a d=3 chain, not 53.999… |
| 2 | Faster slicing test | Applied. Network, path and slices are built once, with target W0−2 (3 slices): 4.45 s → 0.30 s |
| 3 | `basics.py` 27 s stall | **Applied differently: the diagnosis was wrong.** Brute force over 4¹⁰ colourings takes 0.7 s. The time went into `np.einsum(optimize="greedy")`: its path never builds an intermediate larger than the largest input, so on Petersen with q=4 it ends in one 15-index einsum (25 s). `einsum_reference` now uses `opt_einsum` (4 ms), and brute force stays for every q. Script: 27 s → 1.2 s |
| 4 | Basis-state inputs, QFT via the network | Applied. `to_network(..., initial=)`, with a one-line length assert against silent partial traces; the QFT network is checked from every basis input, n ≤ 4 |
| 5 | Norm test, observables | Applied. A norm assert in the open-network test and the norm printed by `circuits.py`. The doubled-network paragraph cites Gray & Kourtis §4.5, Eq. 18. Not implemented in code |
| 6 | Restructure `PLAN.md` | Applied. Part I exact / Part II approximate; canonical form in 05 (Bridgeman & Chubb Eq. 3.45); TEBD named in 06 and removed from 08. The REPORT's unsourced numbers are not copied |
| 7 | Ladder as MPS overlap, cycle as Tr(Tⁿ) | Applied (topic 00 README). Transfer matrix per Bridgeman & Chubb §3.3.1; the eigenvalues of J − I give the chromatic polynomial |
| 8 | #P-hardness, Potts model | Applied with sources: Bridgeman & Chubb §1.5 (refs [8], [9]) and Sokal, arXiv:math/0503607, §2.2. Not "Valiant 1979" |
| 9 | GEMM cost, cost in `bubble`, complex FLOPs | Applied. `bubble` returns the multiply-add count, and the test checks linear vs ≥ 2ⁿ (Bridgeman & Chubb §1.4). 8 FLOPs per complex multiply-add, cited as Gray & Kourtis Sec. 2 (6 for the multiplication, 2 for the addition), not "6 multiplications and 2 additions" |
| 10 | Width bounds | Applied in modified form. Line-graph treewidth and degree from Markov & Shi Prop. 4.2, Lemma 4.4 and the m-ary tree example. W ≥ L proved by a cut argument using Bollobás & Leader (1991) and not called the area law. W ≤ L+1 from the row frontier. **Not applied:** the QFT "K_n, W = Θ(n)" claim (unsourced) |
| 11 | Slicing as sums outside, per-step cost | Applied. 2^\|s∖s_v\|·C(v), Gray & Kourtis §4.7.1. **Not applied:** "strictly C_s > C" (unproved) and slicing = Schrödinger–Feynman (unsourced) |
| 12 | Haar U(4) vs Sycamore gates | **Applied corrected.** "Sycamore-like" means the couplings only. fSim has no exact low-rank split (Gray & Kourtis §4.6), not "operator Schmidt rank 2" |
| 13 | Uniform network format, dead code | Applied, and extended at the author's request: topics 02 and 03 now use (tensor, legs) lists like 00–01, and script outputs are unchanged. `I2` and the equally unused `X` were removed |
| 14 | Test `slice_random`, stable Haar test | Applied. The new test also asserts C_greedy ≤ C_random; the Haar test uses its own RNG |

Human review of the topics is still pending (see each README).
