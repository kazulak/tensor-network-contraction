# 03 - Slicing

**Source:** Gray & Kourtis, *Hyper-optimized tensor network contraction*, arXiv:2002.01935,
Sec. 4.7.1 ([PDF](../../references/arXiv-2002.01935.pdf)).

**Idea:** Pick a set of indices s, fix their values, and contract each of the
d_sliced = ∏ w(e) resulting networks independently, then sum. Each piece needs less memory
(width W_s < W) and they are embarrassingly parallel. But each costs *at least* C/d_sliced, so
the total work C_s ≥ C. Indices are chosen greedily, one at a time, until a target width is met.

## What is here

`slicing.py` (~125 lines):

- A single amplitude of a random 4×4-qubit circuit, depth 12, with Sycamore-like couplings and
  Haar U(4) gates, as a list of (tensor, legs) like topics 00–02.
- A fixed contraction tree from `opt_einsum` greedy.
- The width and total cost of that tree with any set of indices sliced.
- A greedy slicer and a random slicer for comparison.
- An explicit sliced contraction that sums over all slice values.

## Result

Same tree, sliced to a target width (seed 0):

| target W | memory saved | greedy: #sliced | greedy: C_s/C | random: C_s/C (median of 20) |
|---|---|---|---|---|
| 15 | 2× | 1 | 1.02× | 1.6·10⁴× |
| 13 | 8× | 3 | 1.14× | 3.6·10¹² × |
| 12 | 16× | 5 | 1.77× | 4.5·10¹³ × |
| 11 | 32× | 8 | 5.1× | 1.9·10¹⁷ × |
| 10 | 64× | 12 | 27× | 5.4·10¹⁸ × |
| 8 | 256× | 20 | 1345× | 3.0·10²⁵ × |

- **The first few slices are almost free:** 8× less memory for 14% more work. After that the
  overhead grows much faster than the memory saving.
- **Which indices you slice matters enormously.** Most indices do not cross the widest
  intermediate, so slicing them only multiplies the work.

## Why the overhead looks like this

- **Slicing is moving sums outside.** A network is nested sums over its indices, and slicing
  performs some of them last (Gray & Kourtis §4.7.1). On a bond this is the same as inserting
  I = Σₖ |k⟩⟨k| and pulling the sum out of the contraction.
- **Cost per step.** Take a pairwise step v whose legs are s_v, with unsliced cost C(v), and
  slice a set s.
  - Each of the 2^|s| slices pays 2^|s_v ∖ s|.
  - The total is therefore 2^|s ∖ s_v| · C(v) ≥ C(v), which sums to C_s ≥ C. `width_cost`
    computes exactly this.
- **Consequences.**
  - Steps whose legs already include every sliced index cost nothing extra.
  - Every other step is repeated in each slice where it is unchanged. These are the
    "redundantly repeated contractions" of §4.7.1.
  - Greedy slicing picks indices of the widest, most expensive steps, so the first few slices
    are almost free. Random slicing picks indices that most steps lack, so the repeats multiply.

**About "Sycamore-like":** this refers only to the coupling scheme, four alternating
nearest-neighbour patterns. The gates are Haar-random U(4), not Sycamore's fSim gates. Neither
has an exact low-rank decomposition across the two qubits (Gray & Kourtis §4.6 for fSim), unlike
CZ, which splits with bond dimension 2. So no gate splitting applies here.

This topic is the reference answer for the
[2026-09 slicing study](../../agent-evals/2026-09-slicing-study/) agent evaluation.

## Checks

`pytest` (3 tests):

- The sum over all slices equals the unsliced contraction and the state-vector amplitude.
- Greedy slicing reaches the target width, and the total cost never drops below C.
- Random slicing also reaches the target width, but costs at least as much as greedy slicing.

```bash
python slicing.py      # ~20 s
pytest
```

**Author:** code by Claude Code (Claude Opus 5.5), directed by T. Kazulak. Human review: pending.
