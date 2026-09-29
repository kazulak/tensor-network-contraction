# Check of the panel report

Claude Code (Claude Opus 5.5) spot-checked [REPORT.md](REPORT.md) on 2026-09-28 by rerunning the
code and small scripts. Panel: Gemini 3.8 Flash (High), three roles, three rounds, all 10 agent
calls completed.

## Confirmed

| Report item | Check | Result |
|---|---|---|
| 1. `order.py` cost is inexact | `path_width_cost` on a 3-tensor chain | d=3: C = 53.999999999999986, not 54; d=7: 686.0000000000003. Exact for powers of 2 only, so no current test fails. |
| 2. Slow slicing test | `pytest --durations` | `test_sum_over_slices_equals_amplitude` takes 4.45 s of 5.65 s. |
| 3. `basics.py` stalls | `time python basics.py` | 27 s. P(Petersen, 4) = 12960 confirmed by vectorised brute force. |
| 4. QFT network never exercised off \|0…0⟩ | `circuits.py:72`, `test_circuits.py:53` | `to_network` has no initial-state argument; the DFT test uses `statevector` only. |
| 13. Dead `I2` | grep | `circuits.py:12`, never used. |
| 14. `slice_random` untested | grep | Only used in `slicing.py:120`, not in the tests. |

## Wrong or unsupported. Do not apply as written.

- **Item 12: Sycamore fSim gates have operator Schmidt rank 2.** False. fSim(π/2, π/6) has
  operator Schmidt rank **4** (computed; iSWAP is also 4, CZ is 2). The point worth keeping is
  that "Sycamore-like" in topic 03 refers to the coupling layout only.
- **Item 9: a complex MAC is "6 real multiplications and 2 real additions".** The total of 8
  real FLOPs is right, but it is 4 multiplications and 4 additions.
- **Item 10: QFT "embeds K_n, forcing W = Θ(n)", so tensor networks cannot beat the FFT.**
  No source given, and it sits badly with the known result that the (approximate) QFT circuit
  can be simulated efficiently by tensor contraction (Aharonov, Landau & Makowsky 2006). Leave out
  unless checked against a source.
- **Item 11: "strictly C_s > C for any connected network".** The per-step formula
  C_s(v) = 2^{|s∖(s_l∪s_r)|}·C(v) is correct and gives C_s ≥ C. The strict inequality is not
  proved and is not needed.
- **Item 10: "W ≥ L is the entanglement area law".** A graph cut bound, not the area law. The
  analogy is fine as intuition, but should not be stated as the same thing.
- **Item 8: "#3-COL is #P-complete (Valiant 1979)".** The claim is right, the attribution is
  loose. Cite a source that states it for colourings before adding.
- The prose around the learning path contains filler numbers ("L = 53 as in Sycamore",
  "treewidth exceeds 40", "petabytes") with no source. Do not copy it.

## Verdict

As a first pass it is useful. Every concrete code finding (items 1–4, 13, 14) is real and
cheap to fix. The plan restructure (item 6) and the explanations for newcomers (items 5, 7, 8, 10,
11) are good ideas. But the physics and maths prose must be rewritten and sourced rather than
pasted: it contains one plain physics error (item 12) and several unsourced overclaims, all
presented with the same confidence as the correct parts. None of the three panellists caught
these in the debate rounds.
