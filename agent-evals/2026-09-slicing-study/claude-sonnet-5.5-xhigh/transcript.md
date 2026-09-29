The study is finished. It answers both parts of the question, and everything is in the working directory (`README.md`, the code, the pytest file and one PNG).

**Correctness.**
- All 28 pytest checks pass. On four small circuits (7–9 qubits, 1D and 2D), the unsliced contraction, the sum over slices and the state-vector amplitude agree to about 7e-17, against the 1e-10 requirement. The gates are checked to be unitary and Haar-distributed.
- The largest intermediate and the multiply-add counts I report match what the contraction actually executes.
- Six deliberately injected bugs were all caught by the tests. That check was a scratch copy and is not saved.

**How much extra computation does slicing cost?** It depends heavily on geometry and on how much memory you save. The table below uses path-finder seed 0; the README table also has min–max over three seeds.

| Memory saving | 2D grids (6×6 and 8×8) | 1D brickwork (30q and 36q) |
|---|---|---|
| 4× smaller | 1.06–1.11× FLOPs | 2.8–3.74× (36q), 14–122× (30q) |
| 16× smaller | 1.24–1.52× | 10.3–31.6× (36q), 541–4.95e5× (30q) |
| 256× smaller | 5.47–74.8× (6×6), 16.9× (8×8) | 4.23e6× (36q), 9.31e5–8.74e17× (30q) |

Each row spans the best and worst cost-aware strategy I ran, with the same tree for the fixed-tree strategies and a re-optimised tree for `ctg-reconf`. Two cells come from one strategy only, since `ctg-reconf` did not run for 256× on the large instances. The overhead per halving of memory grows as more is saved.

**Does the choice of indices matter?** Yes, by orders of magnitude:
- Picking bonds at random needs 44–433 slices. Picking random bonds from the largest tensors is far worse than my cost-aware greedy rule. For example, 1.22e6× versus 16.9× at 256× smaller on the 8×8 grid.
- Re-optimising the contraction order while slicing (cotengra's `slice_and_reconfigure`) helped most in 1D, e.g. 541× versus 4.95e5× at 16× smaller on the 30-qubit circuit.
- The starting contraction order also shifts overheads by more than 10× across three path-finder seeds.
- A tree optimised only for memory, with no slicing, never got below about half the memory.

**Things you should know:**
- **Runtime.** It runs the four instances in 4 parallel processes, taking 3 m 52 s wall time on this 8-core machine. Run one after another it would take about 9 minutes, longer than your "few minutes" target.
- **Weak path finder.** `kahypar` and `optuna` aren't installed, so the path finder is random greedy plus reconfiguration and its trees are not optimal. Overheads are relative to those trees.
- **Size and scope.** Only 4 instances and 3 seeds. `slice_and_reconfigure` ran on one seed only, and only up to 16× on the large instances. Counts assume no reuse between slices, so real overheads could be lower.
- **Sycamore-like.** The 2D circuits imitate only Sycamore's ABCDCDAB layer order, on a plain square lattice with Haar gates and no single-qubit gates.
- **Code size and skill tooling.** The total is exactly 300 physical lines including blanks and comments. I read the dataviz skill's files and ran its palette validator, which sit outside this directory. The chart has a legend and per-series markers but no direct line labels.
- **References.** They are cited from memory and I did not look them up. The four I cite are Mezzadri (math-ph/0609050), Markov & Shi (quant-ph/0511069), Gray & Kourtis (2002.01935) and Arute et al. (1910.11333). I did not cite the original index-fixing papers because I'm unsure of their IDs.

Files are in `<working directory>`:
- README.md
- overhead_vs_memory.png
- tnslice.py
- run_study.py
- test_tnslice.py
- results.json
- results.md
- run_study.log
- pytest_output.txt
