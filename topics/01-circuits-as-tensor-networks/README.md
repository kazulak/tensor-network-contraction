# 01 - Quantum circuits as tensor networks

**Sources:**
- Nielsen & Chuang, *Quantum Computation and Quantum Information*: Ch. 4 (gates) and Sec. 5.1 (QFT).
- Markov & Shi, *Simulating quantum computation by contracting tensor networks*,
  arXiv:quant-ph/0511069, Sec. 3.
- Mezzadri, *How to generate random matrices from the classical compact groups*,
  arXiv:math-ph/0609050 (Haar-random unitaries).

**Idea:** A circuit becomes a tensor network by turning each gate into a tensor and each wire
segment into an index. Closing the network with ⟨x| gives one amplitude ⟨x|C|0…0⟩. Leaving the
final legs open gives the full output state.

## What is here

`circuits.py` (~130 lines):

- Complex gates (`H, S, T, CNOT, SWAP`, controlled phase) and Haar-random U(4) gates.
- Example circuits: GHZ, the Nielsen & Chuang QFT, and a brickwork of random gates.
- `statevector`: a reference simulator.
- `to_network` + `contract`: the same circuit as a labelled tensor network, contracted with
  `opt_einsum`.

Qubit 0 is the leftmost (most significant) bit.

## Checks

`pytest` (7 tests):

- Every gate is unitary, and the random gates are genuinely complex.
- GHZ amplitudes are exactly 1/√2 on |0…0⟩ and |1…1⟩, and 0 elsewhere.
- **Closed network = state vector:** every amplitude of a random 5-qubit circuit matches.
- **Open network = state vector:** the whole output state of a 7-qubit circuit matches.
- **QFT = DFT:** the circuit's matrix equals 2^(−n/2)·exp(2πi·xk/2ⁿ) for n = 1…5.

These checks target two easy mistakes: real orthogonal matrices passed off as "Haar-random",
and a "QFT" that is not unitary.

```bash
python circuits.py
pytest
```

**Author:** code by Claude Code (Claude Opus 5.5), directed by T. Kazulak. Human review: pending.
