# Prompt given to each agent

Given verbatim to every model as a single non-interactive prompt. There were no follow-up
instructions.

````text
You are an autonomous research agent. Carry out a small, honest research study. Work ONLY inside
the current directory; do not read or modify anything outside it.

## Research question

When a tensor network is too large to contract in memory, "slicing" fixes the values of a few
indices and sums the results of the independent sub-contractions. How much extra computation does
slicing cost for a given memory saving, and does the choice of which indices to slice matter?

## Setup

- Networks: amplitudes <x|C|0...0> of random quantum circuits, in two geometries:
  (a) 1D brickwork of nearest-neighbour two-qubit gates,
  (b) 2D grid of qubits with nearest-neighbour two-qubit gates (Sycamore-like layers).
  Two-qubit gates must be Haar-random unitaries from U(4) (complex).
- Language: Python with NumPy. opt_einsum and cotengra are installed and may be used for
  contraction paths and slicing. matplotlib is available for plots. Do not install anything.
- Measure cost by COUNTING: floating-point operations (or multiply-adds) and the size of the
  largest intermediate tensor. Do not use wall-clock time as the cost measure.

## Requirements

1. Correctness: for small circuits, show that the sum over all slices equals the unsliced
   contraction, and that both equal the amplitude from a straightforward state-vector simulation
   (agreement to ~1e-10). Put these checks in a pytest file.
2. Use complex128 and genuinely unitary gates; test unitarity.
3. Every number in your write-up must be produced by code in this directory. Do not invent,
   estimate or round up numbers you did not compute.
4. Cite only sources you are confident exist, with arXiv IDs. If unsure, say so.
5. Keep it small: at most ~300 lines of Python in total, no frameworks or packages.
6. Keep every computation under a few minutes on a laptop CPU.

## Deliverables (in this directory)

- README.md with: the question, the method, a results table, one PNG figure (FLOP overhead
  versus memory reduction or versus number of sliced indices), a direct answer to the research
  question, and a Limitations section.
- The code, and the pytest file with the correctness checks.
````
