# Topics

One idea per folder, taken from a named textbook or paper, implemented minimally and checked
against an exact reference. The rules are in [GUIDELINES.md](../GUIDELINES.md).

| # | Topic | Source | Status |
|---|---|---|---|
| 00 | [Tensor network basics](00-tensor-network-basics/) | Bridgeman & Chubb §1.3–1.5 | done |
| 01 | [Circuits as tensor networks](01-circuits-as-tensor-networks/) | Nielsen & Chuang Ch. 4, §5.1; Markov & Shi §3 | done |
| 02 | [Contraction order and cost](02-contraction-order-and-cost/) | Gray & Kourtis §2; Markov & Shi Thm 1.1 | done |
| 03 | [Slicing](03-slicing/) | Gray & Kourtis §4.7.1 | done |
| 04 | Parallel contraction | Huang et al. 2020; Williams et al. 2009 | planned |
| 05 | MPS, canonical form and SVD truncation | Orús 2014; Schollwöck 2011 | planned |
| 06 | MPS circuit simulation (TEBD) | Vidal 2003; Zhou et al. 2020 | planned |
| 07 | DMRG | Schollwöck 2011 | planned |

Each topic folder is self-contained (it only needs `numpy` and `opt_einsum`), so it can be
reimplemented in another language next to the Python version, e.g. `julia/` or `c/`.

```bash
pip install -r requirements.txt
pytest topics/
```

Full citations are in [references/](../references/).
