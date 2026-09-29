"""Contraction order decides cost; the best achievable cost is set by the graph (treewidth).

Sources: Gray & Kourtis, arXiv:2002.01935, Sec. 2, Eqs. (2), (3), (5): contraction width W
         and contraction cost C of a contraction tree.
         Markov & Shi, arXiv:quant-ph/0511069, Theorem 1.1: optimal cost is exp(O(treewidth)).
A network is a list of (tensor, legs) pairs, as in topics 00-01. W and C only read the legs and
the tensor shapes, so no intermediate tensor is built. A contraction path is in opt_einsum
format: each step (i, j) removes items i and j and appends their result.
"""
import math
import numpy as np
import opt_einsum as oe
from opt_einsum.paths import DynamicProgramming


def ring(n):
    """Legs of n tensors on a cycle, each with a left and a right leg."""
    return [((i - 1) % n, i) for i in range(n)]


def grid(L):
    """Legs of an L x L square lattice, open boundary; one label per lattice edge."""
    edges = {}
    for r in range(L):
        for c in range(L):
            if c + 1 < L:
                edges[(r, c), (r, c + 1)] = len(edges)
            if r + 1 < L:
                edges[(r, c), (r + 1, c)] = len(edges)
    return [tuple(l for (u, v), l in edges.items() if (r, c) in (u, v))
            for r in range(L) for c in range(L)]


def random_network(legs, dims, rng):
    """Random complex tensors on the given legs; dims maps each label to its dimension."""
    shapes = [[dims[l] for l in t] for t in legs]
    return [(rng.normal(size=s) + 1j * rng.normal(size=s), tuple(t)) for s, t in zip(shapes, legs)]


def path_width_cost(network, path):
    """Width W = max_v log2(size of tensor v); cost C = sum over steps of prod of dims involved.

    Sizes are exact integer products (summing log2 of dims is off by rounding when a dim is not
    a power of 2); only W is turned into log2 at the end.
    """
    dims = {l: d for T, t in network for l, d in zip(t, T.shape)}
    size = lambda s: math.prod(dims[l] for l in s)
    items = [frozenset(t) for _, t in network]
    W, C = max(size(t) for t in items), 0
    for i, j in path:
        a, b = items[i], items[j]
        items = [t for k, t in enumerate(items) if k not in (i, j)]
        new = a ^ b                                   # Eq. (1): shared legs disappear
        C += size(a | b)                              # Eq. (5): vertex congestion
        W = max(W, size(new))                         # Eqs. (2)-(3): edge congestion
        items.append(new)
    return math.log2(W), C


def contract_along(network, path):
    """Contract pairwise along the path; each step is a small einsum over relabelled legs."""
    items = list(network)
    for i, j in path:
        (A, a), (B, b) = items[i], items[j]
        items = [x for k, x in enumerate(items) if k not in (i, j)]
        out = tuple(l for l in a + b if (l in a) != (l in b))
        ids = {l: k for k, l in enumerate(dict.fromkeys(a + b))}
        items.append((np.einsum(A, [ids[l] for l in a], B, [ids[l] for l in b],
                                [ids[l] for l in out]), out))
    return items[0][0]


# --- Orders ------------------------------------------------------------------------

def bubbling_path(n_tensors, order):
    """Absorb tensors one at a time in the given order (a 'bubbling', Bridgeman & Chubb 1.4)."""
    pos = list(range(n_tensors))                      # what sits at each current position
    path = [tuple(sorted((pos.index(order[0]), pos.index(order[1]))))]
    pos = [p for p in pos if p not in order[:2]] + ["acc"]
    for t in order[2:]:
        path.append((pos.index(t), len(pos) - 1))
        pos = [p for p in pos if p != t]
    return path


def optimiser_path(network, optimize):
    eq = ",".join("".join(oe.get_symbol(l) for l in t) for _, t in network) + "->"
    return oe.contract_path(eq, *[T.shape for T, _ in network], shapes=True,
                            optimize=optimize)[0]


ORDERS = ["random bubbling", "row-by-row bubbling", "greedy", "optimal (min W)"]


def orders(network, rng):
    n = len(network)
    return dict(zip(ORDERS, [
        bubbling_path(n, list(rng.permutation(n))),
        bubbling_path(n, list(range(n))),
        optimiser_path(network, "greedy"),
        optimiser_path(network, DynamicProgramming(minimize="size")),
    ]))


if __name__ == "__main__":
    rng = np.random.default_rng(0)                    # draws the random bubbling orders
    entries = np.random.default_rng(1)                # tensor entries; W and C ignore them
    for name, nets in [("ring", [(n, ring(n)) for n in (8, 16, 32, 64)]),
                       ("grid", [(f"{L}x{L}", grid(L)) for L in (2, 3, 4, 5, 6)])]:
        print(f"\n{name} (bond dim 2): width W / log2 cost C")
        print(f"{'size':>6} | " + " | ".join(f"{k:>22}" for k in ORDERS))
        for size, legs in nets:
            network = random_network(legs, {l: 2 for t in legs for l in t}, entries)
            res = [path_width_cost(network, p) for p in orders(network, rng).values()]
            print(f"{size:>6} | " + " | ".join(f"{f'W={w:.0f}, log2 C={math.log2(c):.1f}':>22}"
                                            for w, c in res))
