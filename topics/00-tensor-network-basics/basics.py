"""Tensor network basics: pairwise contraction, bubbling order, and counting colourings.

Source: Bridgeman & Chubb, arXiv:1603.03039, Sections 1.3-1.5.
A tensor is a numpy array plus a tuple of integer labels, one per leg. Every label
appears on at most two tensors; a shared label is summed over.
"""
import itertools
import numpy as np
import opt_einsum as oe


def contract_pair(A, a, B, b):
    """Contract two labelled tensors as a matrix product: transpose -> reshape -> matmul."""
    shared = [l for l in a if l in b]
    free_a = [l for l in a if l not in shared]
    free_b = [l for l in b if l not in shared]
    # Move shared legs to the end of A and the front of B, then flatten to matrices.
    At = A.transpose([a.index(l) for l in free_a + shared])
    Bt = B.transpose([b.index(l) for l in shared + free_b])
    k = int(np.prod([A.shape[a.index(l)] for l in shared]))
    M = At.reshape(-1, k) @ Bt.reshape(k, -1)
    shape = [A.shape[a.index(l)] for l in free_a] + [B.shape[b.index(l)] for l in free_b]
    return M.reshape(shape), tuple(free_a + free_b)


def pair_cost(a, b, dims):
    """Multiply-adds of one pairwise contraction = product of dims of all legs involved."""
    return int(np.prod([dims[l] for l in set(a) | set(b)]))


def bubble(tensors, order):
    """Contract tensors[order[0]], then absorb the others one at a time (a 'bubbling').

    Returns the result, the largest rank of the stored tensor along the way, and the total
    number of multiply-adds.
    """
    dims = {l: d for T, t in tensors for l, d in zip(t, T.shape)}
    T, t = tensors[order[0]]
    max_rank, cost = len(t), 0
    for i in order[1:]:
        cost += pair_cost(t, tensors[i][1], dims)
        T, t = contract_pair(T, t, *tensors[i])
        max_rank = max(max_rank, len(t))
    return T, max_rank, cost


def einsum_reference(tensors):
    """Contract the whole network with opt_einsum (the reference answer).

    Not np.einsum(optimize="greedy"): its path never builds an intermediate larger than the
    largest input, so on the Petersen network with q=4 it ends in one 15-index einsum (25 s).
    """
    args = [x for T, t in tensors for x in (T, list(t))]
    return oe.contract(*args, [], optimize="greedy")


# --- Ladder network (Section 1.4) --------------------------------------------------

def ladder(n, d, rng):
    """Closed ladder of length n: top row 0..n-1, bottom row n..2n-1, rungs between them."""
    top = lambda i: ("t", i)
    bot = lambda i: ("b", i)
    rung = lambda i: ("r", i)
    labels = {}
    tensors = []
    for row, h in (("top", top), ("bot", bot)):
        for i in range(n):
            legs = [rung(i)]
            if i > 0:
                legs.append(h(i - 1))  # bond to the left neighbour
            if i < n - 1:
                legs.append(h(i))      # bond to the right neighbour
            ids = tuple(labels.setdefault(l, len(labels)) for l in legs)
            shape = (d,) * len(ids)
            tensors.append((rng.normal(size=shape) + 1j * rng.normal(size=shape), ids))
    return tensors


def ladder_orders(n):
    """Bad order: along the top, then back along the bottom. Good order: rung by rung."""
    along = list(range(n)) + list(range(2 * n - 1, n - 1, -1))
    rungs = [i for k in range(n) for i in (k, n + k)]
    return along, rungs


# --- Counting q-colourings (Section 1.5) -------------------------------------------

def colouring_network(edges, n_vertices, q):
    """'e' (all legs equal) on each vertex, 'n' (legs differ) on each edge midpoint."""
    tensors, legs = [], {v: [] for v in range(n_vertices)}
    for k, (u, v) in enumerate(edges):
        lu, lv = 2 * k, 2 * k + 1                  # vertex u -- lu -- n -- lv -- vertex v
        tensors.append((np.ones((q, q)) - np.eye(q), (lu, lv)))
        legs[u].append(lu)
        legs[v].append(lv)
    for v in range(n_vertices):
        deg = len(legs[v])
        e = np.zeros((q,) * deg)
        for c in range(q):
            e[(c,) * deg] = 1.0
        tensors.append((e, tuple(legs[v])))
    return tensors


def count_colourings_brute(edges, n_vertices, q):
    return sum(all(c[u] != c[v] for u, v in edges)
               for c in itertools.product(range(q), repeat=n_vertices))


PETERSEN = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)] \
         + [(5 + i, 5 + (i + 2) % 5) for i in range(5)]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Ladder (d=2): largest stored rank and multiply-adds per bubbling, Eqs. (1.14)-(1.17)")
    print(" n | along top, then bottom: rank, cost | rung by rung: rank, cost")
    for n in range(2, 11):
        net = ladder(n, 2, rng)
        along, rungs = ladder_orders(n)
        (x1, r1, c1), (x2, r2, c2) = bubble(net, along), bubble(net, rungs)
        assert np.isclose(x1, x2) and np.isclose(x1, einsum_reference(net))
        print(f"{n:2d} | {r1:27d}, {c1:5d} | {r2:17d}, {c2:5d}")

    print("\nNumber of q-colourings via tensor network, Eq. (1.19)")
    for name, edges, nv in [("triangle", [(0, 1), (1, 2), (2, 0)], 3),
                            ("cycle C6", [(i, (i + 1) % 6) for i in range(6)], 6),
                            ("Petersen", PETERSEN, 10)]:
        for q in (2, 3, 4):
            tn = einsum_reference(colouring_network(edges, nv, q)).real
            print(f"{name:9s} q={q}: TN = {tn:8.0f}   brute force = {count_colourings_brute(edges, nv, q)}")
