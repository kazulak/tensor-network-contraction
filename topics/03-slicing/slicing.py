"""Slicing: fix some indices, contract the pieces independently, sum. Less memory, more work.

Source: Gray & Kourtis, arXiv:2002.01935, Sec. 4.7.1. Slicing a set of indices s gives
d_sliced = prod_{e in s} w(e) independent networks; each costs >= C / d_sliced, so the total
sliced cost C_s >= C. Indices are chosen greedily, one at a time, until a target width is met.
The network is a single amplitude <x|C|0..0> of a random 2D-grid circuit with Haar U(4) gates.
"""
import itertools
import math
import numpy as np
import opt_einsum as oe


def haar_u4(rng):
    Z = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def grid_circuit(rows, cols, depth, rng):
    """Sycamore-like: each layer couples nearest neighbours in one of four patterns."""
    q = lambda r, c: r * cols + c
    patterns = [[(q(r, c), q(r, c + 1)) for r in range(rows) for c in range(p, cols - 1, 2)]
                for p in (0, 1)]
    patterns += [[(q(r, c), q(r + 1, c)) for r in range(p, rows - 1, 2) for c in range(cols)]
                 for p in (0, 1)]
    return [(haar_u4(rng), pair) for layer in range(depth) for pair in patterns[layer % 4]]


def statevector(circ, n):
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1
    for U, qs in circ:
        psi = np.moveaxis(np.tensordot(U.reshape(2, 2, 2, 2), psi, axes=((2, 3), qs)), (0, 1), qs)
    return psi.reshape(-1)


def amplitude_network(circ, n, bitstring):
    """Tensors and integer labels for <bitstring|C|0..0> (as in topic 01)."""
    wire, nxt = list(range(n)), n
    tensors = [np.array([1, 0], dtype=complex) for _ in range(n)]
    labels = [(q,) for q in range(n)]
    for U, (a, b) in circ:
        tensors.append(U.reshape(2, 2, 2, 2))
        labels.append((nxt, nxt + 1, wire[a], wire[b]))
        wire[a], wire[b], nxt = nxt, nxt + 1, nxt + 2
    for q, bit in enumerate(bitstring):
        tensors.append(np.eye(2, dtype=complex)[int(bit)])
        labels.append((wire[q],))
    return tensors, labels


def find_path(labels):
    eq = ",".join("".join(oe.get_symbol(l) for l in t) for t in labels) + "->"
    return oe.contract_path(eq, *[(2,) * len(t) for t in labels], shapes=True,
                            optimize="greedy")[0]


def width_cost(labels, path, sliced=()):
    """W and total cost C_s of the tree, with the sliced indices removed (all dims are 2)."""
    items = [frozenset(t) - set(sliced) for t in labels]
    W, C = max(len(t) for t in items), 0
    for i, j in path:
        a, b = items[i], items[j]
        items = [t for k, t in enumerate(items) if k not in (i, j)] + [a ^ b]
        W, C = max(W, len(a ^ b)), C + 2 ** len(a | b)
    return W, C * 2 ** len(sliced)                     # d_sliced copies of the sliced network


def slice_greedy(labels, path, target_W):
    """Add the index giving the smallest (width, total cost) until the width is <= target."""
    sliced = []
    candidates = sorted({l for t in labels for l in t})
    while width_cost(labels, path, sliced)[0] > target_W:
        best = min((l for l in candidates if l not in sliced),
                   key=lambda l: width_cost(labels, path, sliced + [l]))
        sliced.append(best)
    return sliced


def slice_random(labels, path, target_W, rng):
    candidates = list(rng.permutation(sorted({l for t in labels for l in t})))
    sliced = []
    while width_cost(labels, path, sliced)[0] > target_W:
        sliced.append(candidates.pop())
    return sliced


def contract_sliced(tensors, labels, path, sliced):
    """Sum over all values of the sliced indices of the contraction along the same path."""
    total = 0
    for values in itertools.product((0, 1), repeat=len(sliced)):
        fix = dict(zip(sliced, values))
        items = []
        for T, t in zip(tensors, labels):
            T = T[tuple(fix.get(l, slice(None)) for l in t)]
            items.append((T, tuple(l for l in t if l not in fix)))
        for i, j in path:
            (A, a), (B, b) = items[i], items[j]
            items = [x for k, x in enumerate(items) if k not in (i, j)]
            out = tuple(l for l in a + b if (l in a) != (l in b))
            ids = {l: k for k, l in enumerate(dict.fromkeys(a + b))}
            items.append((np.einsum(A, [ids[l] for l in a], B, [ids[l] for l in b],
                                    [ids[l] for l in out]), out))
        total += items[0][0]
    return total


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols, depth = 4, 4, 12
    circ = grid_circuit(rows, cols, depth, rng)
    _, labels = amplitude_network(circ, rows * cols, "0" * rows * cols)
    path = find_path(labels)
    W0, C0 = width_cost(labels, path)
    print(f"{rows}x{cols} grid, depth {depth}: {len(labels)} tensors, W = {W0}, log2 C = {math.log2(C0):.2f}")
    print(" target W | memory saved | greedy: #sliced, C_s/C | random, median of 20: #sliced, C_s/C")
    for target in range(W0 - 1, W0 - 9, -1):
        g = slice_greedy(labels, path, target)
        rand = [slice_random(labels, path, target, rng) for _ in range(20)]
        r_over = np.median([width_cost(labels, path, s)[1] / C0 for s in rand])
        r_num = np.median([len(s) for s in rand])
        print(f" {target:8d} | {2 ** (W0 - target):11d}x | {len(g):14d}, {width_cost(labels, path, g)[1] / C0:7.2f}x"
              f" | {r_num:28.0f}, {r_over:7.1e}x")
