"""Amplitude tensor networks of random circuits: build, count contraction cost, choose slices, contract."""
import functools
import itertools
import math
import operator
import numpy as np
import cotengra as ctg

# ---------- circuits ----------
def haar_u4(rng):
    """Haar-random U(4): QR of a complex Ginibre matrix, phases of diag(R) removed (Mezzadri, arXiv:math-ph/0609050)."""
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))

def brickwork(n, depth):
    """Nearest-neighbour gates on a line, alternating even/odd bonds."""
    return [[(q, q + 1) for q in range(l % 2, n - 1, 2)] for l in range(depth)]

def grid(lx, ly, depth):
    """Sycamore-like layers on an lx x ly grid: coupler patterns in the order ABCDCDAB (repeated), with
    A/B/C/D = horizontal-even / vertical-even / horizontal-odd / vertical-odd nearest-neighbour bonds."""
    q = lambda r, c: r * ly + c
    h = lambda p: [(q(r, c), q(r, c + 1)) for r in range(lx) for c in range(p, ly - 1, 2)]
    v = lambda p: [(q(r, c), q(r + 1, c)) for r in range(p, lx - 1, 2) for c in range(ly)]
    pat = dict(A=h(0), B=v(0), C=h(1), D=v(1))
    return [pat[k] for k in ("ABCDCDAB" * depth)[:depth]]

def random_gates(layers, rng):
    return [[haar_u4(rng) for _ in layer] for layer in layers]

def statevector_amp(n, layers, gates, x):
    """<x|C|0...0> by applying each gate to the full 2^n state vector."""
    psi = np.zeros((2,) * n, complex)
    psi[(0,) * n] = 1
    for layer, us in zip(layers, gates):
        for (a, b), u in zip(layer, us):
            psi = np.moveaxis(np.tensordot(u.reshape(2, 2, 2, 2), psi, axes=([2, 3], [a, b])), [0, 1], [a, b])
    return psi[tuple(x)]

# ---------- tensor network ----------
def network(n, layers, gates, x):
    """Gate tensors (out_a, out_b, in_a, in_b), |0> inputs and <x| outputs absorbed; every index is a dim-2 bond in two tensors."""
    wire, k, ts, ins = list(range(n)), n, [], []
    fixed = {q: np.array([1, 0], complex) for q in range(n)}
    for layer, us in zip(layers, gates):
        for (a, b), u in zip(layer, us):
            ts.append(u.reshape(2, 2, 2, 2))
            ins.append((k, k + 1, wire[a], wire[b]))
            wire[a], wire[b], k = k, k + 1, k + 2
    assert all(w >= n for w in wire), "every qubit needs at least one gate"
    fixed.update({wire[q]: np.eye(2, dtype=complex)[x[q]] for q in range(n)})
    for j, i in enumerate(ins):
        for ax in reversed(range(4)):
            if i[ax] in fixed:
                ts[j] = np.tensordot(ts[j], fixed[i[ax]], axes=([ax], [0]))
    return ts, [tuple(l for l in i if l not in fixed) for i in ins]

def find_tree(ins, repeats=64, minimize="flops"):
    """Contraction tree from cotengra's hyper-optimizer (greedy candidates; kahypar/optuna are not installed)."""
    sym = ctg.get_symbol
    opt = ctg.HyperOptimizer(methods=["greedy"], max_repeats=repeats, parallel=False, minimize=minimize)
    return opt.search([''.join(sym(l) for l in i) for i in ins], '', {sym(l): 2 for i in ins for l in i})

# ---------- cost counting (one bitmask per tensor: bit l set <=> index l present) ----------
def masks(ins):
    return [sum(1 << l for l in i) for i in ins]

def bits(m):
    return [1 << i for i in range(m.bit_length()) if m >> i & 1]

def stats(path, ms, S=0):
    """Cost of ONE slice of the pairwise contraction `path` (SSA pairs) with index-bitmask S sliced: (multiply-adds,
    log2 largest intermediate, node masks). Contracting A,B costs 2^|A u B|; every index sits in exactly two tensors
    and the output is a scalar, so the result is A xor B. Cost over all slices = macs << popcount(S)."""
    nodes, macs = [m & ~S for m in ms], 0
    for a, b in path:
        macs += 1 << (nodes[a] | nodes[b]).bit_count()
        nodes.append(nodes[a] ^ nodes[b])
    return macs, max(c.bit_count() for c in nodes[len(ms):]), nodes

def slice_to(path, ms, target, how, rng=None):
    """Add sliced indices until the largest intermediate has <= 2**target elements. 'uniform': random index; 'memory':
    random index of an over-target intermediate; 'greedy': such an index with least log2-cost rise per unit of excess removed."""
    S, allbits, n0 = 0, functools.reduce(operator.or_, ms), len(ms)
    excess = lambda nodes: sum(max(0, c.bit_count() - target) for c in nodes[n0:])
    while True:
        macs, big, nodes = stats(path, ms, S)
        if big <= target:
            return S
        over = [c for c in nodes[n0:] if c.bit_count() > target]
        pool = bits(allbits & ~S if how == "uniform" else functools.reduce(operator.or_, over))
        if how == "greedy":
            def key(c):
                m2, _, nd = stats(path, ms, S | c)
                d = excess(nodes) - excess(nd)
                return math.log2(2 * m2 / macs) / d, -d
            S |= min(pool, key=key)
        else:
            S |= pool[rng.integers(len(pool))]

# ---------- execution ----------
def contract(ts, ins, path, sliced=()):
    """Sum over all values of the sliced indices of the contraction along `path`: (value, multiply-adds done, largest intermediate)."""
    total, macs, big = 0, 0, 0
    for vals in itertools.product((0, 1), repeat=len(sliced)):
        fix = dict(zip(sliced, vals))
        A = [t[tuple(fix.get(l, slice(None)) for l in i)] for t, i in zip(ts, ins)]
        I = [[l for l in i if l not in fix] for i in ins]
        for a, b in path:
            sh = [l for l in I[a] if l in I[b]]
            A.append(np.tensordot(A[a], A[b], ([I[a].index(l) for l in sh], [I[b].index(l) for l in sh])))
            I.append([l for l in I[a] if l not in sh] + [l for l in I[b] if l not in sh])
            macs += 2 ** len(set(I[a]) | set(I[b]))
            big = max(big, A[-1].size)
        total = total + A[-1]
    return complex(total), macs, big
