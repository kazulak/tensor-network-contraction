"""Correctness checks. Run `python -m pytest -q -s test_tnslice.py` to see the observed deviations."""
import random
import warnings
import numpy as np
import pytest
import cotengra as ctg
from tnslice import *

warnings.filterwarnings("ignore")
CIRCUITS = {"1D n=8 d=6": (8, brickwork(8, 6)), "1D n=7 d=9": (7, brickwork(7, 9)),
            "2D 3x3 d=8": (9, grid(3, 3, 8)), "2D 2x4 d=6": (8, grid(2, 4, 6))}

def setup(n, layers, seed=0):
    rng = np.random.default_rng(seed)
    gates, x = random_gates(layers, rng), rng.integers(0, 2, n)
    random.seed(seed), np.random.seed(seed)
    ts, ins = network(n, layers, gates, x)
    return gates, x, ts, ins, find_tree(ins, repeats=16)

def chosen_slices(how, ins, tree, m):   # (slice bitmask, path, log2 target size) chosen by strategy `how` for a 2^m memory reduction
    ms, path = masks(ins), tree.get_ssa_path()
    tgt, back = stats(path, ms)[1] - m, {ctg.get_symbol(l): l for i in ins for l in i}
    if how in ("uniform", "memory", "greedy"):
        return slice_to(path, ms, tgt, how, np.random.default_rng(0)), path, tgt
    t = tree.slice(target_size=2 ** tgt) if how == "ctg-slice" else tree.slice_and_reconfigure(target_size=2 ** tgt)
    return sum(1 << back[s] for s in t.sliced_inds), t.get_ssa_path(), tgt

def test_gates_are_haar_unitaries():
    rng = np.random.default_rng(0)
    us = [haar_u4(rng) for _ in range(4000)]
    assert all(u.dtype == np.complex128 and np.abs(u.conj().T @ u - np.eye(4)).max() < 1e-13 for u in us)
    # Haar: E|Tr U|^2 = 1; a QR without the phase fix fails this
    assert abs(np.mean([abs(np.trace(u)) ** 2 for u in us]) - 1) < 0.08

def test_layers_are_nearest_neighbour_matchings():
    for layers, adj in [(brickwork(9, 7), lambda a, b: b - a == 1), (grid(3, 4, 8), lambda a, b: b - a == 4 or (b - a == 1 and a % 4 < 3))]:
        for layer in layers:
            qs = [q for pair in layer for q in pair]
            assert len(qs) == len(set(qs)) and all(adj(a, b) for a, b in layer)

def test_statevector_matches_dense_matrices():
    rng = np.random.default_rng(1)
    u1, u2 = haar_u4(rng), haar_u4(rng)
    dense = np.kron(np.eye(2), u2) @ np.kron(u1, np.eye(2))       # gate 1 on qubits (0,1), then gate 2 on (1,2)
    for x in range(8):
        assert abs(statevector_amp(3, [[(0, 1)], [(1, 2)]], [[u1], [u2]], [x >> 2 & 1, x >> 1 & 1, x & 1]) - dense[x, 0]) < 1e-14

@pytest.mark.parametrize("name", CIRCUITS)
def test_network_matches_statevector_and_counts(name):
    n, layers = CIRCUITS[name]
    gates, x, ts, ins, tree = setup(n, layers)
    path, ms = tree.get_ssa_path(), masks(ins)
    assert all(t.dtype == np.complex128 for t in ts)
    v, macs, big = contract(ts, ins, path)
    sv = statevector_amp(n, layers, gates, x)
    print(f"\n{name}: |amp|={abs(sv):.3e}  |unsliced - statevector|={abs(v - sv):.2e}")
    assert abs(v - sv) < 1e-10 and abs(v) > 1e-6
    assert (macs, big) == (stats(path, ms)[0], 2 ** stats(path, ms)[1])   # counted == executed
    assert (macs, big) == (tree.contraction_cost(), tree.max_size())      # counted == cotengra's own count

@pytest.mark.parametrize("name", CIRCUITS)
@pytest.mark.parametrize("how", ["uniform", "memory", "greedy", "ctg-slice", "ctg-reconf"])
def test_sum_over_slices_equals_unsliced_and_statevector(name, how):
    n, layers = CIRCUITS[name]
    gates, x, ts, ins, tree = setup(n, layers)
    ms = masks(ins)
    S, path, tgt = chosen_slices(how, ins, tree, 2)
    assert stats(path, ms, S)[1] <= tgt                            # strategy reached its memory target (counted)
    S = sum(bits(S)[:7])          # sum over slices is exact for ANY index set: execute all 2^k slices of <= 7 chosen indices
    sliced = [l for l in range(S.bit_length()) if S >> l & 1]
    assert len(sliced) >= 2
    macs1, big1, _ = stats(path, ms, S)
    v_all, _, _ = contract(ts, ins, path)
    v_sl, macs, big = contract(ts, ins, path, sliced)
    sv = statevector_amp(n, layers, gates, x)
    print(f"\n{name} {how} k={len(sliced)}: |sliced sum - unsliced|={abs(v_sl - v_all):.2e}  |sliced sum - statevector|={abs(v_sl - sv):.2e}")
    assert abs(v_sl - v_all) < 1e-10 and abs(v_sl - sv) < 1e-10
    assert macs == macs1 << len(sliced) and big == 2 ** big1      # counted cost == executed cost over all 2^k slices

def test_study_scale_instance_sliced_equals_unsliced():
    _, _, ts, ins, tree = setup(36, grid(6, 6, 12))    # 36-qubit study instance (no state vector): whole greedy slice set, 16x smaller
    S, path, tgt = chosen_slices("greedy", ins, tree, 4)
    sliced = [l for l in range(S.bit_length()) if S >> l & 1]
    (v0, macs0, _), (v1, macs1, big1) = contract(ts, ins, path), contract(ts, ins, path, sliced)
    print(f"\n2D 6x6 d12, k={len(sliced)}: |amp|={abs(v0):.3e} rel. diff={abs(v1 - v0) / abs(v0):.2e} MACs sliced/unsliced={macs1 / macs0:.3f}")
    assert abs(v1 - v0) < 1e-12 * abs(v0) and big1 <= 2 ** tgt and macs1 == stats(path, masks(ins), S)[0] << len(sliced)
