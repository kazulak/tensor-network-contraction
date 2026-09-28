import numpy as np
from slicing import (grid_circuit, statevector, amplitude_network, find_path, width_cost,
                     slice_greedy, contract_sliced)


def test_sum_over_slices_equals_amplitude():
    rng = np.random.default_rng(1)
    circ = grid_circuit(3, 3, 6, rng)
    psi = statevector(circ, 9)
    for x in (0, 5, 300):
        bits = format(x, "09b")
        tensors, labels = amplitude_network(circ, 9, bits)
        path = find_path(labels)
        sliced = slice_greedy(labels, path, width_cost(labels, path)[0] - 3)
        assert len(sliced) >= 3
        assert np.isclose(contract_sliced(tensors, labels, path, []), psi[x])
        assert np.isclose(contract_sliced(tensors, labels, path, sliced), psi[x])


def test_slicing_reaches_target_and_never_saves_work():
    circ = grid_circuit(3, 4, 8, np.random.default_rng(2))
    _, labels = amplitude_network(circ, 12, "0" * 12)
    path = find_path(labels)
    W0, C0 = width_cost(labels, path)
    for target in (W0 - 1, W0 - 3):
        W, C = width_cost(labels, path, slice_greedy(labels, path, target))
        assert W <= target and C >= C0          # Gray & Kourtis 4.7.1: C_s >= C
