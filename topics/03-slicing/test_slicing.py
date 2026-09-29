import numpy as np
from slicing import (grid_circuit, statevector, amplitude_network, find_path, width_cost,
                     slice_greedy, slice_random, contract_sliced)


def test_sum_over_slices_equals_amplitude():
    circ = grid_circuit(3, 3, 6, np.random.default_rng(1))
    psi = statevector(circ, 9)
    network = amplitude_network(circ, 9, "0" * 9)
    path = find_path(network)                   # the legs do not depend on the bitstring
    sliced = slice_greedy(network, path, width_cost(network, path)[0] - 2)
    assert len(sliced) >= 3
    for x in (0, 5, 300):
        network = amplitude_network(circ, 9, format(x, "09b"))
        assert np.isclose(contract_sliced(network, path, []), psi[x])
        assert np.isclose(contract_sliced(network, path, sliced), psi[x])


def test_slicing_reaches_target_and_never_saves_work():
    circ = grid_circuit(3, 4, 8, np.random.default_rng(2))
    network = amplitude_network(circ, 12, "0" * 12)
    path = find_path(network)
    W0, C0 = width_cost(network, path)
    for target in (W0 - 1, W0 - 3):
        W, C = width_cost(network, path, slice_greedy(network, path, target))
        assert W <= target and C >= C0          # Gray & Kourtis 4.7.1: C_s >= C


def test_random_slicing_reaches_target_but_costs_more():
    circ = grid_circuit(3, 4, 8, np.random.default_rng(2))
    network = amplitude_network(circ, 12, "0" * 12)
    path = find_path(network)
    W0, C0 = width_cost(network, path)
    target = W0 - 1
    W_r, C_r = width_cost(network, path, slice_random(network, path, target, np.random.default_rng(3)))
    _, C_g = width_cost(network, path, slice_greedy(network, path, target))
    assert W_r <= target and C0 <= C_g <= C_r
