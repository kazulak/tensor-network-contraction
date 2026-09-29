import math
import numpy as np
import pytest
from order import (ring, grid, random_network, path_width_cost, contract_along, bubbling_path,
                   optimiser_path, orders, DynamicProgramming)

rng = np.random.default_rng(3)
MIN_W = DynamicProgramming(minimize="size")


def network_of(legs, d=2):
    return random_network(legs, {l: d for t in legs for l in t}, rng)


@pytest.mark.parametrize("legs", [ring(8), grid(3)])
def test_every_order_gives_the_same_value(legs):
    network = network_of(legs)
    args = [x for T, t in network for x in (T, list(t))]
    reference = np.einsum(*args, [])
    for path in orders(network, rng).values():
        assert np.isclose(contract_along(network, path), reference)


def test_matrix_chain_width_and_cost():
    network = random_network([(0, 1), (1, 2), (2, 3)], {0: 2, 1: 32, 2: 4, 3: 64}, rng)
    W, C = path_width_cost(network, [(0, 1), (0, 1)])           # (AB)C
    assert C == 2 * 32 * 4 + 2 * 4 * 64
    assert W == math.log2(4 * 64)                               # largest tensor is C itself


def test_cost_is_exact_integer():
    # Summing log2 of the dims gave C = 53.999999999999986 here, not 54.
    W, C = path_width_cost(network_of([(0, 1), (1, 2), (2, 3)], d=3), [(0, 1), (0, 1)])
    assert C == 27 + 27 and isinstance(C, int)


def test_bubbling_path_follows_the_order():
    # Order 2, 0, 3, 1: contract items 0 and 2 -> list [1, 3, acc]; absorb 3 at position 1
    # -> list [1, acc]; absorb 1 at position 0. Four tensors take three pairwise steps.
    assert bubbling_path(4, [2, 0, 3, 1]) == [(0, 2), (1, 2), (0, 1)]


def test_ring_width_is_constant():
    for n in (4, 8, 16, 32):
        network = network_of(ring(n))
        W, _ = path_width_cost(network, optimiser_path(network, MIN_W))
        assert W == 2


def test_grid_width_grows_with_side():
    # Bridgeman & Chubb 1.4: any order has an intermediate whose boundary is ~ the side L.
    for L in (2, 3, 4, 5):
        network = network_of(grid(L))
        W_opt, _ = path_width_cost(network, optimiser_path(network, MIN_W))
        W_row, _ = path_width_cost(network, bubbling_path(L * L, list(range(L * L))))
        assert L <= W_opt <= W_row <= L + 1
