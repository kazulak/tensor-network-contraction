import math
import numpy as np
import pytest
from order import (ring, grid, path_width_cost, contract_along, bubbling_path, optimiser_path,
                   orders, random_tensors, DynamicProgramming)

rng = np.random.default_rng(3)
MIN_W = DynamicProgramming(minimize="size")


def dims_of(labels, d=2):
    return {l: d for t in labels for l in t}


@pytest.mark.parametrize("labels", [ring(8), grid(3)])
def test_every_order_gives_the_same_value(labels):
    dims = dims_of(labels)
    tensors = random_tensors(labels, dims, rng)
    args = [x for T, t in zip(tensors, labels) for x in (T, list(t))]
    reference = np.einsum(*args, [])
    for path in orders(labels, dims, rng).values():
        assert np.isclose(contract_along(tensors, labels, path), reference)


def test_matrix_chain_width_and_cost():
    labels, dims = [(0, 1), (1, 2), (2, 3)], {0: 2, 1: 32, 2: 4, 3: 64}
    W, C = path_width_cost(labels, dims, [(0, 1), (0, 1)])     # (AB)C
    assert C == 2 * 32 * 4 + 2 * 4 * 64
    assert W == math.log2(4 * 64)                               # largest tensor is C itself


def test_bubbling_path_follows_the_order():
    # Order 2, 0, 3, 1: contract items 0 and 2 -> list [1, 3, acc]; absorb 3 at position 1
    # -> list [1, acc]; absorb 1 at position 0. Four tensors take three pairwise steps.
    assert bubbling_path(4, [2, 0, 3, 1]) == [(0, 2), (1, 2), (0, 1)]


def test_ring_width_is_constant():
    for n in (4, 8, 16, 32):
        labels = ring(n)
        W, _ = path_width_cost(labels, dims_of(labels), optimiser_path(labels, dims_of(labels), MIN_W))
        assert W == 2


def test_grid_width_grows_with_side():
    # Bridgeman & Chubb 1.4: any order has an intermediate whose boundary is ~ the side L.
    for L in (2, 3, 4, 5):
        labels = grid(L)
        dims = dims_of(labels)
        W_opt, _ = path_width_cost(labels, dims, optimiser_path(labels, dims, MIN_W))
        W_row, _ = path_width_cost(labels, dims, bubbling_path(L * L, list(range(L * L))))
        assert L <= W_opt <= W_row <= L + 1
