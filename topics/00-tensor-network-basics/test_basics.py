import numpy as np
from basics import (contract_pair, pair_cost, bubble, einsum_reference, ladder, ladder_orders,
                    colouring_network, count_colourings_brute, PETERSEN)

rng = np.random.default_rng(1)


def rand(*shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_contract_pair_matches_einsum():
    A, B = rand(2, 3, 4, 5), rand(5, 3, 6)
    C, c = contract_pair(A, (0, 1, 2, 3), B, (3, 1, 4))
    assert c == (0, 2, 4)
    assert np.allclose(C, np.einsum("abcd,dbe->ace", A, B))


def test_matrix_chain_cost():
    dims = {0: 2, 1: 30, 2: 3, 3: 40}          # A(0,1) B(1,2) C(2,3)
    ab_then_c = pair_cost((0, 1), (1, 2), dims) + pair_cost((0, 2), (2, 3), dims)
    a_then_bc = pair_cost((1, 2), (2, 3), dims) + pair_cost((0, 1), (1, 3), dims)
    assert ab_then_c == 2 * 30 * 3 + 2 * 3 * 40
    assert a_then_bc == 30 * 3 * 40 + 2 * 30 * 40
    assert ab_then_c < a_then_bc


def test_ladder_bubblings():
    # Section 1.4: along the top the stored rank reaches n and the cost is exponential;
    # rung by rung the rank never exceeds 3 and the cost is linear in n.
    rung_costs = []
    for n in range(3, 9):
        net = ladder(n, 2, rng)
        along, rungs = ladder_orders(n)
        (x1, r1, c1), (x2, r2, c2) = bubble(net, along), bubble(net, rungs)
        assert np.isclose(x1, einsum_reference(net)) and np.isclose(x2, x1)
        assert r1 == n and r2 <= 3
        assert c1 >= 2 ** n
        rung_costs.append(c2)
    assert len(set(np.diff(rung_costs))) == 1      # same extra cost for every extra rung


def test_colourings_cycle_chromatic_polynomial():
    # Number of proper q-colourings of the cycle C_n is (q-1)^n + (-1)^n (q-1).
    for n in range(3, 8):
        edges = [(i, (i + 1) % n) for i in range(n)]
        for q in (2, 3, 4):
            tn = einsum_reference(colouring_network(edges, n, q))
            assert np.isclose(tn, (q - 1) ** n + (-1) ** n * (q - 1))


def test_colourings_petersen():
    for q in (2, 3):
        tn = einsum_reference(colouring_network(PETERSEN, 10, q))
        assert np.isclose(tn, count_colourings_brute(PETERSEN, 10, q))
