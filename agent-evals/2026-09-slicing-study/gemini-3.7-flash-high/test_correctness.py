"""Pytest suite verifying numerical correctness of quantum circuit slicing.
Checks:
1. Genuine unitarity of complex128 gates (U @ U.conj().T == I).
2. Exact 3-way agreement (to < 1e-10) between state-vector simulation,
   unsliced tensor network contraction, and sliced tensor network contraction.
"""

import numpy as np
import pytest
from slicing_study import haar_u4, build_1d_circuit, build_2d_circuit, simulate_sv, circuit_to_tn, contract_sliced_tn
import cotengra as ctg
import opt_einsum as oe


def test_gate_unitarity():
    """Verify that Haar-generated 2-qubit gates are genuinely unitary in complex128."""
    rng = np.random.default_rng(101)
    for _ in range(20):
        u = haar_u4(rng)
        assert u.dtype == np.complex128
        identity = np.eye(4, dtype=np.complex128)
        assert np.linalg.norm(u @ u.conj().T - identity) < 1e-14
        assert np.linalg.norm(u.conj().T @ u - identity) < 1e-14


@pytest.mark.parametrize("n_qubits,depth", [(4, 4), (6, 4)])
@pytest.mark.parametrize("num_slices", [2, 4])
def test_1d_brickwork_correctness(n_qubits, depth, num_slices):
    """Verify state-vector, unsliced TN, and sliced TN match within 1e-10 on 1D circuits."""
    rng = np.random.default_rng(42)
    gates = build_1d_circuit(n_qubits, depth, rng)
    x = [q % 2 for q in range(n_qubits)]

    amp_sv = simulate_sv(n_qubits, gates, x)

    inputs, tensors, size_dict, _ = circuit_to_tn(n_qubits, gates, x)
    opt = ctg.GreedyOptimizer()
    tree = opt.search(inputs, [], size_dict)

    # Unsliced contraction
    amp_unsliced = oe.contract(*[item for pair in zip(tensors, inputs) for item in pair], [], optimize=tree.get_path())

    # Sliced contraction
    sf = ctg.SliceFinder(tree, target_slices=num_slices)
    ix_sl, _ = sf.search()
    t_sl = tree.copy()
    for ix in ix_sl:
        t_sl.remove_ind_(ix)
    amp_sliced = contract_sliced_tn(tree, t_sl, tensors)

    assert abs(amp_sv - amp_unsliced) < 1e-10
    assert abs(amp_unsliced - amp_sliced) < 1e-10
    assert abs(amp_sv - amp_sliced) < 1e-10


@pytest.mark.parametrize("nrows,ncols,depth", [(2, 2, 4), (2, 3, 4)])
@pytest.mark.parametrize("num_slices", [2, 4])
def test_2d_grid_correctness(nrows, ncols, depth, num_slices):
    """Verify state-vector, unsliced TN, and sliced TN match within 1e-10 on 2D grid circuits."""
    rng = np.random.default_rng(84)
    n_qubits = nrows * ncols
    gates = build_2d_circuit(nrows, ncols, depth, rng)
    x = [(q + 1) % 2 for q in range(n_qubits)]

    amp_sv = simulate_sv(n_qubits, gates, x)

    inputs, tensors, size_dict, _ = circuit_to_tn(n_qubits, gates, x)
    opt = ctg.GreedyOptimizer()
    tree = opt.search(inputs, [], size_dict)

    amp_unsliced = oe.contract(*[item for pair in zip(tensors, inputs) for item in pair], [], optimize=tree.get_path())

    sf = ctg.SliceFinder(tree, target_slices=num_slices)
    ix_sl, _ = sf.search()
    t_sl = tree.copy()
    for ix in ix_sl:
        t_sl.remove_ind_(ix)
    amp_sliced = contract_sliced_tn(tree, t_sl, tensors)

    assert abs(amp_sv - amp_unsliced) < 1e-10
    assert abs(amp_unsliced - amp_sliced) < 1e-10
    assert abs(amp_sv - amp_sliced) < 1e-10
