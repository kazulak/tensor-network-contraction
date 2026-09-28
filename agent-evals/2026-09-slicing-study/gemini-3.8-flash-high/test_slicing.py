"""Correctness and unitarity tests for random quantum circuit tensor networks."""
import numpy as np
import pytest
import cotengra as ctg
from circuits import haar_unitary, build_1d_circuit, build_2d_sycamore_circuit, simulate_statevector, build_circuit_tn


def test_unitarity():
    """Verify that Haar-random unitaries are genuinely unitary to machine precision."""
    rng = np.random.default_rng(2026)
    for _ in range(20):
        u = haar_unitary(4, rng)
        assert u.dtype == np.complex128
        identity = np.eye(4, dtype=np.complex128)
        error = np.max(np.abs(u.conj().T @ u - identity))
        assert error < 1e-14, f"Unitarity error {error} exceeds tolerance"


def test_1d_brickwork_correctness():
    """Verify state-vector, unsliced TN, and sliced TN agree to ~1e-10 on 1D circuit."""
    n_qubits, depth = 8, 6
    rng = np.random.default_rng(101)
    gates = build_1d_circuit(n_qubits, depth, rng)
    x = [0, 1, 0, 1, 0, 0, 1, 0]

    amp_sv = simulate_statevector(n_qubits, gates, x)
    tensors, indices, size_dict = build_circuit_tn(n_qubits, gates, x)

    opt = ctg.AutoOptimizer(progbar=False)
    tree = ctg.array_contract_tree(indices, (), size_dict, optimize=opt)
    amp_tn = tree.contract(tensors)

    tree_sliced = tree.slice(target_slices=8, seed=42)
    amp_sliced = tree_sliced.contract(tensors)

    assert np.abs(amp_sv - amp_tn) < 1e-10
    assert np.abs(amp_tn - amp_sliced) < 1e-10
    assert np.abs(amp_sv - amp_sliced) < 1e-10


def test_2d_sycamore_correctness():
    """Verify state-vector, unsliced TN, and sliced TN agree to ~1e-10 on 2D Sycamore circuit."""
    H, W, n_cycles = 3, 3, 1
    n_qubits = H * W
    rng = np.random.default_rng(202)
    gates = build_2d_sycamore_circuit(H, W, n_cycles, rng)
    x = [1, 0, 0, 1, 1, 0, 0, 0, 1]

    amp_sv = simulate_statevector(n_qubits, gates, x)
    tensors, indices, size_dict = build_circuit_tn(n_qubits, gates, x)

    opt = ctg.AutoOptimizer(progbar=False)
    tree = ctg.array_contract_tree(indices, (), size_dict, optimize=opt)
    amp_tn = tree.contract(tensors)

    tree_sliced = tree.slice(target_slices=8, seed=42)
    amp_sliced = tree_sliced.contract(tensors)

    assert np.abs(amp_sv - amp_tn) < 1e-10
    assert np.abs(amp_tn - amp_sliced) < 1e-10
    assert np.abs(amp_sv - amp_sliced) < 1e-10
