import itertools
import numpy as np
import pytest
import cotengra as ctg
import opt_einsum as oe

from slicing_study import (
    generate_haar_unitary,
    state_vector_simulate,
    circuit_to_tn,
    build_1d_brickwork,
    build_2d_grid,
)


def test_gate_unitarity():
    r"""Verify that generated 2-qubit gates are strictly unitary (U^\dagger U = I)."""
    np.random.seed(123)
    for _ in range(10):
        u = generate_haar_unitary()
        u_mat = u.reshape((4, 4))
        identity_diff = np.max(np.abs(u_mat.conj().T @ u_mat - np.eye(4, dtype=np.complex128)))
        assert identity_diff < 1e-14, f"Unitarity failed with max diff {identity_diff}"


def test_1d_brickwork_correctness():
    """Verify SV amp == Unsliced TN amp == Sliced TN sum to ~1e-10 for 1D brickwork circuit."""
    N, L = 6, 4
    gates = build_1d_brickwork(N, L)
    target_bitstring = (1, 0, 1, 0, 0, 1)

    # State vector
    sv_amp = state_vector_simulate(N, gates, target_bitstring)

    # Unsliced TN
    tensors, inputs = circuit_to_tn(N, gates, target_bitstring)
    path, info = oe.contract_path(*[item for pair in zip(tensors, inputs) for item in pair])
    tree = ctg.ContractionTree.from_info(info)
    unsliced_tn_amp = tree.contract(tensors)

    # Sliced TN
    sf = ctg.SliceFinder(tree, target_size=4)
    sliced_inds, _ = sf.search()

    sliced_inds_list = sorted(list(sliced_inds))
    n_slices = len(sliced_inds_list)
    sliced_tn_sum = 0.0 + 0.0j

    for val_tuple in itertools.product([0, 1], repeat=n_slices):
        val_dict = dict(zip(sliced_inds_list, val_tuple))
        sliced_tensors = []
        for t, idx in zip(tensors, inputs):
            sl = [slice(None)] * t.ndim
            for axis, wire in enumerate(idx):
                if wire in val_dict:
                    v = val_dict[wire]
                    sl[axis] = slice(v, v + 1)
            sliced_tensors.append(t[tuple(sl)])
        sliced_tn_sum += tree.contract(sliced_tensors)

    assert np.abs(sv_amp - unsliced_tn_amp) < 1e-10, f"SV vs Unsliced TN diff: {np.abs(sv_amp - unsliced_tn_amp)}"
    assert np.abs(unsliced_tn_amp - sliced_tn_sum) < 1e-10, f"Unsliced vs Sliced TN diff: {np.abs(unsliced_tn_amp - sliced_tn_sum)}"


def test_2d_grid_correctness():
    """Verify SV amp == Unsliced TN amp == Sliced TN sum to ~1e-10 for 2D grid circuit."""
    N_row, N_col, L = 3, 3, 4
    N = N_row * N_col
    gates = build_2d_grid(N_row, N_col, L)
    target_bitstring = (0, 1, 0, 1, 1, 0, 1, 0, 1)

    # State vector
    sv_amp = state_vector_simulate(N, gates, target_bitstring)

    # Unsliced TN
    tensors, inputs = circuit_to_tn(N, gates, target_bitstring)
    path, info = oe.contract_path(*[item for pair in zip(tensors, inputs) for item in pair])
    tree = ctg.ContractionTree.from_info(info)
    unsliced_tn_amp = tree.contract(tensors)

    # Sliced TN
    sf = ctg.SliceFinder(tree, target_size=8)
    sliced_inds, _ = sf.search()

    sliced_inds_list = sorted(list(sliced_inds))
    n_slices = len(sliced_inds_list)
    sliced_tn_sum = 0.0 + 0.0j

    for val_tuple in itertools.product([0, 1], repeat=n_slices):
        val_dict = dict(zip(sliced_inds_list, val_tuple))
        sliced_tensors = []
        for t, idx in zip(tensors, inputs):
            sl = [slice(None)] * t.ndim
            for axis, wire in enumerate(idx):
                if wire in val_dict:
                    v = val_dict[wire]
                    sl[axis] = slice(v, v + 1)
            sliced_tensors.append(t[tuple(sl)])
        sliced_tn_sum += tree.contract(sliced_tensors)

    assert np.abs(sv_amp - unsliced_tn_amp) < 1e-10, f"SV vs Unsliced TN diff: {np.abs(sv_amp - unsliced_tn_amp)}"
    assert np.abs(unsliced_tn_amp - sliced_tn_sum) < 1e-10, f"Unsliced vs Sliced TN diff: {np.abs(unsliced_tn_amp - sliced_tn_sum)}"
