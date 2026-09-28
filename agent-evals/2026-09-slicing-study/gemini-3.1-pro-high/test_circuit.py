import numpy as np
import pytest
import opt_einsum as oe
from circuit import build_1d_circuit, build_2d_circuit, get_tensor_network, apply_gate_sv

def state_vector_sim(n_qubits, gates):
    state = np.zeros((2,) * n_qubits, dtype=np.complex128)
    state[(0,) * n_qubits] = 1.0
    for q1, q2, gate in gates:
        state = apply_gate_sv(state, gate, q1, q2, n_qubits)
    return state[(0,) * n_qubits]

def contract_sliced(tensors, eqs, sliced_inds):
    def dfs(current_tensors, current_eqs, inds_left):
        if not inds_left:
            args = []
            for t, eq in zip(current_tensors, current_eqs):
                args.extend([t, eq])
            return oe.contract(*args)
        
        ind = inds_left[0]
        res = 0
        for v in [0, 1]:
            next_tensors = []
            next_eqs = []
            for t, eq in zip(current_tensors, current_eqs):
                if ind in eq:
                    axis = eq.index(ind)
                    next_tensors.append(np.take(t, v, axis=axis))
                    next_eqs.append([e for e in eq if e != ind])
                else:
                    next_tensors.append(t)
                    next_eqs.append(eq)
            res += dfs(next_tensors, next_eqs, inds_left[1:])
        return res
    return dfs(tensors, eqs, sliced_inds)

def test_unitarity():
    from circuit import haar_unitary
    U = haar_unitary(4)
    assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)

def test_1d_contraction():
    n_qubits = 6
    depth = 4
    gates = build_1d_circuit(n_qubits, depth)
    sv_amp = state_vector_sim(n_qubits, gates)
    
    tensors, eqs = get_tensor_network(n_qubits, gates)
    args = []
    for t, eq in zip(tensors, eqs):
        args.extend([t, eq])
    unsliced_amp = oe.contract(*args)
    assert np.isclose(sv_amp, unsliced_amp, atol=1e-10)
    
    # Slice 2 indices
    all_inds = list(set([ind for eq in eqs for ind in eq]))
    sliced_inds = all_inds[:2]
    sliced_amp = contract_sliced(tensors, eqs, sliced_inds)
    assert np.isclose(sv_amp, sliced_amp, atol=1e-10)

def test_2d_contraction():
    rows, cols, depth = 3, 3, 4
    n_qubits = rows * cols
    gates = build_2d_circuit(rows, cols, depth)
    sv_amp = state_vector_sim(n_qubits, gates)
    
    tensors, eqs = get_tensor_network(n_qubits, gates)
    args = []
    for t, eq in zip(tensors, eqs):
        args.extend([t, eq])
    unsliced_amp = oe.contract(*args)
    assert np.isclose(sv_amp, unsliced_amp, atol=1e-10)

    # Slice explicitly 1 index
    all_inds = list(set([ind for eq in eqs for ind in eq]))
    sliced_inds = all_inds[:1]
    sliced_amp = contract_sliced(tensors, eqs, sliced_inds)
    assert np.isclose(sv_amp, sliced_amp, atol=1e-10)
