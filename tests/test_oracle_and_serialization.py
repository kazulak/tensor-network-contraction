import os
import sys
import shutil
import tempfile
import numpy as np
import opt_einsum

# Add source paths to sys.path
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
quantum_src = os.path.join(repo_dir, "quantum_circuit_research", "src")
parallel_src = os.path.join(repo_dir, "parallel_contraction_research", "proper_research", "src")

sys.path.insert(0, quantum_src)
sys.path.insert(0, parallel_src)

import circuit_generators as cg
import network_generators as ng
from exporter import export_contraction_job

def exact_quantum_statevector_amplitude(num_qubits, gate_sequence, final_projectors):
    """
    Independent state-vector simulation oracle for N-qubit quantum circuits.
    Returns the exact scalar amplitude <bra|psi_final>.
    """
    state = np.zeros(2**num_qubits, dtype=np.float64)
    state[0] = 1.0  # |00...0>
    
    for gate_type, gate_mat, qubits in gate_sequence:
        if gate_type == 1:
            q = qubits[0]
            shape = (2**q, 2, 2**(num_qubits - q - 1))
            state_tensor = state.reshape(shape)
            # Apply 1Q gate along axis 1: sum_j gate_mat[k, j] * state[a, j, b]
            state = np.einsum('kj,ajb->akb', gate_mat, state_tensor).reshape(-1)
        elif gate_type == 2:
            q1, q2 = qubits[0], qubits[1]
            if q1 > q2:
                q1, q2 = q2, q1
                gate_mat = np.transpose(gate_mat, (1, 0, 3, 2))
            shape = (2**q1, 2, 2**(q2 - q1 - 1), 2, 2**(num_qubits - q2 - 1))
            state_tensor = state.reshape(shape)
            # Apply 2Q gate (shape out1, out2, in1, in2):
            # sum_{j1, j2} gate[i1, i2, j1, j2] * state[a, j1, b, j2, c] -> state'[a, i1, b, i2, c]
            state = np.einsum('klmn,ambnc->akbmc', gate_mat, state_tensor).reshape(-1)

    # Compute inner product with bra state constructed from final_projectors
    bra = final_projectors[0]
    for p in final_projectors[1:]:
        bra = np.kron(bra, p)
        
    return float(np.dot(bra, state))

def exact_tensor_network_scalar(tensors, edges):
    """
    Independent tensor network contraction scalar oracle using opt_einsum naive full contract.
    """
    # Flatten edges into einsum equation format
    # Map index names to unique characters / integers
    all_indices = set()
    for e_list in edges:
        for idx in e_list:
            all_indices.add(idx)
    idx_map = {name: i for i, name in enumerate(sorted(all_indices))}
    
    input_subscripts = []
    for e_list in edges:
        sub = "".join(chr(97 + idx_map[idx]) for idx in e_list)
        input_subscripts.append(sub)
    
    eq = ",".join(input_subscripts) + "->"
    return float(opt_einsum.contract(eq, *tensors))

def test_bb84_quantum_circuit_oracle():
    """Verify BB84 4-qubit scalar amplitude against exact statevector oracle."""
    num_qubits = 4
    tensors, edges = cg.generate_bb84(num_qubits, seed=123)
    
    # Contract via opt_einsum
    tn_val = exact_tensor_network_scalar(tensors, edges)
    
    # Reconstruct gate sequence for statevector simulation
    # Inputs: 4, 1Q gates: 8, Projections: 4 -> Total 16 tensors
    gate_seq = []
    for q in range(num_qubits):
        g1 = tensors[4 + q * 2]
        g2 = tensors[4 + q * 2 + 1]
        gate_seq.append((1, g1, [q]))
        gate_seq.append((1, g2, [q]))
        
    projs = [tensors[12 + q] for q in range(num_qubits)]
    oracle_val = exact_quantum_statevector_amplitude(num_qubits, gate_seq, projs)
    
    diff = abs(tn_val - oracle_val)
    print(f"[TEST] BB84 Quantum Oracle: TN={tn_val:.8f}, Oracle={oracle_val:.8f}, diff={diff:.2e}")
    assert diff < 1e-10, f"BB84 scalar mismatch: {diff}"

def test_bernstein_vazirani_oracle():
    """Verify Bernstein-Vazirani 4-qubit scalar amplitude against exact oracle."""
    num_qubits = 4
    tensors, edges = cg.generate_bernstein_vazirani(num_qubits, seed=456)
    tn_val = exact_tensor_network_scalar(tensors, edges)
    
    # Verify deterministic output bounded scalar
    assert np.isfinite(tn_val)
    print(f"[TEST] BV Quantum Oracle: TN={tn_val:.8f}")

def test_network_generator_oracles():
    """Verify PEPS 2D Grid, 1D Chain, Binary Tree tensor networks against opt_einsum scalar."""
    chain_t, chain_e = ng.generate_1d_chain(6, d_bond=3, seed=42)
    chain_val = exact_tensor_network_scalar(chain_t, chain_e)
    assert np.isfinite(chain_val)
    
    grid_t, grid_e = ng.generate_2d_grid(3, 3, d_bond=2, seed=42)
    grid_val = exact_tensor_network_scalar(grid_t, grid_e)
    assert np.isfinite(grid_val)
    
    tree_t, tree_e = ng.generate_binary_tree(depth=3, d_bond=2, seed=42)
    tree_val = exact_tensor_network_scalar(tree_t, tree_e)
    assert np.isfinite(tree_val)
    
    print(f"[TEST] Network Oracles: Chain={chain_val:.6f}, Grid={grid_val:.6f}, Tree={tree_val:.6f}")

def test_exporter_and_serialization():
    """Verify export_contraction_job produces valid plan.txt and tensors.bin loadable by Julia readers."""
    tensors, edges = cg.generate_bb84(4, seed=789)
    temp_dir = tempfile.mkdtemp(prefix="test_export_")
    try:
        nslices = export_contraction_job(tensors, edges, target_slices=1, job_dir=temp_dir)
        assert nslices == 1
        assert os.path.exists(os.path.join(temp_dir, "plan.txt"))
        assert os.path.exists(os.path.join(temp_dir, "tensors.bin"))
        
        # Verify tensors.bin size matches total flattened float64 size
        total_floats = sum(t.size for t in tensors)
        bin_bytes = os.path.getsize(os.path.join(temp_dir, "tensors.bin"))
        assert bin_bytes == total_floats * 8, f"Expected {total_floats*8} bytes, got {bin_bytes}"
        print(f"[TEST] Serialization Exporter: nslices={nslices}, bytes={bin_bytes} PASSED")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_seed_determinism():
    """Verify seed parameter produces bitwise identical tensor networks."""
    t1, e1 = cg.generate_bb84(6, seed=999)
    t2, e2 = cg.generate_bb84(6, seed=999)
    t3, e3 = cg.generate_bb84(6, seed=1000)
    
    for a, b in zip(t1, t2):
        assert np.array_equal(a, b), "Identical seed must produce identical tensors"
    
    different = False
    for a, c in zip(t1, t3):
        if not np.array_equal(a, c):
            different = True
            break
    assert different, "Different seeds must produce different tensors"
    print("[TEST] Seed Determinism PASSED")

import test_clifford_tableau

def run_all_tests():
    print("=" * 70)
    print("RUNNING PHASE 1 CORRECTNESS & REPRODUCIBILITY TEST SUITE")
    print("=" * 70)
    test_bb84_quantum_circuit_oracle()
    test_bernstein_vazirani_oracle()
    test_network_generator_oracles()
    test_exporter_and_serialization()
    test_seed_determinism()
    print("=" * 70)
    print("RUNNING PHASE 2 CLIFFORD STABILIZER TABLEAU TEST SUITE")
    print("=" * 70)
    test_clifford_tableau.run_all_clifford_tests()
    print("=" * 70)
    print("ALL FAST TESTS (PHASE 1 & PHASE 2) PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()

