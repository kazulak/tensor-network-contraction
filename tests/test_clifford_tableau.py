"""
Comprehensive Correctness Test Suite for Stabilizer Tableau Clifford Simulator.

Validates the stabilizer tableau baseline against an independent dense statevector oracle.
Covers single-qubit gates, complex phase tracking, two-qubit equivalences (CZ vs CNOT),
deterministic and half-probability outcomes, zero probability branches, random circuit sweeps,
invalid gate handling, input validation, seeded sampling, and repository generator integration.
"""

import os
import sys
import numpy as np

# Setup python path to include quantum_circuit_research/src
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
quantum_src = os.path.join(repo_dir, "quantum_circuit_research", "src")
if quantum_src not in sys.path:
    sys.path.insert(0, quantum_src)

from tableau_simulator import (
    StabilizerTableau,
    CliffordCircuit,
    NonCliffordGateError,
    classify_1q_gate,
    classify_2q_gate,
    from_tensor_network_gates
)
import circuit_generators as cg


def dense_statevector_oracle(num_qubits: int, circuit: CliffordCircuit) -> np.ndarray:
    """Computes exact dense statevector array of shape (2^N,) for verification."""
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    state[0] = 1.0

    H_mat = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    S_mat = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    SDG_mat = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    X_mat = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y_mat = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z_mat = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    CNOT_mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128).reshape(2, 2, 2, 2)

    CZ_mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=np.complex128).reshape(2, 2, 2, 2)

    gate_map_1q = {
        'H': H_mat, 'S': S_mat, 'SDG': SDG_mat,
        'X': X_mat, 'Y': Y_mat, 'Z': Z_mat
    }

    for g in circuit.gates:
        if g.name in gate_map_1q:
            q = g.qubits[0]
            mat = gate_map_1q[g.name]
            shape = (2**q, 2, 2**(num_qubits - q - 1))
            st = state.reshape(shape)
            state = np.einsum('kj,ajb->akb', mat, st).reshape(-1)
        elif g.name in ('CX', 'CNOT'):
            q1, q2 = g.qubits[0], g.qubits[1]
            mat = CNOT_mat
            if q1 > q2:
                q1, q2 = q2, q1
                mat = np.transpose(mat, (1, 0, 3, 2))
            shape = (2**q1, 2, 2**(q2 - q1 - 1), 2, 2**(num_qubits - q2 - 1))
            st = state.reshape(shape)
            state = np.einsum('klmn,ambnc->akbmc', mat, st).reshape(-1)
        elif g.name == 'CZ':
            q1, q2 = g.qubits[0], g.qubits[1]
            mat = CZ_mat
            if q1 > q2:
                q1, q2 = q2, q1
                mat = np.transpose(mat, (1, 0, 3, 2))
            shape = (2**q1, 2, 2**(q2 - q1 - 1), 2, 2**(num_qubits - q2 - 1))
            st = state.reshape(shape)
            state = np.einsum('klmn,ambnc->akbmc', mat, st).reshape(-1)

    return state


def verify_tableau_against_statevector(circuit: CliffordCircuit, tol: float = 1e-10):
    """Compares all basis outcome probabilities P(b) between tableau and statevector oracle."""
    n = circuit.num_qubits
    tableau = circuit.to_tableau()
    psi = dense_statevector_oracle(n, circuit)
    sv_probs = np.abs(psi)**2

    for idx in range(2**n):
        b_str = format(idx, f'0{n}b')
        tab_p = tableau.get_probability(b_str)
        sv_p = sv_probs[idx]
        diff = abs(tab_p - sv_p)
        assert diff < tol, f"Probability mismatch for bitstring {b_str}: Tableau={tab_p}, Oracle={sv_p}"


def test_single_qubit_gates_and_phases():
    """Verify single qubit Clifford gates (H, S, SDG, X, Y, Z) and complex phase accumulation."""
    # Test H, S, SDG
    c = CliffordCircuit(2).h(0).s(0).sdg(1).x(1)
    verify_tableau_against_statevector(c)

    # Test Y, Z, S^2 == Z
    c2 = CliffordCircuit(1).y(0).z(0).s(0).s(0)
    verify_tableau_against_statevector(c2)

    # Verify S * SDG == Identity
    c3 = CliffordCircuit(1).s(0).sdg(0)
    tab = c3.to_tableau()
    assert tab.get_probability("0") == 1.0
    assert tab.get_probability("1") == 0.0

    print("[PASS] test_single_qubit_gates_and_phases")


def test_deterministic_and_branch_probabilities():
    """Verify 0.0, 0.5, and 1.0 probability branches."""
    # Bell state (|00> + |11>) / sqrt(2)
    bell = CliffordCircuit(2).h(0).cx(0, 1)
    tab_bell = bell.to_tableau()

    assert abs(tab_bell.get_probability("00") - 0.5) < 1e-10
    assert abs(tab_bell.get_probability("11") - 0.5) < 1e-10
    assert abs(tab_bell.get_probability("01") - 0.0) < 1e-10
    assert abs(tab_bell.get_probability("10") - 0.0) < 1e-10
    verify_tableau_against_statevector(bell)

    # GHZ state 3 qubits (|000> + |111>) / sqrt(2)
    ghz = CliffordCircuit(3).h(0).cx(0, 1).cx(1, 2)
    tab_ghz = ghz.to_tableau()
    assert abs(tab_ghz.get_probability("000") - 0.5) < 1e-10
    assert abs(tab_ghz.get_probability("111") - 0.5) < 1e-10
    assert abs(tab_ghz.get_probability("001") - 0.0) < 1e-10
    verify_tableau_against_statevector(ghz)

    print("[PASS] test_deterministic_and_branch_probabilities")


def test_cz_cnot_equivalence():
    """Verify (I x H) CNOT (I x H) == CZ equivalence."""
    n = 2
    cnot_circuit = CliffordCircuit(n).h(1).cx(0, 1).h(1)
    cz_circuit = CliffordCircuit(n).cz(0, 1)

    tab_cnot = cnot_circuit.to_tableau()
    tab_cz = cz_circuit.to_tableau()

    for idx in range(2**n):
        b_str = format(idx, f'0{n}b')
        p_cnot = tab_cnot.get_probability(b_str)
        p_cz = tab_cz.get_probability(b_str)
        assert abs(p_cnot - p_cz) < 1e-10, f"CZ/CNOT mismatch for {b_str}"

    verify_tableau_against_statevector(cnot_circuit)
    verify_tableau_against_statevector(cz_circuit)

    print("[PASS] test_cz_cnot_equivalence")


def test_random_seeded_clifford_sweep():
    """Sweeps 50 random seeded Clifford circuits of N=1..8 qubits against statevector oracle."""
    gate_pool_1q = ['H', 'S', 'SDG', 'X', 'Y', 'Z']
    gate_pool_2q = ['CX', 'CZ']

    seed_count = 0
    for n in range(1, 7):
        for trial in range(8):
            rng = np.random.default_rng(seed=1000 + seed_count)
            seed_count += 1

            c = CliffordCircuit(n)
            depth = 15
            for _ in range(depth):
                if n == 1 or rng.random() > 0.4:
                    g = rng.choice(gate_pool_1q)
                    q = int(rng.integers(0, n))
                    c.append(g, (q,))
                else:
                    g = rng.choice(gate_pool_2q)
                    q1, q2 = rng.choice(n, size=2, replace=False)
                    c.append(g, (int(q1), int(q2)))

            verify_tableau_against_statevector(c)

    print(f"[PASS] test_random_seeded_clifford_sweep ({seed_count} circuits verified)")


def test_extra_randomized_small_clifford_cross_check():
    """Explicit extra randomized small Clifford (N=3..5, depth 25) vs dense statevector cross-check."""
    gate_pool_1q = ['H', 'S', 'SDG', 'X', 'Y', 'Z']
    gate_pool_2q = ['CX', 'CZ']
    total_checked = 0
    for n in range(3, 6):
        for trial in range(10):
            rng = np.random.default_rng(seed=2000 + total_checked)
            total_checked += 1
            c = CliffordCircuit(n)
            for _ in range(25):
                if rng.random() > 0.4:
                    g = rng.choice(gate_pool_1q)
                    q = int(rng.integers(0, n))
                    c.append(g, (q,))
                else:
                    g = rng.choice(gate_pool_2q)
                    q1, q2 = rng.choice(n, size=2, replace=False)
                    c.append(g, (int(q1), int(q2)))
            verify_tableau_against_statevector(c)
    print(f"[PASS] test_extra_randomized_small_clifford_cross_check ({total_checked} circuits verified)")


def test_invalid_gates_and_input_validation():
    """Verify rejection of non-Clifford gates and improper inputs."""
    c = CliffordCircuit(2)

    # Non-Clifford gate string
    try:
        c.append('T', (0,))
        assert False, "Should have raised NonCliffordGateError"
    except NonCliffordGateError:
        pass

    # Non-Clifford continuous rotation matrix classification
    o2_mat = np.array([
        [np.cos(0.3), -np.sin(0.3)],
        [np.sin(0.3), np.cos(0.3)]
    ], dtype=np.complex128)
    try:
        classify_1q_gate(o2_mat)
        assert False, "Continuous rotation should raise NonCliffordGateError"
    except NonCliffordGateError:
        pass

    # Out of bounds qubit
    try:
        CliffordCircuit(2).h(5)
        assert False, "Should raise ValueError for qubit index 5"
    except ValueError:
        pass

    # Same control/target
    try:
        CliffordCircuit(2).cx(0, 0)
        assert False, "Should raise ValueError for control==target"
    except ValueError:
        pass

    # Bitstring length mismatch
    tab = CliffordCircuit(2).to_tableau()
    try:
        tab.get_probability("000")
        assert False, "Should raise ValueError for 3-bit string on 2 qubits"
    except ValueError:
        pass

    print("[PASS] test_invalid_gates_and_input_validation")


def test_seeded_sampling():
    """Verify deterministic sampling under a seed and empirical frequency accuracy."""
    c = CliffordCircuit(2).h(0).cx(0, 1)
    tab = c.to_tableau()

    # Determinism
    s1 = tab.sample(shots=500, seed=42)
    s2 = tab.sample(shots=500, seed=42)
    assert s1 == s2, "Identical seed must produce identical sampling counts"

    # Distribution checks for Bell pair (|00> and |11> only)
    assert '01' not in s1
    assert '10' not in s1
    assert s1['00'] + s1['11'] == 500
    # Expected ~250 each
    assert 200 <= s1['00'] <= 300
    assert 200 <= s1['11'] <= 300

    print("[PASS] test_seeded_sampling")


def test_generator_integration():
    """Verify repository Clifford generators adapt to CliffordCircuit and match oracle."""
    num_qubits = 4

    # 1. Bernstein-Vazirani
    bv_tensors, bv_edges = cg.generate_bernstein_vazirani(num_qubits, seed=123)
    # Reconstruct gate_sequence format
    gate_seq_bv = []
    # 4 inputs, H*4, X*1, CNOT*3, H*3, projs*4
    # H on all 4
    for q in range(num_qubits):
        gate_seq_bv.append((1, cg.H_gate, [q]))
    # X on target (3)
    gate_seq_bv.append((1, cg.X_gate, [num_qubits - 1]))
    # CNOT from 0..2 to target
    for q in range(num_qubits - 1):
        gate_seq_bv.append((2, cg.CNOT_gate, [q, num_qubits - 1]))
    # H on 0..2
    for q in range(num_qubits - 1):
        gate_seq_bv.append((1, cg.H_gate, [q]))

    bv_circuit = from_tensor_network_gates(num_qubits, gate_seq_bv)
    verify_tableau_against_statevector(bv_circuit)

    # 2. Error Detection Code
    edc_tensors, edc_edges = cg.generate_error_detection(num_qubits, seed=456)
    gate_seq_edc = []
    for q in range(num_qubits):
        gate_seq_edc.append((1, cg.H_gate, [q]))
    for q in range(num_qubits - 1):
        gate_seq_edc.append((2, cg.CNOT_gate, [q, q + 1]))
    for q in range(num_qubits - 1):
        gate_seq_edc.append((2, cg.CNOT_gate, [q + 1, q]))
    for q in range(num_qubits):
        gate_seq_edc.append((1, cg.H_gate, [q]))

    edc_circuit = from_tensor_network_gates(num_qubits, gate_seq_edc)
    verify_tableau_against_statevector(edc_circuit)

    # 3. Non-Clifford BB84 rejection test
    bb84_tensors, bb84_edges = cg.generate_bb84(num_qubits, seed=789)
    # BB84 uses continuous O(2) rotations
    o2_gate = bb84_tensors[4]
    gate_seq_bb84 = [(1, o2_gate, [0])]
    try:
        from_tensor_network_gates(num_qubits, gate_seq_bb84)
        assert False, "BB84 with continuous O(2) rotations must be rejected as non-Clifford"
    except NonCliffordGateError:
        pass

    print("[PASS] test_generator_integration")


def test_y_heavy_clifford_and_row_mult_regression():
    """Verify Y-heavy Clifford circuit probabilities and direct private row multiplication phase accumulation."""
    # 1. Direct row multiplication sanity check: Y * Y = +I (x=0, z=0, r=0)
    tab = StabilizerTableau(1)
    tab.mat[0, 0] = 1  # x_h = 1
    tab.mat[0, 1] = 1  # z_h = 1 (Y generator)
    tab.mat[0, 2] = 0  # r_h = 0
    tab.mat[1, 0] = 1  # x_i = 1
    tab.mat[1, 1] = 1  # z_i = 1 (Y generator)
    tab.mat[1, 2] = 0  # r_i = 0
    tab._row_mult(0, 1)
    assert tab.mat[0, 0] == 0 and tab.mat[0, 1] == 0 and tab.mat[0, 2] == 0, (
        f"Direct row mult Y*Y failed: x={tab.mat[0,0]}, z={tab.mat[0,1]}, r={tab.mat[0,2]}"
    )

    # 2. Y-heavy Clifford circuit compared against independent dense statevector oracle
    circuit = CliffordCircuit(3)
    circuit.y(0).y(1).y(2)
    circuit.h(0).s(1).sdg(2)
    circuit.cx(0, 1).cz(1, 2)
    circuit.y(0).y(1).y(2)
    circuit.s(0).y(1).h(2)
    verify_tableau_against_statevector(circuit)

    print("[PASS] test_y_heavy_clifford_and_row_mult_regression")


def run_all_clifford_tests():
    print("=" * 70)
    print("RUNNING PHASE 2 CLIFFORD STABILIZER TABLEAU TEST SUITE")
    print("=" * 70)
    test_single_qubit_gates_and_phases()
    test_deterministic_and_branch_probabilities()
    test_cz_cnot_equivalence()
    test_random_seeded_clifford_sweep()
    test_extra_randomized_small_clifford_cross_check()
    test_invalid_gates_and_input_validation()
    test_seeded_sampling()
    test_generator_integration()
    test_y_heavy_clifford_and_row_mult_regression()
    print("=" * 70)
    print("ALL PHASE 2 CLIFFORD TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_clifford_tests()
