import itertools
import numpy as np
from circuits import (H, S, T, CNOT, SWAP, controlled_phase, haar_unitary, ghz, qft,
                      random_circuit, statevector, to_network, contract, amplitude)

rng = np.random.default_rng(2)


def is_unitary(U):
    return np.allclose(U.conj().T @ U, np.eye(len(U)))


def test_gates_are_unitary():
    for U in (H, S, T, CNOT, SWAP, controlled_phase(0.3), haar_unitary(4, rng)):
        assert is_unitary(U)


def test_haar_unitary_is_complex():
    # A real orthogonal matrix is not a Haar U(4) sample, which is complex.
    assert np.abs(haar_unitary(4, np.random.default_rng(0)).imag).max() > 0.1


def test_statevector_single_gates():
    # |00> --(H on 0)--> (|00> + |10>)/sqrt2 with qubit 0 as the most significant bit.
    assert np.allclose(statevector([(H, (0,))], 2), [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])
    # CNOT with control 1, target 0 acting on |01> gives |11>.
    assert np.allclose(statevector([(CNOT, (1, 0))], 2, x=0b01), np.eye(4)[0b11])


def test_ghz_amplitudes():
    n = 4
    for bits in itertools.product("01", repeat=n):
        b = "".join(bits)
        expected = 1 / np.sqrt(2) if b in ("0000", "1111") else 0
        assert np.isclose(amplitude(ghz(n), n, b), expected)


def test_closed_network_equals_statevector():
    n = 5
    circ = random_circuit(n, 4, rng)
    psi = statevector(circ, n)
    for x in range(2 ** n):
        assert np.isclose(amplitude(circ, n, format(x, f"0{n}b")), psi[x])


def test_open_network_equals_statevector():
    n = 7
    circ = random_circuit(n, 5, rng)
    tensors, out = to_network(circ, n)
    psi_tn = contract(tensors, out).reshape(-1)
    assert np.allclose(psi_tn, statevector(circ, n))
    assert np.isclose(np.linalg.norm(psi_tn), 1)


def test_qft_matches_dft():
    # Nielsen & Chuang Sec. 5.1: QFT|x> = 2^{-n/2} sum_k exp(2 pi i x k / 2^n) |k>.
    for n in (1, 2, 3, 4, 5):
        N = 2 ** n
        F = np.array([statevector(qft(n), n, x) for x in range(N)]).T
        expected = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
        assert np.allclose(F, expected)


def test_qft_network_matches_dft():
    # The same, contracted as a network from every basis input |x>. On |0...0> alone the
    # controlled phases never act, so this is what checks their tensors and the SWAPs.
    for n in (1, 2, 3, 4):
        N = 2 ** n
        for x in range(N):
            psi = contract(*to_network(qft(n), n, initial=format(x, f"0{n}b"))).reshape(-1)
            assert np.allclose(psi, np.exp(2j * np.pi * x * np.arange(N) / N) / np.sqrt(N))
