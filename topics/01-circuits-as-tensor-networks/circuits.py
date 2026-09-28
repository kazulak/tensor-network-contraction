"""Quantum circuits as tensor networks, checked against a state-vector simulator.

Sources: Nielsen & Chuang, Ch. 4 (gates) and Sec. 5.1 (QFT);
         Markov & Shi, arXiv:quant-ph/0511069, Sec. 3 (circuits as tensor networks);
         Mezzadri, arXiv:math-ph/0609050 (Haar-random unitaries).
Convention: qubit 0 is the leftmost (most significant) bit, as in Nielsen & Chuang.
A circuit is a list of (U, qubits), U a (2^k x 2^k) unitary acting on k qubits.
"""
import numpy as np
import opt_einsum as oe

I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def controlled_phase(phi):
    return np.diag([1, 1, 1, np.exp(1j * phi)])


def haar_unitary(dim, rng):
    """Haar-random U(dim): QR of a complex Gaussian matrix, with R's diagonal phases fixed."""
    Z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


# --- Circuits ------------------------------------------------------------------------

def ghz(n):
    return [(H, (0,))] + [(CNOT, (q, q + 1)) for q in range(n - 1)]


def qft(n):
    """Nielsen & Chuang Sec. 5.1: H and controlled-R_k on each qubit, then reverse with SWAPs."""
    circ = []
    for j in range(n):
        circ.append((H, (j,)))
        for k, c in enumerate(range(j + 1, n), start=2):
            circ.append((controlled_phase(2 * np.pi / 2 ** k), (c, j)))
    circ += [(SWAP, (q, n - 1 - q)) for q in range(n // 2)]
    return circ


def random_circuit(n, depth, rng):
    """Brickwork of Haar-random two-qubit gates on a line."""
    return [(haar_unitary(4, rng), (q, q + 1))
            for layer in range(depth) for q in range(layer % 2, n - 1, 2)]


# --- Reference: state vector -------------------------------------------------------

def statevector(circ, n, x=0):
    """Apply each gate to the n-qubit state |x> reshaped as a (2,)*n tensor."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[x] = 1
    psi = psi.reshape((2,) * n)
    for U, qs in circ:
        k = len(qs)
        G = U.reshape((2,) * 2 * k)                    # (out_1..out_k, in_1..in_k)
        psi = np.tensordot(G, psi, axes=(range(k, 2 * k), qs))
        psi = np.moveaxis(psi, range(k), qs)           # put the output legs back in place
    return psi.reshape(-1)


# --- Tensor network ----------------------------------------------------------------

def to_network(circ, n, bitstring=None):
    """One tensor per initial qubit, per gate and (optionally) per final projector <x|.

    Each wire segment gets a fresh label. Without a bitstring, the n final wire labels
    stay open and the network is the full output state; with one, it is closed and
    its value is the amplitude <x|C|0...0>.
    """
    wire = list(range(n))                              # current label on each qubit wire
    next_label = n
    tensors = [(np.array([1, 0], dtype=complex), (q,)) for q in range(n)]
    for U, qs in circ:
        k = len(qs)
        out = tuple(range(next_label, next_label + k))
        next_label += k
        tensors.append((U.reshape((2,) * 2 * k), out + tuple(wire[q] for q in qs)))
        for q, l in zip(qs, out):
            wire[q] = l
    if bitstring is None:
        return tensors, tuple(wire)
    for q, b in enumerate(bitstring):
        tensors.append((np.eye(2, dtype=complex)[int(b)], (wire[q],)))
    return tensors, ()


def contract(tensors, output):
    """Contract a labelled network with opt_einsum's greedy path."""
    args = [x for Tn, t in tensors for x in (Tn, list(t))]
    return oe.contract(*args, list(output), optimize="greedy")


def amplitude(circ, n, bitstring):
    return contract(*to_network(circ, n, bitstring))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 3
    print("GHZ(3) amplitudes from closed networks:")
    for b in ("000", "011", "111"):
        print(f"  <{b}|GHZ> = {amplitude(ghz(n), n, b):.6f}")

    n, depth = 8, 6
    circ = random_circuit(n, depth, rng)
    tensors, out = to_network(circ, n)
    psi_tn = contract(tensors, out).reshape(-1)
    err = np.max(np.abs(psi_tn - statevector(circ, n)))
    print(f"\nRandom brickwork n={n}, depth={depth}: {len(tensors)} tensors, "
          f"max |TN - state vector| = {err:.1e}")

    n = 5
    x = 13
    psi = statevector(qft(n), n, x)
    dft = np.exp(2j * np.pi * x * np.arange(2 ** n) / 2 ** n) / np.sqrt(2 ** n)
    print(f"QFT n={n} on |{x}>: max |circuit - DFT formula| = {np.max(np.abs(psi - dft)):.1e}")
