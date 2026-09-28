"""Circuit generation, state-vector simulation, and tensor network construction."""
import numpy as np


def haar_unitary(dim=4, rng=None):
    """Generate a Haar-random unitary from U(dim) via QR decomposition."""
    rng = np.random.default_rng() if rng is None else rng
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return (q * (d / np.abs(d))).astype(np.complex128)


def build_1d_circuit(n_qubits, depth, rng=None):
    """1D brickwork circuit with nearest-neighbour Haar-random U(4) gates."""
    rng = np.random.default_rng(42) if rng is None else rng
    gates = []
    for d in range(depth):
        for q in range(d % 2, n_qubits - 1, 2):
            gates.append((q, q + 1, haar_unitary(4, rng)))
    return gates


def build_2d_sycamore_circuit(H, W, n_cycles, rng=None):
    """2D grid circuit with Sycamore-like ABCD nearest-neighbour gate layers."""
    rng = np.random.default_rng(42) if rng is None else rng
    q_idx = lambda r, c: r * W + c
    A, B, C, D = [], [], [], []
    for r in range(H):
        for c in range(W - 1):
            (A if (r + c) % 2 == 0 else B).append((q_idx(r, c), q_idx(r, c + 1)))
    for r in range(H - 1):
        for c in range(W):
            (C if (r + c) % 2 == 0 else D).append((q_idx(r, c), q_idx(r + 1, c)))
    gates = []
    for _ in range(n_cycles):
        for pattern in (A, B, C, D):
            for q0, q1 in pattern:
                gates.append((q0, q1, haar_unitary(4, rng)))
    return gates


def simulate_statevector(n_qubits, gates, x=None):
    """Exact state-vector simulation returning amplitude <x|C|0...0>."""
    psi = np.zeros([2] * n_qubits, dtype=np.complex128)
    psi.flat[0] = 1.0
    for q0, q1, u in gates:
        u4 = u.reshape(2, 2, 2, 2)
        p_in = list(range(n_qubits))
        out0, out1 = n_qubits, n_qubits + 1
        p_out = list(range(n_qubits))
        p_out[q0], p_out[q1] = out0, out1
        psi = np.einsum(u4, [out0, out1, q0, q1], psi, p_in, p_out)
    idx = 0 if x is None else sum(b * (1 << (n_qubits - 1 - i)) for i, b in enumerate(x))
    return psi.flat[idx]


def build_circuit_tn(n_qubits, gates, x=None):
    """Construct tensor network (tensors, indices, size_dict) for amplitude <x|C|0...0>."""
    wire = [f"q_{i}_0" for i in range(n_qubits)]
    next_id = 1
    tensors, indices = [], []
    for i in range(n_qubits):
        s0 = np.array([1.0, 0.0], dtype=np.complex128)
        tensors.append(s0)
        indices.append((wire[i],))
    for q0, q1, u in gates:
        in0, in1 = wire[q0], wire[q1]
        out0, out1 = f"q_{q0}_{next_id}", f"q_{q1}_{next_id}"
        next_id += 1
        wire[q0], wire[q1] = out0, out1
        tensors.append(u.reshape(2, 2, 2, 2))
        indices.append((out0, out1, in0, in1))
    for i in range(n_qubits):
        sx = np.zeros(2, dtype=np.complex128)
        sx[0 if x is None else x[i]] = 1.0
        tensors.append(sx)
        indices.append((wire[i],))
    size_dict = {ix: 2 for term in indices for ix in term}
    return tensors, indices, size_dict
