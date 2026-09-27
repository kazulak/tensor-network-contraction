"""
Stabilizer Tableau Simulator for Clifford Circuits.

Grounded in the binary symplectic tableau representation described by:
    Scott Aaronson and Daniel Gottesman, "Improved Simulation of Stabilizer Circuits",
    Phys. Rev. A 70, 052328 (2004) [arXiv:quant-ph/0406196].

This module provides polynomial-time O(N^2) per Clifford gate simulation and
O(N^3) exact probability evaluation for stabilizer quantum states.
"""

import numpy as np


class NonCliffordGateError(ValueError):
    """Raised when a non-Clifford gate is encountered in a Clifford-only context."""
    pass


class CliffordGate:
    """Minimal representation of a Clifford gate."""
    def __init__(self, name: str, qubits: tuple):
        self.name = name.upper()
        self.qubits = tuple(qubits)

    def __repr__(self):
        return f"CliffordGate({self.name}, qubits={self.qubits})"


class CliffordCircuit:
    """Container for a sequence of Clifford gates on N qubits."""
    SUPPORTED_GATES = {'H', 'S', 'SDG', 'X', 'Y', 'Z', 'CX', 'CNOT', 'CZ'}

    def __init__(self, num_qubits: int):
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        self.num_qubits = num_qubits
        self.gates = []

    def _validate_qubit(self, q: int):
        if not (0 <= q < self.num_qubits):
            raise ValueError(f"Qubit index {q} out of range [0, {self.num_qubits - 1}]")

    def h(self, q: int):
        self._validate_qubit(q)
        self.gates.append(CliffordGate('H', (q,)))
        return self

    def s(self, q: int):
        self._validate_qubit(q)
        self.gates.append(CliffordGate('S', (q,)))
        return self

    def sdg(self, q: int):
        self._validate_qubit(q)
        self.gates.append(CliffordGate('SDG', (q,)))
        return self

    def x(self, q: int):
        self._validate_qubit(q)
        self.gates.append(CliffordGate('X', (q,)))
        return self

    def y(self, q: int):
        self._validate_qubit(q)
        self.gates.append(CliffordGate('Y', (q,)))
        return self

    def z(self, q: int):
        self._validate_qubit(q)
        self.gates.append(CliffordGate('Z', (q,)))
        return self

    def cx(self, control: int, target: int):
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError(f"Control and target qubits must differ, got {control}")
        self.gates.append(CliffordGate('CX', (control, target)))
        return self

    def cnot(self, control: int, target: int):
        return self.cx(control, target)

    def cz(self, q1: int, q2: int):
        self._validate_qubit(q1)
        self._validate_qubit(q2)
        if q1 == q2:
            raise ValueError(f"Qubits must differ for CZ, got {q1}")
        self.gates.append(CliffordGate('CZ', (q1, q2)))
        return self

    def append(self, name: str, qubits: tuple):
        name_up = name.upper()
        if name_up not in self.SUPPORTED_GATES:
            raise NonCliffordGateError(f"Gate '{name}' is not a supported Clifford gate.")
        for q in qubits:
            self._validate_qubit(q)
        self.gates.append(CliffordGate(name_up, qubits))
        return self

    def to_tableau(self) -> 'StabilizerTableau':
        tab = StabilizerTableau(self.num_qubits)
        for gate in self.gates:
            tab.apply_gate(gate.name, *gate.qubits)
        return tab


def g_phase_exponent(x1: int, z1: int, x2: int, z2: int) -> int:
    """
    Computes the exponent g in {-1, 0, 1} (mod 4) of i in the product of two single-qubit Pauli operators
    under the Hermitian convention P = (-1)^r i^(xz) X^x Z^z such that P1 * P2 = i^g * P3.
    As defined in Aaronson & Gottesman (2004), Section III, Table I.
    """
    if x1 == 0 and z1 == 0:
        return 0
    elif x1 == 1 and z1 == 1:
        return int(z2) - int(x2)
    elif x1 == 1 and z1 == 0:
        return int(z2) * (2 * int(x2) - 1)
    else:  # x1 == 0, z1 == 1
        return int(x2) * (1 - 2 * int(z2))



class StabilizerTableau:
    """
    Aaronson-Gottesman Binary Symplectic Stabilizer Tableau.

    Represents an N-qubit stabilizer state using a (2N) x (2N + 1) binary array.
    Rows 0..N-1: Destabilizers R_1..R_N
    Rows N..2N-1: Stabilizers R_{N+1}..R_{2N}
    Cols 0..N-1: X matrix
    Cols N..2N-1: Z matrix
    Col 2N: Phase vector r (0 for +1, 1 for -1)
    """

    def __init__(self, num_qubits: int):
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        self.n = num_qubits
        self.mat = np.zeros((2 * self.n, 2 * self.n + 1), dtype=np.uint8)

        # Destabilizers: R_i = +X_i
        for i in range(self.n):
            self.mat[i, i] = 1

        # Stabilizers: R_{N+i} = +Z_i
        for i in range(self.n):
            self.mat[self.n + i, self.n + i] = 1

    def copy(self) -> 'StabilizerTableau':
        new_tab = StabilizerTableau.__new__(StabilizerTableau)
        new_tab.n = self.n
        new_tab.mat = self.mat.copy()
        return new_tab

    def _row_mult(self, h: int, i: int):
        """Row h <- Row h * Row i (Pauli multiplication of generators)."""
        n = self.n
        x_h = self.mat[h, :n]
        z_h = self.mat[h, n:2*n]
        x_i = self.mat[i, :n]
        z_i = self.mat[i, n:2*n]

        r_h = int(self.mat[h, 2*n])
        r_i = int(self.mat[i, 2*n])

        g_sum = sum(
            g_phase_exponent(int(x_h[k]), int(z_h[k]), int(x_i[k]), int(z_i[k]))
            for k in range(n)
        )
        new_r = (r_h + r_i + g_sum // 2) % 2

        self.mat[h, :n] ^= x_i
        self.mat[h, n:2*n] ^= z_i
        self.mat[h, 2*n] = new_r


    def h(self, a: int):
        """Hadamard gate on qubit a."""
        n = self.n
        x_col = self.mat[:, a].copy()
        z_col = self.mat[:, a + n].copy()
        r_col = self.mat[:, 2 * n].copy()

        self.mat[:, 2 * n] = r_col ^ (x_col & z_col)
        self.mat[:, a] = z_col
        self.mat[:, a + n] = x_col
        return self

    def s(self, a: int):
        """Phase gate S on qubit a."""
        n = self.n
        x_col = self.mat[:, a].copy()
        z_col = self.mat[:, a + n].copy()
        r_col = self.mat[:, 2 * n].copy()

        self.mat[:, 2 * n] = r_col ^ (x_col & z_col)
        self.mat[:, a + n] = z_col ^ x_col
        return self

    def sdg(self, a: int):
        """Phase dagger S^dag (S^3) on qubit a."""
        n = self.n
        x_col = self.mat[:, a].copy()
        z_col = self.mat[:, a + n].copy()
        r_col = self.mat[:, 2 * n].copy()

        self.mat[:, 2 * n] = r_col ^ (x_col & (1 ^ z_col))
        self.mat[:, a + n] = z_col ^ x_col
        return self

    def x(self, a: int):
        """Pauli X gate on qubit a."""
        n = self.n
        z_col = self.mat[:, a + n].copy()
        r_col = self.mat[:, 2 * n].copy()
        self.mat[:, 2 * n] = r_col ^ z_col
        return self

    def y(self, a: int):
        """Pauli Y gate on qubit a."""
        n = self.n
        x_col = self.mat[:, a].copy()
        z_col = self.mat[:, a + n].copy()
        r_col = self.mat[:, 2 * n].copy()
        self.mat[:, 2 * n] = r_col ^ x_col ^ z_col
        return self

    def z(self, a: int):
        """Pauli Z gate on qubit a."""
        n = self.n
        x_col = self.mat[:, a].copy()
        r_col = self.mat[:, 2 * n].copy()
        self.mat[:, 2 * n] = r_col ^ x_col
        return self

    def cx(self, a: int, b: int):
        """CNOT (Control a, Target b)."""
        n = self.n
        x_a = self.mat[:, a].copy()
        z_a = self.mat[:, a + n].copy()
        x_b = self.mat[:, b].copy()
        z_b = self.mat[:, b + n].copy()
        r_col = self.mat[:, 2 * n].copy()

        self.mat[:, 2 * n] = r_col ^ (x_a & z_b & (x_b ^ z_a ^ 1))
        self.mat[:, b] = x_b ^ x_a
        self.mat[:, a + n] = z_a ^ z_b
        return self

    def cnot(self, a: int, b: int):
        return self.cx(a, b)

    def cz(self, a: int, b: int):
        """CZ gate (Control a, Target b)."""
        n = self.n
        x_a = self.mat[:, a].copy()
        z_a = self.mat[:, a + n].copy()
        x_b = self.mat[:, b].copy()
        z_b = self.mat[:, b + n].copy()
        r_col = self.mat[:, 2 * n].copy()

        self.mat[:, 2 * n] = r_col ^ (x_a & x_b)
        self.mat[:, a + n] = z_a ^ x_b
        self.mat[:, b + n] = z_b ^ x_a
        return self

    def apply_gate(self, name: str, *qubits):
        name_up = name.upper()
        if name_up == 'H':
            self.h(qubits[0])
        elif name_up == 'S':
            self.s(qubits[0])
        elif name_up == 'SDG':
            self.sdg(qubits[0])
        elif name_up == 'X':
            self.x(qubits[0])
        elif name_up == 'Y':
            self.y(qubits[0])
        elif name_up == 'Z':
            self.z(qubits[0])
        elif name_up in ('CX', 'CNOT'):
            self.cx(qubits[0], qubits[1])
        elif name_up == 'CZ':
            self.cz(qubits[0], qubits[1])
        else:
            raise NonCliffordGateError(f"Unsupported gate '{name}'.")

    def measure(self, p: int, rng=None) -> tuple:
        """
        Measures qubit p in computational basis.
        Returns (outcome_bit, is_random).
        """
        n = self.n
        if not (0 <= p < n):
            raise ValueError(f"Qubit index {p} out of range [0, {n-1}]")

        stabs_x = self.mat[n:2*n, p]
        pos = np.where(stabs_x == 1)[0]

        if len(pos) > 0:
            p_stab_idx = n + pos[0]

            for j in range(2 * n):
                if j != p_stab_idx and self.mat[j, p] == 1:
                    self._row_mult(j, p_stab_idx)

            destab_idx = p_stab_idx - n
            self.mat[destab_idx, :] = self.mat[p_stab_idx, :].copy()

            if rng is not None:
                outcome = int(rng.choice([0, 1]))
            else:
                outcome = int(np.random.choice([0, 1]))

            self.mat[p_stab_idx, :] = 0
            self.mat[p_stab_idx, p + n] = 1
            self.mat[p_stab_idx, 2 * n] = outcome

            return outcome, True
        else:
            scratch = np.zeros(2 * n + 1, dtype=np.uint8)
            for i in range(n):
                if self.mat[i, p] == 1:
                    stab_row_idx = n + i
                    self._row_mult_scratch(scratch, self.mat[stab_row_idx, :])

            outcome = int(scratch[2 * n])
            return outcome, False

    def _row_mult_scratch(self, scratch_row, row_i):
        n = self.n
        x_h = scratch_row[:n]
        z_h = scratch_row[n:2*n]
        x_i = row_i[:n]
        z_i = row_i[n:2*n]

        r_h = int(scratch_row[2*n])
        r_i = int(row_i[2*n])

        g_sum = sum(
            g_phase_exponent(int(x_h[k]), int(z_h[k]), int(x_i[k]), int(z_i[k]))
            for k in range(n)
        )
        new_r = (r_h + r_i + g_sum // 2) % 2

        scratch_row[:n] ^= x_i
        scratch_row[n:2*n] ^= z_i
        scratch_row[2*n] = new_r


    def get_probability(self, bitstring) -> float:
        """
        Evaluates exact probability P(b) of computational basis bitstring b in O(N^3) time.
        Does not mutate self.
        """
        if isinstance(bitstring, str):
            bits = [int(c) for c in bitstring]
        else:
            bits = list(bitstring)

        if len(bits) != self.n:
            raise ValueError(f"Bitstring length ({len(bits)}) must match num_qubits ({self.n})")

        tab = self.copy()
        prob = 1.0

        for q in range(self.n):
            target_bit = bits[q]

            stabs_x = tab.mat[tab.n:2*tab.n, q]
            pos = np.where(stabs_x == 1)[0]

            if len(pos) > 0:
                prob *= 0.5
                p_stab_idx = tab.n + pos[0]

                for j in range(2 * tab.n):
                    if j != p_stab_idx and tab.mat[j, q] == 1:
                        tab._row_mult(j, p_stab_idx)

                destab_idx = p_stab_idx - tab.n
                tab.mat[destab_idx, :] = tab.mat[p_stab_idx, :].copy()

                tab.mat[p_stab_idx, :] = 0
                tab.mat[p_stab_idx, q + tab.n] = 1
                tab.mat[p_stab_idx, 2 * tab.n] = target_bit
            else:
                scratch = np.zeros(2 * tab.n + 1, dtype=np.uint8)
                for i in range(tab.n):
                    if tab.mat[i, q] == 1:
                        stab_row_idx = tab.n + i
                        tab._row_mult_scratch(scratch, tab.mat[stab_row_idx, :])

                det_outcome = int(scratch[2 * tab.n])
                if target_bit != det_outcome:
                    return 0.0

        return prob

    def sample(self, shots: int = 100, seed=None) -> dict:
        """Samples shots computational basis bitstrings from the state."""
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")

        rng = np.random.default_rng(seed)
        counts = {}

        for _ in range(shots):
            tab = self.copy()
            bits = []
            for q in range(self.n):
                bit, _ = tab.measure(q, rng=rng)
                bits.append(str(bit))
            b_str = "".join(bits)
            counts[b_str] = counts.get(b_str, 0) + 1

        return counts

    def get_stabilizer_strings(self) -> list:
        """Returns string representation of the N stabilizer generators."""
        n = self.n
        pauli_chars = ['I', 'X', 'Z', 'Y']
        res = []
        for i in range(n, 2 * n):
            phase = "-" if self.mat[i, 2 * n] == 1 else "+"
            chars = []
            for q in range(n):
                x = self.mat[i, q]
                z = self.mat[i, q + n]
                idx = x + 2 * z
                chars.append(pauli_chars[idx])
            res.append(phase + "".join(chars))
        return res


# Standard matrix reference definitions for Clifford gate classification
_H_MAT = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
_S_MAT = np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128)
_SDG_MAT = np.array([[1.0, 0.0], [0.0, -1j]], dtype=np.complex128)
_X_MAT = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y_MAT = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
_Z_MAT = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_I_MAT = np.eye(2, dtype=np.complex128)

_1Q_CLIFFORD_DICT = {
    'H': _H_MAT,
    'S': _S_MAT,
    'SDG': _SDG_MAT,
    'X': _X_MAT,
    'Y': _Y_MAT,
    'Z': _Z_MAT,
    'I': _I_MAT,
}

_CNOT_MAT_4X4 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)

_CZ_MAT_4X4 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=np.complex128)


def _matrix_match_up_to_phase(A: np.ndarray, B: np.ndarray, tol: float = 1e-6) -> bool:
    """Returns True if matrix A equals e^(i phi) * B within tolerance."""
    if A.shape != B.shape:
        return False
    # Find max element in B to determine reference phase
    max_idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
    if np.abs(B[max_idx]) < 1e-12 or np.abs(A[max_idx]) < 1e-12:
        return False
    phase_ratio = A[max_idx] / B[max_idx]
    if np.abs(np.abs(phase_ratio) - 1.0) > tol:
        return False
    diff = A - phase_ratio * B
    return np.max(np.abs(diff)) < tol


def classify_1q_gate(mat: np.ndarray) -> str:
    """Classifies a 2x2 unitary matrix as a standard Clifford gate name or raises NonCliffordGateError."""
    mat_c = np.asarray(mat, dtype=np.complex128)
    for name, ref in _1Q_CLIFFORD_DICT.items():
        if _matrix_match_up_to_phase(mat_c, ref):
            return name
    raise NonCliffordGateError("1-qubit matrix is not a recognized Clifford gate.")


def classify_2q_gate(mat: np.ndarray) -> str:
    """Classifies a 4x4 or 2x2x2x2 unitary matrix as CNOT or CZ or raises NonCliffordGateError."""
    mat_c = np.asarray(mat, dtype=np.complex128)
    if mat_c.ndim == 4:
        mat_c = mat_c.reshape((4, 4))
    if _matrix_match_up_to_phase(mat_c, _CNOT_MAT_4X4):
        return 'CX'
    if _matrix_match_up_to_phase(mat_c, _CZ_MAT_4X4):
        return 'CZ'
    raise NonCliffordGateError("2-qubit matrix is not a recognized Clifford gate (CNOT/CZ).")


def from_tensor_network_gates(num_qubits: int, gate_sequence: list) -> CliffordCircuit:
    """
    Adapter converting a gate sequence tuple list [(gate_type, gate_mat, qubits), ...]
    into a CliffordCircuit object. Raises NonCliffordGateError if any non-Clifford gate is present.
    """
    circuit = CliffordCircuit(num_qubits)
    for item in gate_sequence:
        gate_type, gate_mat, qubits = item[0], item[1], item[2]
        if gate_type == 1:
            name = classify_1q_gate(gate_mat)
            if name != 'I':
                circuit.append(name, qubits)
        elif gate_type == 2:
            name = classify_2q_gate(gate_mat)
            circuit.append(name, qubits)
        else:
            raise NonCliffordGateError(f"Unsupported gate type: {gate_type}")
    return circuit
