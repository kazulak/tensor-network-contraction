import numpy as np

def haar_unitary(dim=4):
    Z = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q @ np.diag(ph)

def apply_gate_sv(state, gate, q1, q2, n_qubits):
    state = np.moveaxis(state, [q1, q2], [-2, -1])
    state_shape = state.shape
    state = state.reshape(-1, 4)
    gate_matrix = gate.reshape(4, 4)
    state = state @ gate_matrix.T
    state = state.reshape(state_shape)
    state = np.moveaxis(state, [-2, -1], [q1, q2])
    return state

def build_1d_circuit(n_qubits, depth):
    gates = []
    for d in range(depth):
        start = d % 2
        for i in range(start, n_qubits - 1, 2):
            gates.append((i, i+1, haar_unitary(4).reshape(2, 2, 2, 2)))
    return gates

def build_2d_circuit(rows, cols, depth):
    gates = []
    def idx(r, c): return r * cols + c
    
    patterns = [
        [((r, c), (r, c+1)) for r in range(rows) for c in range(0, cols-1, 2)], # H even
        [((r, c), (r+1, c)) for r in range(0, rows-1, 2) for c in range(cols)], # V even
        [((r, c), (r, c+1)) for r in range(rows) for c in range(1, cols-1, 2)], # H odd
        [((r, c), (r+1, c)) for r in range(1, rows-1, 2) for c in range(cols)], # V odd
    ]
    
    for d in range(depth):
        pat = patterns[d % 4]
        for (r1, c1), (r2, c2) in pat:
            q1, q2 = idx(r1, c1), idx(r2, c2)
            gates.append((q1, q2, haar_unitary(4).reshape(2, 2, 2, 2)))
    return gates

def get_tensor_network(n_qubits, gates):
    tensors = []
    eqs = []
    wire_counts = {q: 0 for q in range(n_qubits)}
    
    def get_wire(q, count):
        return f"q{q}_{count}"

    zero_state = np.array([1.0, 0.0], dtype=np.complex128)
    for q in range(n_qubits):
        tensors.append(zero_state)
        eqs.append([get_wire(q, 0)])

    for q1, q2, gate in gates:
        w_in1 = get_wire(q1, wire_counts[q1])
        w_in2 = get_wire(q2, wire_counts[q2])
        wire_counts[q1] += 1
        wire_counts[q2] += 1
        w_out1 = get_wire(q1, wire_counts[q1])
        w_out2 = get_wire(q2, wire_counts[q2])
        
        tensors.append(gate)
        eqs.append([w_out1, w_out2, w_in1, w_in2])
        
    for q in range(n_qubits):
        tensors.append(zero_state)
        eqs.append([get_wire(q, wire_counts[q])])
        
    return tensors, eqs
