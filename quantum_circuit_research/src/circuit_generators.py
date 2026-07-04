import numpy as np

# Standard real gates
H_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)
X_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

# CNOT gate: shape (out1, out2, in1, in2)
CNOT_gate = np.zeros((2, 2, 2, 2), dtype=np.float64)
CNOT_gate[0, 0, 0, 0] = 1.0
CNOT_gate[0, 1, 0, 1] = 1.0
CNOT_gate[1, 1, 1, 0] = 1.0
CNOT_gate[1, 0, 1, 1] = 1.0

# CZ gate: shape (out1, out2, in1, in2)
CZ_gate = np.zeros((2, 2, 2, 2), dtype=np.float64)
CZ_gate[0, 0, 0, 0] = 1.0
CZ_gate[0, 1, 0, 1] = 1.0
CZ_gate[1, 0, 1, 0] = 1.0
CZ_gate[1, 1, 1, 1] = -1.0

def get_random_o2():
    theta = np.random.uniform(0, 2 * np.pi)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)

class CircuitBuilder:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.tensors = []
        self.edges = []
        self.q_versions = [0] * num_qubits
        
        # Add |0> inputs
        for q in range(num_qubits):
            self.tensors.append(np.array([1.0, 0.0], dtype=np.float64))
            self.edges.append([f"q_{q}_0"])
            
    def apply_1q_gate(self, gate_arr, q):
        v = self.q_versions[q]
        in_leg = f"q_{q}_{v}"
        out_leg = f"q_{q}_{v+1}"
        
        self.tensors.append(gate_arr)
        self.edges.append([out_leg, in_leg])
        self.q_versions[q] += 1
        
    def apply_2q_gate(self, gate_arr, q1, q2):
        v1 = self.q_versions[q1]
        v2 = self.q_versions[q2]
        
        in_leg1 = f"q_{q1}_{v1}"
        in_leg2 = f"q_{q2}_{v2}"
        out_leg1 = f"q_{q1}_{v1+1}"
        out_leg2 = f"q_{q2}_{v2+1}"
        
        self.tensors.append(gate_arr)
        self.edges.append([out_leg1, out_leg2, in_leg1, in_leg2])
        self.q_versions[q1] += 1
        self.q_versions[q2] += 1
        
    def close_circuit(self):
        for q in range(self.num_qubits):
            v = self.q_versions[q]
            final_leg = f"q_{q}_{v}"
            proj = np.array([1.0, 0.0], dtype=np.float64) if np.random.rand() > 0.5 else np.array([0.0, 1.0], dtype=np.float64)
            self.tensors.append(proj)
            self.edges.append([final_leg])
            
        return self.tensors, self.edges

# 1. BB84 Protocol (BB_n)
# Qubits: n, 1Q Gates: 2n, 2Q Gates: 0
def generate_bb84(num_qubits):
    builder = CircuitBuilder(num_qubits)
    for q in range(num_qubits):
        builder.apply_1q_gate(get_random_o2(), q)
        builder.apply_1q_gate(get_random_o2(), q)
    return builder.close_circuit()

# 2. Bernstein–Vazirani (BV_n)
# Qubits: n, 1Q Gates: 2n, 2Q Gates: n-1
def generate_bernstein_vazirani(num_qubits):
    builder = CircuitBuilder(num_qubits)
    target = num_qubits - 1
    
    # 1. H on all qubits
    for q in range(num_qubits):
        builder.apply_1q_gate(H_gate, q)
        
    # 2. X on target
    builder.apply_1q_gate(X_gate, target)
    
    # 3. CNOT oracle from query qubits to target
    for q in range(num_qubits - 1):
        builder.apply_2q_gate(CNOT_gate, q, target)
        
    # 4. H on query qubits
    for q in range(num_qubits - 1):
        builder.apply_1q_gate(H_gate, q)
        
    return builder.close_circuit()

# 3. Error Detection Code (EDC_n)
# Qubits: n, 1Q Gates: 2n, 2Q Gates: 2n-2
def generate_error_detection(num_qubits):
    builder = CircuitBuilder(num_qubits)
    
    # 1. H on all qubits (n gates)
    for q in range(num_qubits):
        builder.apply_1q_gate(H_gate, q)
        
    # 2. CNOTs forward (n-1 gates)
    for q in range(num_qubits - 1):
        builder.apply_2q_gate(CNOT_gate, q, q + 1)
        
    # 3. CNOTs backward (n-1 gates)
    for q in range(num_qubits - 1):
        builder.apply_2q_gate(CNOT_gate, q + 1, q)
        
    # 4. H on all qubits (n gates)
    for q in range(num_qubits):
        builder.apply_1q_gate(H_gate, q)
        
    return builder.close_circuit()

# 4. Hidden Subgroup Problem (HS_2n)
# Qubits: 2n, 1Q Gates: 6n, 2Q Gates: 2n
def generate_hidden_subgroup(num_qubits):
    # num_qubits is 2n, so n = num_qubits // 2
    n = num_qubits // 2
    builder = CircuitBuilder(num_qubits)
    
    # 1. H on all 2n qubits (2n gates)
    for q in range(num_qubits):
        builder.apply_1q_gate(H_gate, q)
        
    # 2. Oracle CNOTs (2n gates)
    for i in range(n):
        builder.apply_2q_gate(CNOT_gate, i, i + n)
        builder.apply_2q_gate(CNOT_gate, i + n, i)
        
    # 3. Register H (n gates)
    for q in range(n):
        builder.apply_1q_gate(H_gate, q)
        
    # 4. Random rotation gates (3n gates)
    for q in range(num_qubits):
        builder.apply_1q_gate(get_random_o2(), q)
    for q in range(n):
        # We need n more single qubit gates to reach exactly 6n
        builder.apply_1q_gate(get_random_o2(), q)
        
    return builder.close_circuit()

# 5. Quantum Random Number Generator (QRNG_n)
# Qubits: n, 1Q Gates: n, 2Q Gates: 0
def generate_qrng(num_qubits):
    builder = CircuitBuilder(num_qubits)
    for q in range(num_qubits):
        builder.apply_1q_gate(H_gate, q)
    return builder.close_circuit()

# 6. Exclusive-OR (XOR_n)
# Qubits: n, 1Q Gates: 0, 2Q Gates: n-1
def generate_xor(num_qubits):
    builder = CircuitBuilder(num_qubits)
    target = num_qubits - 1
    for q in range(num_qubits - 1):
        builder.apply_2q_gate(CNOT_gate, q, target)
    return builder.close_circuit()

# 7. Sycamore-like Random Circuit (Reference)
def generate_sycamore_like(rows, cols, depth):
    num_qubits = rows * cols
    builder = CircuitBuilder(num_qubits)
    
    def get_q_idx(r, c):
        return r * cols + c

    patterns = [
        lambda d: [(get_q_idx(r, c), get_q_idx(r + 1, c)) for r in range(0, rows - 1, 2) for c in range(cols)],
        lambda d: [(get_q_idx(r, c), get_q_idx(r, c + 1)) for r in range(rows) for c in range(0, cols - 1, 2)],
        lambda d: [(get_q_idx(r, c), get_q_idx(r + 1, c)) for r in range(1, rows - 1, 2) for c in range(cols)],
        lambda d: [(get_q_idx(r, c), get_q_idx(r, c + 1)) for r in range(rows) for c in range(1, cols - 1, 2)]
    ]

    for d in range(depth):
        for q in range(num_qubits):
            builder.apply_1q_gate(get_random_o2(), q)
            
        pairs = patterns[d % len(patterns)](d)
        for q1, q2 in pairs:
            builder.apply_2q_gate(CZ_gate, q1, q2)
            
    return builder.close_circuit()

# 8. Random Circuit (Arbitrary Connectivity)
def generate_random_arbitrary(num_qubits, depth):
    builder = CircuitBuilder(num_qubits)
    for d in range(depth):
        for q in range(num_qubits):
            builder.apply_1q_gate(get_random_o2(), q)
        active_qubits = list(range(num_qubits))
        np.random.shuffle(active_qubits)
        for i in range(0, len(active_qubits) - 1, 2):
            q1 = active_qubits[i]
            q2 = active_qubits[i+1]
            
            M = np.random.normal(size=(4, 4))
            Q, R = np.linalg.qr(M)
            d_mat = np.diag(R)
            ph = d_mat / np.abs(d_mat)
            Q = Q * ph
            builder.apply_2q_gate(Q.reshape(2, 2, 2, 2), q1, q2)
            
    return builder.close_circuit()
