import json
import random
import numpy as np
import matplotlib.pyplot as plt
import opt_einsum as oe
import cotengra as ctg
from collections import Counter


def generate_haar_unitary(seed=None):
    """Generate a Haar-random 4x4 unitary matrix reshaped to (2,2,2,2)."""
    if seed is not None:
        np.random.seed(seed)
    z = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return (q * ph).reshape((2, 2, 2, 2))


def state_vector_simulate(N, gates, target_bitstring):
    """Perform exact state-vector simulation of a quantum circuit."""
    state = np.zeros([2] * N, dtype=np.complex128)
    state[(0,) * N] = 1.0

    for q1, q2, U in gates:
        perm = [i for i in range(N) if i not in (q1, q2)] + [q1, q2]
        state_perm = np.transpose(state, perm)
        state_mat = state_perm.reshape((-1, 4))
        new_state_mat = state_mat @ U.reshape((4, 4)).T
        new_state = new_state_mat.reshape([2] * (N - 2) + [2, 2])

        inv_perm = np.zeros(N, dtype=int)
        for idx, pos in enumerate(perm):
            inv_perm[pos] = idx
        state = np.transpose(new_state, inv_perm)

    return state[tuple(target_bitstring)]


def build_1d_brickwork(N=24, L=16):
    """Build a 1D brickwork circuit with N qubits and depth L."""
    gates = []
    for l in range(L):
        start = l % 2
        for q in range(start, N - 1, 2):
            gates.append((q, q + 1, generate_haar_unitary()))
    return gates


def build_2d_grid(N_row=5, N_col=5, L=6):
    """Build a 2D grid circuit (Sycamore-like layers) with N_row x N_col qubits and depth L."""
    gates = []
    for l in range(L):
        for r in range(N_row):
            for c in range(N_col):
                q1 = r * N_col + c
                if l % 2 == 0 and c + 1 < N_col:
                    q2 = r * N_col + (c + 1)
                    gates.append((q1, q2, generate_haar_unitary()))
                elif l % 2 == 1 and r + 1 < N_row:
                    q2 = (r + 1) * N_col + c
                    gates.append((q1, q2, generate_haar_unitary()))
    return gates


def circuit_to_tn(N, gates, target_bitstring=None):
    """Convert circuit gates into tensor network format (tensors and index tuples)."""
    if target_bitstring is None:
        target_bitstring = (0,) * N

    wire_count = 0

    def get_wire():
        nonlocal wire_count
        w = oe.get_symbol(wire_count)
        wire_count += 1
        return w

    curr_wires = [get_wire() for _ in range(N)]
    tensors, indices = [], []

    for q in range(N):
        v0 = np.array([1.0, 0.0], dtype=np.complex128)
        tensors.append(v0)
        indices.append([curr_wires[q]])

    for q1, q2, U in gates:
        in1, in2 = curr_wires[q1], curr_wires[q2]
        out1, out2 = get_wire(), get_wire()
        curr_wires[q1], curr_wires[q2] = out1, out2
        tensors.append(U)
        indices.append([out1, out2, in1, in2])

    for q in range(N):
        vq = np.zeros(2, dtype=np.complex128)
        vq[target_bitstring[q]] = 1.0
        tensors.append(vq)
        indices.append([curr_wires[q]])

    return tensors, [tuple(idx) for idx in indices]


def analyze_circuit(name, N, gates, target_slice_counts=[4, 16, 64, 256, 1024, 4096, 16384]):
    """Analyze slicing cost and memory reduction for a given circuit geometry."""
    tensors, inputs = circuit_to_tn(N, gates)
    path, info = oe.contract_path(*[item for pair in zip(tensors, inputs) for item in pair])
    tree = ctg.ContractionTree.from_info(info)

    unsliced_flops = float(tree.total_flops())
    unsliced_max_size = float(tree.max_size())

    all_inds = Counter([ind for idx in inputs for ind in idx])
    internal_inds = [ind for ind, count in all_inds.items() if count >= 2]

    results = []
    for target_s in target_slice_counts:
        sf = ctg.SliceFinder(tree, target_slices=target_s)
        opt_inds, opt_costs = sf.search()

        k = len(opt_inds)
        opt_tot_flops = float(opt_costs.flops * (2**k))
        opt_max_size = float(opt_costs.size)
        opt_overhead = opt_tot_flops / unsliced_flops
        opt_mem_red = unsliced_max_size / opt_max_size

        rnd_sizes, rnd_flops_list = [], []
        for seed_idx in range(5):
            random.seed(1000 + seed_idx)
            sample_inds = random.sample(internal_inds, k)
            t_copy = tree.copy()
            for ix in sample_inds:
                t_copy.remove_ind(ix)
            rnd_sizes.append(float(t_copy.max_size()))
            rnd_flops_list.append(float(t_copy.total_flops() * (2**k)))

        rnd_max_size = float(np.mean(rnd_sizes))
        rnd_tot_flops = float(np.mean(rnd_flops_list))
        rnd_overhead = rnd_tot_flops / unsliced_flops
        rnd_mem_red = unsliced_max_size / rnd_max_size

        results.append({
            'k': k,
            'n_slices': 2**k,
            'opt_max_size': opt_max_size,
            'opt_mem_red': opt_mem_red,
            'opt_tot_flops': opt_tot_flops,
            'opt_overhead': opt_overhead,
            'rnd_max_size': rnd_max_size,
            'rnd_mem_red': rnd_mem_red,
            'rnd_tot_flops': rnd_tot_flops,
            'rnd_overhead': rnd_overhead
        })

    return {
        'name': name,
        'N': N,
        'unsliced_flops': unsliced_flops,
        'unsliced_max_size': unsliced_max_size,
        'slicing_data': results
    }


def run_study():
    np.random.seed(42)
    random.seed(42)

    gates_1d = build_1d_brickwork(24, 16)
    res_1d = analyze_circuit("1D Brickwork (24q, L=16)", 24, gates_1d)

    gates_2d = build_2d_grid(5, 5, 6)
    res_2d = analyze_circuit("2D Grid (5x5, L=6)", 25, gates_2d)

    all_data = {'1d_brickwork': res_1d, '2d_grid': res_2d}
    with open("results.json", "w") as f:
        json.dump(all_data, f, indent=2)

    plt.figure(figsize=(8.5, 5.5))
    k_2d = [r['k'] for r in res_2d['slicing_data']]
    plt.plot(k_2d, [r['opt_overhead'] for r in res_2d['slicing_data']], 'o-', color='#1f77b4', lw=2, label='2D Grid (5x5) - Optimal Slicing')
    plt.plot(k_2d, [r['rnd_overhead'] for r in res_2d['slicing_data']], 's--', color='#aec7e8', lw=1.5, label='2D Grid (5x5) - Random Slicing')

    k_1d = [r['k'] for r in res_1d['slicing_data']]
    plt.plot(k_1d, [r['opt_overhead'] for r in res_1d['slicing_data']], 'd-', color='#d62728', lw=2, label='1D Brickwork (24q) - Optimal Slicing')
    plt.plot(k_1d, [r['rnd_overhead'] for r in res_1d['slicing_data']], 'x--', color='#ff9896', lw=1.5, label='1D Brickwork (24q) - Random Slicing')
    plt.plot(k_2d, [2**k for k in k_2d], 'k:', alpha=0.5, label='Naive $2^k$ FLOP Multiplier')

    plt.yscale('log')
    plt.xlabel('Number of Sliced Indices ($k$)', fontsize=11)
    plt.ylabel('FLOP Overhead Factor ($F_{\\text{sliced}} / F_{\\text{unsliced}}$)', fontsize=11)
    plt.title('FLOP Overhead vs. Sliced Indices by Geometry & Index Choice', fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=9.5, loc='upper left')
    plt.tight_layout()
    plt.savefig("flops_overhead_vs_memory.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_study()
