"""Slicing study for tensor network contraction of random quantum circuits.
Measures FLOPs and peak intermediate tensor sizes for 1D and 2D geometries.
"""

import numpy as np
import opt_einsum as oe
import cotengra as ctg
import matplotlib.pyplot as plt


def haar_u4(rng):
    """Generate a Haar-random U(4) unitary matrix in complex128 (Mezzadri 2007)."""
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return (q * ph).astype(np.complex128)


def build_1d_circuit(n_qubits, depth, rng):
    """1D brickwork circuit with nearest-neighbour two-qubit Haar gates."""
    gates = []
    for d in range(depth):
        for q in range(d % 2, n_qubits - 1, 2):
            gates.append((q, q + 1, haar_u4(rng)))
    return gates


def build_2d_circuit(nrows, ncols, depth, rng):
    """2D grid circuit with Sycamore-like ABCD coupling layers."""
    gates = []
    for d in range(depth):
        p = d % 4
        if p == 0:
            pairs = [(r * ncols + c, (r + 1) * ncols + c) for r in range(0, nrows - 1, 2) for c in range(ncols)]
        elif p == 1:
            pairs = [(r * ncols + c, (r + 1) * ncols + c) for r in range(1, nrows - 1, 2) for c in range(ncols)]
        elif p == 2:
            pairs = [(r * ncols + c, r * ncols + c + 1) for r in range(nrows) for c in range(0, ncols - 1, 2)]
        elif p == 3:
            pairs = [(r * ncols + c, r * ncols + c + 1) for r in range(nrows) for c in range(1, ncols - 1, 2)]
        for q0, q1 in pairs:
            gates.append((q0, q1, haar_u4(rng)))
    return gates


def simulate_sv(n_qubits, gates, x_bitstring):
    """Compute amplitude <x|C|0...0> via exact state-vector simulation."""
    psi = np.zeros([2] * n_qubits, dtype=np.complex128)
    psi.flat[0] = 1.0
    for q0, q1, u in gates:
        u_t = u.reshape(2, 2, 2, 2)
        psi = np.tensordot(u_t, psi, axes=([2, 3], [q0, q1]))
        psi = np.moveaxis(psi, [0, 1], [q0, q1])
    return psi[tuple(x_bitstring)]


def circuit_to_tn(n_qubits, gates, x_bitstring=None):
    """Convert circuit to tensor network equation, tensors, and internal edges."""
    edge_cnt = 0
    curr = {q: f'in_{q}' for q in range(n_qubits)}
    inputs = [[curr[q]] for q in range(n_qubits)]
    tensors = [np.array([1.0, 0.0], dtype=np.complex128) for _ in range(n_qubits)]
    internal_edges = []

    for q0, q1, u in gates:
        o0, o1 = f'e_{edge_cnt}', f'e_{edge_cnt + 1}'
        edge_cnt += 2
        inputs.append([o0, o1, curr[q0], curr[q1]])
        tensors.append(u.reshape(2, 2, 2, 2))
        internal_edges.extend([o0, o1])
        curr[q0], curr[q1] = o0, o1

    if x_bitstring is not None:
        for q in range(n_qubits):
            inputs.append([curr[q]])
            v = np.zeros(2, dtype=np.complex128)
            v[x_bitstring[q]] = 1.0
            tensors.append(v)

    size_dict = {ix: 2 for ix in set(sum(inputs, []))}
    return inputs, tensors, size_dict, internal_edges


def contract_sliced_tn(tree, sliced_tree, tensors):
    """Contract all slices of a sliced contraction tree using opt_einsum."""
    sl_inputs = sliced_tree.get_inputs_sliced()
    path = sliced_tree.get_path()
    total = 0.0 + 0.0j
    for s in range(sliced_tree.nslices):
        sl_arrays = sliced_tree.slice_arrays(tensors, s)
        total += oe.contract(*[x for pair in zip(sl_arrays, sl_inputs) for x in pair], [], optimize=path)
    return total


def run_benchmark():
    """Run slicing experiments on 1D and 2D circuits and plot results."""
    rng = np.random.default_rng(2026)
    configs = [
        ('1D Brickwork (N=12, D=10)', 12, build_1d_circuit(12, 10, rng)),
        ('2D Sycamore Grid (4x4, D=8)', 16, build_2d_circuit(4, 4, 8, rng))
    ]
    results = {}
    ks = [1, 2, 3, 4, 5, 6, 7, 8]

    for label, n_q, gates in configs:
        inputs, tensors, size_dict, internal_edges = circuit_to_tn(n_q, gates, [0] * n_q)
        opt = ctg.GreedyOptimizer()
        tree_0 = opt.search(inputs, [], size_dict)
        f0 = tree_0.total_flops()
        m0 = tree_0.max_size()

        rows = []
        for k in ks:
            # Greedy slicing via SliceFinder
            sf = ctg.SliceFinder(tree_0, target_slices=2**k)
            ix_sl, _ = sf.search()
            t_g = tree_0.copy()
            for ix in ix_sl:
                t_g.remove_ind_(ix)

            # Random internal edge slicing (10 trials)
            r_m, r_f = [], []
            for _ in range(10):
                t_r = tree_0.copy()
                for ix in rng.choice(internal_edges, size=k, replace=False):
                    t_r.remove_ind_(ix)
                r_m.append(t_r.max_size())
                r_f.append(t_r.total_flops())

            # Peripheral / boundary slicing
            t_b = tree_0.copy()
            for q in range(k):
                t_b.remove_ind_(f'in_{q}')

            rows.append({
                'k': k, 'slices': 2**k,
                'g_size': t_g.max_size(), 'g_flops': t_g.total_flops(), 'g_oh': t_g.total_flops() / f0,
                'r_size': np.mean(r_m), 'r_flops': np.mean(r_f), 'r_oh': np.mean(r_f) / f0,
                'b_size': t_b.max_size(), 'b_flops': t_b.total_flops(), 'b_oh': t_b.total_flops() / f0,
            })
        results[label] = {'f0': f0, 'm0': m0, 'rows': rows}

    plot_results(results, ks)
    return results


def plot_results(results, ks):
    """Generate publication-ready 2-panel figure comparing slicing strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    colors = {'1D': '#1f77b4', '2D': '#2ca02c', 'Rand': '#d62728', 'Bnd': '#7f7f7f'}

    # Panel (a): FLOP Overhead vs Sliced Indices
    ax1.plot(ks, [2**k for k in ks], 'k--', label=r'Theoretical naive ($2^k$)', alpha=0.6)
    for name, key, col in [('1D Brickwork (N=12, D=10)', '1D', colors['1D']),
                           ('2D Sycamore Grid (4x4, D=8)', '2D', colors['2D'])]:
        d = results[name]
        g_oh = [r['g_oh'] for r in d['rows']]
        r_oh = [r['r_oh'] for r in d['rows']]
        ax1.plot(ks, g_oh, 'o-', color=col, lw=2, label=f'{key} Greedy (SliceFinder)')
        ax1.plot(ks, r_oh, 's--', color=col, alpha=0.45, label=f'{key} Random')

    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Sliced Indices ($k$)')
    ax1.set_ylabel('FLOP Overhead ($C_{\\mathrm{sliced}} / C_0$)')
    ax1.set_title('(a) Computational Overhead vs. Slicing Depth')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=8, framealpha=0.9)

    # Panel (b): Memory Reduction Factor vs Sliced Indices
    for name, key, col in [('1D Brickwork (N=12, D=10)', '1D', colors['1D']),
                           ('2D Sycamore Grid (4x4, D=8)', '2D', colors['2D'])]:
        d = results[name]
        m0 = d['m0']
        g_red = [m0 / r['g_size'] for r in d['rows']]
        r_red = [m0 / r['r_size'] for r in d['rows']]
        ax2.plot(ks, g_red, 'o-', color=col, lw=2, label=f'{key} Greedy')
        ax2.plot(ks, r_red, 's--', color=col, alpha=0.45, label=f'{key} Random')

    ax2.set_yscale('log', base=2)
    ax2.set_xlabel('Number of Sliced Indices ($k$)')
    ax2.set_ylabel('Memory Reduction Factor ($M_0 / M_{\\mathrm{sliced}}$)')
    ax2.set_title('(b) Peak Memory Reduction vs. Slicing Depth')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('slicing_study.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    res = run_benchmark()
    for name, data in res.items():
        print(f"\n### {name}")
        print(f"Baseline: FLOPs = {data['f0']:,}, Max Tensor Size = {data['m0']:,}")
        print("| $k$ | Slices | Greedy MaxSize | Greedy FLOPs | Greedy Overhead | Rand MaxSize | Rand Overhead | Boundary OH |")
        print("|---|---|---|---|---|---|---|---|")
        for r in data['rows']:
            print(f"| {r['k']} | {r['slices']} | {r['g_size']:,} | {r['g_flops']:,} | {r['g_oh']:.3f}x | {r['r_size']:.1f} | {r['r_oh']:.2f}x | {r['b_oh']:.2f}x |")
