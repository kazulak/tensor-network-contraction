"""Slicing benchmark: FLOP overhead vs memory reduction in 1D and 2D tensor networks."""
import json
import warnings
import matplotlib.pyplot as plt
import numpy as np
import cotengra as ctg
from circuits import build_1d_circuit, build_2d_sycamore_circuit, build_circuit_tn

warnings.filterwarnings("ignore")


def run_benchmark():
    opt = ctg.AutoOptimizer(progbar=False)
    rng = np.random.default_rng(42)
    results = {}

    configs = [
        ("1D_brickwork", 14, lambda: build_1d_circuit(14, 10, rng), [256, 128, 64, 32]),
        ("2D_sycamore", 16, lambda: build_2d_sycamore_circuit(4, 4, 2, rng), [8192, 4096, 2048, 1024, 512, 256]),
    ]

    for name, n_qubits, circuit_fn, targets in configs:
        gates = circuit_fn()
        tensors, indices, size_dict = build_circuit_tn(n_qubits, gates)
        tree = ctg.array_contract_tree(indices, (), size_dict, optimize=opt)
        m0, c0 = tree.max_size(), tree.total_flops()

        guided_data = []
        for target in targets:
            tsl = tree.slice(target_size=target, seed=42)
            k = int(np.log2(tsl.nslices))
            guided_data.append({
                "target": target, "k": k, "nslices": tsl.nslices,
                "max_size": tsl.max_size(), "mem_red": m0 / tsl.max_size(),
                "flops": tsl.total_flops(), "overhead": tsl.total_flops() / c0,
            })

        all_inds = list(tree.size_dict.keys())
        # Peripheral indices (boundary qubits)
        peripheral_inds = [term[0] for term in tree.inputs if len(term) == 1]

        random_data, subopt_data = [], []
        for row in guided_data:
            k = row["k"]
            if k == 0:
                continue
            r_sizes, r_flops = [], []
            for _ in range(25):
                trand = tree.copy()
                for ix in rng.choice(all_inds, size=k, replace=False):
                    trand.remove_ind_(ix)
                r_sizes.append(trand.max_size())
                r_flops.append(trand.total_flops())
            random_data.append({
                "k": k, "nslices": 2**k, "mean_size": float(np.mean(r_sizes)),
                "min_size": int(np.min(r_sizes)), "mem_red": m0 / float(np.mean(r_sizes)),
                "mean_flops": float(np.mean(r_flops)), "mean_overhead": float(np.mean(r_flops) / c0),
            })

            tbad = tree.copy()
            for ix in peripheral_inds[:k]:
                tbad.remove_ind_(ix)
            subopt_data.append({
                "k": k, "nslices": 2**k, "max_size": tbad.max_size(),
                "mem_red": m0 / tbad.max_size(), "flops": tbad.total_flops(),
                "overhead": tbad.total_flops() / c0,
            })

        results[name] = {
            "baseline": {"max_size": m0, "flops": c0},
            "guided": guided_data, "random": random_data, "suboptimal": subopt_data,
        }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_results(results)
    return results


def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    # Panel A: FLOP Overhead vs Memory Reduction Factor
    ax = axes[0]
    for name, marker, color in [("1D_brickwork", "o", "#1f77b4"), ("2D_sycamore", "s", "#ff7f0e")]:
        g, r, s = results[name]["guided"], results[name]["random"], results[name]["suboptimal"]
        lbl = "1D" if "1D" in name else "2D"
        ax.plot([d["mem_red"] for d in g], [d["overhead"] for d in g],
                f"-{marker}", color=color, lw=2.2, ms=6, label=f"{lbl} Guided")
        ax.plot([d["mem_red"] for d in r], [d["mean_overhead"] for d in r],
                f"--{marker}", color=color, alpha=0.7, lw=1.5, ms=5, label=f"{lbl} Random")
        ax.scatter([d["mem_red"] for d in s], [d["overhead"] for d in s],
                   color=color, marker="x", s=40, alpha=0.8, label=f"{lbl} Suboptimal")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Memory Reduction Factor ($M_0 / M$)", fontsize=11)
    ax.set_ylabel("FLOP Overhead ($C / C_0$)", fontsize=11)
    ax.set_title("(a) FLOP Overhead vs Memory Saving", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # Panel B: Peak Tensor Size vs Sliced Indices
    ax = axes[1]
    for name, marker, color in [("1D_brickwork", "o", "#1f77b4"), ("2D_sycamore", "s", "#ff7f0e")]:
        g, r, s = results[name]["guided"], results[name]["random"], results[name]["suboptimal"]
        lbl = "1D" if "1D" in name else "2D"
        ax.plot([d["k"] for d in g], [d["max_size"] for d in g],
                f"-{marker}", color=color, lw=2.2, ms=6, label=f"{lbl} Guided")
        ax.plot([d["k"] for d in r], [d["mean_size"] for d in r],
                f"--{marker}", color=color, alpha=0.7, lw=1.5, ms=5, label=f"{lbl} Random")
        ax.plot([d["k"] for d in s], [d["max_size"] for d in s],
                f":{marker}", color=color, alpha=0.5, lw=1.2, ms=4, label=f"{lbl} Suboptimal")

    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of Sliced Indices ($k$)", fontsize=11)
    ax.set_ylabel("Peak Intermediate Tensor Size ($M$)", fontsize=11)
    ax.set_title("(b) Memory Reduction vs Sliced Indices", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig("slicing_study.png")
    plt.close()


if __name__ == "__main__":
    res = run_benchmark()
    print("Benchmark complete. Results saved to results.json and slicing_study.png.")
