import os
import json
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results")
json_path = os.path.join(results_dir, "advanced_scientific_results.json")

def set_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

def plot_roofline(data):
    points = data.get("roofline_points", [])
    if not points:
        return
        
    hw = data.get("hardware", {})
    peak_gflops = hw.get("peak_fp64_gflops", 422.4)
    peak_bw = hw.get("peak_mem_bandwidth_gb_s", 51.2)
    ridge = hw.get("machine_balance_ridge", 8.25)
    
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    # Roofline boundary lines
    i_min, i_max = 0.05, 50.0
    i_arr = np.logspace(np.log10(i_min), np.log10(i_max), 200)
    perf_roof = np.minimum(peak_gflops, peak_bw * i_arr)
    
    ax.plot(i_arr, perf_roof, color="#E63946", linewidth=2.5, label="AMD Ryzen 5 5600 Theoretical Ceiling")
    ax.axhline(peak_gflops, color="#E63946", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(ridge, color="#555555", linestyle=":", linewidth=1.2, label=f"Machine Balance Ridge ({ridge:.2f} FLOPs/B)")
    
    # Scatter points grouped by category
    cats = {
        "small": {"color": "#2E86AB", "label": "Small Contractions (|C| < 10⁴)", "alpha": 0.55, "size": 18},
        "medium": {"color": "#F6AE2D", "label": "Medium Contractions (10⁴ ≤ |C| < 10⁶)", "alpha": 0.85, "size": 45},
        "large": {"color": "#D90429", "label": "Large Intermediates (|C| ≥ 10⁶)", "alpha": 0.95, "size": 90}
    }
    
    for cat_name, cat_props in cats.items():
        sub = [p for p in points if p["category"] == cat_name and p["intensity"] > 0 and p["gflops"] > 0]
        if sub:
            x_vals = [p["intensity"] for p in sub]
            y_vals = [p["gflops"] for p in sub]
            ax.scatter(x_vals, y_vals, c=cat_props["color"], label=cat_props["label"],
                       alpha=cat_props["alpha"], s=cat_props["size"], edgecolors="none")
                       
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 50)
    ax.set_ylim(0.001, 1000)
    
    ax.set_xlabel("Operational Intensity (FLOPs / Byte Transferred)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Attained Performance (GFLOPS - FP64)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Empirical Roofline Model: Contraction Steps Across Operational Regimes", fontsize=12, fontweight="bold", pad=12)
    
    # Annotations
    ax.text(0.12, 10, "MEMORY-BOUND REGIME\n(Index Transposition & Permutation)", fontsize=9, fontweight="bold", color="#2E86AB", bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))
    ax.text(12, 150, "COMPUTE-BOUND REGIME\n(Dense Level-3 GEMM)", fontsize=9, fontweight="bold", color="#D90429", bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))
    
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5)
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, "roofline_model_analysis.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_component_decomposition(data):
    exp1 = data.get("experiment_1_topologies", {})
    if not exp1:
        return
        
    labels = []
    pct_perm = []
    pct_alloc = []
    pct_gemm = []
    
    for topo, configs in exp1.items():
        largest = configs[-1]
        labels.append(f"{topo}\n({largest['size_label']})")
        pct_perm.append(largest["pct_perm"])
        pct_alloc.append(largest["pct_alloc"])
        pct_gemm.append(largest["pct_gemm"])
        
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    y = np.arange(len(labels))
    height = 0.55
    
    c_perm = "#3A86FF"  # Blue: Transposition
    c_alloc = "#8338EC" # Purple: Allocation
    c_gemm = "#FF006E"  # Magenta/Crimson: GEMM
    
    ax.barh(y, pct_alloc, height, label="Memory Allocation & Reshaping", color=c_alloc, alpha=0.9)
    ax.barh(y, pct_perm, height, left=pct_alloc, label="Index Permutation (permutedims)", color=c_perm, alpha=0.9)
    left_gemm = [a + p for a, p in zip(pct_alloc, pct_perm)]
    ax.barh(y, pct_gemm, height, left=left_gemm, label="Core BLAS GEMM Multiplication", color=c_gemm, alpha=0.9)
    
    ax.set_xlabel("Proportion of Total Contraction Time (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Computational Bottleneck Decomposition: Permutation vs. Allocation vs. GEMM", fontsize=12, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, "computational_component_decomposition.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_strong_scaling(data):
    exp2 = data.get("experiment_2_strong_scaling", {})
    if not exp2:
        return
        
    circuits = list(exp2.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    
    colors = {"Tree": "#EE6C4D", "Intra": "#3D5A80", "Hybrid": "#293241"}
    
    for idx, cname in enumerate(circuits[:4]):
        ax = axes[idx]
        cd = exp2[cname]
        th = cd["threads"]
        
        t_tree_1 = cd["tree_times"][0]
        t_intra_1 = cd["intra_times"][0]
        t_hyb_1 = cd["hybrid_times"][0]
        
        sp_tree = [t_tree_1 / t if t > 0 else 1.0 for t in cd["tree_times"]]
        sp_intra = [t_intra_1 / t if t > 0 else 1.0 for t in cd["intra_times"]]
        sp_hyb = [t_hyb_1 / t if t > 0 else 1.0 for t in cd["hybrid_times"]]
        
        ax.plot(th, sp_tree, marker="s", linewidth=2.4, color=colors["Tree"], label=f"Method B: Tree-Level (T₁={t_tree_1*1000:.1f}ms)")
        ax.plot(th, sp_intra, marker="o", linewidth=2.2, color=colors["Intra"], label=f"Method A: Intra-Tensor (T₁={t_intra_1*1000:.1f}ms)")
        ax.plot(th, sp_hyb, marker="^", linewidth=2.2, color=colors["Hybrid"], label=f"Method C: Combined (T₁={t_hyb_1*1000:.1f}ms)")
        
        # Ideal linear speedup reference
        ax.plot([1, 6], [1, 6], linestyle=":", color="#888888", label="Ideal Linear (up to 6 cores)")
        ax.axhline(1.0, linestyle="--", color="#E63946", alpha=0.7, label="Base 1-Thread (1.0x)")
        
        ax.set_title(f"{cname}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Worker Threads (P)", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Parallel Speedup S(P) = T₁ / T_P", fontsize=9.5, fontweight="bold")
        ax.set_xticks(th)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=8)
        
    plt.suptitle("Strong Scaling & Multi-Thread Speedup Curves (P = 1 to 12 Threads)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "multi_thread_strong_scaling.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_treewidth_phase_transition(data):
    exp1 = data.get("experiment_1_topologies", {})
    if not exp1:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#00A878", "#6B2D5C"]
    
    for idx, (topo, configs) in enumerate(exp1.items()):
        labels = [c["size_label"] for c in configs]
        # Approximate size order index
        x = np.arange(len(labels))
        gflops = [c["achieved_gflops"] for c in configs]
        intensities = [c["intensity"] for c in configs]
        
        ax.plot(labels, intensities, marker="o", linewidth=2.2, label=topo, color=colors[idx % len(colors)])
        
    ax.set_ylabel("Operational Intensity (FLOPs / Byte)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Operational Intensity Scaling: Bounded Structures vs. Volume-Law Haar", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5)
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, "treewidth_phase_transition.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_parallel_decision_matrix(data):
    exp1 = data.get("experiment_1_topologies", {})
    if not exp1:
        return
        
    topos = list(exp1.keys())
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    x = np.arange(len(topos))
    width = 0.26
    
    labels = []
    sp_intra = []
    sp_tree = []
    sp_hybr = []
    
    for topo in topos:
        largest = exp1[topo][-1]
        labels.append(f"{topo}\n({largest['size_label']})")
        sp_intra.append(largest["speedup_intra"])
        sp_tree.append(largest["speedup_tree"])
        sp_hybr.append(largest["speedup_hybrid"])
        
    b1 = ax.bar(x - width, sp_intra, width, label="Method A: Intra-Tensor (BLAS GEMM)", color="#3D5A80", alpha=0.9)
    b2 = ax.bar(x, sp_tree, width, label="Method B: Tree-Level (Task DAG)", color="#EE6C4D", alpha=0.9)
    b3 = ax.bar(x + width, sp_hybr, width, label="Method C: Combined Hybrid", color="#293241", alpha=0.95)
    
    ax.axhline(1.0, color="#E63946", linestyle="--", linewidth=1.5, label="Sequential Baseline (1.0x)")
    
    ax.set_ylabel("Speedup on 6 Cores (Relative to Sequential)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Large-Scale Parallel Speedup Matrix Across Quantum Topologies", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9)
    
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}x",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=35)
                        
    plt.tight_layout()
    out_path = os.path.join(results_dir, "parallel_decision_boundary_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    if not os.path.exists(json_path):
        print(f"Results file {json_path} does not exist. Run research runner first.")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    set_style()
    print("Generating advanced scientific figures...")
    plot_roofline(data)
    plot_component_decomposition(data)
    plot_strong_scaling(data)
    plot_treewidth_phase_transition(data)
    plot_parallel_decision_matrix(data)
    print("All 5 scientific research figures successfully generated!")

if __name__ == "__main__":
    main()
