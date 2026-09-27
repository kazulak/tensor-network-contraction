import os
import json
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results")
json_path = os.path.join(results_dir, "parallel_research_results.json")

def set_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

def plot_compute_time_distribution(data):
    topos = list(data.keys())
    # Take largest scale for each topology
    labels = []
    pct_small = []
    pct_med = []
    pct_large = []
    
    for topo in topos:
        largest = data[topo][-1]
        labels.append(f"{topo}\n({largest['size_label']})")
        pct_small.append(largest["pct_small"])
        pct_med.append(largest["pct_medium"])
        pct_large.append(largest["pct_large"])
        
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    y = np.arange(len(labels))
    height = 0.55
    
    c_small = "#2E86AB"  # Deep blue
    c_med = "#F6AE2D"    # Warm amber
    c_large = "#F26419"  # Crimson / Coral
    
    bars_s = ax.barh(y, pct_small, height, label="Small Contractions (|C| < 10⁴)", color=c_small, alpha=0.92)
    bars_m = ax.barh(y, pct_med, height, left=pct_small, label="Medium Contractions (10⁴ ≤ |C| < 10⁶)", color=c_med, alpha=0.92)
    left_l = [s + m for s, m in zip(pct_small, pct_med)]
    bars_l = ax.barh(y, pct_large, height, left=left_l, label="Large Intermediates (|C| ≥ 10⁶)", color=c_large, alpha=0.92)
    
    ax.set_xlabel("Share of Total Contraction Time (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Compute Time Distribution: Small vs. Large Intermediate Tensors Across Circuit Classes", fontsize=12, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9)
    
    # Add percentage text inside bars where large enough
    for idx, (s, m, l) in enumerate(zip(pct_small, pct_med, pct_large)):
        if s > 12:
            ax.text(s / 2, idx, f"{s:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=8)
        if m > 12:
            ax.text(s + m / 2, idx, f"{m:.1f}%", va="center", ha="center", color="black", fontweight="bold", fontsize=8)
        if l > 12:
            ax.text(s + m + l / 2, idx, f"{l:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=8)
            
    plt.tight_layout()
    out_path = os.path.join(results_dir, "compute_time_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_step_progression(data):
    topos = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#00A878", "#6B2D5C"]
    
    for idx, topo in enumerate(topos):
        largest = data[topo][-1]
        steps = largest["step_details"]
        times = [s["time"] for s in steps]
        cum_times = np.cumsum(times)
        total = cum_times[-1] if cum_times[-1] > 0 else 1.0
        cum_pct = (cum_times / total) * 100.0
        x_pct = np.linspace(0, 100, len(cum_pct))
        
        ax.plot(x_pct, cum_pct, label=f"{topo} ({largest['size_label']})", color=colors[idx % len(colors)], linewidth=2.2, alpha=0.85)
        
    ax.set_xlabel("Contraction Progress (% of Total Steps)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Cumulative Compute Time (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Contraction Step Cost Accumulation: Linear Walk vs. Peak Treewidth S-Curves", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=8.5)
    
    # Annotate the diagonal linear reference
    ax.plot([0, 100], [0, 100], linestyle=":", color="#888888", label="Uniform Cost Reference (Linear)")
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, "step_progression_analysis.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_parallel_speedup_matrix(data):
    topos = list(data.keys())
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    x = np.arange(len(topos))
    width = 0.26
    
    labels = []
    sp_intra = []
    sp_tree = []
    sp_hybr = []
    
    for topo in topos:
        largest = data[topo][-1]
        labels.append(f"{topo}\n({largest['size_label']})")
        sp_intra.append(largest["speedup_intra"])
        sp_tree.append(largest["speedup_tree"])
        sp_hybr.append(largest["speedup_hybrid"])
        
    c_intra = "#3D5A80"
    c_tree = "#EE6C4D"
    c_hybr = "#293241"
    
    b1 = ax.bar(x - width, sp_intra, width, label="Method A: Intra-Tensor Parallel (BLAS GEMM)", color=c_intra, alpha=0.9)
    b2 = ax.bar(x, sp_tree, width, label="Method B: Tree-Level Parallel (Task DAG)", color=c_tree, alpha=0.9)
    b3 = ax.bar(x + width, sp_hybr, width, label="Method C: Combined Hybrid (Dynamic DAG + GEMM)", color=c_hybr, alpha=0.95)
    
    # Reference line at 1.0x (Sequential baseline)
    ax.axhline(1.0, color="#E63946", linestyle="--", linewidth=1.5, label="Sequential Baseline (1.0x)")
    
    ax.set_ylabel("Parallel Speedup (Relative to Sequential)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Parallel Speedup Comparison Across Quantum Circuit Classes (6 Cores)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9)
    
    # Value annotations on top of bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}x",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=40)
                        
    plt.tight_layout()
    out_path = os.path.join(results_dir, "parallel_speedup_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_scaling_crossover(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # Subplot 1: Structured / Low Entanglement scaling (1D Brickwork)
    bw_data = data.get("1D Local (Brickwork)", [])
    if bw_data:
        sizes = [d["size_label"].split(",")[0] for d in bw_data]
        sp_i = [d["speedup_intra"] for d in bw_data]
        sp_t = [d["speedup_tree"] for d in bw_data]
        sp_h = [d["speedup_hybrid"] for d in bw_data]
        
        ax1.plot(sizes, sp_i, marker="o", linewidth=2, label="Intra-Tensor", color="#3D5A80")
        ax1.plot(sizes, sp_t, marker="s", linewidth=2, label="Tree-Level", color="#EE6C4D")
        ax1.plot(sizes, sp_h, marker="^", linewidth=2.5, label="Combined Hybrid", color="#293241")
        ax1.axhline(1.0, color="#E63946", linestyle=":", label="Seq Baseline")
        ax1.set_title("1D Local Brickwork Scaling\n(Tree-Level Dominates Low-Rank Subtrees)", fontsize=10.5, fontweight="bold")
        ax1.set_xlabel("Circuit Width", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Parallel Speedup", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(frameon=True, facecolor="white", fontsize=8.5)
        
    # Subplot 2: Heavy 2D / Random scaling (Random Arbitrary)
    rand_data = data.get("Random Arbitrary (Haar Volume)", [])
    if rand_data:
        sizes = [d["size_label"] for d in rand_data]
        sp_i = [d["speedup_intra"] for d in rand_data]
        sp_t = [d["speedup_tree"] for d in rand_data]
        sp_h = [d["speedup_hybrid"] for d in rand_data]
        
        ax2.plot(sizes, sp_i, marker="o", linewidth=2, label="Intra-Tensor", color="#3D5A80")
        ax2.plot(sizes, sp_t, marker="s", linewidth=2, label="Tree-Level", color="#EE6C4D")
        ax2.plot(sizes, sp_h, marker="^", linewidth=2.5, label="Combined Hybrid", color="#293241")
        ax2.axhline(1.0, color="#E63946", linestyle=":", label="Seq Baseline")
        ax2.set_title("Random Arbitrary Haar Scaling\n(High Treewidth: Intra-Tensor & Hybrid Surge)", fontsize=10.5, fontweight="bold")
        ax2.set_xlabel("Circuit Scale", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Parallel Speedup", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True, facecolor="white", fontsize=8.5)
        
    plt.tight_layout()
    out_path = os.path.join(results_dir, "scaling_crossover_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    if not os.path.exists(json_path):
        print(f"Results file {json_path} does not exist yet. Run research sweep first.")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    set_style()
    print("Generating research plots...")
    plot_compute_time_distribution(data)
    plot_step_progression(data)
    plot_parallel_speedup_matrix(data)
    plot_scaling_crossover(data)
    print("All 4 research plots successfully generated!")

if __name__ == "__main__":
    main()
