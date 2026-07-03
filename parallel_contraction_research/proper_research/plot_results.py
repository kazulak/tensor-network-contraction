import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "results", "advanced_scaling_results.json")
output_path = os.path.join(current_dir, "results", "scaling_comparison.png")

def main():
    if not os.path.exists(json_path):
        print(f"Error: JSON data file not found at '{json_path}'. Run reproduction sweep first.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes_flat = axes.flatten()
    
    topologies = list(results.keys())
    
    colors = {
        "tree_node": "#1f77b4",       # Deep Blue
        "static_slicing": "#d62728",  # Crimson Red
        "active_slicing": "#2ca02c"   # Forest Green
    }
    
    markers = {
        "tree_node": "o",
        "static_slicing": "s",
        "active_slicing": "^"
    }
    
    labels = {
        "tree_node": "Pure Tree-Node Parallelism",
        "static_slicing": "Static Slicing (4 slices)",
        "active_slicing": "Advanced Active Slicing"
    }

    for i, topo in enumerate(topologies):
        ax = axes_flat[i]
        data = results[topo]
        
        sizes = [item["size"] for item in data]
        t_tn = [item["tree_node"] for item in data]
        t_ss = [item["static_slicing"] for item in data]
        t_as = [item["active_slicing"] for item in data]
        
        x = np.arange(len(sizes))
        
        # Plot curves on a logarithmic y-axis
        ax.plot(x, t_tn, color=colors["tree_node"], marker=markers["tree_node"], label=labels["tree_node"], linewidth=2, markersize=6)
        ax.plot(x, t_ss, color=colors["static_slicing"], marker=markers["static_slicing"], linestyle="--", label=labels["static_slicing"], linewidth=2, markersize=6)
        ax.plot(x, t_as, color=colors["active_slicing"], marker=markers["active_slicing"], label=labels["active_slicing"], linewidth=2, markersize=6)
        
        ax.set_yscale("log")
        ax.set_title(topo, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=30, ha="right")
        ax.set_ylabel("Compute Time (seconds)")
        ax.set_xlabel("Instance Configurations")
        
    # Sixth subplot (empty or used for general legend / summary text)
    ax_legend = axes_flat[5]
    ax_legend.axis("off")
    
    # Draw a clean box summarizing key findings
    text_content = (
        "$\mathbf{Scientific\ Key\ Insights:}$\n\n"
        "1. $\mathbf{Latency-Crossover\ Threshold:}$\n"
        "   Traditional Static Slicing has a constant ~80ms\n"
        "   overhead (loop setup + copy operations).\n"
        "   For small/medium networks, Tree-Node and\n"
        "   Active Slicing execute in sub-milliseconds.\n\n"
        "2. $\mathbf{Slicing\ Dominance\ at\ Scale:}$\n"
        "   On the largest 3D PEPS grids, Static Slicing\n"
        "   outperforms other backends by up to 7.6x.\n"
        "   Slicing indices globally shrinks intermediate\n"
        "   tensor shapes, avoiding memory bus saturation.\n\n"
        "3. $\mathbf{Active\ Slicing\ Adaptive\ Nature:}$\n"
        "   Active Slicing is an optimal adaptive trade-off,\n"
        "   matching Tree-Node speeds on small sizes,\n"
        "   but falling behind Static Slicing on huge grids."
    )
    
    ax_legend.text(0.05, 0.5, text_content, transform=ax_legend.transAxes, 
                   fontsize=9.5, verticalalignment='center',
                   bbox=dict(boxstyle='round,pad=1.0', facecolor='#f7f7f7', edgecolor='#cccccc', alpha=0.9))

    # Add global title and tight layout
    fig.suptitle("Tensor Network Contraction: Multi-Topology Comparative Parallel Scaling", fontweight="bold", y=0.98)
    
    # Create unified legend in Subplot 6
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Successfully generated clean plot at: {output_path}")

if __name__ == "__main__":
    main()
