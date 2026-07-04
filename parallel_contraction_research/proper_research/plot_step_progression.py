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
    'figure.titlesize': 15,
    'legend.fontsize': 8.5,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "results", "step_profiling_results.json")
output_path = os.path.join(current_dir, "results", "step_cost_progression_v2.png")

def main():
    if not os.path.exists(json_path):
        print(f"Error: JSON data file not found at '{json_path}'. Run orchestration script first.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    # 2x5 grid: Row 1 = Cumulative Time (%), Row 2 = Absolute Time (s, log scale)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), dpi=300)
    
    topologies = list(results.keys())
    
    # Consistent color encoding for sizes: Small, Medium, Large
    size_colors = [
        "#7fcdbb",  # Small (Turquoise)
        "#41b6c4",  # Medium (Teal)
        "#0c2c84"   # Large (Deep Ocean Blue)
    ]
    
    for col_idx, topo_name in enumerate(topologies):
        size_configs = results[topo_name]
        
        # Row 1 ax: Cumulative (%)
        ax_cum = axes[0, col_idx]
        # Row 2 ax: Absolute (seconds, log)
        ax_abs = axes[1, col_idx]
        
        for size_idx, config in enumerate(size_configs):
            size_label = config["size_label"]
            seq_data = config["sequential"]
            sliced_data = config["sliced"]
            
            # 1. Sequential timing
            seq_times = [step["time"] for step in seq_data]
            seq_cum_time = np.cumsum(seq_times)
            
            if seq_cum_time[-1] > 0:
                seq_cum_time_pct = (seq_cum_time / seq_cum_time[-1]) * 100
            else:
                seq_cum_time_pct = np.zeros_like(seq_cum_time)
                
            # 2. Sliced parallel timing
            sliced_times = [step["time_per_slice"] for step in sliced_data]
            sliced_cum_time = np.cumsum(sliced_times)
            
            if sliced_cum_time[-1] > 0:
                sliced_cum_time_pct = (sliced_cum_time / sliced_cum_time[-1]) * 100
            else:
                sliced_cum_time_pct = np.zeros_like(sliced_cum_time)
                
            # Prepend 0.0 so all curves start exactly at (0, 0)
            seq_cum_time_pct = np.insert(seq_cum_time_pct, 0, 0.0)
            sliced_cum_time_pct = np.insert(sliced_cum_time_pct, 0, 0.0)
            
            seq_cum_time_sec = np.insert(seq_cum_time, 0, 0.0)
            sliced_cum_time_sec = np.insert(sliced_cum_time, 0, 0.0)
            
            x_steps = np.linspace(0, 100, len(seq_cum_time_pct))
            
            color = size_colors[size_idx % len(size_colors)]
            
            # --- Plot Row 1: Cumulative percentage ---
            ax_cum.plot(x_steps, seq_cum_time_pct, label=f"Seq: {size_label}", color=color, linestyle="-", linewidth=2.2)
            ax_cum.plot(x_steps, sliced_cum_time_pct, label=f"Par: {size_label}", color=color, linestyle="--", linewidth=2.2)
            
            # --- Plot Row 2: Absolute seconds (log scale) ---
            # To handle log scale correctly, we plot from index 1 onward (skipping the 0.0 origin value)
            ax_abs.plot(x_steps[1:], seq_cum_time_sec[1:], label=f"Seq: {size_label}", color=color, linestyle="-", linewidth=2.2)
            ax_abs.plot(x_steps[1:], sliced_cum_time_sec[1:], label=f"Par: {size_label}", color=color, linestyle="--", linewidth=2.2)

        # Labels & limits for Row 1 (Cumulative)
        ax_cum.set_title(topo_name, fontweight="bold", pad=10)
        ax_cum.set_xlabel("Contraction Progress (%)")
        ax_cum.set_ylabel("Cumulative Cost (%)")
        ax_cum.set_xlim(0, 100)
        ax_cum.set_ylim(0, 105)
        ax_cum.legend(loc="upper left", frameon=True)
        ax_cum.axhline(90, color="gray", linestyle=":", alpha=0.3)
        ax_cum.axhline(99, color="red", linestyle=":", alpha=0.3)
        
        # Labels & limits for Row 2 (Absolute)
        ax_abs.set_yscale("log")
        ax_abs.set_xlabel("Contraction Progress (%)")
        ax_abs.set_ylabel("Absolute Time (seconds)")
        ax_abs.set_xlim(0, 100)
        ax_abs.legend(loc="upper left", frameon=True)

    fig.suptitle("Comparative Cost Progression: Sequential vs. Parallel Sliced Contraction (Scale-Dependent)", fontweight="bold", y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Successfully generated 2x5 step progression plot at: {output_path}")

if __name__ == "__main__":
    main()
