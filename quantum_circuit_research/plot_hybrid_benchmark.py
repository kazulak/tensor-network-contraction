import os
import json
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "results", "hybrid_benchmark_results.json")
output_path = os.path.join(current_dir, "results", "hybrid_speedup_comparison.png")

def main():
    if not os.path.exists(json_path):
        print(f"Error: results file not found at {json_path}")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    topologies = list(results.keys())
    
    # Extract times in milliseconds
    seq_times = [results[k]["sequential_time"] * 1000 for k in topologies]
    static_times = [results[k]["static_sliced_time"] * 1000 for k in topologies]
    hybrid_times = [results[k]["hybrid_time"] * 1000 for k in topologies]

    x = np.arange(len(topologies))
    width = 0.25

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Curated premium color palette
    rects1 = ax.bar(x - width, seq_times, width, label="Baseline Sequential (1 Thread)", color="#2A2A72", edgecolor="none", alpha=0.9)
    rects2 = ax.bar(x, static_times, width, label="Baseline Sliced (4 Threads)", color="#009FFD", edgecolor="none", alpha=0.9)
    rects3 = ax.bar(x + width, hybrid_times, width, label="Hybrid Adaptive AHPC (4 Threads)", color="#9E0059", edgecolor="none", alpha=0.9)

    ax.set_ylabel("Execution Time (ms) - Log Scale", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Performance Comparison: Sequential vs. Sliced vs. Hybrid Contractor", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=15, ha="right", fontsize=9, fontweight="semibold")
    
    # Use logarithmic scale to handle timings spanning 3 orders of magnitude
    ax.set_yscale("log")
    
    # Add minor grid lines for readability on log scale
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)

    # Add numeric labels above the columns for clarity
    for rect in rects1 + rects2 + rects3:
        height = rect.get_height()
        # Format label text
        if height >= 100:
            label_text = f"{height:.0f}ms"
        elif height >= 1:
            label_text = f"{height:.1f}ms"
        else:
            label_text = f"{height:.2f}ms"
            
        ax.annotate(label_text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, rotation=45, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Successfully generated comparative performance plot at: {output_path}")

if __name__ == "__main__":
    main()
