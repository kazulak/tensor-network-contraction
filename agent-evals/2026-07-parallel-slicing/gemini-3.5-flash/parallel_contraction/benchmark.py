import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import cotengra as ctg

# Import our package modules
from parallel_contraction.network_builder import build_random_regular, build_2d_grid_norm
from parallel_contraction.parallel_contractor import contract_sliced_parallel, contract_sliced_serial

def run_benchmarks(worker_counts=[1, 2, 4, 6, 8, 12], seed=42, output_json="benchmark_results.json", output_plot="speedup_plot.png"):
    print("=" * 70)
    print("Starting Tensor Network Parallel Contraction Benchmark Pipeline")
    print("=" * 70)
    
    # 1. Define benchmark cases
    cases_config = [
        {"name": "Random Regular n=40, D=8", "family": "Random Regular Graph", "type": "rr", "n": 40, "D": 8, "slices": 64},
        {"name": "Random Regular n=44, D=8", "family": "Random Regular Graph", "type": "rr", "n": 44, "D": 8, "slices": 64},
        {"name": "2D Grid L=5x5, D=7", "family": "2D Grid Network", "type": "grid", "L": 5, "D": 7, "slices": 343},
        {"name": "2D Grid L=6x6, D=5", "family": "2D Grid Network", "type": "grid", "L": 6, "D": 5, "slices": 125}
    ]
    
    results = {}
    
    for case_idx, cfg in enumerate(cases_config):
        name = cfg["name"]
        print(f"\n--- Case {case_idx + 1}: {name} ---")
        
        # Build tensor network
        if cfg["type"] == "rr":
            tn = build_random_regular(n=cfg["n"], reg=3, D=cfg["D"], seed=seed)
        else:
            tn = build_2d_grid_norm(L=cfg["L"], D=cfg["D"], seed=seed)
            
        arrays = [t.data for t in tn]
        
        # Optimize contraction path *without* slicing to get a clean baseline
        print("Finding optimal unsliced plan...")
        opt_unsliced = ctg.HyperOptimizer(methods=['greedy'], seed=seed)
        tree_unsliced = tn.contraction_tree(optimize=opt_unsliced)
        unsliced_flops = tree_unsliced.total_flops()
        unsliced_width = tree_unsliced.contraction_width()
        
        print("Measuring unsliced serial contraction time...")
        t0 = time.time()
        res_unsliced = tree_unsliced.contract(arrays)
        unsliced_time = time.time() - t0
        print(f"  Unsliced FLOPs: {unsliced_flops:.2e} | Width: {unsliced_width:.2f} | Time: {unsliced_time:.4f}s")
        
        # Optimize contraction path *with* slicing
        print("Finding sliced plan...")
        opt_sliced = ctg.HyperOptimizer(
            methods=['greedy'],
            slicing_opts={'target_slices': cfg["slices"]},
            seed=seed
        )
        tree_sliced = tn.contraction_tree(optimize=opt_sliced)
        sliced_flops = tree_sliced.total_flops()
        sliced_width = tree_sliced.contraction_width()
        nslices = tree_sliced.nslices
        
        # Flop inflation calculation
        flop_inflation = sliced_flops / unsliced_flops
        
        print("Measuring sliced serial contraction time...")
        sliced_serial_times = []
        res_serial = None
        for _ in range(3):
            t0 = time.time()
            res_serial = contract_sliced_serial(tree_sliced, arrays)
            sliced_serial_times.append(time.time() - t0)
        sliced_serial_time = min(sliced_serial_times)
        
        # Sliced time overhead factor
        time_overhead = sliced_serial_time / unsliced_time
        
        print(f"  Sliced FLOPs: {sliced_flops:.2e} | Width: {sliced_width:.2f} | Time: {sliced_serial_time:.4f}s")
        print(f"  FLOP Inflation: {flop_inflation:.2f}x | Sliced Time Overhead: {time_overhead:.2f}x")
        
        case_results = {
            "name": name,
            "family": cfg["family"],
            "tensors": tn.num_tensors,
            "indices": tn.num_indices,
            "nslices": nslices,
            "unsliced_flops": float(unsliced_flops),
            "unsliced_width": float(unsliced_width),
            "unsliced_time": float(unsliced_time),
            "sliced_flops": float(sliced_flops),
            "sliced_width": float(sliced_width),
            "sliced_serial_time": float(sliced_serial_time),
            "flop_inflation": float(flop_inflation),
            "time_overhead": float(time_overhead),
            "parallel_runs": {}
        }
        
        for w in worker_counts:
            print(f"Running parallel contraction with {w} workers...")
            p_times = []
            res_parallel = None
            for _ in range(3):
                t0 = time.time()
                res_parallel = contract_sliced_parallel(tree_sliced, arrays, w)
                p_times.append(time.time() - t0)
            p_time = min(p_times)
            
            # Check correctness
            error = float(np.abs(res_serial - res_parallel))
            agree = bool(np.allclose(res_serial, res_parallel, rtol=1e-12, atol=1e-12))
            
            # Speedup relative to sliced serial
            speedup_sliced = sliced_serial_time / p_time
            efficiency_sliced = speedup_sliced / w
            
            # Speedup relative to optimal unsliced serial (Net Speedup)
            net_speedup = unsliced_time / p_time
            
            print(f"  Workers={w:2d} | Time={p_time:.4f}s | Speedup (Sliced)={speedup_sliced:.2f}x | Efficiency={efficiency_sliced:.2f} | Net Speedup={net_speedup:.2f}x | Agree={agree}")
            
            case_results["parallel_runs"][str(w)] = {
                "time": p_time,
                "speedup_sliced": speedup_sliced,
                "efficiency_sliced": efficiency_sliced,
                "net_speedup": net_speedup,
                "agree": agree,
                "error": error
            }
            
        results[name] = case_results
        
    # Save results to JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Markdown Table
    table = []
    table.append("| Case Name | Tensors | Unsliced Width | Unsliced Time (s) | Slices | Sliced Width | Sliced Serial (s) | FLOP Inflation | Workers | Parallel Time (s) | Speedup (vs Sliced) | Parallel Efficiency | Net Speedup (vs Unsliced) | Correct |")
    table.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    for name, data in results.items():
        # Add serial row
        table.append(f"| **{name}** | {data['tensors']} | {data['unsliced_width']:.1f} | {data['unsliced_time']:.4f} | {data['nslices']} | {data['sliced_width']:.1f} | {data['sliced_serial_time']:.4f} | {data['flop_inflation']:.1f}x | Serial | {data['sliced_serial_time']:.4f} | 1.00x | 1.00 | {data['unsliced_time']/data['sliced_serial_time']:.2f}x | Yes |")
        for w in worker_counts:
            p = data["parallel_runs"][str(w)]
            agree_str = "Yes" if p["agree"] else "No"
            table.append(f"| | | | | | | | | {w} | {p['time']:.4f} | {p['speedup_sliced']:.2f}x | {p['efficiency_sliced']:.2f} | {p['net_speedup']:.2f}x | {agree_str} |")
            
    markdown_table = "\n".join(table)
    with open("benchmark_table.md", "w") as f:
        f.write(markdown_table)
        
    # Plot Speedup & Efficiency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for idx, (name, data) in enumerate(results.items()):
        ws = [int(w) for w in data["parallel_runs"].keys()]
        speedups = [data["parallel_runs"][str(w)]["speedup_sliced"] for w in ws]
        efficiencies = [data["parallel_runs"][str(w)]["efficiency_sliced"] for w in ws]
        
        ax1.plot(ws, speedups, label=name, marker=markers[idx], color=colors[idx], linewidth=2, markersize=8)
        ax2.plot(ws, efficiencies, label=name, marker=markers[idx], color=colors[idx], linewidth=2, markersize=8)
        
    ax1.plot(worker_counts, worker_counts, label="Ideal Speedup", linestyle="--", color="gray", alpha=0.7)
    ax1.set_title("Speedup vs. Number of Workers (Relative to Sliced Serial)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Number of Workers (Processes)", fontsize=11)
    ax1.set_ylabel("Speedup (Sliced Serial Time / Parallel Time)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xticks(worker_counts)
    
    ax2.axhline(y=1.0, label="Ideal Efficiency", linestyle="--", color="gray", alpha=0.7)
    ax2.set_title("Parallel Efficiency vs. Number of Workers", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel("Number of Workers (Processes)", fontsize=11)
    ax2.set_ylabel("Efficiency (Speedup / Workers)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.set_xticks(worker_counts)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(markdown_table)
    return results
