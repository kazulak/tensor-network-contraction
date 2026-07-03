import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from src.network_generators import (
    generate_1d_chain,
    generate_2d_grid,
    generate_3d_grid,
    generate_random_regular,
    generate_binary_tree
)
from src.contractors import ContractionEngine

def run_deeper_benchmarks(worker_counts=[1, 2, 4, 8, 12], output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("Executing Deeper Research: Parallel Tensor-Network Contraction Scaling Sweep")
    print("=" * 80)
    
    # Define experimental configurations (More cases & diverse topologies)
    cases = [
        {"name": "1D MPS Chain (n=30, d_bond=16)", "gen": lambda: generate_1d_chain(30, d_bond=16), "slices": 16},
        {"name": "2D PEPS Grid (5x5, d_bond=4)", "gen": lambda: generate_2d_grid(5, 5, d_bond=4), "slices": 64},
        {"name": "3D PEPS Grid (3x3x3, d_bond=2)", "gen": lambda: generate_3d_grid(3, 3, 3, d_bond=2), "slices": 32},
        {"name": "Random Regular Graph (n=30, deg=3)", "gen": lambda: generate_random_regular(30, degree=3, d_bond=4), "slices": 64},
        {"name": "Binary Tree Network (depth=5)", "gen": lambda: generate_binary_tree(5, d_bond=8), "slices": 16}
    ]
    
    results = {}
    
    for case_idx, c in enumerate(cases):
        name = c["name"]
        print(f"\n[Case {case_idx + 1}/{len(cases)}] {name}")
        tensors, edges = c["gen"]()
        
        # Initialize contraction engines
        print("  Optimizing contraction paths...")
        # Unsliced (optimal baseline)
        engine_unsliced = ContractionEngine(tensors, edges, target_slices=0, optimize_time=2.0)
        # Sliced
        engine_sliced = ContractionEngine(tensors, edges, target_slices=c["slices"], optimize_time=2.0)
        
        unsliced_flops = float(engine_unsliced.tree.total_flops())
        unsliced_width = float(engine_unsliced.tree.contraction_width())
        sliced_flops = float(engine_sliced.sliced_tree.total_flops())
        sliced_width = float(engine_sliced.sliced_tree.contraction_width())
        flop_inflation = sliced_flops / unsliced_flops
        
        # 1. Run sequential baselines
        print("  Running sequential baselines...")
        res_seq_unsliced, time_seq_unsliced = engine_unsliced.contract_sequential_unsliced()
        res_seq_sliced, time_seq_sliced = engine_sliced.contract_sequential_sliced()
        
        print(f"    Unsliced: Width={unsliced_width:.1f} | Time={time_seq_unsliced:.4f}s | FLOPs={unsliced_flops:.2e}")
        print(f"    Sliced:   Width={sliced_width:.1f} | Time={time_seq_sliced:.4f}s | FLOPs={sliced_flops:.2e} | Inflation={flop_inflation:.2f}x")
        
        # 2. Verify baseline correctness
        correct_baseline = np.allclose(res_seq_unsliced, res_seq_sliced, rtol=1e-5, atol=1e-8)
        if not correct_baseline:
            print("    [WARNING] Sliced baseline differs from unsliced baseline!")
            
        case_results = {
            "name": name,
            "tensors": len(tensors),
            "unsliced_flops": unsliced_flops,
            "unsliced_width": unsliced_width,
            "unsliced_time": time_seq_unsliced,
            "sliced_flops": sliced_flops,
            "sliced_width": sliced_width,
            "sliced_serial_time": time_seq_sliced,
            "flop_inflation": flop_inflation,
            "nslices": engine_sliced.nslices,
            "process_pool": {},
            "thread_pool": {}
        }
        
        # 3. Sweep workers for ProcessPoolExecutor
        print("  Evaluating ProcessPoolExecutor (multiprocessing)...")
        for w in worker_counts:
            # Run 3 times to get minimum duration (reduces noise)
            times = []
            res = None
            for _ in range(2):
                r, t = engine_sliced.contract_parallel_processes(w)
                times.append(t)
                res = r
            p_time = min(times)
            is_correct = np.allclose(res_seq_unsliced, res, rtol=1e-5, atol=1e-8)
            speedup = time_seq_sliced / p_time
            net_speedup = time_seq_unsliced / p_time
            
            print(f"    Workers={w:2d}: Time={p_time:.4f}s | Speedup={speedup:.2f}x | Net Speedup={net_speedup:.2f}x | Correct={is_correct}")
            case_results["process_pool"][str(w)] = {
                "time": p_time,
                "speedup": speedup,
                "net_speedup": net_speedup,
                "correct": bool(is_correct)
            }
            
        # 4. Sweep workers for ThreadPoolExecutor
        print("  Evaluating ThreadPoolExecutor (multithreading)...")
        for w in worker_counts:
            times = []
            res = None
            for _ in range(2):
                r, t = engine_sliced.contract_parallel_threads(w)
                times.append(t)
                res = r
            t_time = min(times)
            is_correct = np.allclose(res_seq_unsliced, res, rtol=1e-5, atol=1e-8)
            speedup = time_seq_sliced / t_time
            net_speedup = time_seq_unsliced / t_time
            
            print(f"    Workers={w:2d}: Time={t_time:.4f}s | Speedup={speedup:.2f}x | Net Speedup={net_speedup:.2f}x | Correct={is_correct}")
            case_results["thread_pool"][str(w)] = {
                "time": t_time,
                "speedup": speedup,
                "net_speedup": net_speedup,
                "correct": bool(is_correct)
            }
            
        results[name] = case_results
        
    # Save raw JSON data
    with open(os.path.join(output_dir, "deeper_research_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Plots
    print("\nGenerating performance scaling plots...")
    generate_comparison_plots(results, worker_counts, output_dir)
    
    # Generate Markdown Table
    print("Compiling final markdown results table...")
    generate_markdown_report(results, worker_counts, output_dir)
    
    print("\nResearch benchmark run completed successfully.")

def generate_comparison_plots(results, worker_counts, output_dir):
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 4 * len(results)))
    if len(results) == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for idx, (name, data) in enumerate(results.items()):
        ax_speedup = axes[idx, 0]
        ax_time = axes[idx, 1]
        
        ws = [int(w) for w in worker_counts]
        proc_speedups = [data["process_pool"][str(w)]["speedup"] for w in worker_counts]
        thread_speedups = [data["thread_pool"][str(w)]["speedup"] for w in worker_counts]
        
        proc_times = [data["process_pool"][str(w)]["time"] for w in worker_counts]
        thread_times = [data["thread_pool"][str(w)]["time"] for w in worker_counts]
        
        # Plot Speedups
        ax_speedup.plot(ws, proc_speedups, label="Processes (ProcessPool)", marker='o', color='#1f77b4', linewidth=2)
        ax_speedup.plot(ws, thread_speedups, label="Threads (ThreadPool)", marker='s', color='#2ca02c', linewidth=2)
        ax_speedup.plot(ws, ws, label="Linear Speedup", linestyle="--", color="gray", alpha=0.7)
        ax_speedup.set_title(f"Speedup: {name}\n(Relative to Sliced Serial)")
        ax_speedup.set_xlabel("Number of Workers")
        ax_speedup.set_ylabel("Speedup factor")
        ax_speedup.grid(True, linestyle=":", alpha=0.6)
        ax_speedup.legend()
        ax_speedup.set_xticks(ws)
        
        # Plot Runtimes
        ax_time.plot(ws, proc_times, label="Processes", marker='o', color='#1f77b4', linewidth=2)
        ax_time.plot(ws, thread_times, label="Threads", marker='s', color='#2ca02c', linewidth=2)
        ax_time.axhline(y=data["unsliced_time"], label="Ideal Unsliced Serial", linestyle="-.", color="red", alpha=0.7)
        ax_time.axhline(y=data["sliced_serial_time"], label="Sliced Serial Baseline", linestyle=":", color="purple", alpha=0.7)
        ax_time.set_title(f"Execution Time: {name}")
        ax_time.set_xlabel("Number of Workers")
        ax_time.set_ylabel("Time (seconds)")
        ax_time.set_yscale('log')
        ax_time.grid(True, linestyle=":", alpha=0.6)
        ax_time.legend()
        ax_time.set_xticks(ws)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scaling_comparison.png"), dpi=300)
    plt.close()

def generate_markdown_report(results, worker_counts, output_dir):
    lines = []
    lines.append("# Deeper Research: Scaling & Efficiency Report")
    lines.append("\nThis report compiles the benchmarking run comparing multi-processing (ProcessPool) vs multi-threading (ThreadPool) backends across multiple tensor network topologies.")
    
    for name, data in results.items():
        lines.append(f"\n## Topology: {name}")
        lines.append(f"- **Tensors**: {data['tensors']}")
        lines.append(f"- **Slices**: {data['nslices']}")
        lines.append(f"- **Unsliced Width**: {data['unsliced_width']:.1f} | Sliced Width: {data['sliced_width']:.1f}")
        lines.append(f"- **FLOP Inflation**: {data['flop_inflation']:.2f}x")
        lines.append(f"- **Unsliced Time**: {data['unsliced_time']:.4f}s | Sliced Serial Time: {data['sliced_serial_time']:.4f}s")
        lines.append("\n| Workers | Backend | Time (s) | Relative Speedup | Net Speedup (vs Unsliced) | Correct |")
        lines.append("|---|---|---|---|---|---|")
        
        # Add Sliced Serial Row
        lines.append(f"| Serial | Sliced Serial | {data['sliced_serial_time']:.4f} | 1.00x | {data['unsliced_time']/data['sliced_serial_time']:.3f}x | Yes |")
        
        for w in worker_counts:
            p = data["process_pool"][str(w)]
            t = data["thread_pool"][str(w)]
            
            p_correct = "Yes" if p["correct"] else "No"
            t_correct = "Yes" if t["correct"] else "No"
            
            lines.append(f"| {w} | ProcessPool | {p['time']:.4f} | {p['speedup']:.2f}x | {p['net_speedup']:.2f}x | {p_correct} |")
            lines.append(f"| | ThreadPool | {t['time']:.4f} | {t['speedup']:.2f}x | {t['net_speedup']:.2f}x | {t_correct} |")
            
    report_content = "\n".join(lines)
    with open(os.path.join(output_dir, "scaling_comparison_report.md"), "w") as f:
        f.write(report_content)
