import os
import json
import subprocess
import shutil
import numpy as np
from src.circuit_generators import (
    generate_bb84,
    generate_bernstein_vazirani,
    generate_error_detection,
    generate_hidden_subgroup,
    generate_xor,
    generate_sycamore_like,
    generate_random_arbitrary
)
from src.exporter import export_contraction_job

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results")
os.makedirs(results_dir, exist_ok=True)

job_dir = os.path.join(results_dir, "benchmark_job")
julia_project = "/home/tom/repos/tensor-network-contraction/parallel_contraction_research/proper_research/src/"

def run_baseline(mode, threads=1):
    cmd = [
        "julia",
        f"-t", str(threads),
        f"--project={julia_project}",
        os.path.join(current_dir, "src", "step_profiler.jl"),
        job_dir,
        mode
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Baseline error ({mode}):", res.stderr)
        return None
    # Parse JSON from stdout
    lines = res.stdout.splitlines()
    json_line = None
    for line in reversed(lines):
        if line.strip().startswith("[") and line.strip().endswith("]"):
            json_line = line
            break
    if json_line is None:
        return None
    metrics = json.loads(json_line)
    
    # Return total time as sum
    total_time = sum(step["time_per_slice"] if mode == "sliced" else step["time"] for step in metrics)
    return total_time

def run_hybrid(threads=4):
    cmd = [
        "julia",
        f"-t", str(threads),
        f"--project={julia_project}",
        os.path.join(current_dir, "src", "hybrid_scheduler.jl"),
        job_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Hybrid scheduler error:", res.stderr)
        return None
    # Parse dict from stdout
    lines = res.stdout.splitlines()
    json_line = None
    for line in reversed(lines):
        if line.strip().startswith("{") and line.strip().endswith("}"):
            json_line = line
            break
    if json_line is None:
        return None
    data = json.loads(json_line)
    return data["elapsed"], data["result"]

def main():
    print("=" * 80)
    print("STARTING ADVANCED HYBRID ADAPTIVE SCHEDULER COMPARATIVE BENCHMARK")
    print("=" * 80)
    
    configs = {
        "BB84 Protocol (BB_24)": lambda: generate_bb84(24),
        "Bernstein-Vazirani (BV_24)": lambda: generate_bernstein_vazirani(24),
        "Exclusive-OR (XOR_24)": lambda: generate_xor(24),
        "Sycamore Grid (4x4, D=8)": lambda: generate_sycamore_like(4, 4, 8),
        "Random Arbitrary (N=18, D=18)": lambda: generate_random_arbitrary(18, 18)
    }
    
    benchmark_results = {}
    
    for name, generator_fn in configs.items():
        print(f"\nBenchmarking topology: {name}")
        
        # Clean job folder
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
            
        tensors, edges = generator_fn()
        
        # Sliced plan with 8 target slices
        nslices = export_contraction_job(tensors, edges, target_slices=8, job_dir=job_dir)
        
        # 1. Run sequential baseline (1 thread)
        t_seq = run_baseline("sequential", threads=1)
        
        # 2. Run static sliced baseline (4 threads)
        t_static = run_baseline("sliced", threads=4)
        
        # 3. Run Hybrid Adaptive Scheduler (4 threads)
        hybrid_out = run_hybrid(threads=4)
        if hybrid_out is None:
            print("  * Skipping due to hybrid runner failure.")
            continue
        t_hybrid, res_hybrid = hybrid_out
        
        if t_seq is None or t_static is None:
            print("  * Skipping due to baseline failure.")
            continue
            
        # Calculate speedup ratios
        speedup_vs_seq = t_seq / t_hybrid if t_hybrid > 0 else 0.0
        speedup_vs_static = t_static / t_hybrid if t_hybrid > 0 else 0.0
        
        print(f"  -> Sequential:  {t_seq*1000:.2f}ms")
        print(f"  -> Static sliced: {t_static*1000:.2f}ms")
        print(f"  -> Hybrid AHPC:  {t_hybrid*1000:.2f}ms (Speedup: {speedup_vs_seq:.2f}x vs Seq, {speedup_vs_static:.2f}x vs Static)")
        print(f"  -> Numerical Result: {res_hybrid:.10f}")
        
        benchmark_results[name] = {
            "sequential_time": t_seq,
            "static_sliced_time": t_static,
            "hybrid_time": t_hybrid,
            "speedup_vs_seq": speedup_vs_seq,
            "speedup_vs_static": speedup_vs_static,
            "result_value": res_hybrid
        }
        
    # Clean up temp job dir
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        
    # Save raw results
    out_json = os.path.join(results_dir, "hybrid_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(benchmark_results, f, indent=4)
        
    print(f"\nBenchmark finished! Raw results saved to '{out_json}'.")

if __name__ == "__main__":
    main()
