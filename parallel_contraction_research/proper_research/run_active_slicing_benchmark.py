import os
import sys
import time
import subprocess
import numpy as np

# Ensure proper_research root in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.network_generators import generate_2d_grid
from src.exporter import export_contraction_job

def run_active_slicing(job_dir, threads):
    julia_script = os.path.join(current_dir, "src", "active_slicing_contractor.jl")
    cmd = ["julia", f"--threads={threads}", julia_script, job_dir]
    
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    total_time = time.perf_counter() - start
    
    if proc.returncode != 0:
        raise RuntimeError(f"Julia failed: {proc.stderr}")
        
    lines = proc.stdout.splitlines()
    res_val, contract_time = None, None
    for line in lines:
        if line.startswith("ACTIVE_SLICING_RESULT:"):
            res_val = float(line.split()[1])
        elif line.startswith("ACTIVE_SLICING_TIME:"):
            contract_time = float(line.split()[1])
    return res_val, contract_time, total_time

def run_static_slicing(job_dir, threads):
    # This runs the traditional hybrid contractor that executes static slices in parallel
    julia_script = os.path.join(current_dir, "src", "hybrid_contractor.jl")
    cmd = ["julia", f"--threads={threads}", julia_script, job_dir]
    
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    total_time = time.perf_counter() - start
    
    if proc.returncode != 0:
         raise RuntimeError(f"Julia failed: {proc.stderr}")
         
    lines = proc.stdout.splitlines()
    res_val, contract_time = None, None
    for line in lines:
        if line.startswith("Result:"):
            res_val = float(line.split()[1])
        elif line.startswith("Contraction Time:"):
            contract_time = float(line.split()[2]) # format is: Contraction Time: X s
    return res_val, contract_time, total_time

def run_tree_node(job_dir, threads):
    julia_script = os.path.join(current_dir, "src", "tree_parallel_contractor.jl")
    cmd = ["julia", f"--threads={threads}", julia_script, job_dir, "1000.0"]
    
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    total_time = time.perf_counter() - start
    
    if proc.returncode != 0:
         raise RuntimeError(f"Julia failed: {proc.stderr}")
         
    lines = proc.stdout.splitlines()
    res_val, contract_time = None, None
    for line in lines:
        if line.startswith("TREE_NODE_RESULT:"):
            res_val = float(line.split()[1])
        elif line.startswith("TREE_NODE_TIME:"):
            contract_time = float(line.split()[1])
    return res_val, contract_time, total_time

def main():
    print("=" * 80)
    print("      COMPARING ADVANCED ACTIVE SLICING VS STATIC SLICING")
    print("=" * 80)
    
    # Generate PEPS grid
    # To compare correctly:
    # 1. For Active Slicing and Tree-Node, we export the job with 1 slice (unsliced tree)
    # 2. For Static Slicing, we export the job with 4 slices (traditional sliced tree)
    # We will use the exact same grid values to ensure comparison is correct.
    print("Generating 4x4 Grid with bond dimension 4...")
    tensors, edges = generate_2d_grid(rows=4, cols=4, d_bond=4)
    
    unsliced_job_dir = os.path.join(current_dir, "results", "active_unsliced_job")
    sliced_job_dir = os.path.join(current_dir, "results", "active_sliced_job")
    
    # Export unsliced tree (for Active Slicing & Tree-Node)
    export_contraction_job(tensors, edges, target_slices=1, job_dir=unsliced_job_dir)
    
    # Export sliced tree (for Static Slicing - 4 slices)
    export_contraction_job(tensors, edges, target_slices=4, job_dir=sliced_job_dir)
    
    threads = 4
    
    print(f"\nRunning benchmarks with {threads} Julia threads...")
    
    # 1. Pure Tree-Node Parallelism
    print("\n[Method 1] Running Pure Tree-Node Parallelism (Unsliced)...")
    res_tn, c_tn, t_tn = run_tree_node(unsliced_job_dir, threads)
    print(f"  Result: {res_tn:.6f} | Contraction: {c_tn:.4f}s | Total: {t_tn:.4f}s")
    
    # 2. Traditional Static Slicing
    print("\n[Method 2] Running Traditional Static Slicing (4 slices from start)...")
    res_ss, c_ss, t_ss = run_static_slicing(sliced_job_dir, threads)
    print(f"  Result: {res_ss:.6f} | Contraction: {c_ss:.4f}s | Total: {t_ss:.4f}s")
    
    # 3. Active Slicing Parallelism
    print("\n[Method 3] Running Advanced Active Slicing (Delays slicing)...")
    res_as, c_as, t_as = run_active_slicing(unsliced_job_dir, threads)
    print(f"  Result: {res_as:.6f} | Contraction: {c_as:.4f}s | Total: {t_as:.4f}s")
    
    print("\n" + "=" * 80)
    print("                      COMPARATIVE REPORT")
    print("=" * 80)
    print(f"{'Method':<30} | {'Contraction Time (s)':<22} | {'Result Matches?'}")
    print("-" * 80)
    
    match_tn = "Yes" if np.abs(res_tn - res_as) / (np.abs(res_as) + 1e-15) < 1e-5 else "No"
    match_ss = "Yes" if np.abs(res_ss - res_as) / (np.abs(res_as) + 1e-15) < 1e-5 else "No"
    
    print(f"{'Pure Tree-Node':<30} | {c_tn:<22.4f} | {match_tn} ({res_tn:.4f})")
    print(f"{'Traditional Static Slicing':<30} | {c_ss:<22.4f} | {match_ss} ({res_ss:.4f})")
    print(f"{'Advanced Active Slicing':<30} | {c_as:<22.4f} | Yes ({res_as:.4f})")
    print("-" * 80)
    
    speedup = c_ss / c_as
    print(f"Active Slicing Speedup vs Static Slicing: {speedup:.2f}x faster compute!")
    print("=" * 80)
    
    # Clean up folders
    subprocess.run(["rm", "-rf", unsliced_job_dir, sliced_job_dir])

if __name__ == "__main__":
    main()
