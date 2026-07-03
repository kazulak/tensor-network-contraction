import os
import sys
import time
import json
import subprocess
import numpy as np

# Ensure proper_research root in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.network_generators import (
    generate_2d_grid, 
    generate_random_regular, 
    generate_binary_tree,
    generate_1d_chain,
    generate_3d_grid
)
from src.exporter import export_contraction_job

def run_active_slicing(job_dir, threads):
    julia_script = os.path.join(current_dir, "src", "active_slicing_contractor.jl")
    cmd = ["julia", f"--threads={threads}", julia_script, job_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Julia Active Slicing failed: {proc.stderr}")
    lines = proc.stdout.splitlines()
    contract_time = None
    for line in lines:
        if line.startswith("ACTIVE_SLICING_TIME:"):
            contract_time = float(line.split()[1])
    return contract_time

def run_static_slicing(job_dir, threads):
    julia_script = os.path.join(current_dir, "src", "hybrid_contractor.jl")
    cmd = ["julia", f"--threads={threads}", julia_script, job_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Julia Static Slicing failed: {proc.stderr}")
    lines = proc.stdout.splitlines()
    contract_time = None
    for line in lines:
        if line.startswith("Contraction Time:"):
            contract_time = float(line.split()[2])
    return contract_time

def run_tree_node(job_dir, threads):
    julia_script = os.path.join(current_dir, "src", "tree_parallel_contractor.jl")
    cmd = ["julia", f"--threads={threads}", julia_script, job_dir, "1000.0"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Julia Tree-Node failed: {proc.stderr}")
    lines = proc.stdout.splitlines()
    contract_time = None
    for line in lines:
        if line.startswith("TREE_NODE_TIME:"):
            contract_time = float(line.split()[1])
    return contract_time

def main():
    print("=" * 80)
    print("        LAUNCHING ADVANCED MULTI-TOPOLOGY 7-SIZE SCALING SWEEP")
    print("=" * 80)
    
    # 5 Topologies, each with 7 sizes
    topologies = {
        "1D MPS Chain": [
            ("N=10, D=4", lambda: generate_1d_chain(10, d_bond=4)),
            ("N=15, D=4", lambda: generate_1d_chain(15, d_bond=4)),
            ("N=20, D=4", lambda: generate_1d_chain(20, d_bond=4)),
            ("N=25, D=4", lambda: generate_1d_chain(25, d_bond=4)),
            ("N=30, D=4", lambda: generate_1d_chain(30, d_bond=4)),
            ("N=35, D=4", lambda: generate_1d_chain(35, d_bond=4)),
            ("N=40, D=4", lambda: generate_1d_chain(40, d_bond=4)),
        ],
        "2D PEPS Grid": [
            ("3x3, D=3", lambda: generate_2d_grid(rows=3, cols=3, d_bond=3)),
            ("3x3, D=4", lambda: generate_2d_grid(rows=3, cols=3, d_bond=4)),
            ("4x3, D=4", lambda: generate_2d_grid(rows=4, cols=3, d_bond=4)),
            ("4x4, D=4", lambda: generate_2d_grid(rows=4, cols=4, d_bond=4)),
            ("5x4, D=4", lambda: generate_2d_grid(rows=5, cols=4, d_bond=4)),
            ("5x5, D=4", lambda: generate_2d_grid(rows=5, cols=5, d_bond=4)),
            ("6x5, D=4", lambda: generate_2d_grid(rows=6, cols=5, d_bond=4)),
        ],
        "3D PEPS Grid": [
            ("2x2x2, D=3", lambda: generate_3d_grid(2, 2, 2, d_bond=3)),
            ("2x2x2, D=4", lambda: generate_3d_grid(2, 2, 2, d_bond=4)),
            ("3x2x2, D=3", lambda: generate_3d_grid(3, 2, 2, d_bond=3)),
            ("3x3x2, D=3", lambda: generate_3d_grid(3, 3, 2, d_bond=3)),
            ("3x3x2, D=4", lambda: generate_3d_grid(3, 3, 2, d_bond=4)),
            ("3x3x3, D=3", lambda: generate_3d_grid(3, 3, 3, d_bond=3)),
            ("3x3x3, D=4", lambda: generate_3d_grid(3, 3, 3, d_bond=4)),
        ],
        "Random Regular": [
            ("N=12, D=4", lambda: generate_random_regular(n=12, d_bond=4)),
            ("N=16, D=4", lambda: generate_random_regular(n=16, d_bond=4)),
            ("N=20, D=4", lambda: generate_random_regular(n=20, d_bond=4)),
            ("N=24, D=4", lambda: generate_random_regular(n=24, d_bond=4)),
            ("N=28, D=4", lambda: generate_random_regular(n=28, d_bond=4)),
            ("N=32, D=4", lambda: generate_random_regular(n=32, d_bond=4)),
            ("N=36, D=4", lambda: generate_random_regular(n=36, d_bond=4)),
        ],
        "Binary Tree": [
            ("depth=3, D=4", lambda: generate_binary_tree(depth=3, d_bond=4)),
            ("depth=3, D=6", lambda: generate_binary_tree(depth=3, d_bond=6)),
            ("depth=4, D=4", lambda: generate_binary_tree(depth=4, d_bond=4)),
            ("depth=4, D=6", lambda: generate_binary_tree(depth=4, d_bond=6)),
            ("depth=5, D=4", lambda: generate_binary_tree(depth=5, d_bond=4)),
            ("depth=5, D=5", lambda: generate_binary_tree(depth=5, d_bond=5)),
            ("depth=6, D=4", lambda: generate_binary_tree(depth=6, d_bond=4)),
        ]
    }
    
    threads = 4
    results = {}
    
    unsliced_job_dir = os.path.join(current_dir, "results", "adv_unsliced")
    sliced_job_dir = os.path.join(current_dir, "results", "adv_sliced")
    
    for topo_name, sizes in topologies.items():
        print(f"\n--- Topology: {topo_name} ---")
        results[topo_name] = []
        
        for size_label, generator in sizes:
            print(f"Running size: {size_label}...")
            tensors, edges = generator()
            
            # Export unsliced (slices=1) for Tree-Node and Active Slicing
            export_contraction_job(tensors, edges, target_slices=1, job_dir=unsliced_job_dir)
            
            # Export sliced (slices=4) for Static Slicing
            export_contraction_job(tensors, edges, target_slices=4, job_dir=sliced_job_dir)
            
            try:
                # 1. Tree-Node
                t_tn = run_tree_node(unsliced_job_dir, threads)
                # 2. Static Slicing
                t_ss = run_static_slicing(sliced_job_dir, threads)
                # 3. Active Slicing
                t_as = run_active_slicing(unsliced_job_dir, threads)
                
                results[topo_name].append({
                    "size": size_label,
                    "tree_node": t_tn,
                    "static_slicing": t_ss,
                    "active_slicing": t_as
                })
                
                print(f"  Tree-Node: {t_tn:.4f}s | Static Slicing: {t_ss:.4f}s | Active Slicing: {t_as:.4f}s")
            except Exception as e:
                print(f"  Error running {size_label}: {e}")
            finally:
                subprocess.run(["rm", "-rf", unsliced_job_dir, sliced_job_dir])
                
    # Save results to JSON
    json_path = os.path.join(current_dir, "results", "advanced_scaling_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Markdown report
    report = "# Advanced Multi-Topology 7-Size Comparative Scaling Report\n\n"
    report += "This report evaluates three parallel tensor contraction backends: "
    report += "Pure Tree-Node Parallelism, Traditional Static Slicing (4 slices), and Advanced Active Slicing.\n"
    report += "All measurements exclude JIT compilation through double-run warmups, executed on 4 CPU threads.\n\n"
    
    for topo_name, data in results.items():
        report += f"### Topology: {topo_name}\n\n"
        report += "| Instance Size | Pure Tree-Node (s) | Static Slicing (4 slices) (s) | Advanced Active Slicing (s) | Active Slicing Speedup |\n"
        report += "|---|---|---|---|---|\n"
        for item in data:
            speedup = item["static_slicing"] / item["active_slicing"]
            report += f"| {item['size']} | {item['tree_node']:.4f}s | {item['static_slicing']:.4f}s | {item['active_slicing']:.4f}s | {speedup:.2f}x |\n"
        report += "\n"
        
    report_path = os.path.join(current_dir, "results", "advanced_scaling_report.md")
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"\nAdvanced sweep complete! Report saved to '{report_path}'.")

if __name__ == "__main__":
    main()
