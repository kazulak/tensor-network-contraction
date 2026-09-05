import os
import json
import subprocess
import shutil
import time
import numpy as np

from src.circuit_generators import (
    generate_bb84,
    generate_bernstein_vazirani,
    generate_xor,
    generate_1d_brickwork,
    generate_sycamore_like,
    generate_qft,
    generate_random_arbitrary
)
from src.exporter import export_contraction_job

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results")
os.makedirs(results_dir, exist_ok=True)
job_dir = os.path.join(results_dir, "research_bench_job")

NUM_THREADS = 6

def run_julia_json(script_name, args):
    script_path = os.path.join(current_dir, "src", script_name)
    cmd = ["julia", "-t", str(NUM_THREADS), script_path] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {script_name}: {res.stderr}")
        return None
    
    # Parse JSON from last JSON-like line
    lines = res.stdout.strip().splitlines()
    json_line = None
    for line in reversed(lines):
        line_s = line.strip()
        if (line_s.startswith("{") and line_s.endswith("}")) or (line_s.startswith("[") and line_s.endswith("]")):
            json_line = line_s
            break
            
    if json_line is None:
        print(f"Failed to find JSON in output of {script_name}: {res.stdout}")
        return None
        
    return json.loads(json_line)

def main():
    print("=" * 85)
    print("STARTING TENSOR NETWORK CONTRACTION RESEARCH SWEEP (7 TOPOLOGIES x 4 SIZES)")
    print(f"Testing: Sequential Baseline (1 thread) vs. Intra-Tensor vs. Tree-Level vs. Combined Hybrid ({NUM_THREADS} threads)")
    print("=" * 85)
    
    topologies = {
        "Zero-Entanglement (BB84)": [
            ("N=12", lambda: generate_bb84(12)),
            ("N=18", lambda: generate_bb84(18)),
            ("N=24", lambda: generate_bb84(24)),
            ("N=30", lambda: generate_bb84(30)),
        ],
        "Star-Graph (Bernstein-Vazirani)": [
            ("N=12", lambda: generate_bernstein_vazirani(12)),
            ("N=18", lambda: generate_bernstein_vazirani(18)),
            ("N=24", lambda: generate_bernstein_vazirani(24)),
            ("N=30", lambda: generate_bernstein_vazirani(30)),
        ],
        "Tree-Graph (Exclusive-OR)": [
            ("N=12", lambda: generate_xor(12)),
            ("N=18", lambda: generate_xor(18)),
            ("N=24", lambda: generate_xor(24)),
            ("N=30", lambda: generate_xor(30)),
        ],
        "1D Local (Brickwork)": [
            ("N=10, D=10", lambda: generate_1d_brickwork(10, 10)),
            ("N=14, D=10", lambda: generate_1d_brickwork(14, 10)),
            ("N=18, D=10", lambda: generate_1d_brickwork(18, 10)),
            ("N=22, D=10", lambda: generate_1d_brickwork(22, 10)),
        ],
        "2D Planar (Sycamore Grid)": [
            ("3x3, D=6", lambda: generate_sycamore_like(3, 3, 6)),
            ("3x4, D=6", lambda: generate_sycamore_like(3, 4, 6)),
            ("4x4, D=6", lambda: generate_sycamore_like(4, 4, 6)),
            ("4x5, D=6", lambda: generate_sycamore_like(4, 5, 6)),
        ],
        "All-to-All Structured (QFT)": [
            ("N=8", lambda: generate_qft(8)),
            ("N=10", lambda: generate_qft(10)),
            ("N=12", lambda: generate_qft(12)),
            ("N=14", lambda: generate_qft(14)),
        ],
        "Random Arbitrary (Haar Volume)": [
            ("N=10, D=10", lambda: generate_random_arbitrary(10, 10)),
            ("N=12, D=12", lambda: generate_random_arbitrary(12, 12)),
            ("N=14, D=14", lambda: generate_random_arbitrary(14, 14)),
            ("N=16, D=16", lambda: generate_random_arbitrary(16, 16)),
        ],
    }
    
    final_database = {}
    
    for topo_name, size_configs in topologies.items():
        print(f"\n=======================================================")
        print(f"TOPOLOGY: {topo_name}")
        print(f"=======================================================")
        final_database[topo_name] = []
        
        for size_label, generator_fn in size_configs:
            print(f"\n>>> Running Config: {topo_name} [{size_label}]")
            
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)
                
            tensors, edges = generator_fn()
            # Export unsliced plan (target_slices=1) to contract exact contraction tree
            export_contraction_job(tensors, edges, target_slices=1, job_dir=job_dir)
            
            # 1. Step Profiler (Sequential Baseline & Detailed breakdown)
            prof_data = run_julia_json("profiling_instrument.jl", [job_dir])
            if prof_data is None:
                print("Failed step profiler, skipping.")
                continue
                
            seq_time = prof_data["total_time"]
            final_scalar = prof_data["final_scalar"]
            pct_small = prof_data["pct_small"]
            pct_med = prof_data["pct_medium"]
            pct_large = prof_data["pct_large"]
            peak_size = prof_data["peak_tensor_size"]
            
            print(f"  [Sequential Base] Time: {seq_time*1000:.2f}ms | Peak Size: {peak_size:,} | Time Dist: Small={pct_small:.1f}%, Med={pct_med:.1f}%, Large={pct_large:.1f}%")
            
            # 2. Method A: Intra-Tensor Parallel Contractor
            intra_data = run_julia_json("intra_tensor_contractor.jl", [job_dir, str(NUM_THREADS)])
            intra_time = intra_data["elapsed"] if intra_data else None
            intra_res = intra_data["result"] if intra_data else None
            speedup_intra = (seq_time / intra_time) if (intra_time and intra_time > 0) else 0.0
            
            # 3. Method B: Tree-Level Parallel Contractor
            tree_data = run_julia_json("tree_level_contractor.jl", [job_dir])
            tree_time = tree_data["elapsed"] if tree_data else None
            tree_res = tree_data["result"] if tree_data else None
            speedup_tree = (seq_time / tree_time) if (tree_time and tree_time > 0) else 0.0
            
            # 4. Method C: Combined Hybrid Contractor
            hybrid_data = run_julia_json("hybrid_combined_contractor.jl", [job_dir, str(NUM_THREADS)])
            hybrid_time = hybrid_data["elapsed"] if hybrid_data else None
            hybrid_res = hybrid_data["result"] if hybrid_data else None
            speedup_hybrid = (seq_time / hybrid_time) if (hybrid_time and hybrid_time > 0) else 0.0
            
            # Verification: numerical difference
            max_err = 0.0
            for r in [intra_res, tree_res, hybrid_res]:
                if r is not None:
                    max_err = max(max_err, abs(r - final_scalar))
                    
            print(f"  [Method A: Intra] Time: {intra_time*1000:.2f}ms | Speedup: {speedup_intra:.2f}x")
            print(f"  [Method B: Tree ] Time: {tree_time*1000:.2f}ms | Speedup: {speedup_tree:.2f}x")
            print(f"  [Method C: Hybr ] Time: {hybrid_time*1000:.2f}ms | Speedup: {speedup_hybrid:.2f}x")
            print(f"  [Numerical Err  ] Max abs error: {max_err:.2e}")
            
            config_result = {
                "size_label": size_label,
                "sequential_time": seq_time,
                "intra_time": intra_time,
                "tree_time": tree_time,
                "hybrid_time": hybrid_time,
                "speedup_intra": speedup_intra,
                "speedup_tree": speedup_tree,
                "speedup_hybrid": speedup_hybrid,
                "pct_small": pct_small,
                "pct_medium": pct_med,
                "pct_large": pct_large,
                "time_small": prof_data["time_small"],
                "time_medium": prof_data["time_medium"],
                "time_large": prof_data["time_large"],
                "flops_small": prof_data["flops_small"],
                "flops_medium": prof_data["flops_medium"],
                "flops_large": prof_data["flops_large"],
                "peak_tensor_size": peak_size,
                "num_steps": prof_data["num_steps"],
                "max_numerical_error": max_err,
                "final_scalar": final_scalar,
                "step_details": prof_data["step_details"]
            }
            final_database[topo_name].append(config_result)
            
    # Cleanup temp job folder
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        
    out_json = os.path.join(results_dir, "parallel_research_results.json")
    with open(out_json, "w") as f:
        json.dump(final_database, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"All 28 configurations evaluated successfully!")
    print(f"Raw research data saved to: {out_json}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
