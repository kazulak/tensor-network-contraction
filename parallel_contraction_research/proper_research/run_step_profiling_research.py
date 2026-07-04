import os
import sys
import json
import subprocess

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

def run_step_profiler(job_dir, mode, threads=4):
    julia_script = os.path.join(current_dir, "src", "step_profiler.jl")
    cmd = ["julia", f"--threads={threads}", julia_script, job_dir, mode]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Julia step profiler failed ({mode}): {proc.stderr}")
    
    # The last line should contain the JSON output
    lines = proc.stdout.splitlines()
    json_line = None
    for line in reversed(lines):
        if line.strip().startswith("[") and line.strip().endswith("]"):
            json_line = line
            break
            
    if json_line is None:
        raise RuntimeError(f"Could not find JSON output in stdout. Output: {proc.stdout}")
        
    return json.loads(json_line)

def main():
    print("=" * 80)
    # Define topologies and 3 sizes per topology to investigate
    topologies = {
        "1D MPS Chain": [
            ("N=50, D=16", lambda: generate_1d_chain(50, d_bond=16)),
            ("N=100, D=24", lambda: generate_1d_chain(100, d_bond=24)),
            ("N=150, D=32", lambda: generate_1d_chain(150, d_bond=32))
        ],
        "2D PEPS Grid": [
            ("4x4, D=4", lambda: generate_2d_grid(rows=4, cols=4, d_bond=4)),
            ("5x5, D=4", lambda: generate_2d_grid(rows=5, cols=5, d_bond=4)),
            ("6x6, D=4", lambda: generate_2d_grid(rows=6, cols=6, d_bond=4))
        ],
        "3D PEPS Grid": [
            ("3x2x2, D=3", lambda: generate_3d_grid(3, 2, 2, d_bond=3)),
            ("3x3x2, D=4", lambda: generate_3d_grid(3, 3, 2, d_bond=4)),
            ("3x3x3, D=4", lambda: generate_3d_grid(3, 3, 3, d_bond=4))
        ],
        "Random Regular": [
            ("N=16, D=4", lambda: generate_random_regular(16, d_bond=4)),
            ("N=24, D=4", lambda: generate_random_regular(24, d_bond=4)),
            ("N=32, D=4", lambda: generate_random_regular(32, d_bond=4))
        ],
        "Binary Tree": [
            ("depth=4, D=4", lambda: generate_binary_tree(depth=4, d_bond=4)),
            ("depth=5, D=5", lambda: generate_binary_tree(depth=5, d_bond=5)),
            ("depth=6, D=6", lambda: generate_binary_tree(depth=6, d_bond=6))
        ]
    }
    
    results = {}
    
    unsliced_job_dir = os.path.join(current_dir, "results", "prof_unsliced")
    sliced_job_dir = os.path.join(current_dir, "results", "prof_sliced")
    
    for topo_name, size_configs in topologies.items():
        results[topo_name] = []
        print(f"\n--- Profiling Topology: {topo_name} ---")
        
        for size_label, generator in size_configs:
            print(f"  * Size Configuration: {size_label}")
            tensors, edges = generator()
            
            # Export jobs
            export_contraction_job(tensors, edges, target_slices=1, job_dir=unsliced_job_dir)
            export_contraction_job(tensors, edges, target_slices=4, job_dir=sliced_job_dir)
            
            try:
                # Run sequential profile
                seq_metrics = run_step_profiler(unsliced_job_dir, "sequential", threads=1)
                
                # Run sliced parallel profile
                sliced_metrics = run_step_profiler(sliced_job_dir, "sliced", threads=4)
                
                results[topo_name].append({
                    "size_label": size_label,
                    "sequential": seq_metrics,
                    "sliced": sliced_metrics
                })
                
                seq_total_time = sum(step["time"] for step in seq_metrics)
                seq_root_time = seq_metrics[-1]["time"]
                seq_root_pct = (seq_root_time / seq_total_time) * 100 if seq_total_time > 0 else 0
                print(f"    Sequential: Total={seq_total_time:.4f}s | Root={seq_root_time:.4f}s ({seq_root_pct:.2f}%)")
                
            except Exception as e:
                print(f"    Error profiling size {size_label}: {e}")
            finally:
                subprocess.run(["rm", "-rf", unsliced_job_dir, sliced_job_dir])
            
    # Save the accumulated results
    output_json_path = os.path.join(current_dir, "results", "step_profiling_results.json")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nOrchestration sweep complete! Data saved to '{output_json_path}'.")

if __name__ == "__main__":
    main()
