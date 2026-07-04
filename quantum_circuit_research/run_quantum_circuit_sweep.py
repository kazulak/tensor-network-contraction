import os
import json
import subprocess
import shutil
from src.circuit_generators import (
    generate_bb84,
    generate_bernstein_vazirani,
    generate_error_detection,
    generate_hidden_subgroup,
    generate_qrng,
    generate_xor,
    generate_sycamore_like
)
from src.exporter import export_contraction_job

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results")
os.makedirs(results_dir, exist_ok=True)

unsliced_job_dir = os.path.join(results_dir, "prof_unsliced")
sliced_job_dir = os.path.join(results_dir, "prof_sliced")

# Julia environment project path
julia_project = "/home/tom/repos/tensor-network-contraction/parallel_contraction_research/proper_research/src/"

def run_julia_profiler(job_dir, mode, threads=1):
    cmd = [
        "julia",
        f"-t", str(threads),
        f"--project={julia_project}",
        os.path.join(current_dir, "src", "step_profiler.jl"),
        job_dir,
        mode
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running Julia profiler in {job_dir}:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None
    
    # Parse JSON from stdout lines
    lines = result.stdout.splitlines()
    json_line = None
    for line in reversed(lines):
        if line.strip().startswith("[") and line.strip().endswith("]"):
            json_line = line
            break
            
    if json_line is None:
        print(f"Error: Could not find JSON output in Julia stdout. Output: {result.stdout}")
        return None
        
    return json.loads(json_line)

def main():
    print("=" * 80)
    print("STARTING 7x7 QUANTUM CIRCUIT TENSOR NETWORK CONTRACTION SWEEP")
    print("=" * 80)
    
    # 7 topologies, each with 7 size configurations
    topologies = {
        "BB84 Protocol (BB_n)": [
            ("N=6", lambda: generate_bb84(6)),
            ("N=8", lambda: generate_bb84(8)),
            ("N=10", lambda: generate_bb84(10)),
            ("N=12", lambda: generate_bb84(12)),
            ("N=14", lambda: generate_bb84(14)),
            ("N=16", lambda: generate_bb84(16)),
            ("N=18", lambda: generate_bb84(18))
        ],
        "Bernstein-Vazirani (BV_n)": [
            ("N=6", lambda: generate_bernstein_vazirani(6)),
            ("N=8", lambda: generate_bernstein_vazirani(8)),
            ("N=10", lambda: generate_bernstein_vazirani(10)),
            ("N=12", lambda: generate_bernstein_vazirani(12)),
            ("N=14", lambda: generate_bernstein_vazirani(14)),
            ("N=16", lambda: generate_bernstein_vazirani(16)),
            ("N=18", lambda: generate_bernstein_vazirani(18))
        ],
        "Error Detection Code (EDC_n)": [
            ("N=6", lambda: generate_error_detection(6)),
            ("N=8", lambda: generate_error_detection(8)),
            ("N=10", lambda: generate_error_detection(10)),
            ("N=12", lambda: generate_error_detection(12)),
            ("N=14", lambda: generate_error_detection(14)),
            ("N=16", lambda: generate_error_detection(16)),
            ("N=18", lambda: generate_error_detection(18))
        ],
        "Hidden Subgroup (HS_2n)": [
            ("N=6", lambda: generate_hidden_subgroup(6)),
            ("N=8", lambda: generate_hidden_subgroup(8)),
            ("N=10", lambda: generate_hidden_subgroup(10)),
            ("N=12", lambda: generate_hidden_subgroup(12)),
            ("N=14", lambda: generate_hidden_subgroup(14)),
            ("N=16", lambda: generate_hidden_subgroup(16)),
            ("N=18", lambda: generate_hidden_subgroup(18))
        ],
        "Quantum Random Number Generator (QRNG_n)": [
            ("N=6", lambda: generate_qrng(6)),
            ("N=8", lambda: generate_qrng(8)),
            ("N=10", lambda: generate_qrng(10)),
            ("N=12", lambda: generate_qrng(12)),
            ("N=14", lambda: generate_qrng(14)),
            ("N=16", lambda: generate_qrng(16)),
            ("N=18", lambda: generate_qrng(18))
        ],
        "Exclusive-OR (XOR_n)": [
            ("N=6", lambda: generate_xor(6)),
            ("N=8", lambda: generate_xor(8)),
            ("N=10", lambda: generate_xor(10)),
            ("N=12", lambda: generate_xor(12)),
            ("N=14", lambda: generate_xor(14)),
            ("N=16", lambda: generate_xor(16)),
            ("N=18", lambda: generate_xor(18))
        ],
        "Sycamore-like Reference": [
            ("2x3, D=6", lambda: generate_sycamore_like(2, 3, 6)),
            ("2x4, D=6", lambda: generate_sycamore_like(2, 4, 6)),
            ("3x3, D=6", lambda: generate_sycamore_like(3, 3, 6)),
            ("3x4, D=6", lambda: generate_sycamore_like(3, 4, 6)),
            ("4x4, D=6", lambda: generate_sycamore_like(4, 4, 6)),
            ("4x5, D=6", lambda: generate_sycamore_like(4, 5, 6)),
            ("5x5, D=6", lambda: generate_sycamore_like(5, 5, 6))
        ]
    }
    
    final_results = {}
    
    for topo_name, size_configs in topologies.items():
        print(f"\n--- Profiling Topology: {topo_name} ---")
        final_results[topo_name] = []
        
        for size_label, generator_fn in size_configs:
            print(f"  * Size Configuration: {size_label}")
            
            # 1. Clean directories
            for d in [unsliced_job_dir, sliced_job_dir]:
                if os.path.exists(d):
                    shutil.rmtree(d)
                    
            # 2. Generate network tensors and edges
            tensors, edges = generator_fn()
            
            # 3. Export unsliced
            nslices_unsliced = export_contraction_job(tensors, edges, target_slices=1, job_dir=unsliced_job_dir)
            # 4. Export sliced (target 4 slices)
            nslices_sliced = export_contraction_job(tensors, edges, target_slices=4, job_dir=sliced_job_dir)
            
            # 5. Run sequential (1 thread)
            seq_metrics = run_julia_profiler(unsliced_job_dir, "sequential", threads=1)
            
            # 6. Run parallel sliced (4 threads)
            sliced_metrics = run_julia_profiler(sliced_job_dir, "sliced", threads=4)
            
            if seq_metrics is None or sliced_metrics is None:
                print(f"    WARNING: Sweep step failed for config {size_label}")
                continue
                
            # Log key results
            seq_total_time = sum(step["time"] for step in seq_metrics)
            seq_root_time = seq_metrics[-1]["time"]
            seq_root_pct = (seq_root_time / seq_total_time) * 100 if seq_total_time > 0 else 0.0
            
            print(f"    Sequential: Total={seq_total_time:.4f}s | Root={seq_root_time:.4f}s ({seq_root_pct:.2f}%)")
            
            final_results[topo_name].append({
                "size_label": size_label,
                "sequential": seq_metrics,
                "sliced": sliced_metrics
            })
            
    # Clean up large temp directories
    for d in [unsliced_job_dir, sliced_job_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
            
    # Save sweep results
    out_json = os.path.join(results_dir, "quantum_scaling_results.json")
    with open(out_json, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nSweep complete! Data saved to '{out_json}'.")

if __name__ == "__main__":
    main()
