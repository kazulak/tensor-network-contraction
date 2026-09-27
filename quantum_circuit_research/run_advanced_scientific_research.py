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
job_dir = os.path.join(results_dir, "adv_scientific_job")

DEFAULT_THREADS = 6

def run_julia_json(script_name, args, num_threads=DEFAULT_THREADS):
    script_path = os.path.join(current_dir, "src", script_name)
    cmd = ["julia", "-t", str(num_threads), script_path] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {script_name}: {res.stderr}")
        return None
    
    lines = res.stdout.strip().splitlines()
    json_line = None
    for line in reversed(lines):
        line_s = line.strip()
        if (line_s.startswith("{") and line_s.endswith("}")) or (line_s.startswith("[") and line_s.endswith("]")):
            json_line = line_s
            break
            
    if json_line is None:
        print(f"Failed to find JSON in {script_name}: {res.stdout}")
        return None
        
    return json.loads(json_line)

def run_experiment_1():
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: LARGE-SCALE ENTANGLEMENT & COMPONENT DECOMPOSITION SWEEP")
    print(f"Running across 7 Topologies up to 100 qubits on {DEFAULT_THREADS} Threads")
    print("=" * 90)
    
    topologies = {
        "Zero-Entanglement (BB84)": [
            ("N=20", lambda: generate_bb84(20)),
            ("N=40", lambda: generate_bb84(40)),
            ("N=60", lambda: generate_bb84(60)),
            ("N=80", lambda: generate_bb84(80)),
            ("N=100", lambda: generate_bb84(100)),
        ],
        "Star-Graph (Bernstein-Vazirani)": [
            ("N=20", lambda: generate_bernstein_vazirani(20)),
            ("N=40", lambda: generate_bernstein_vazirani(40)),
            ("N=60", lambda: generate_bernstein_vazirani(60)),
            ("N=80", lambda: generate_bernstein_vazirani(80)),
            ("N=100", lambda: generate_bernstein_vazirani(100)),
        ],
        "Tree-Graph (Exclusive-OR)": [
            ("N=20", lambda: generate_xor(20)),
            ("N=40", lambda: generate_xor(40)),
            ("N=60", lambda: generate_xor(60)),
            ("N=80", lambda: generate_xor(80)),
            ("N=100", lambda: generate_xor(100)),
        ],
        "1D Local (Brickwork)": [
            ("N=14, D=10", lambda: generate_1d_brickwork(14, 10)),
            ("N=18, D=10", lambda: generate_1d_brickwork(18, 10)),
            ("N=22, D=10", lambda: generate_1d_brickwork(22, 10)),
            ("N=26, D=10", lambda: generate_1d_brickwork(26, 10)),
        ],
        "2D Planar (Sycamore Grid)": [
            ("3x3, D=8", lambda: generate_sycamore_like(3, 3, 8)),
            ("3x4, D=8", lambda: generate_sycamore_like(3, 4, 8)),
            ("4x4, D=8", lambda: generate_sycamore_like(4, 4, 8)),
            ("4x4, D=10", lambda: generate_sycamore_like(4, 4, 10)),
            ("5x5, D=8", lambda: generate_sycamore_like(5, 5, 8)),
        ],
        "All-to-All Structured (QFT)": [
            ("N=10", lambda: generate_qft(10)),
            ("N=12", lambda: generate_qft(12)),
            ("N=14", lambda: generate_qft(14)),
            ("N=16", lambda: generate_qft(16)),
            ("N=18", lambda: generate_qft(18)),
        ],
        "Random Haar Volume": [
            ("N=10, D=10", lambda: generate_random_arbitrary(10, 10)),
            ("N=12, D=12", lambda: generate_random_arbitrary(12, 12)),
            ("N=14, D=14", lambda: generate_random_arbitrary(14, 14)),
            ("N=16, D=14", lambda: generate_random_arbitrary(16, 14)),
            ("N=18, D=14", lambda: generate_random_arbitrary(18, 14)),
        ],
    }
    
    exp1_data = {}
    all_roofline_points = []
    
    for topo_name, size_configs in topologies.items():
        print(f"\n--- TOPOLOGY: {topo_name} ---")
        exp1_data[topo_name] = []
        
        for size_label, generator_fn in size_configs:
            print(f">>> Config: {topo_name} [{size_label}]")
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)
                
            tensors, edges = generator_fn()
            export_contraction_job(tensors, edges, target_slices=1, job_dir=job_dir)
            
            # 1. Advanced Scientific Profiler (Detailed component & roofline metrics)
            sci_data = run_julia_json("advanced_scientific_profiler.jl", [job_dir], num_threads=1)
            if sci_data is None:
                continue
                
            seq_time = sci_data["total_time"]
            pct_perm = sci_data["pct_perm"]
            pct_gemm = sci_data["pct_gemm"]
            pct_alloc = sci_data["pct_alloc"]
            intensity = sci_data["overall_intensity"]
            achieved_gflops = sci_data["overall_gflops"]
            
            # Collect step-level roofline samples
            for st in sci_data["steps"]:
                if st["flops"] > 0 and st["time_total"] > 0:
                    all_roofline_points.append({
                        "topology": topo_name,
                        "intensity": st["intensity"],
                        "gflops": st["gflops"],
                        "category": st["category"],
                        "flops": st["flops"],
                        "size_C": st["size_C"]
                    })
                    
            print(f"  [Profile] Time: {seq_time*1000:.2f}ms | Perm: {pct_perm:.1f}%, GEMM: {pct_gemm:.1f}%, Alloc: {pct_alloc:.1f}% | Intensity: {intensity:.2f} FLOPs/B | GFLOPS: {achieved_gflops:.2f}")
            
            # 2. Method A: Intra-Tensor
            intra_data = run_julia_json("intra_tensor_contractor.jl", [job_dir, str(DEFAULT_THREADS)], num_threads=DEFAULT_THREADS)
            intra_time = intra_data["elapsed"] if intra_data else None
            speedup_intra = (seq_time / intra_time) if (intra_time and intra_time > 0) else 0.0
            
            # 3. Method B: Tree-Level
            tree_data = run_julia_json("tree_level_contractor.jl", [job_dir], num_threads=DEFAULT_THREADS)
            tree_time = tree_data["elapsed"] if tree_data else None
            speedup_tree = (seq_time / tree_time) if (tree_time and tree_time > 0) else 0.0
            
            # 4. Method C: Combined Hybrid
            hybrid_data = run_julia_json("hybrid_combined_contractor.jl", [job_dir, str(DEFAULT_THREADS)], num_threads=DEFAULT_THREADS)
            hybrid_time = hybrid_data["elapsed"] if hybrid_data else None
            speedup_hybrid = (seq_time / hybrid_time) if (hybrid_time and hybrid_time > 0) else 0.0
            
            print(f"  [Speedups] Intra: {speedup_intra:.2f}x | Tree: {speedup_tree:.2f}x | Hybrid: {speedup_hybrid:.2f}x")
            
            exp1_data[topo_name].append({
                "size_label": size_label,
                "seq_time": seq_time,
                "intra_time": intra_time,
                "tree_time": tree_time,
                "hybrid_time": hybrid_time,
                "speedup_intra": speedup_intra,
                "speedup_tree": speedup_tree,
                "speedup_hybrid": speedup_hybrid,
                "pct_perm": pct_perm,
                "pct_gemm": pct_gemm,
                "pct_alloc": pct_alloc,
                "intensity": intensity,
                "achieved_gflops": achieved_gflops,
                "total_flops": sci_data["total_flops"],
                "total_bytes": sci_data["total_bytes"]
            })
            
    return exp1_data, all_roofline_points

def run_experiment_2():
    print("\n" + "=" * 90)
    print("EXPERIMENT 2: MULTI-THREAD STRONG SCALING & AMDAHL'S LAW STUDY")
    print("Testing across Worker Thread Counts: P = [1, 2, 4, 6, 8, 12]")
    print("=" * 90)
    
    scaling_circuits = {
        "Star-Graph BV (N=60)": lambda: generate_bernstein_vazirani(60),
        "1D Brickwork (N=22, D=10)": lambda: generate_1d_brickwork(22, 10),
        "Random Haar (N=14, D=14)": lambda: generate_random_arbitrary(14, 14),
        "Random Haar (N=16, D=14)": lambda: generate_random_arbitrary(16, 14),
    }
    
    thread_counts = [1, 2, 4, 6, 8, 12]
    exp2_data = {}
    
    for circuit_name, gen_fn in scaling_circuits.items():
        print(f"\n>>> Scaling Study: {circuit_name}")
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
            
        tensors, edges = gen_fn()
        export_contraction_job(tensors, edges, target_slices=1, job_dir=job_dir)
        
        # Measure base sequential time
        base_data = run_julia_json("advanced_scientific_profiler.jl", [job_dir], num_threads=1)
        t_base = base_data["total_time"] if base_data else 1.0
        
        circuit_scaling = {
            "t_base": t_base,
            "threads": thread_counts,
            "intra_times": [],
            "tree_times": [],
            "hybrid_times": [],
            "intra_speedup": [],
            "tree_speedup": [],
            "hybrid_speedup": [],
            "intra_efficiency": [],
            "tree_efficiency": [],
            "hybrid_efficiency": []
        }
        
        for P in thread_counts:
            intra_d = run_julia_json("intra_tensor_contractor.jl", [job_dir, str(P)], num_threads=P)
            t_intra = intra_d["elapsed"] if intra_d else t_base
            
            tree_d = run_julia_json("tree_level_contractor.jl", [job_dir], num_threads=P)
            t_tree = tree_d["elapsed"] if tree_d else t_base
            
            hyb_d = run_julia_json("hybrid_combined_contractor.jl", [job_dir, str(P)], num_threads=P)
            t_hyb = hyb_d["elapsed"] if hyb_d else t_base
            
            s_intra = t_base / t_intra if t_intra > 0 else 1.0
            s_tree = t_base / t_tree if t_tree > 0 else 1.0
            s_hyb = t_base / t_hyb if t_hyb > 0 else 1.0
            
            e_intra = s_intra / P
            e_tree = s_tree / P
            e_hyb = s_hyb / P
            
            circuit_scaling["intra_times"].append(t_intra)
            circuit_scaling["tree_times"].append(t_tree)
            circuit_scaling["hybrid_times"].append(t_hyb)
            circuit_scaling["intra_speedup"].append(s_intra)
            circuit_scaling["tree_speedup"].append(s_tree)
            circuit_scaling["hybrid_speedup"].append(s_hyb)
            circuit_scaling["intra_efficiency"].append(e_intra)
            circuit_scaling["tree_efficiency"].append(e_tree)
            circuit_scaling["hybrid_efficiency"].append(e_hyb)
            
            print(f"  P={P:2d} threads | Tree: {s_tree:.2f}x (E={e_tree*100:.1f}%) | Intra: {s_intra:.2f}x (E={e_intra*100:.1f}%) | Hybrid: {s_hyb:.2f}x (E={e_hyb*100:.1f}%)")
            
        exp2_data[circuit_name] = circuit_scaling
        
    return exp2_data

def main():
    start_total = time.time()
    
    exp1_data, roofline_points = run_experiment_1()
    exp2_data = run_experiment_2()
    
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        
    final_output = {
        "hardware": {
            "cpu_model": "AMD Ryzen 5 5600 6-Core Processor",
            "physical_cores": 6,
            "logical_threads": 12,
            "peak_fp64_gflops": 422.4,
            "peak_mem_bandwidth_gb_s": 51.2,
            "machine_balance_ridge": 8.25
        },
        "experiment_1_topologies": exp1_data,
        "experiment_2_strong_scaling": exp2_data,
        "roofline_points": roofline_points
    }
    
    out_json = os.path.join(results_dir, "advanced_scientific_results.json")
    with open(out_json, "w") as f:
        json.dump(final_output, f, indent=2)
        
    total_duration = time.time() - start_total
    print("\n" + "=" * 90)
    print(f"ADVANCED SCIENTIFIC RESEARCH COMPLETED IN {total_duration:.1f} SECONDS!")
    print(f"Data saved to: {out_json}")
    print("=" * 90)

if __name__ == "__main__":
    main()
