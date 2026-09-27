import os
import sys
import time
import shutil
import tempfile
import numpy as np

# Ensure proper_research root and repo root in path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(repo_dir, "quantum_circuit_research"))

from src.network_generators import generate_2d_grid, generate_1d_chain
from src.exporter import export_contraction_job
from src.circuit_generators import generate_bb84

def run_smoke_benchmark():
    print("=" * 70)
    print("Running Real Pipeline Tensor Network Smoke Benchmark...")
    print("=" * 70)
    start_time = time.perf_counter()
    
    # 1. Test 2D PEPS Network Generation
    tensors_grid, edges_grid = generate_2d_grid(rows=3, cols=3, d_bond=3, seed=42)
    print(f"Generated 3x3 PEPS grid ({len(tensors_grid)} tensors).")
    
    # 2. Test Contraction Export & Slicing Plan Generation
    temp_dir = tempfile.mkdtemp(prefix="smoke_job_")
    try:
        nslices = export_contraction_job(tensors_grid, edges_grid, target_slices=2, job_dir=temp_dir)
        plan_file = os.path.join(temp_dir, "plan.txt")
        tensors_file = os.path.join(temp_dir, "tensors.bin")
        
        assert os.path.exists(plan_file), "plan.txt was not created"
        assert os.path.exists(tensors_file), "tensors.bin was not created"
        print(f"Successfully exported contraction job (nslices={nslices}).")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    # 3. Test Quantum Circuit Generator
    q_tensors, q_edges = generate_bb84(num_qubits=4, seed=123)
    print(f"Generated 4-qubit BB84 quantum circuit network ({len(q_tensors)} tensors).")
    
    elapsed = time.perf_counter() - start_time
    print(f"Smoke benchmark completed successfully in {elapsed:.4f} seconds.")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_benchmark()
