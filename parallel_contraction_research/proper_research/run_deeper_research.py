import os
import sys

# Set environment variables to restrict internal BLAS multithreading.
# This prevents different processes from thrashing the same physical cores.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Ensure the root of proper_research is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.benchmark_runner import run_deeper_benchmarks

if __name__ == "__main__":
    # We will test on worker counts of 1, 2, 4, 8, and 12
    # Output results will be saved in the 'results' subfolder
    run_deeper_benchmarks(worker_counts=[1, 2, 4, 8, 12], output_dir=os.path.join(current_dir, "results"))
