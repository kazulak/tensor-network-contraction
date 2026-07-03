import os
# Limit BLAS threads per worker to prevent CPU oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from parallel_contraction.benchmark import run_benchmarks

if __name__ == '__main__':
    run_benchmarks()
