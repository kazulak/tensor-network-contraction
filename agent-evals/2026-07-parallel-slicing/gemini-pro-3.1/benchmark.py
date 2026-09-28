import os
# Limit BLAS threads for each worker to avoid oversubscription
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import quimb.tensor as qtn
import cotengra as ctg
import time
import numpy as np
import concurrent.futures

# Global variables for workers to avoid passing large data
GLOBAL_TREE = None
GLOBAL_ARRAYS = None

def init_worker(tree, arrays):
    global GLOBAL_TREE, GLOBAL_ARRAYS
    GLOBAL_TREE = tree
    GLOBAL_ARRAYS = arrays

def contract_slice_worker(i):
    # Eagerly compute one slice
    return GLOBAL_TREE.contract_slice(GLOBAL_ARRAYS, i, backend='numpy')

def run_benchmark():
    L = 12
    D = 3
    print(f"Creating 2D Grid TN {L}x{L} with bond dimension {D}...")
    tn = qtn.TN2D_rand(L, L, D=D)
    arrays = [t.data for t in tn]
    
    print("Finding contraction path...")
    # Target slices: enough to keep workers busy, e.g., 64-256
    opt_sliced = ctg.HyperOptimizer(
        max_time=10.0,
        methods=['kahypar', 'greedy'],
        slicing_opts={'target_slices': 64},
        progbar=True
    )
    
    tree = tn.contraction_tree(opt_sliced)
    print(f"Tree has {tree.nslices} slices.")
    print(f"Total FLOPs: {np.log10(tree.total_flops()):.2f} (log10)")
    
    print("Contracting serially...")
    t0 = time.time()
    res_serial = tree.contract(arrays, backend='numpy')
    t_serial = time.time() - t0
    print(f"Serial time: {t_serial:.4f}s")
    
    num_workers_list = [1, 2, 4, 8]
    results = []
    
    for workers in num_workers_list:
        print(f"Contracting in parallel with {workers} workers...")
        t0 = time.time()
        
        # Parallel contraction over slices
        # We use a multiprocessing pool
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(tree, arrays)
        ) as pool:
            slice_results = list(pool.map(contract_slice_worker, range(tree.nslices)))
            
        res_parallel = sum(slice_results)
        t_parallel = time.time() - t0
        
        speedup = t_serial / t_parallel
        efficiency = speedup / workers
        diff = np.max(np.abs(res_serial - res_parallel))
        print(f"  Workers: {workers}, Time: {t_parallel:.4f}s, Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}, Diff: {diff:.2e}")
        
        results.append({
            'workers': workers,
            'time': t_parallel,
            'speedup': speedup,
            'efficiency': efficiency,
            'diff': diff
        })

    print("\nBenchmark Summary:")
    print("Workers | Time (s) | Speedup | Efficiency")
    print("-" * 45)
    print(f"{'Serial':>7} | {t_serial:>8.4f} | {'1.00':>7} | {'1.00':>10}")
    for res in results:
        print(f"{res['workers']:>7} | {res['time']:>8.4f} | {res['speedup']:>7.2f} | {res['efficiency']:>10.2f}")

if __name__ == "__main__":
    run_benchmark()
