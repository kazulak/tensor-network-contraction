import quimb.tensor as qtn
import cotengra as ctg
import time
import numpy as np
import concurrent.futures

def run_test():
    print("Creating TN...")
    # 2D Grid TN
    # To get a measurable time, size needs to be big enough but not too big
    # Let's try 10x10 with bond dimension 2 or 3
    tn = qtn.TN2D_rand(10, 10, D=3)
    
    print("Finding contraction path...")
    # Find a tree without slicing first
    opt = ctg.HyperOptimizer(
        max_time=2.0,
        methods=['greedy', 'kahypar'],
        progbar=False
    )
    
    # We can ask for a tree with slices directly
    opt_sliced = ctg.HyperOptimizer(
        max_time=2.0,
        methods=['greedy', 'kahypar'],
        slicing_opts={'target_slices': 64},
        progbar=False
    )
    
    tree = tn.contraction_tree(opt_sliced)
    print(f"Tree has {tree.nslices} slices.")
    
    # Arrays
    arrays = [t.data for t in tn]
    
    print("Contracting serially...")
    t0 = time.time()
    res_serial = tree.contract(arrays, backend='numpy')
    t_serial = time.time() - t0
    print(f"Serial time: {t_serial:.4f}s")
    
    print("Contracting in parallel...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        t0 = time.time()
        res_parallel = tree.contract(arrays, pool=pool, backend='numpy')
        t_parallel = time.time() - t0
    print(f"Parallel time: {t_parallel:.4f}s")
    
    print(f"Difference: {abs(res_serial - res_parallel)}")

if __name__ == "__main__":
    run_test()
