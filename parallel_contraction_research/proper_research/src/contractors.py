import os
import time
import numpy as np
import cotengra as ctg
import quimb.tensor as qtn
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Globals for caching in worker processes to eliminate serialization overhead
_global_tree = None
_global_arrays = None

def init_process_worker(tree, arrays):
    """Caches the contraction tree and arrays in process-level global variables."""
    global _global_tree, _global_arrays
    _global_tree = tree
    _global_arrays = arrays

def run_slice_process(i):
    """Executes a single slice contraction on the cached globals."""
    global _global_tree, _global_arrays
    return _global_tree.contract_slice(_global_arrays, i)

class ContractionEngine:
    def __init__(self, tensors, edges, target_slices=0, optimize_time=5.0):
        # Use quimb to wrap the network, ensuring index and backend alignment
        self.tn = qtn.TensorNetwork([
            qtn.Tensor(data=np.ascontiguousarray(t), inds=e) 
            for t, e in zip(tensors, edges)
        ])
        self.tensors = [t.data for t in self.tn]
        self.target_slices = target_slices
        
        # Configure the hyper-optimizer
        if target_slices > 1:
            opt = ctg.HyperOptimizer(
                methods=['greedy'],
                slicing_opts={'target_slices': target_slices},
                max_time=optimize_time,
                parallel=False
            )
            self.sliced_tree = self.tn.contraction_tree(optimize=opt)
            self.tree = self.sliced_tree
            self.nslices = self.sliced_tree.nslices
            print(f"  [Engine] Path optimized: sliced into {self.nslices} slices. Sliced width: {self.sliced_tree.contraction_width()}")
        else:
            opt = ctg.HyperOptimizer(
                methods=['greedy'],
                max_time=optimize_time,
                parallel=False
            )
            self.tree = self.tn.contraction_tree(optimize=opt)
            self.sliced_tree = None
            self.nslices = 1
            print(f"  [Engine] Path optimized: single unsliced tree. Contraction width: {self.tree.contraction_width()}")

    def contract_sequential_unsliced(self):
        """Runs the optimal sequential contraction without slicing."""
        if self.tree is None:
            raise ValueError("Contraction path not optimized.")
        start = time.perf_counter()
        res = self.tree.contract(self.tensors)
        duration = time.perf_counter() - start
        return res, duration

    def contract_sequential_sliced(self):
        """Runs the sliced contraction sequentially (one slice at a time)."""
        if self.sliced_tree is None:
            return self.contract_sequential_unsliced()
            
        start = time.perf_counter()
        res = sum(self.sliced_tree.contract_slice(self.tensors, i) for i in range(self.nslices))
        duration = time.perf_counter() - start
        return res, duration

    def contract_parallel_processes(self, num_workers):
        """Runs sliced contraction using ProcessPoolExecutor with cached initialization."""
        if self.sliced_tree is None:
            raise ValueError("Slicing must be enabled to run in parallel.")
            
        start = time.perf_counter()
        
        if num_workers == 1:
            res = sum(self.sliced_tree.contract_slice(self.tensors, i) for i in range(self.nslices))
            return res, time.perf_counter() - start
            
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_process_worker,
            initargs=(self.sliced_tree, self.tensors)
        ) as executor:
            results = list(executor.map(run_slice_process, range(self.nslices)))
            
        res = sum(results)
        duration = time.perf_counter() - start
        return res, duration

    def contract_parallel_threads(self, num_workers):
        """Runs sliced contraction using ThreadPoolExecutor (shared memory, GIL bound)."""
        if self.sliced_tree is None:
            raise ValueError("Slicing must be enabled to run in parallel.")
            
        start = time.perf_counter()
        
        if num_workers == 1:
            res = sum(self.sliced_tree.contract_slice(self.tensors, i) for i in range(self.nslices))
            return res, time.perf_counter() - start
            
        def run_slice(i):
            return self.sliced_tree.contract_slice(self.tensors, i)
            
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(run_slice, range(self.nslices)))
            
        res = sum(results)
        duration = time.perf_counter() - start
        return res, duration
