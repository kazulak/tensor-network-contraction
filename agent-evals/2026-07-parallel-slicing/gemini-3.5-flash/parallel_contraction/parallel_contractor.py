import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Global variables in worker processes to store shared state and avoid re-serialization
_global_tree = None
_global_arrays = None

def init_worker(tree, arrays):
    """Initializer function run in worker processes to cache the tree and arrays."""
    global _global_tree, _global_arrays
    _global_tree = tree
    _global_arrays = arrays

def run_slice_global(i):
    """Worker function to compute a single slice contraction."""
    global _global_tree, _global_arrays
    return _global_tree.contract_slice(_global_arrays, i)

def contract_sliced_parallel(tree, arrays, num_workers):
    """Contracts a sliced tensor network in parallel.
    
    Args:
        tree (cotengra.ContractionTree): Sliced contraction tree.
        arrays (list of np.ndarray): Raw tensor data.
        num_workers (int): Number of worker processes.
        
    Returns:
        np.ndarray: The contracted output tensor (scalar in our closed cases).
    """
    if num_workers == 1:
        # Avoid process pool overhead for a single worker
        return sum(tree.contract_slice(arrays, i) for i in range(tree.nslices))
        
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(tree, arrays)) as executor:
        results = list(executor.map(run_slice_global, range(tree.nslices)))
    return sum(results)

def contract_sliced_serial(tree, arrays):
    """Contracts a sliced tensor network in serial.
    
    Args:
        tree (cotengra.ContractionTree): Sliced contraction tree.
        arrays (list of np.ndarray): Raw tensor data.
        
    Returns:
        np.ndarray: The contracted output tensor.
    """
    return tree.contract(arrays)
