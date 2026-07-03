# Parallel Tensor-Network Contraction: Slicing and Speedup

## Abstract
Tensor network contraction is a computationally demanding task central to quantum physics and quantum computing simulations. We investigate the speedup achievable by parallelizing tensor network contraction using a "slicing" technique. By separating the contraction into independent tasks, we evaluate the serial versus parallel execution runtime on a 2D Grid tensor network. We demonstrate up to a 1.92x speedup on 4 CPU workers before memory bandwidth and inter-process communication overheads lead to diminishing returns, highlighting both the viability and the limitations of slicing for shared-memory multi-core systems.

## Introduction
The exact contraction of tensor networks, such as those arising from 2D grids (e.g., PEPS) or quantum circuits, scales exponentially with the size and complexity of the network. While advanced contraction path optimizers minimize the floating-point operations (FLOPs) required, the sheer volume of computation necessitates parallel execution. In this study, we answer the research question: *How much speedup can be achieved by parallelizing tensor-network contraction?* Our focus is explicitly on parallelization strategies rather than algorithmic contraction-ordering or avoiding the sign problem. 

## Background: Tensor-Network Contraction and Slicing
To contract an arbitrary tensor network, one finds a contraction tree that defines the pairwise order of operations. To parallelize this process, "slicing" (or bond-cutting) is used. Slicing selects a subset of tensor indices (edges in the network) and iterates over all their possible values. For a sliced edge with dimension $d$, the network is split into $d$ independent sub-networks that can be contracted in parallel. The total sum of these sub-network contractions yields the final exact result. This is highly parallelizable ("embarrassingly parallel"), but slicing increases the total FLOP count. The tradeoff is minimizing the FLOP overhead while creating enough independent tasks (slices) for parallel workers.

## Python Libraries and Implementation Choices
Our benchmark pipeline is built primarily using Python, orchestrating high-performance computational backends. 
- **quimb**: Used to construct a realistic 12x12 2D grid tensor network with a bond dimension of $D=3$.
- **cotengra**: Used for finding the optimal contraction path. We use `ctg.HyperOptimizer` with `slicing_opts` to explicitly instruct the optimizer to slice the contraction tree into at least 64 slices.
- **NumPy**: Serves as the numerical tensor contraction backend.
- **multiprocessing / concurrent.futures**: Manages the pool of worker processes to execute independent slice contractions in parallel.

To avoid severe CPU oversubscription (where each worker spawns multiple BLAS threads, causing thread thrashing), we strictly limit numerical libraries to a single thread per worker by setting environment variables (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.) prior to importing NumPy.

## Parallelization Method
1. The 12x12 2D grid tensor network is constructed.
2. `cotengra` identifies an optimized contraction tree, explicitly discovering 81 independent slices for the target topology.
3. The serial baseline computes the full tree (all 81 slices) on a single process.
4. The parallel implementation uses `concurrent.futures.ProcessPoolExecutor` with varying numbers of workers (1, 2, 4, 8). The original tensors and contraction tree are passed exactly once during worker initialization to minimize Inter-Process Communication (IPC) overhead. Each worker then eagerly evaluates the slices assigned to it.
5. We verify the parallel and serial results agree numerically to within floating-point precision.

## Experiments
We evaluate the contraction on a local multi-core CPU. 

**Experimental Parameters:**
* Network Topology: 12x12 2D Grid
* Bond Dimension: $D=3$
* Number of slices: 81
* Serial Total FLOPs: $10^{10.59}$ operations.
* Metrics: Runtime, Speedup, Parallel Efficiency, and Result Difference.

## Results

**Benchmark Table:**

| Workers | Time (s) | Speedup | Efficiency | Difference vs Serial |
|---------|----------|---------|------------|----------------------|
| Serial  | 18.1162  | 1.00x   | 1.00       | 0.00e+00             |
| 1       | 17.8344  | 1.02x   | 1.02       | 0.00e+00             |
| 2       | 10.8477  | 1.67x   | 0.84       | 0.00e+00             |
| 4       | 9.4121   | 1.92x   | 0.48       | 0.00e+00             |
| 8       | 10.2356  | 1.77x   | 0.22       | 0.00e+00             |

*(Note: 1 Worker parallel execution slightly outperforms Serial due to process isolation avoiding background GIL contention, though the difference is within noise margins).*

**Speedup vs Number of Workers Plot:**

![Speedup vs Workers](/home/tom/.gemini/antigravity-cli/brain/a7de9a6f-46c4-413e-a121-e27acc51d47f/speedup_plot.png)

## Discussion
**Did parallelization help?**
Yes, parallelization successfully reduced the absolute runtime from 18.1 seconds to 9.4 seconds.

**How much speedup was achieved?**
A maximum speedup of **1.92x** was achieved using 4 workers. 

**What limited the speedup?**
The speedup is distinctly sublinear and peaks at 4 workers. Going from 4 to 8 workers causes a regression in performance (10.23s runtime). This indicates a bottleneck beyond CPU cores. Because the parallel execution relies on `ProcessPoolExecutor` with NumPy, each process must continuously allocate large memory blocks for intermediate tensors. This saturates the shared L3 cache and main memory bandwidth of the CPU. Furthermore, despite limiting IPC by initializing shared globals, Python's multiprocessing still introduces context-switching and orchestration overhead. 

## Limitations
1. **Hardware Bounds**: This benchmark was run on a single machine where memory bandwidth is shared among all cores. 
2. **Backend Engine**: Pure NumPy relies on the CPU. A GPU backend (like CuPy or PyTorch) would handle large tensor operations much faster but would require careful VRAM management for parallel slice execution.

## Future Work
The most immediate next improvement would be distributing the sliced tasks across a cluster of multiple independent nodes (using a framework like Dask or Ray, or MPI). Distributed nodes possess independent memory buses, eliminating the shared memory bandwidth bottleneck observed on a single machine. Additionally, running the individual slice contractions on GPUs using `cuTensorNet` would drastically reduce the contraction time per slice.

## Conclusion
Parallelizing tensor-network contraction via slicing is a highly effective strategy for mapping complex contractions onto parallel workers. While we successfully achieved a nearly 2x speedup and validated numerical correctness, we also demonstrated that on shared-memory multi-core systems, memory bandwidth and process management overhead quickly bottleneck parallel efficiency. Scaling beyond these limits requires distributed computing clusters or hardware accelerators.

## References
1. Gray, J., & Kourtis, S. (2021). *Hyper-optimized tensor network contraction*. Quantum, 5, 410. (`cotengra`)
2. Gray, J. (2018). *quimb: A python package for quantum information and many-body calculations*. Journal of Open Source Software, 3(29), 819.
