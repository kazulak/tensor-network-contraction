# Prompt given to each agent (experiment 01)

Recovered from the repository README at commit d5ecf54. Given verbatim to each agent.

````text
You are an autonomous research agent. Your task is to do a small research project on **parallel tensor-network contraction using Python**.

## Research question

**How much speedup can be achieved by parallelizing tensor-network contraction?**

Focus only on parallelization. Do not study the sign problem. Do not invent a new contraction-ordering algorithm. You may use existing contraction-ordering tools only as part of the implementation.

## Requirements

Use **Python** as the main language.

Do not use naive Python loops, raw `numpy.einsum`, or Python threading as the main method. Python should be used with serious tensor-network and parallel-computing tools.

Recommended libraries to consider:

* `quimb` for building tensor networks,
* `cotengra` for contraction planning, slicing, and optimized contraction trees,
* NumPy/SciPy, CuPy, PyTorch, JAX, or cuTensorNet as numerical backends,
* multiprocessing, MPI, Dask, Ray, or joblib for parallel execution.

A strong solution will likely use **slicing**: split a tensor-network contraction into many independent slice contractions, run them in parallel, and combine the results.

## What to build

Create a small reproducible benchmark pipeline that:

1. Builds one or more tensor-network examples.
2. Computes a contraction plan using existing libraries.
3. Runs a serial baseline.
4. Runs a parallel version.
5. Checks that serial and parallel results agree.
6. Measures speedup.

Use at least one realistic benchmark family, such as:

* random tensor networks,
* 2D grid tensor networks,
* quantum-circuit tensor networks,
* Ising-model tensor networks,
* PEPS-like tensor networks.

Use more than one benchmark family if feasible.

## Metrics to report

Report:

* serial runtime,
* parallel runtime,
* number of workers,
* speedup,
* parallel efficiency,
* tensor-network size,
* number of slices or tasks,
* numerical error versus baseline.

Use:

```text
speedup = serial_time / parallel_time
parallel_efficiency = speedup / number_of_workers
```

Control obvious timing issues. In particular, explain how you avoid CPU oversubscription from BLAS or nested parallelism.

## Deliverables

Produce:

1. A short research paper in Markdown or Typst.
2. A small Python code structure or implementation.
3. Reproduction instructions.
4. A benchmark table.
5. At least one plot, preferably speedup versus number of workers.
6. A clear answer to the research question.

## Suggested paper structure

Use this structure:

1. Title
2. Abstract
3. Introduction
4. Background: tensor-network contraction and slicing
5. Python libraries and implementation choices
6. Parallelization method
7. Experiments
8. Results
9. Discussion
10. Limitations
11. Future work
12. Conclusion
13. References

## Important guidance

Be honest. Do not fabricate benchmark results. If full implementation is not possible, provide a clear experimental plan, code skeleton, and expected bottlenecks.

The final paper should clearly answer:

* Did parallelization help?
* How much speedup was achieved or expected?
* What limited the speedup?
* What would be the next improvement?

````
