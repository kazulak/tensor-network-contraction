import quimb.tensor as qtn
import cotengra as ctg
import time
import os

print(f"cotengra version: {ctg.__version__}")

# create a simple TN
tn = qtn.TN2D_rand(12, 12, D=2)
inputs, output, size_dict = tn.get_inputs(), tn.outer_inds(), tn.ind_sizes()

# optimizer with slicing
opt = ctg.HyperOptimizer(
    methods=['kahypar', 'greedy'],
    max_time=5,
    slicing_opts={'target_slices': 16}
)

tree = opt.search(inputs, output, size_dict)
print(f"Found tree with {tree.nslices} slices.")
print(f"Total FLOPs: {tree.total_flops()}")
