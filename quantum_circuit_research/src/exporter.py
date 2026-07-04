import os
import json
import numpy as np
import cotengra as ctg

def export_contraction_job(tensors, edges, target_slices, job_dir="job_data"):
    """
    Exports a tensor network and its sliced contraction plan to a folder.
    
    Args:
        tensors: list of numpy arrays
        edges: list of lists of strings (original index names)
        target_slices: int, number of target slices
        job_dir: str, target directory for export
    """
    os.makedirs(job_dir, exist_ok=True)
    
    # 1. Build size dict
    size_dict = {}
    for t, e_list in zip(tensors, edges):
        for dim_idx, idx in enumerate(e_list):
            size_dict[idx] = t.shape[dim_idx]
            
    # Determine output indices (empty list for closed networks)
    idx_counts = {}
    for edge_list in edges:
        for idx in edge_list:
            idx_counts[idx] = idx_counts.get(idx, 0) + 1
    output_indices = [idx for idx, count in idx_counts.items() if count == 1]
    
    # 2. Find sliced contraction path
    opt = ctg.HyperOptimizer(
        methods=['greedy'],
        slicing_opts={'target_slices': target_slices} if target_slices > 1 else None,
        max_time=2.0,
        parallel=False
    )
    tree = opt.search(edges, output_indices, size_dict)
    nslices = tree.nslices
    
    # 3. Save raw tensors as a single binary file to avoid hundreds of separate disk writes
    with open(os.path.join(job_dir, "tensors.bin"), "wb") as f:
        for t in tensors:
            t_arr = t.flatten(order='F')
            f.write(t_arr.tobytes())
        
    # 4. Reconstruct contraction tree steps with original index names
    node_inds = {i: list(edges[i]) for i in range(len(tensors))}
    steps = []
    
    for parent, (left, right) in tree.children.items():
        inds_left = node_inds[left]
        inds_right = node_inds[right]
        
        # Contracted legs
        inds_contract = [idx for idx in inds_left if idx in inds_right]
        
        # Output legs of the parent node (uncontracted left followed by uncontracted right)
        inds_parent = [idx for idx in inds_left if idx not in inds_contract] + \
                      [idx for idx in inds_right if idx not in inds_contract]
                      
        node_inds[parent] = inds_parent
        
        steps.append({
            "parent": parent,
            "left": left,
            "right": right,
            "contract_legs": inds_contract
        })
        
    # 5. Write metadata plan.txt
    with open(os.path.join(job_dir, "plan.txt"), "w") as f:
        f.write(f"{nslices}\n")
        
        # Sliced indices
        sliced_parts = []
        for name, info in tree.sliced_inds.items():
            sliced_parts.append(f"{name}:{info.size}")
        f.write(" ".join(sliced_parts) + "\n")
        
        # Tensors
        f.write(f"{len(tensors)}\n")
        for idx, (t, e_list) in enumerate(zip(tensors, edges)):
            shape_str = ",".join(str(s) for s in t.shape)
            edges_str = ",".join(e_list)
            f.write(f"{shape_str}|{edges_str}\n")
            
        # Steps
        f.write(f"{len(steps)}\n")
        for s in steps:
            legs_str = ",".join(s["contract_legs"]) if s["contract_legs"] else "NONE"
            f.write(f"{s['parent']} {s['left']} {s['right']} {legs_str}\n")
            
    print(f"Exported contraction job successfully to '{job_dir}'. nslices={nslices}")
    return nslices
