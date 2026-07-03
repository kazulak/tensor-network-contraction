using Sockets
using LinearAlgebra

# Structure representing step descriptors
struct ContractionStep
    parent::Int
    left::Int
    right::Int
    left_inds::Vector{String}
    right_inds::Vector{String}
    contract_inds::Vector{String}
    out_inds::Vector{String}
    flops::Float64  # Estimated arithmetic complexity
end

function read_tree_plan(job_dir::String)
    plan_path = joinpath(job_dir, "plan.txt")
    lines = readlines(plan_path)
    
    line_idx = 1
    nslices = parse(Int, lines[line_idx])
    line_idx += 2  # skip nslices and sliced_inds line
    
    num_tensors = parse(Int, lines[line_idx])
    line_idx += 1
    
    # Store shapes and original index list for each tensor
    tensor_shapes = Vector{Vector{Int}}()
    tensor_indices = Vector{Vector{String}}()
    
    for i in 1:num_tensors
        parts = split(lines[line_idx], "|")
        shape = isempty(parts[1]) ? Int[] : [parse(Int, s) for s in split(parts[1], ",")]
        inds = isempty(parts[2]) ? String[] : [String(s) for s in split(parts[2], ",")]
        push!(tensor_shapes, shape)
        push!(tensor_indices, inds)
        line_idx += 1
    end
    
    num_steps = parse(Int, lines[line_idx])
    line_idx += 1
    
    steps = Vector{ContractionStep}()
    
    # Keep track of active indices of each intermediate tensor
    node_indices = Dict{Int, Vector{String}}()
    for i in 1:num_tensors
        node_indices[i-1] = tensor_indices[i]
    end
    
    for i in 1:num_steps
        parts = split(lines[line_idx])
        parent = parse(Int, parts[1])
        left = parse(Int, parts[2])
        right = parse(Int, parts[3])
        contract_inds = parts[4] == "NONE" ? String[] : [String(s) for s in split(parts[4], ",")]
        
        # Get input legs of the children
        inds_left = node_indices[left]
        inds_right = node_indices[right]
        
        # Output legs of parent: left free legs, then right free legs
        inds_parent = filter(x -> !(x in contract_inds), inds_left)
        append!(inds_parent, filter(x -> !(x in contract_inds), inds_right))
        node_indices[parent] = inds_parent
        
        # Simple FLOP estimation: 2.0^(number of unique indices involved)
        unique_inds = unique(vcat(inds_left, inds_right))
        flops = 2.0^(length(unique_inds))
        
        push!(steps, ContractionStep(parent, left, right, inds_left, inds_right, contract_inds, inds_parent, flops))
        line_idx += 1
    end
    
    return num_tensors, steps
end

# Core pairwise tensor contractor (performing transposes and gemm)
function contract_pair(A::Array{Float64}, B::Array{Float64}, left_inds::Vector{String}, right_inds::Vector{String}, contract_inds::Vector{String}, out_inds::Vector{String})
    # If there are no contraction legs, it is an outer product
    if isempty(contract_inds)
        A_flat = reshape(A, :)
        B_flat = reshape(B, :)
        C_flat = A_flat * B_flat'
        return reshape(C_flat, vcat(size(A)..., size(B)...)...)
    end
    
    # Identify positions of contraction indices (1-based in Julia)
    c_pos_A = [findfirst(x -> x == i, left_inds) for i in contract_inds]
    c_pos_B = [findfirst(x -> x == i, right_inds) for i in contract_inds]
    
    # Identify positions of free indices
    free_pos_A = filter(x -> !(x in c_pos_A), 1:ndims(A))
    free_pos_B = filter(x -> !(x in c_pos_B), 1:ndims(B))
    
    # Permute A so contraction indices are at the end
    perm_A = vcat(free_pos_A, c_pos_A)
    A_perm = permutedims(A, perm_A)
    
    # Permute B so contraction indices are at the beginning
    perm_B = vcat(c_pos_B, free_pos_B)
    B_perm = permutedims(B, perm_B)
    
    # Flatten into matrices
    size_free_A = [size(A, i) for i in free_pos_A]
    size_c = [size(A, i) for i in c_pos_A]
    size_free_B = [size(B, i) for i in free_pos_B]
    
    mat_A = reshape(A_perm, prod(size_free_A; init=1), prod(size_c; init=1))
    mat_B = reshape(B_perm, prod(size_c; init=1), prod(size_free_B; init=1))
    
    # Matrix Multiplication
    mat_C = mat_A * mat_B
    
    # Reconstruct tensor shape
    C_raw = reshape(mat_C, vcat(size_free_A, size_free_B)...)
    
    # Map raw C indices to target output indices
    raw_out_inds = vcat([left_inds[i] for i in free_pos_A], [right_inds[i] for i in free_pos_B])
    perm_C = [findfirst(x -> x == i, raw_out_inds) for i in out_inds]
    
    if isempty(perm_C)
        return C_raw
    else
        return permutedims(C_raw, perm_C)
    end
end

# Load initial tensors from binary files
function load_tensors_only(job_dir::String, num_initial::Int)
    tensors = Vector{Array{Float64}}(undef, num_initial)
    plan_lines = readlines(joinpath(job_dir, "plan.txt"))
    
    # Parse shapes sequentially
    line_idx = 4
    for i in 1:num_initial
        parts = split(plan_lines[line_idx], "|")
        shape = isempty(parts[1]) ? Int[] : [parse(Int, x) for x in split(parts[1], ",")]
        
        t_path = joinpath(job_dir, "tensors", "$(i-1).bin")
        arr_size = prod(shape; init=1)
        data = Vector{Float64}(undef, arr_size)
        read!(t_path, data)
        
        if isempty(shape)
            tensors[i] = fill(data[1])
        else
            tensors[i] = reshape(data, shape...)
        end
        line_idx += 1
    end
    return tensors
end

# Recursive contractor function
function contract_recursive(node_idx::Int, num_initial::Int, steps::Vector{ContractionStep}, parent_to_step::Dict{Int, ContractionStep}, node_cache::Dict{Int, Array{Float64}}, threshold::Float64)
    # Leaf node: return initial tensor
    if node_idx < num_initial
        return node_cache[node_idx]
    end
    
    step = parent_to_step[node_idx]
    
    if step.flops < threshold
        # Contract sequentially on current thread
        left_tensor = contract_recursive(step.left, num_initial, steps, parent_to_step, node_cache, threshold)
        right_tensor = contract_recursive(step.right, num_initial, steps, parent_to_step, node_cache, threshold)
        return contract_pair(left_tensor, right_tensor, step.left_inds, step.right_inds, step.contract_inds, step.out_inds)
    else
        # Fork: Spawn left child as an async task, compute right child on current thread
        task_left = Threads.@spawn contract_recursive(step.left, num_initial, steps, parent_to_step, node_cache, threshold)
        right_tensor = contract_recursive(step.right, num_initial, steps, parent_to_step, node_cache, threshold)
        
        left_tensor = fetch(task_left)
        return contract_pair(left_tensor, right_tensor, step.left_inds, step.right_inds, step.contract_inds, step.out_inds)
    end
end

function run_tree_parallel_contraction(job_dir::String, threshold::Float64)
    num_initial, steps = read_tree_plan(job_dir)
    initial_tensors = load_tensors_only(job_dir, num_initial)
    
    # Initialize cache with initial tensors
    node_cache = Dict{Int, Array{Float64}}()
    for i in 1:num_initial
        node_cache[i-1] = initial_tensors[i]
    end
    
    # Map parent indices to steps
    parent_to_step = Dict{Int, ContractionStep}()
    for step in steps
        parent_to_step[step.parent] = step
    end
    
    # The root of the tree is the parent of the last step
    root_idx = steps[end].parent
    
    # Warmup
    contract_recursive(root_idx, num_initial, steps, parent_to_step, node_cache, threshold)
    
    # Start timed contraction
    t_start = time_ns()
    result_tensor = contract_recursive(root_idx, num_initial, steps, parent_to_step, node_cache, threshold)
    t_end = time_ns()
    
    dur = (t_end - t_start) / 1e9
    return result_tensor[1], dur
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia tree_parallel_contractor.jl <job_dir> <threshold_flops>")
        exit(1)
    end
    job_dir = ARGS[1]
    threshold = parse(Float64, ARGS[2])
    
    res, dur = run_tree_parallel_contraction(job_dir, threshold)
    println("TREE_NODE_RESULT: $res")
    println("TREE_NODE_TIME: $dur")
end
