using Sockets
using LinearAlgebra

struct ContractionStep
    parent::Int
    left::Int
    right::Int
    left_inds::Vector{String}
    right_inds::Vector{String}
    contract_inds::Vector{String}
    out_inds::Vector{String}
end

# Representation of a tensor which might be unsliced (single array) or actively sliced (vector of arrays)
struct ActiveTensor
    is_sliced::Bool
    data::Union{Array{Float64}, Vector{Array{Float64}}}
end

function read_active_plan(job_dir::String)
    plan_path = joinpath(job_dir, "plan.txt")
    lines = readlines(plan_path)
    
    line_idx = 1
    nslices = parse(Int, lines[line_idx])
    line_idx += 2
    
    num_tensors = parse(Int, lines[line_idx])
    line_idx += 1
    
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
        
        inds_left = node_indices[left]
        inds_right = node_indices[right]
        
        inds_parent = filter(x -> !(x in contract_inds), inds_left)
        append!(inds_parent, filter(x -> !(x in contract_inds), inds_right))
        node_indices[parent] = inds_parent
        
        push!(steps, ContractionStep(parent, left, right, inds_left, inds_right, contract_inds, inds_parent))
        line_idx += 1
    end
    
    return num_tensors, steps
end

# Core tensor contraction
function contract_dense(A::Array{Float64}, B::Array{Float64}, left_inds::Vector{String}, right_inds::Vector{String}, contract_inds::Vector{String}, out_inds::Vector{String})
    if isempty(contract_inds)
        return reshape(reshape(A, :) * reshape(B, :)', vcat(size(A)..., size(B)...)...)
    end
    
    c_pos_A = [findfirst(x -> x == i, left_inds) for i in contract_inds]
    c_pos_B = [findfirst(x -> x == i, right_inds) for i in contract_inds]
    
    free_pos_A = filter(x -> !(x in c_pos_A), 1:ndims(A))
    free_pos_B = filter(x -> !(x in c_pos_B), 1:ndims(B))
    
    perm_A = vcat(free_pos_A, c_pos_A)
    perm_B = vcat(c_pos_B, free_pos_B)
    
    A_perm = permutedims(A, perm_A)
    B_perm = permutedims(B, perm_B)
    
    size_free_A = [size(A, i) for i in free_pos_A]
    size_c = [size(A, i) for i in c_pos_A]
    size_free_B = [size(B, i) for i in free_pos_B]
    
    mat_A = reshape(A_perm, prod(size_free_A; init=1), prod(size_c; init=1))
    mat_B = reshape(B_perm, prod(size_c; init=1), prod(size_free_B; init=1))
    
    mat_C = mat_A * mat_B
    C_raw = reshape(mat_C, vcat(size_free_A, size_free_B)...)
    
    raw_out_inds = vcat([left_inds[i] for i in free_pos_A], [right_inds[i] for i in free_pos_B])
    perm_C = [findfirst(x -> x == i, raw_out_inds) for i in out_inds]
    
    if isempty(perm_C)
        return C_raw
    else
        return permutedims(C_raw, perm_C)
    end
end

# Load initial tensors
function load_tensors_only(job_dir::String, num_initial::Int)
    tensors = Vector{Array{Float64}}(undef, num_initial)
    plan_lines = readlines(joinpath(job_dir, "plan.txt"))
    line_idx = 4
    for i in 1:num_initial
        parts = split(plan_lines[line_idx], "|")
        shape = isempty(parts[1]) ? Int[] : [parse(Int, x) for x in split(parts[1], ",")]
        t_path = joinpath(job_dir, "tensors", "$(i-1).bin")
        arr_size = prod(shape; init=1)
        data = Vector{Float64}(undef, arr_size)
        read!(t_path, data)
        tensors[i] = isempty(shape) ? fill(data[1]) : reshape(data, shape...)
        line_idx += 1
    end
    return tensors
end

# Projects a tensor along index 'e' at coordinate 'idx'
function project_tensor(A::Array{Float64}, inds::Vector{String}, e::String, idx::Int)
    dim_pos = findfirst(x -> x == e, inds)
    if dim_pos == nothing
        return A
    end
    # Select slice along dim_pos
    colons = Any[Colon() for _ in 1:ndims(A)]
    colons[dim_pos] = idx:idx
    # Return view (zero-copy) or copy. To keep it simple, we do view
    val = view(A, colons...)
    # We want to drop the sliced dimension of size 1 if needed, or keep it.
    # To keep shapes aligned with out_inds, we preserve size 1 as in cotengra
    return Array(val)
end

# Recursive contractor with active slicing
function contract_active(node_idx::Int, num_initial::Int, steps::Vector{ContractionStep}, parent_to_step::Dict{Int, ContractionStep}, initial_tensors::Vector{Array{Float64}}, sliced_edge::String, d_slice::Int)
    # Leaf node
    if node_idx < num_initial
        t = initial_tensors[node_idx+1]
        t_inds = parent_to_step[steps[1].parent].left_inds # fallback to find original inds
        # For simplicity, look up original indices of this leaf tensor
        # We can scan the steps to find where this leaf is left or right child
        inds = String[]
        for s in steps
            if s.left == node_idx
                inds = s.left_inds
                break
            elseif s.right == node_idx
                inds = s.right_inds
                break
            end
        end
        
        if sliced_edge in inds
            # Active Slicing Boundary: slice the leaf tensor
            slices = Vector{Array{Float64}}(undef, d_slice)
            for i in 1:d_slice
                slices[i] = project_tensor(t, inds, sliced_edge, i)
            end
            return ActiveTensor(true, slices)
        else
            return ActiveTensor(false, t)
        end
    end
    
    step = parent_to_step[node_idx]
    
    # Recursively contract children
    # We can spawn them in parallel if they are unsliced
    # For this advanced contractor, we run children concurrently
    task_left = Threads.@spawn contract_active(step.left, num_initial, steps, parent_to_step, initial_tensors, sliced_edge, d_slice)
    B = contract_active(step.right, num_initial, steps, parent_to_step, initial_tensors, sliced_edge, d_slice)
    A = fetch(task_left)
    
    # Determine output shapes and indices
    # We contract A and B
    if !A.is_sliced && !B.is_sliced
        # Case 1: Both unsliced
        res = contract_dense(A.data, B.data, step.left_inds, step.right_inds, step.contract_inds, step.out_inds)
        return ActiveTensor(false, res)
        
    elseif A.is_sliced && !B.is_sliced
        # Case 2: A is sliced, B is unsliced
        # Contract each slice of A with B in parallel!
        slices = Vector{Array{Float64}}(undef, d_slice)
        Threads.@threads for i in 1:d_slice
            slices[i] = contract_dense(A.data[i], B.data, step.left_inds, step.right_inds, step.contract_inds, step.out_inds)
        end
        return ActiveTensor(true, slices)
        
    elseif !A.is_sliced && B.is_sliced
        # Case 3: A is unsliced, B is sliced
        slices = Vector{Array{Float64}}(undef, d_slice)
        Threads.@threads for i in 1:d_slice
            slices[i] = contract_dense(A.data, B.data[i], step.left_inds, step.right_inds, step.contract_inds, step.out_inds)
        end
        return ActiveTensor(true, slices)
        
    else
        # Case 4: Both are sliced
        if sliced_edge in step.contract_inds
            # Case 4a: Sliced edge is being contracted out (The Root Summation!)
            # C = sum(A_i * B_i)
            # Compute terms in parallel
            terms = Vector{Array{Float64}}(undef, d_slice)
            Threads.@threads for i in 1:d_slice
                terms[i] = contract_dense(A.data[i], B.data[i], step.left_inds, step.right_inds, step.contract_inds, step.out_inds)
            end
            # Sum up terms
            res = sum(terms)
            return ActiveTensor(false, res)
        else
            # Case 4b: Sliced edge is a free leg of both, remains in output
            slices = Vector{Array{Float64}}(undef, d_slice)
            Threads.@threads for i in 1:d_slice
                slices[i] = contract_dense(A.data[i], B.data[i], step.left_inds, step.right_inds, step.contract_inds, step.out_inds)
            end
            return ActiveTensor(true, slices)
        end
    end
end

function run_active_slicing(job_dir::String)
    num_initial, steps = read_active_plan(job_dir)
    initial_tensors = load_tensors_only(job_dir, num_initial)
    
    parent_to_step = Dict{Int, ContractionStep}()
    for step in steps
        parent_to_step[step.parent] = step
    end
    
    # Identify the sliced edge from the root contraction (the very last step)
    root_step = steps[end]
    if isempty(root_step.contract_inds)
        println("Error: Root contraction does not contract any indices (disconnected network).")
        exit(1)
    end
    
    # Choose the first contracted index at the root as the active sliced index
    sliced_edge = root_step.contract_inds[1]
    
    # Find its dimension by looking up the leaf tensor that contains it
    d_slice = 0
    for s in steps
        # Look in left child indices
        pos = findfirst(x -> x == sliced_edge, s.left_inds)
        if pos != nothing && s.left < num_initial
            d_slice = size(initial_tensors[s.left+1], pos)
            break
        end
        # Look in right child indices
        pos = findfirst(x -> x == sliced_edge, s.right_inds)
        if pos != nothing && s.right < num_initial
            d_slice = size(initial_tensors[s.right+1], pos)
            break
        end
    end
    
    if d_slice == 0
        d_slice = 4 # fallback dimension
    end
    
    println("Active Slicing edge identified: '$sliced_edge' (dimension: $d_slice)")
    
    root_idx = root_step.parent
    
    # Warmup
    contract_active(root_idx, num_initial, steps, parent_to_step, initial_tensors, sliced_edge, d_slice)
    
    # Start timed contraction
    t_start = time_ns()
    res_tensor = contract_active(root_idx, num_initial, steps, parent_to_step, initial_tensors, sliced_edge, d_slice)
    t_end = time_ns()
    
    dur = (t_end - t_start) / 1e9
    
    val = res_tensor.is_sliced ? res_tensor.data[1][1] : res_tensor.data[1]
    return val, dur
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia active_slicing_contractor.jl <job_dir>")
        exit(1)
    end
    job_dir = ARGS[1]
    res, dur = run_active_slicing(job_dir)
    println("ACTIVE_SLICING_RESULT: $res")
    println("ACTIVE_SLICING_TIME: $dur")
end
