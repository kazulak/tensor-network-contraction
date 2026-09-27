using JSON
using LinearAlgebra
using Base.Threads

struct TensorInfo
    shape::Vector{Int}
    edges::Vector{String}
end

struct ContractionStep
    parent::Int
    left::Int
    right::Int
    legs::Vector{String}
end

function read_plan(job_dir::String)
    plan_path = joinpath(job_dir, "plan.txt")
    lines = readlines(plan_path)
    
    nslices = parse(Int, lines[1])
    
    sliced_inds = Dict{String, Int}()
    sliced_names = Vector{String}()
    if !isempty(strip(lines[2]))
        for part in split(lines[2])
            name, size_str = split(part, ":")
            sliced_inds[name] = parse(Int, size_str)
            push!(sliced_names, String(name))
        end
    end
    
    num_tensors = parse(Int, lines[3])
    tensors_info = Vector{TensorInfo}()
    line_idx = 4
    for i in 1:num_tensors
        parts = split(lines[line_idx], "|")
        shape = isempty(parts[1]) ? Int[] : [parse(Int, s) for s in split(parts[1], ",")]
        edges = isempty(parts[2]) ? String[] : [String(s) for s in split(parts[2], ",")]
        push!(tensors_info, TensorInfo(shape, edges))
        line_idx += 1
    end
    
    num_steps = parse(Int, lines[line_idx])
    line_idx += 1
    steps = Vector{ContractionStep}()
    for i in 1:num_steps
        parts = split(lines[line_idx])
        parent = parse(Int, parts[1])
        left = parse(Int, parts[2])
        right = parse(Int, parts[3])
        legs = parts[4] == "NONE" ? String[] : [String(s) for s in split(parts[4], ",")]
        push!(steps, ContractionStep(parent, left, right, legs))
        line_idx += 1
    end
    
    return nslices, sliced_inds, sliced_names, tensors_info, steps
end

function load_tensors(job_dir::String, infos::Vector{TensorInfo})
    tensors = Vector{Array{Float64}}()
    t_path = joinpath(job_dir, "tensors.bin")
    open(t_path, "r") do io
        for info in infos
            arr_size = prod(info.shape; init=1)
            data = Vector{Float64}(undef, arr_size)
            read!(io, data)
            if isempty(info.shape)
                push!(tensors, fill(data[1]))
            else
                push!(tensors, reshape(data, info.shape...))
            end
        end
    end
    return tensors
end

function tensordot_single(A::AbstractArray{Float64}, B::AbstractArray{Float64}, inds_A::Vector{String}, inds_B::Vector{String}, inds_contract::Vector{String})
    if isempty(inds_contract)
        A_flat = reshape(A, :)
        B_flat = reshape(B, :)
        C_flat = A_flat * B_flat'
        shape_C = vcat(size(A)..., size(B)...)
        return isempty(shape_C) ? fill(C_flat[1]) : reshape(C_flat, shape_C...)
    end

    axes_A_contract = [findfirst(x -> x == idx, inds_A) for idx in inds_contract]
    axes_B_contract = [findfirst(x -> x == idx, inds_B) for idx in inds_contract]
    
    axes_A_free = filter(i -> !(i in axes_A_contract), 1:ndims(A))
    axes_B_free = filter(i -> !(i in axes_B_contract), 1:ndims(B))
    
    perm_A = vcat(axes_A_free, axes_A_contract)
    perm_B = vcat(axes_B_contract, axes_B_free)
    
    A_perm = permutedims(A, perm_A)
    B_perm = permutedims(B, perm_B)
    
    shape_A_free = [size(A, i) for i in axes_A_free]
    shape_A_contract = [size(A, i) for i in axes_A_contract]
    shape_B_contract = [size(B, i) for i in axes_B_contract]
    shape_B_free = [size(B, i) for i in axes_B_free]
    
    size_A_free = prod(shape_A_free; init=1)
    size_A_contract = prod(shape_A_contract; init=1)
    size_B_contract = prod(shape_B_contract; init=1)
    size_B_free = prod(shape_B_free; init=1)
    
    A_mat = reshape(A_perm, size_A_free, size_A_contract)
    B_mat = reshape(B_perm, size_B_contract, size_B_free)
    
    C_mat = A_mat * B_mat
    
    shape_C = vcat(shape_A_free, shape_B_free)
    if isempty(shape_C)
        return fill(C_mat[1])
    else
        return reshape(C_mat, shape_C...)
    end
end

# Recursive contract with task-level tree parallelism
function contract_subtree(node_id::Int, num_initial::Int, parent_to_step::Dict{Int, ContractionStep},
                          node_edges::Vector{Vector{String}}, initial_tensors::Vector{Array{Float64}},
                          cutoff_flops::Float64)
    # If leaf node, return initial tensor
    if node_id <= num_initial
        return initial_tensors[node_id]
    end
    
    step = parent_to_step[node_id]
    left_id = step.left + 1
    right_id = step.right + 1
    
    # Check if this subtree has enough work to justify spawning a task
    if cutoff_flops > 0 && (left_id > num_initial || right_id > num_initial)
        # Fork: compute left on spawned task, right on current thread
        task_L = Threads.@spawn contract_subtree(left_id, num_initial, parent_to_step, node_edges, initial_tensors, cutoff_flops)
        res_R = contract_subtree(right_id, num_initial, parent_to_step, node_edges, initial_tensors, cutoff_flops)
        res_L = fetch(task_L)
        return tensordot_single(res_L, res_R, node_edges[left_id], node_edges[right_id], step.legs)
    else
        # Sequential subtree execution
        res_L = contract_subtree(left_id, num_initial, parent_to_step, node_edges, initial_tensors, cutoff_flops)
        res_R = contract_subtree(right_id, num_initial, parent_to_step, node_edges, initial_tensors, cutoff_flops)
        return tensordot_single(res_L, res_R, node_edges[left_id], node_edges[right_id], step.legs)
    end
end

function contract_tree_parallel(job_dir::String)
    # Ensure individual GEMMs are 1 thread to avoid oversubscription
    BLAS.set_num_threads(1)
    
    nslices, sliced_inds, sliced_names, tensors_info, steps = read_plan(job_dir)
    initial_tensors = load_tensors(job_dir, tensors_info)
    
    num_initial = length(initial_tensors)
    total_tensors = num_initial + length(steps)
    
    # Precompute edge propagation for each node in the tree
    node_edges = Vector{Vector{String}}(undef, total_tensors)
    for idx in 1:num_initial
        node_edges[idx] = copy(tensors_info[idx].edges)
    end
    
    parent_to_step = Dict{Int, ContractionStep}()
    for step in steps
        parent_id = step.parent + 1
        left_id = step.left + 1
        right_id = step.right + 1
        parent_to_step[parent_id] = step
        
        left_free = filter(x -> !(x in step.legs), node_edges[left_id])
        right_free = filter(x -> !(x in step.legs), node_edges[right_id])
        node_edges[parent_id] = vcat(left_free, right_free)
    end
    
    root_id = steps[end].parent + 1
    cutoff_flops = 500.0 # Min work to fork task
    
    result_tensor = contract_subtree(root_id, num_initial, parent_to_step, node_edges, initial_tensors, cutoff_flops)
    return result_tensor[1]
end

if length(ARGS) < 1
    println("Usage: julia tree_level_contractor.jl <job_dir>")
    exit(1)
end

job_dir = ARGS[1]

# Warmup pass
val_warm = contract_tree_parallel(job_dir)

# Profiled pass
t0 = time_ns()
val = contract_tree_parallel(job_dir)
t1 = time_ns()

elapsed = (t1 - t0) / 1e9

results = Dict(
    "elapsed" => elapsed,
    "result" => val,
    "threads" => Threads.nthreads(),
    "mode" => "tree_level"
)
println(JSON.json(results))
