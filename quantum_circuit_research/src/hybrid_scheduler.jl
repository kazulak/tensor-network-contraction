using JSON
using Base.Threads

# Plan and step helper structures
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

function get_slice_array(t::Array{Float64}, edges::Vector{String}, sliced_inds::Dict{String, Int}, sliced_names::Vector{String}, slice_idx::Int)
    reversed_names = reverse(sliced_names)
    val = slice_idx - 1
    
    slice_vals = Dict{String, Int}()
    for name in reversed_names
        sz = sliced_inds[name]
        slice_vals[name] = val % sz
        val = val ÷ sz
    end
    
    idxs = Any[ Colon() for _ in 1:ndims(t) ]
    for name in reversed_names
        pos = findfirst(x -> x == name, edges)
        if pos !== nothing
            s_val = slice_vals[name]
            idxs[pos] = (s_val + 1):(s_val + 1)
        end
    end
    
    return Array(view(t, idxs...))
end

function tensordot(A::AbstractArray{Float64}, B::AbstractArray{Float64}, inds_A::Vector{String}, inds_B::Vector{String}, inds_contract::Vector{String})
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

# Thread-safe, type-stable function executing the sequence of steps for a single slice.
function execute_slice_steps(s::Int, nslices::Int, total_tensors::Int, num_initial::Int, steps::Vector{ContractionStep},
                             is_node_sliced::Vector{Bool}, is_step_sliced::Vector{Bool},
                             sliced_tensors::Vector{Union{Nothing, Vector{Array{Float64}}}},
                             unsliced_tensors::Vector{Union{Nothing, Array{Float64}}},
                             node_edges::Vector{Vector{String}})
    
    thread_tensors = Vector{Array{Float64}}(undef, total_tensors)
    
    # 1. Populate initial tensors
    for i in 1:num_initial
        if is_node_sliced[i]
            thread_tensors[i] = sliced_tensors[i][s]
        else
            thread_tensors[i] = unsliced_tensors[i]
        end
    end
    
    # 2. Copy unsliced pre-computations
    for (s_idx, step) in enumerate(steps)
        if !is_step_sliced[s_idx]
            parent_idx = step.parent + 1
            thread_tensors[parent_idx] = unsliced_tensors[parent_idx]
        end
    end
    
    # 3. Execute sliced steps sequentially
    for (s_idx, step) in enumerate(steps)
        if is_step_sliced[s_idx]
            parent_idx = step.parent + 1
            left_idx = step.left + 1
            right_idx = step.right + 1
            
            A_s = thread_tensors[left_idx]
            B_s = thread_tensors[right_idx]
            
            thread_tensors[parent_idx] = tensordot(A_s, B_s, node_edges[left_idx], node_edges[right_idx], step.legs)
        end
    end
    
    root_idx = steps[end].parent + 1
    return thread_tensors[root_idx][1]
end

function contract_hybrid(job_dir::String)
    nslices, sliced_inds, sliced_names, tensors_info, steps = read_plan(job_dir)
    tensors = load_tensors(job_dir, tensors_info)
    
    num_initial = length(tensors)
    total_tensors = num_initial + length(steps)
    num_steps = length(steps)
    
    # 1. Track which nodes and steps are sliced
    is_node_sliced = zeros(Bool, total_tensors)
    for i in 1:num_initial
        is_node_sliced[i] = any(e -> e in sliced_names, tensors_info[i].edges)
    end
    
    # Setup node edges
    node_edges = Vector{Vector{String}}(undef, total_tensors)
    for i in 1:num_initial
        node_edges[i] = copy(tensors_info[i].edges)
    end
    
    # Pre-scan steps to determine sliced vs unsliced nodes and steps
    is_step_sliced = zeros(Bool, num_steps)
    for (s_idx, step) in enumerate(steps)
        parent_idx = step.parent + 1
        left_idx = step.left + 1
        right_idx = step.right + 1
        
        is_node_sliced[parent_idx] = is_node_sliced[left_idx] || is_node_sliced[right_idx]
        is_step_sliced[s_idx] = is_node_sliced[parent_idx]
        
        left_free = filter(x -> !(x in step.legs), node_edges[left_idx])
        right_free = filter(x -> !(x in step.legs), node_edges[right_idx])
        node_edges[parent_idx] = vcat(left_free, right_free)
    end
    
    # Reset node edges for actual contraction execution
    for i in 1:num_initial
        node_edges[i] = copy(tensors_info[i].edges)
    end
    for (s_idx, step) in enumerate(steps)
        parent_idx = step.parent + 1
        left_idx = step.left + 1
        right_idx = step.right + 1
        left_free = filter(x -> !(x in step.legs), node_edges[left_idx])
        right_free = filter(x -> !(x in step.legs), node_edges[right_idx])
        node_edges[parent_idx] = vcat(left_free, right_free)
    end
    
    # 2. Allocate data stores
    unsliced_tensors = Vector{Union{Nothing, Array{Float64}}}(nothing, total_tensors)
    sliced_tensors = Vector{Union{Nothing, Vector{Array{Float64}}}}(nothing, total_tensors)
    
    # 3. Populate initial tensors
    for i in 1:num_initial
        if is_node_sliced[i] && nslices > 1
            sliced_tensors[i] = [get_slice_array(tensors[i], node_edges[i], sliced_inds, sliced_names, s) for s in 1:nslices]
        else
            unsliced_tensors[i] = tensors[i]
        end
    end
    
    # --- PHASE 1: Unsliced Pre-computation (Sequential, executed exactly once!) ---
    for (s_idx, step) in enumerate(steps)
        if !is_step_sliced[s_idx]
            parent_idx = step.parent + 1
            left_idx = step.left + 1
            right_idx = step.right + 1
            A = unsliced_tensors[left_idx]
            B = unsliced_tensors[right_idx]
            unsliced_tensors[parent_idx] = tensordot(A, B, node_edges[left_idx], node_edges[right_idx], step.legs)
        end
    end
    
    # --- PHASE 2: Sliced Contraction (Coarse-Grained Parallel loop over slices) ---
    root_slices = zeros(Float64, nslices)
    
    if nslices > 1
        Threads.@threads for s in 1:nslices
            root_slices[s] = execute_slice_steps(s, nslices, total_tensors, num_initial, steps,
                                                 is_node_sliced, is_step_sliced,
                                                 sliced_tensors, unsliced_tensors, node_edges)
        end
    else
        root_slices[1] = execute_slice_steps(1, nslices, total_tensors, num_initial, steps,
                                             is_node_sliced, is_step_sliced,
                                             sliced_tensors, unsliced_tensors, node_edges)
    end
    
    # Sum over slices
    root_idx = steps[end].parent + 1
    if is_node_sliced[root_idx]
        return sum(root_slices)
    else
        return root_slices[1]
    end
end

if length(ARGS) < 1
    println("Usage: julia hybrid_scheduler.jl <job_dir>")
    exit(1)
end

job_dir = ARGS[1]

# 1. Warm-up run to compile all methods and paths
val_warm = contract_hybrid(job_dir)

# 2. Main profiled run
t0 = time_ns()
val = contract_hybrid(job_dir)
t1 = time_ns()

elapsed = (t1 - t0) / 1e9

# Output JSON results
results = Dict(
    "elapsed" => elapsed,
    "result" => val
)
println(JSON.json(results))
