using JSON

# Include helper types and parsers from hybrid_contractor
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
    
    # Line 1: nslices
    nslices = parse(Int, lines[1])
    
    # Line 2: sliced indices and sizes
    sliced_inds = Dict{String, Int}()
    sliced_names = Vector{String}()
    if !isempty(strip(lines[2]))
        for part in split(lines[2])
            name, size_str = split(part, ":")
            sliced_inds[name] = parse(Int, size_str)
            push!(sliced_names, String(name))
        end
    end
    
    # Line 3: num_tensors
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
    
    # Line line_idx: num_steps
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
    for (i, info) in enumerate(infos)
        t_path = joinpath(job_dir, "tensors", "$(i-1).bin")
        arr_size = prod(info.shape; init=1)
        data = Vector{Float64}(undef, arr_size)
        read!(t_path, data)
        if isempty(info.shape)
            push!(tensors, fill(data[1]))
        else
            push!(tensors, reshape(data, info.shape...))
        end
    end
    return tensors
end

function apply_slice!(local_tensors::Vector{AbstractArray{Float64}}, local_edges::Vector{Vector{String}}, num_initial::Int, sliced_inds::Dict{String, Int}, sliced_names::Vector{String}, slice_idx::Int)
    reversed_names = reverse(sliced_names)
    val = slice_idx - 1
    
    slice_vals = Dict{String, Int}()
    for name in reversed_names
        sz = sliced_inds[name]
        slice_vals[name] = val % sz
        val = val ÷ sz
    end
    
    for t_idx in 1:num_initial
        edges = local_edges[t_idx]
        t = local_tensors[t_idx]
        for name in reversed_names
            pos = findfirst(x -> x == name, edges)
            if pos !== nothing
                s_val = slice_vals[name]
                idxs = ntuple(i -> i == pos ? (s_val+1 : s_val+1) : (:), ndims(t))
                t = view(t, idxs...)
            end
        end
        local_tensors[t_idx] = t
    end
end

function calculate_step_flops(A::AbstractArray{Float64}, B::AbstractArray{Float64}, inds_A::Vector{String}, inds_B::Vector{String}, inds_contract::Vector{String})
    # Determine axes sizes
    if isempty(inds_contract)
        return 2 * length(A) * length(B)
    end
    
    axes_A_contract = [findfirst(x -> x == idx, inds_A) for idx in inds_contract]
    axes_B_contract = [findfirst(x -> x == idx, inds_B) for idx in inds_contract]
    
    axes_A_free = filter(i -> !(i in axes_A_contract), 1:ndims(A))
    axes_B_free = filter(i -> !(i in axes_B_contract), 1:ndims(B))
    
    shape_A_free = [size(A, i) for i in axes_A_free]
    shape_A_contract = [size(A, i) for i in axes_A_contract]
    shape_B_free = [size(B, i) for i in axes_B_free]
    
    size_A_free = prod(shape_A_free; init=1)
    size_A_contract = prod(shape_A_contract; init=1)
    size_B_free = prod(shape_B_free; init=1)
    
    # GEMM FLOPs = 2 * M * N * K
    return 2 * size_A_free * size_B_free * size_A_contract
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

function profile_sequential(job_dir::String)
    nslices, sliced_inds, sliced_names, tensors_info, steps = read_plan(job_dir)
    tensors = load_tensors(job_dir, tensors_info)
    
    num_initial = length(tensors)
    total_tensors = num_initial + length(steps)
    
    local_tensors = Vector{AbstractArray{Float64}}(undef, total_tensors)
    local_edges = Vector{Vector{String}}(undef, total_tensors)
    
    for idx in 1:num_initial
        local_tensors[idx] = tensors[idx]
        local_edges[idx] = copy(tensors_info[idx].edges)
    end
    
    if nslices > 1 && !isempty(sliced_names)
        apply_slice!(local_tensors, local_edges, num_initial, sliced_inds, sliced_names, 1)
    end
    
    step_metrics = []
    
    # 1st warm up run to compile all functions
    for step in steps
        parent_idx = step.parent + 1
        left_idx = step.left + 1
        right_idx = step.right + 1
        local_tensors[parent_idx] = tensordot(local_tensors[left_idx], local_tensors[right_idx], local_edges[left_idx], local_edges[right_idx], step.legs)
        left_free = filter(x -> !(x in step.legs), local_edges[left_idx])
        right_free = filter(x -> !(x in step.legs), local_edges[right_idx])
        local_edges[parent_idx] = vcat(left_free, right_free)
    end
    
    # Reset local variables
    for idx in 1:num_initial
        local_tensors[idx] = tensors[idx]
        local_edges[idx] = copy(tensors_info[idx].edges)
    end
    if nslices > 1 && !isempty(sliced_names)
        apply_slice!(local_tensors, local_edges, num_initial, sliced_inds, sliced_names, 1)
    end
    
    # 2nd run: profiled
    for (s_idx, step) in enumerate(steps)
        parent_idx = step.parent + 1
        left_idx = step.left + 1
        right_idx = step.right + 1
        
        A = local_tensors[left_idx]
        B = local_tensors[right_idx]
        inds_A = local_edges[left_idx]
        inds_B = local_edges[right_idx]
        
        flops = calculate_step_flops(A, B, inds_A, inds_B, step.legs)
        
        t0 = time_ns()
        C = tensordot(A, B, inds_A, inds_B, step.legs)
        t1 = time_ns()
        
        local_tensors[parent_idx] = C
        
        left_free = filter(x -> !(x in step.legs), inds_A)
        right_free = filter(x -> !(x in step.legs), inds_B)
        local_edges[parent_idx] = vcat(left_free, right_free)
        
        duration = (t1 - t0) / 1e9
        push!(step_metrics, Dict(
            "step_index" => s_idx,
            "flops" => flops,
            "time" => duration,
            "shape_A" => size(A),
            "shape_B" => size(B),
            "shape_C" => size(C)
        ))
    end
    
    return step_metrics
end

function profile_sliced_parallel(job_dir::String)
    nslices, sliced_inds, sliced_names, tensors_info, steps = read_plan(job_dir)
    tensors = load_tensors(job_dir, tensors_info)
    
    num_initial = length(tensors)
    total_tensors = num_initial + length(steps)
    num_steps = length(steps)
    
    # Warm-up pass to JIT compile the steps
    warmup_tensors = Vector{AbstractArray{Float64}}(undef, total_tensors)
    warmup_edges = Vector{Vector{String}}(undef, total_tensors)
    for idx in 1:num_initial
        warmup_tensors[idx] = tensors[idx]
        warmup_edges[idx] = copy(tensors_info[idx].edges)
    end
    if nslices > 0 && !isempty(sliced_names)
        apply_slice!(warmup_tensors, warmup_edges, num_initial, sliced_inds, sliced_names, 1)
    end
    for step in steps
        parent_idx = step.parent + 1
        left_idx = step.left + 1
        right_idx = step.right + 1
        warmup_tensors[parent_idx] = tensordot(warmup_tensors[left_idx], warmup_tensors[right_idx], warmup_edges[left_idx], warmup_edges[right_idx], step.legs)
        left_free = filter(x -> !(x in step.legs), warmup_edges[left_idx])
        right_free = filter(x -> !(x in step.legs), warmup_edges[right_idx])
        warmup_edges[parent_idx] = vcat(left_free, right_free)
    end
    
    # Step metrics collection arrays
    step_flops = zeros(Int64, num_steps)
    
    # Compute step FLOPs based on slice 1 (identical across all slices)
    for (s_idx, step) in enumerate(steps)
        left_idx = step.left + 1
        right_idx = step.right + 1
        step_flops[s_idx] = calculate_step_flops(
            warmup_tensors[left_idx], warmup_tensors[right_idx], 
            warmup_edges[left_idx], warmup_edges[right_idx], step.legs
        )
    end
    
    # Allocate thread-local matrix for timings to avoid cache line contention
    # Step timings collection via atomic accumulators to avoid threading race conditions
    step_times = [Threads.Atomic{Float64}(0.0) for _ in 1:num_steps]
    
    # Run the parallel loop
    Threads.@threads for slice_idx in 1:nslices
        local_tensors = Vector{AbstractArray{Float64}}(undef, total_tensors)
        local_edges = Vector{Vector{String}}(undef, total_tensors)
        
        for idx in 1:num_initial
            local_tensors[idx] = tensors[idx]
            local_edges[idx] = copy(tensors_info[idx].edges)
        end
        
        if !isempty(sliced_names)
            apply_slice!(local_tensors, local_edges, num_initial, sliced_inds, sliced_names, slice_idx)
        end
        
        for (s_idx, step) in enumerate(steps)
            parent_idx = step.parent + 1
            left_idx = step.left + 1
            right_idx = step.right + 1
            
            A = local_tensors[left_idx]
            B = local_tensors[right_idx]
            inds_A = local_edges[left_idx]
            inds_B = local_edges[right_idx]
            
            t0 = time_ns()
            C = tensordot(A, B, inds_A, inds_B, step.legs)
            t1 = time_ns()
            
            Threads.atomic_add!(step_times[s_idx], (t1 - t0) / 1e9)
            
            local_tensors[parent_idx] = C
            
            left_free = filter(x -> !(x in step.legs), inds_A)
            right_free = filter(x -> !(x in step.legs), inds_B)
            local_edges[parent_idx] = vcat(left_free, right_free)
        end
    end
    
    step_metrics = []
    for s_idx in 1:num_steps
        t_val = step_times[s_idx][]
        push!(step_metrics, Dict(
            "step_index" => s_idx,
            "flops" => step_flops[s_idx] * nslices,
            "time" => t_val,
            "flops_per_slice" => step_flops[s_idx],
            "time_per_slice" => t_val / nslices
        ))
    end
    
    return step_metrics
end

if length(ARGS) < 2
    println("Usage: julia step_profiler.jl <job_dir> <mode: sequential|sliced>")
    exit(1)
end

job_dir = ARGS[1]
mode = ARGS[2]

if mode == "sequential"
    metrics = profile_sequential(job_dir)
elseif mode == "sliced"
    metrics = profile_sliced_parallel(job_dir)
else
    println("Invalid mode: $mode")
    exit(1)
end

# Output the metrics as JSON to stdout for easy python consumption
println(JSON.json(metrics))
