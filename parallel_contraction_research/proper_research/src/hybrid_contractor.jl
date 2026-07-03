using LinearAlgebra

# Force single-threaded BLAS per Julia thread to avoid core thrashing
BLAS.set_num_threads(1)

struct ContractionStep
    parent::Int
    left::Int
    right::Int
    legs::Vector{String}
end

struct TensorInfo
    shape::Vector{Int}
    edges::Vector{String}
end

function read_plan(job_dir::String)
    plan_path = joinpath(job_dir, "plan.txt")
    lines = readlines(plan_path)
    
    # Line 1: nslices
    nslices = parse(Int, lines[1])
    
    # Line 2: sliced indices and sizes (preserve order)
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
        # Compute total size of array
        arr_size = prod(info.shape; init=1)
        data = Vector{Float64}(undef, arr_size)
        read!(t_path, data)
        # Reshape to original shape
        if isempty(info.shape)
            push!(tensors, fill(data[1]))
        else
            push!(tensors, reshape(data, info.shape...))
        end
    end
    return tensors
end

function tensordot(A::AbstractArray{Float64}, B::AbstractArray{Float64}, inds_A::Vector{String}, inds_B::Vector{String}, inds_contract::Vector{String})
    # If there are no contraction legs, it is an outer product
    if isempty(inds_contract)
        # Outer product
        A_flat = reshape(A, :)
        B_flat = reshape(B, :)
        C_flat = A_flat * B_flat'
        shape_C = vcat(size(A)..., size(B)...)
        return isempty(shape_C) ? fill(C_flat[1]) : reshape(C_flat, shape_C...)
    end

    # 1. Determine axes to contract
    axes_A_contract = [findfirst(x -> x == idx, inds_A) for idx in inds_contract]
    axes_B_contract = [findfirst(x -> x == idx, inds_B) for idx in inds_contract]
    
    # 2. Determine uncontracted axes
    axes_A_free = filter(i -> !(i in axes_A_contract), 1:ndims(A))
    axes_B_free = filter(i -> !(i in axes_B_contract), 1:ndims(B))
    
    # 3. Permute A and B
    perm_A = vcat(axes_A_free, axes_A_contract)
    perm_B = vcat(axes_B_contract, axes_B_free)
    
    A_perm = permutedims(A, perm_A)
    B_perm = permutedims(B, perm_B)
    
    # 4. Reshape to 2D matrices
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
    
    # 5. Matrix multiplication (BLAS GEMM)
    C_mat = A_mat * B_mat
    
    # 6. Reshape to output shape
    shape_C = vcat(shape_A_free, shape_B_free)
    if isempty(shape_C)
        return fill(C_mat[1])
    else
        return reshape(C_mat, shape_C...)
    end
end

function apply_slice!(local_tensors::Vector{AbstractArray{Float64}}, local_edges::Vector{Vector{String}}, num_initial::Int, sliced_inds::Dict{String, Int}, sliced_names::Vector{String}, slice_idx::Int)
    # Mixed-radix coordinate conversion (last index changes fastest, so reverse order)
    reversed_names = reverse(sliced_names)
    val = slice_idx - 1
    
    slice_vals = Dict{String, Int}()
    for name in reversed_names
        sz = sliced_inds[name]
        slice_vals[name] = val % sz
        val = val ÷ sz
    end
    
    # Apply to each initial tensor
    for t_idx in 1:num_initial
        edges = local_edges[t_idx]
        t = local_tensors[t_idx]
        
        # Check if tensor contains any sliced indices
        for name in reversed_names
            pos = findfirst(x -> x == name, edges)
            if pos !== nothing
                s_val = slice_vals[name]
                # Slice along dimension pos preserving size 1 using a view with a unit range
                idxs = ntuple(i -> i == pos ? (s_val+1 : s_val+1) : (:), ndims(t))
                t = view(t, idxs...)
            end
        end
        local_tensors[t_idx] = t
    end
end

function contract_sliced_hybrid(job_dir::String)
    println("--- Julia Hybrid Contractor ---")
    start_total = time_ns()
    
    # 1. Read plan and load tensors
    nslices, sliced_inds, sliced_names, tensors_info, steps = read_plan(job_dir)
    println("Loaded plan. Slices: $nslices, Tensors: $(length(tensors_info)), Steps: $(length(steps))")
    
    tensors = load_tensors(job_dir, tensors_info)
    
    num_initial = length(tensors)
    total_tensors = num_initial + length(steps)
    
    # 2. Parallel contraction loop (GIL-free multithreading)
    slice_results = Vector{Float64}(undef, nslices)
    println("Starting parallel contraction over $nslices slices with $(Threads.nthreads()) Julia threads...")
    
    # 2a. JIT Warm-up Pass (run slice 1 sequentially)
    if nslices > 0
        warmup_tensors = Vector{AbstractArray{Float64}}(undef, total_tensors)
        warmup_edges = Vector{Vector{String}}(undef, total_tensors)
        for idx in 1:num_initial
            warmup_tensors[idx] = tensors[idx]
            warmup_edges[idx] = copy(tensors_info[idx].edges)
        end
        if !isempty(sliced_inds)
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
    end
    
    start_contraction = time_ns()
    
    # Check if multithreading is enabled
    # Threads.@threads performs loop partitioning across worker threads
    Threads.@threads for slice_idx in 1:nslices
        # Preallocate thread-local arrays with enough slots for intermediate nodes
        local_tensors = Vector{AbstractArray{Float64}}(undef, total_tensors)
        local_edges = Vector{Vector{String}}(undef, total_tensors)
        
        # Copy initial tensors and edges
        for idx in 1:num_initial
            local_tensors[idx] = tensors[idx]
            local_edges[idx] = copy(tensors_info[idx].edges)
        end
        
        # Apply slice-specific constraints (setting sliced indices to fixed values)
        if !isempty(sliced_inds)
            apply_slice!(local_tensors, local_edges, num_initial, sliced_inds, sliced_names, slice_idx)
        end
        
        # Perform pairwise contractions according to the tree steps
        for step in steps
            # Julia is 1-indexed, so add 1 to the python 0-indexed IDs
            parent_idx = step.parent + 1
            left_idx = step.left + 1
            right_idx = step.right + 1
            
            local_tensors[parent_idx] = tensordot(
                local_tensors[left_idx], 
                local_tensors[right_idx], 
                local_edges[left_idx], 
                local_edges[right_idx], 
                step.legs
            )
            
            # Update edge list of the parent node (uncontracted left followed by uncontracted right)
            left_free = filter(x -> !(x in step.legs), local_edges[left_idx])
            right_free = filter(x -> !(x in step.legs), local_edges[right_idx])
            local_edges[parent_idx] = vcat(left_free, right_free)
        end
        
        # The last parent node (which is steps[end].parent + 1) contains the final scalar view
        final_parent_idx = steps[end].parent + 1
        slice_results[slice_idx] = local_tensors[final_parent_idx][1]
    end
    
    # 3. Sum up all slice outputs
    final_scalar = sum(slice_results)
    
    end_time = time_ns()
    contraction_seconds = (end_time - start_contraction) / 1e9
    total_seconds = (end_time - start_total) / 1e9
    
    println("Contraction completed successfully!")
    println("Result: $final_scalar")
    println("Contraction Time: $(round(contraction_seconds, digits=6)) seconds")
    println("Total Hybrid Time (IO + Contraction): $(round(total_seconds, digits=6)) seconds")
    
    # Write output to result.txt
    write(joinpath(job_dir, "result.txt"), "$final_scalar\n$contraction_seconds\n$total_seconds")
end

# Check command line arguments to run
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia hybrid_contractor.jl <job_directory>")
        exit(1)
    end
    contract_sliced_hybrid(ARGS[1])
end
