using JSON
using LinearAlgebra

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

function profile_advanced_scientific(job_dir::String)
    BLAS.set_num_threads(1)
    
    nslices, sliced_inds, sliced_names, tensors_info, steps = read_plan(job_dir)
    tensors = load_tensors(job_dir, tensors_info)
    
    num_initial = length(tensors)
    total_tensors = num_initial + length(steps)
    
    local_tensors = Vector{Array{Float64}}(undef, total_tensors)
    local_edges = Vector{Vector{String}}(undef, total_tensors)
    
    for idx in 1:num_initial
        local_tensors[idx] = tensors[idx]
        local_edges[idx] = copy(tensors_info[idx].edges)
    end
    
    # Pre-computation of edge propagation
    for step in steps
        parent_idx = step.parent + 1
        left_idx = step.left + 1
        right_idx = step.right + 1
        left_free = filter(x -> !(x in step.legs), local_edges[left_idx])
        right_free = filter(x -> !(x in step.legs), local_edges[right_idx])
        local_edges[parent_idx] = vcat(left_free, right_free)
    end
    
    # 1. Warm-up pass to compile functions
    for step in steps
        parent_idx = step.parent + 1
        left_idx = step.left + 1
        right_idx = step.right + 1
        A = local_tensors[left_idx]
        B = local_tensors[right_idx]
        inds_A = local_edges[left_idx]
        inds_B = local_edges[right_idx]
        
        if isempty(step.legs)
            local_tensors[parent_idx] = reshape(reshape(A, :) * reshape(B, :)', vcat(size(A)..., size(B)...)...)
        else
            axes_A_contract = [findfirst(x -> x == idx, inds_A) for idx in step.legs]
            axes_B_contract = [findfirst(x -> x == idx, inds_B) for idx in step.legs]
            axes_A_free = filter(i -> !(i in axes_A_contract), 1:ndims(A))
            axes_B_free = filter(i -> !(i in axes_B_contract), 1:ndims(B))
            A_perm = permutedims(A, vcat(axes_A_free, axes_A_contract))
            B_perm = permutedims(B, vcat(axes_B_contract, axes_B_free))
            A_mat = reshape(A_perm, prod([size(A, i) for i in axes_A_free]; init=1), prod([size(A, i) for i in axes_A_contract]; init=1))
            B_mat = reshape(B_perm, prod([size(B, i) for i in axes_B_contract]; init=1), prod([size(B, i) for i in axes_B_free]; init=1))
            C_mat = A_mat * B_mat
            shape_C = vcat([size(A, i) for i in axes_A_free], [size(B, i) for i in axes_B_free])
            local_tensors[parent_idx] = isempty(shape_C) ? fill(C_mat[1]) : reshape(C_mat, shape_C...)
        end
    end
    
    # 2. Reset tensors and run micro-benchmarked pass
    for idx in 1:num_initial
        local_tensors[idx] = tensors[idx]
    end
    
    step_records = []
    total_time_perm = 0.0
    total_time_gemm = 0.0
    total_time_alloc = 0.0
    total_flops = 0
    total_bytes = 0
    
    for (s_idx, step) in enumerate(steps)
        parent_idx = step.parent + 1
        left_idx = step.left + 1
        right_idx = step.right + 1
        
        A = local_tensors[left_idx]
        B = local_tensors[right_idx]
        inds_A = local_edges[left_idx]
        inds_B = local_edges[right_idx]
        
        len_A = length(A)
        len_B = length(B)
        
        t_perm = 0.0
        t_gemm = 0.0
        t_alloc = 0.0
        
        C = nothing
        flops = 0
        
        if isempty(step.legs)
            flops = 2 * len_A * len_B
            t0 = time_ns()
            A_flat = reshape(A, :)
            B_flat = reshape(B, :)
            t1 = time_ns()
            t_alloc += (t1 - t0) / 1e9
            
            t0 = time_ns()
            C_flat = A_flat * B_flat'
            t1 = time_ns()
            t_gemm += (t1 - t0) / 1e9
            
            t0 = time_ns()
            shape_C = vcat(size(A)..., size(B)...)
            C = isempty(shape_C) ? fill(C_flat[1]) : reshape(C_flat, shape_C...)
            t1 = time_ns()
            t_alloc += (t1 - t0) / 1e9
        else
            axes_A_contract = [findfirst(x -> x == idx, inds_A) for idx in step.legs]
            axes_B_contract = [findfirst(x -> x == idx, inds_B) for idx in step.legs]
            axes_A_free = filter(i -> !(i in axes_A_contract), 1:ndims(A))
            axes_B_free = filter(i -> !(i in axes_B_contract), 1:ndims(B))
            
            size_A_free = prod([size(A, i) for i in axes_A_free]; init=1)
            size_A_contract = prod([size(A, i) for i in axes_A_contract]; init=1)
            size_B_contract = prod([size(B, i) for i in axes_B_contract]; init=1)
            size_B_free = prod([size(B, i) for i in axes_B_free]; init=1)
            
            flops = 2 * size_A_free * size_B_free * size_A_contract
            
            # Measure Permutation
            t0 = time_ns()
            A_perm = permutedims(A, vcat(axes_A_free, axes_A_contract))
            B_perm = permutedims(B, vcat(axes_B_contract, axes_B_free))
            t1 = time_ns()
            t_perm = (t1 - t0) / 1e9
            
            # Measure Reshaping
            t0 = time_ns()
            A_mat = reshape(A_perm, size_A_free, size_A_contract)
            B_mat = reshape(B_perm, size_B_contract, size_B_free)
            t1 = time_ns()
            t_alloc = (t1 - t0) / 1e9
            
            # Measure GEMM
            t0 = time_ns()
            C_mat = A_mat * B_mat
            t1 = time_ns()
            t_gemm = (t1 - t0) / 1e9
            
            # Measure Final Output Packaging
            t0 = time_ns()
            shape_C = vcat([size(A, i) for i in axes_A_free], [size(B, i) for i in axes_B_free])
            C = isempty(shape_C) ? fill(C_mat[1]) : reshape(C_mat, shape_C...)
            t1 = time_ns()
            t_alloc += (t1 - t0) / 1e9
        end
        
        local_tensors[parent_idx] = C
        len_C = length(C)
        
        bytes_transferred = (len_A + len_B + len_C) * 8
        step_total_time = t_perm + t_gemm + t_alloc
        
        intensity = bytes_transferred > 0 ? flops / bytes_transferred : 0.0
        achieved_gflops = (step_total_time > 0 && flops > 0) ? (flops / 1e9) / step_total_time : 0.0
        
        total_time_perm += t_perm
        total_time_gemm += t_gemm
        total_time_alloc += t_alloc
        total_flops += flops
        total_bytes += bytes_transferred
        
        category = "small"
        if len_C >= 1_000_000
            category = "large"
        elseif len_C >= 10_000
            category = "medium"
        else
            category = "small"
        end
        
        push!(step_records, Dict(
            "step" => s_idx,
            "flops" => flops,
            "bytes" => bytes_transferred,
            "intensity" => intensity,
            "gflops" => achieved_gflops,
            "time_total" => step_total_time,
            "time_perm" => t_perm,
            "time_gemm" => t_gemm,
            "time_alloc" => t_alloc,
            "size_C" => len_C,
            "category" => category
        ))
    end
    
    total_step_time = total_time_perm + total_time_gemm + total_time_alloc
    
    summary = Dict(
        "total_time" => total_step_time,
        "total_time_perm" => total_time_perm,
        "total_time_gemm" => total_time_gemm,
        "total_time_alloc" => total_time_alloc,
        "pct_perm" => total_step_time > 0 ? (total_time_perm / total_step_time) * 100.0 : 0.0,
        "pct_gemm" => total_step_time > 0 ? (total_time_gemm / total_step_time) * 100.0 : 0.0,
        "pct_alloc" => total_step_time > 0 ? (total_time_alloc / total_step_time) * 100.0 : 0.0,
        "total_flops" => total_flops,
        "total_bytes" => total_bytes,
        "overall_intensity" => total_bytes > 0 ? total_flops / total_bytes : 0.0,
        "overall_gflops" => total_step_time > 0 ? (total_flops / 1e9) / total_step_time : 0.0,
        "steps" => step_records
    )
    
    return summary
end

if length(ARGS) < 1
    println("Usage: julia advanced_scientific_profiler.jl <job_dir>")
    exit(1)
end

job_dir = ARGS[1]
res = profile_advanced_scientific(job_dir)
println(JSON.json(res))
