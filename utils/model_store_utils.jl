using JSON3

function canonical_models_root(script_dir::AbstractString, run_dir::AbstractString)
    # 1. Absolute path: trust it as-is.
    isabspath(run_dir) && return run_dir
    # 2. Resolves from the current working directory (e.g. invoked from the
    #    repo root with run_dir = "data/example_runs/<bank>").
    cwd_path = abspath(run_dir)
    isdir(cwd_path) && return cwd_path
    # 3. Fallback: the original convention, <script_dir>/model_runs/<run_dir>,
    #    used when the user passes only a bank name and a `model_runs/`
    #    symlink lives next to the script.
    return joinpath(script_dir, "model_runs", run_dir)
end

function resolve_model_path(root::AbstractString, model_file::AbstractString)
    base = basename(strip(model_file))
    endswith(base, ".json") || (base = base * ".json")
    isabspath(model_file) && return joinpath(dirname(model_file), base)
    return joinpath(root, base)
end

function resolve_model_paths(root::AbstractString, model_file::Union{Nothing,String})
    if model_file === nothing
        paths = sort([joinpath(root, f) for f in readdir(root)
                      if endswith(f, ".json") && !contains(f, "_chunk_")])
        isempty(paths) && error("No model JSON files found in $root")
        return paths
    end
    path = resolve_model_path(root, model_file)
    isfile(path) || error("Model file not found: $path")
    return [path]
end

"""
    compute_r_unique_equilibrium(A, B, alpha)

Compute growth rates r(α) that plant equilibrium at x* = 1 for the unique_equilibrium mode.
r_i(α) = -[(1-α) Σ_j A[i,j] + α Σ_{j,k} B[j,k,i]]
"""
function compute_r_unique_equilibrium(A::Matrix{Float64}, B::Array{Float64,3}, alpha::Float64)
    n = size(A, 1)
    r = Vector{Float64}(undef, n)
    for i in 1:n
        r[i] = -((1 - alpha) * sum(@view A[i, :]) + alpha * sum(@view B[:, :, i]))
    end
    return r
end

function safe_write_json(path::AbstractString, payload)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON3.write(io, payload)
    end
    mv(tmp, path; force=true)
end
