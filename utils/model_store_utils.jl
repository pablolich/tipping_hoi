using JSON3

function canonical_models_root(script_dir::AbstractString, run_dir::AbstractString)
    isabspath(run_dir) && return run_dir
    return abspath(run_dir)
end

function resolve_model_path(root::AbstractString, model_file::AbstractString)
    base = basename(strip(model_file))
    endswith(base, ".json") || (base = base * ".json")
    isabspath(model_file) && return joinpath(dirname(model_file), base)
    return joinpath(root, base)
end

"""
    resolve_model_paths(root, model_file)

All model JSONs under `root`, recursively.  Recursion is what lets a sharded
scan output (`<run>/<cell>/shard<k>/model_*.json`) be handed to a driver as one
directory instead of one invocation per shard; it is a no-op on the flat bank
layouts under `data/example_runs/`, none of which has a subdirectory.

The `_chunk_` name filter is deliberate and must stay a NAME filter: the chunk
files in the aguade / karatayev / mougi / stouffer banks do carry a valid
`scan_results` payload, so swapping this for a content test would newly pull in
four files per bank that the submitted runs excluded.

Files that are not model payloads at all (a `scan_manifest.json` beside the
shard, the pre-scan `gibbs_refgrid` systems) are *not* filtered here — that
needs the parsed file — but callers are expected to skip a payload with no
`scan_results` rather than die on it.
"""
function resolve_model_paths(root::AbstractString, model_file::Union{Nothing,String})
    if model_file === nothing
        paths = String[]
        for (dir, _, files) in walkdir(root), f in files
            endswith(f, ".json") && !contains(f, "_chunk_") && push!(paths, joinpath(dir, f))
        end
        isempty(paths) && error("No model JSON files found in $root")
        return sort(paths)
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
