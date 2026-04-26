using JSON3

function canonical_models_root(script_dir::AbstractString, run_dir::AbstractString)
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
    newton_find_equilibrium(rhs!, x0; max_iter, ftol, eps_fd, clip_lo)

Find equilibrium of dx/dt = f(x) via Newton's method on the per-capita system
g_i(x) = f_i(x)/x_i = 0, using a finite-difference Jacobian.
`rhs!` has signature f!(du, u, nothing, 0.0).
Returns NamedTuple (success::Bool, x_eq::Vector{Float64}).
"""
function newton_find_equilibrium(rhs!, x0::AbstractVector{<:Real};
                                  max_iter::Int    = 200,
                                  ftol::Float64    = 1e-10,
                                  eps_fd::Float64  = 1e-6,
                                  clip_lo::Float64 = 1e-10)
    n  = length(x0)
    x  = max.(Float64.(x0), clip_lo)
    du = zeros(n)
    g  = zeros(n)
    gp = zeros(n)
    J  = zeros(n, n)

    for _ in 1:max_iter
        rhs!(du, x, nothing, 0.0)
        @. g = du / x
        maximum(abs, g) < ftol && return (success=true, x_eq=copy(x))

        # Finite-difference Jacobian of g(x)
        for j in 1:n
            h    = eps_fd * max(abs(x[j]), 1.0)
            x[j] += h
            rhs!(du, x, nothing, 0.0)
            @. gp = du / x
            @inbounds for i in 1:n; J[i, j] = (gp[i] - g[i]) / h; end
            x[j] -= h
        end

        delta = J \ (-g)

        # Backtracking line search to keep all x positive
        alpha = 1.0
        for _ in 1:20
            all(x .+ alpha .* delta .>= clip_lo) && break
            alpha *= 0.5
        end
        x .= max.(x .+ alpha .* delta, clip_lo)
    end

    rhs!(du, x, nothing, 0.0)
    @. g = du / x
    return (success=maximum(abs, g) < ftol * 1e3, x_eq=copy(x))
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
