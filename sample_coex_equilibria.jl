# Sample equilibria along perturbation rays to cover the coexistence region.
#
# For each direction u_k stored in U, walk dr = s·u_k for s ∈ [0, delta_c_k]
# using HomotopyContinuation's parameter-homotopy Tracker, seeded from the
# baseline equilibrium x_star.  We record (dr, x_star(dr)) samples at K
# uniformly-spaced s values per ray (including s = delta_c_k).  By construction
# the equilibrium condition gives f_nominal(x_star(dr)) = -dr, so the pooled
# samples parameterize the coexistence region and the induced map x → f(x).
#
# The tracker follows the same branch HC followed during the boundary scan,
# so we match the branch structure used by scan_results.  If the tracker
# loses the path before delta_c (should be rare since scan_results already
# reached it), we stop along that ray and report the count kept.

using HomotopyContinuation
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "pipeline_config.jl"))
include(joinpath(@__DIR__, "utils", "math_utils.jl"))
include(joinpath(@__DIR__, "utils", "json_utils.jl"))
include(joinpath(@__DIR__, "utils", "hc_tracker_utils.jl"))
include(joinpath(@__DIR__, "utils", "glvhoi_utils.jl"))
include(joinpath(@__DIR__, "model_store_utils.jl"))

"""
    track_to_target!(ws, x_start, p_start, p_target) -> Union{Vector{Float64}, Nothing}

Follow the parameter homotopy x_start @ p_start → ? @ p_target using the
ScanWorkspace's tracker. Returns the real-valued equilibrium at p_target on
success; returns `nothing` if the tracker fails before reaching the target.
"""
function track_to_target!(ws::ScanWorkspace,
                          x_start::AbstractVector{<:Real},
                          p_start::AbstractVector{<:Real},
                          p_target::AbstractVector{<:Real})
    start_parameters!(ws.tracker, p_start)
    target_parameters!(ws.tracker, p_target)
    init!(ws.tracker, Float64.(x_start), 1.0, 0.0)

    while is_tracking(ws.tracker.state.code)
        HomotopyContinuation.step!(ws.tracker)
    end

    if is_success(ws.tracker.state.code)
        return real.(ws.tracker.state.x)
    else
        return nothing
    end
end

"""
    sample_coex_equilibria(bank; K=10, α_index=1) -> NamedTuple

Sample equilibria along each ray up to the stored critical magnitude δ_c.

Returns `(; dr, x, per_ray_kept, per_ray_target, ray_ids, s_fractions)` where:
  - `dr`  is an (n × N) matrix of perturbation vectors.
  - `x`   is an (n × N) matrix of equilibria at those perturbations.
  - `per_ray_kept[k]`   = number of non-baseline samples that were successfully tracked along ray k.
  - `per_ray_target[k]` = K (the number we tried to place per ray).
  - `ray_ids`  gives the 1-based ray index that produced each sample column.
  - `s_fractions` gives the s / δ_c fraction for each sample column (0 for baseline).

The first column of `dr` and `x` is always the baseline (dr=0, x=x_star).
"""
function sample_coex_equilibria(bank::AbstractDict; K::Int=10, α_index::Int=1)
    ctx = build_hc_system(bank)
    n = ctx.n
    n_dirs = size(ctx.U, 2)
    x_star = Float64.(ctx.x0)

    α = ctx.alpha_grid[α_index]
    ws = ctx.make_workspace(Float64(α))
    ws === nothing && error("make_workspace returned nothing at α=$α (cannot track).")

    scan_dirs = bank["scan_results"][α_index]["directions"]
    length(scan_dirs) == n_dirs ||
        error("scan_results has $(length(scan_dirs)) directions but U has $n_dirs columns.")

    # Baseline sample at dr = 0.
    dr_samples = Vector{Vector{Float64}}()
    x_samples  = Vector{Vector{Float64}}()
    ray_ids    = Int[]
    s_fracs    = Float64[]

    push!(dr_samples, zeros(Float64, n))
    push!(x_samples,  copy(x_star))
    push!(ray_ids, 0)
    push!(s_fracs, 0.0)

    per_ray_kept   = zeros(Int, n_dirs)
    per_ray_target = fill(K, n_dirs)

    for k in 1:n_dirs
        dir = scan_dirs[k]
        δ_c = Float64(dir["delta_c"])
        (isfinite(δ_c) && δ_c > 0) || continue

        u_k = Vector{Float64}(ctx.U[:, k])

        x_prev = copy(x_star)
        p_prev = zeros(Float64, n)

        for i in 1:K
            s_next = δ_c * (i / K)
            p_next = s_next .* u_k

            x_new = track_to_target!(ws, x_prev, p_prev, p_next)
            x_new === nothing && break

            push!(dr_samples, copy(p_next))
            push!(x_samples, copy(x_new))
            push!(ray_ids, k)
            push!(s_fracs, i / K)

            x_prev = x_new
            p_prev = p_next
            per_ray_kept[k] += 1
        end
    end

    N = length(dr_samples)
    dr_mat = Matrix{Float64}(undef, n, N)
    x_mat  = Matrix{Float64}(undef, n, N)
    for j in 1:N
        dr_mat[:, j] .= dr_samples[j]
        x_mat[:,  j] .= x_samples[j]
    end

    return (; dr = dr_mat, x = x_mat,
              per_ray_kept = per_ray_kept,
              per_ray_target = per_ray_target,
              ray_ids = ray_ids,
              s_fractions = s_fracs)
end

"""
    compute_alpha_eff_coex(bank; K=10, α_index=1) -> NamedTuple

Sample the coexistence region with `sample_coex_equilibria`, then regress
f_nominal(x) := -dr(x) onto the affine family a + B·x using ordinary least
squares.  Returns `(; alpha_eff_coex, R2, N, n_kept_total, SS_res, SS_total)`
where `alpha_eff_coex = 1 − R²` is the fraction of L² variation in f that
no affine model can explain over the sampled coexistence points.
"""
function compute_alpha_eff_coex(bank::AbstractDict; K::Int=10, α_index::Int=1)
    s = sample_coex_equilibria(bank; K=K, α_index=α_index)
    N = size(s.x, 2)
    n = size(s.x, 1)

    # By the equilibrium condition f_nominal(x_star(dr)) = -dr.  Build
    # regression matrices:
    #   X  (N × (n+1))  columns = [1, x1, ..., xn]
    #   Y  (N × n)       rows = -dr
    X = Matrix{Float64}(undef, N, n + 1)
    Y = Matrix{Float64}(undef, N, n)
    @inbounds for j in 1:N
        X[j, 1] = 1.0
        for i in 1:n
            X[j, 1 + i] = s.x[i, j]
            Y[j, i]     = -s.dr[i, j]
        end
    end

    # Ordinary least squares: θ ∈ R^{(n+1) × n}.
    θ = X \ Y
    Y_pred = X * θ

    # R² via centred Y (subtract column means so α=0 holds for affine f).
    Y_mean = mean(Y, dims = 1)
    SS_total = sum(abs2, Y .- Y_mean)
    SS_res   = sum(abs2, Y .- Y_pred)

    R2 = SS_total > 0 ? 1 - SS_res / SS_total : 0.0
    α_eff_coex = SS_total > 0 ? SS_res / SS_total : 0.0

    return (; alpha_eff_coex = α_eff_coex,
              R2 = R2,
              N = N,
              n_kept_total = sum(s.per_ray_kept),
              SS_res = SS_res,
              SS_total = SS_total)
end
