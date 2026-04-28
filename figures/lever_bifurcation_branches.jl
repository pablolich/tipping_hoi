#!/usr/bin/env julia
# Compute bifurcation branches + hysteresis integration for the Lever mutualism model
# and save a tidy CSV for downstream Python plotting.
#
# Output: figures/data/lever_bifurcation_branches.csv

using HomotopyContinuation
using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using DataFrames
using CSV

# ---------------------------------------------------------------------------
# System definition
# ---------------------------------------------------------------------------

"""
lever_steady_state_system(Sp, Sa)

Build the steady-state system F(x)=0 where x = [P; A].

Plants (i=1..Sp):
dP_i = P_i * (rP_i + benefit_i - comp_i) + muP
Pollinators (k=1..Sa):
dA_k = A_k * (rA_k - dA + benefit_k - comp_k) + muA

We return F(x) = 0 with
F_P_i = rP_i + benefit_i - comp_i + muP / P_i
F_A_k = rA_k - dA + benefit_k - comp_k + muA / A_k

Note: F contains rational terms due to muP/P_i and muA/A_k.
"""
function lever_steady_state_system(Sp::Int, Sa::Int)
    @var P[1:Sp] A[1:Sa]
    @var rP[1:Sp] rA[1:Sa] hP[1:Sp] hA[1:Sa]
    @var CP[1:Sp,1:Sp] CA[1:Sa,1:Sa]
    @var GP[1:Sp,1:Sa] GA[1:Sa,1:Sp]
    @var muP muA dA

    vars = vcat(P, A)
    eqs = Vector{Any}(undef, Sp + Sa)

    for i in 1:Sp
        m = zero(P[1])
        comp = zero(P[1])
        for k in 1:Sa
            m += GP[i, k] * A[k]
        end
        for j in 1:Sp
            comp += CP[i, j] * P[j]
        end
        benefit = m / (1 + hP[i] * m)
        eqs[i] = rP[i] + benefit - comp + muP / P[i]
    end

    for k in 1:Sa
        m = zero(P[1])
        comp = zero(P[1])
        for i in 1:Sp
            m += GA[k, i] * P[i]
        end
        for l in 1:Sa
            comp += CA[k, l] * A[l]
        end
        benefit = m / (1 + hA[k] * m)
        eqs[Sp + k] = rA[k] - dA + benefit - comp + muA / A[k]
    end

    params = vcat(rP, rA, hP, hA, vec(CP), vec(CA), vec(GP), vec(GA), [muP, muA, dA])

    return System(eqs; variables=vars, parameters=params)
end

# ---------------------------------------------------------------------------
# Parameter struct + sampling
# ---------------------------------------------------------------------------

mutable struct ModelParams
    Sp::Int
    Sa::Int
    rP::Vector{Float64}
    rA::Vector{Float64}
    hP::Vector{Float64}
    hA::Vector{Float64}
    CP::Matrix{Float64}
    CA::Matrix{Float64}
    GP::Matrix{Float64}
    GA::Matrix{Float64}
    muP::Float64
    muA::Float64
    dA::Float64
end

function sample_parameters(adj::BitMatrix; t=0.5, μ=1e-4, rng=Random.default_rng())
    Sp, Sa = size(adj)
    rP = rand(rng, Uniform(0.05, 0.35), Sp)
    rA = rand(rng, Uniform(0.05, 0.35), Sa)
    hP = rand(rng, Uniform(0.15, 0.30), Sp)
    hA = rand(rng, Uniform(0.15, 0.30), Sa)

    CP = rand(rng, Uniform(0.01, 0.05), Sp, Sp)
    CA = rand(rng, Uniform(0.01, 0.05), Sa, Sa)
    for i in 1:Sp
        CP[i, i] = rand(rng, Uniform(0.8, 1.1))
    end
    for k in 1:Sa
        CA[k, k] = rand(rng, Uniform(0.8, 1.1))
    end

    degP = vec(sum(adj; dims=2))
    degA = vec(sum(adj; dims=1))
    GP = zeros(Float64, Sp, Sa)
    GA = zeros(Float64, Sa, Sp)
    for i in 1:Sp, k in 1:Sa
        if adj[i, k]
            γ0P = rand(rng, Uniform(0.8, 1.2))
            γ0A = rand(rng, Uniform(0.8, 1.2))
            GP[i, k] = γ0P / (degP[i]^t)
            GA[k, i] = γ0A / (degA[k]^t)
        end
    end
    return ModelParams(Sp, Sa, rP, rA, hP, hA, CP, CA, GP, GA, μ, μ, 0.0)
end

# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------

function lever_mutualism!(du, u, p::ModelParams, t)
    Sp, Sa = p.Sp, p.Sa
    P = @view u[1:Sp]
    A = @view u[Sp+1:Sp+Sa]
    dP = @view du[1:Sp]
    dA_vec = @view du[Sp+1:Sp+Sa]

    for i in 1:Sp
        m = 0.0
        comp = 0.0
        for k in 1:Sa
            m += p.GP[i, k] * A[k]
        end
        for j in 1:Sp
            comp += p.CP[i, j] * P[j]
        end
        benefit = (m == 0.0) ? 0.0 : m / (1.0 + p.hP[i] * m)
        dP[i] = p.rP[i] * P[i] + benefit * P[i] - comp * P[i] + p.muP
    end

    for k in 1:Sa
        m = 0.0
        comp = 0.0
        for i in 1:Sp
            m += p.GA[k, i] * P[i]
        end
        for l in 1:Sa
            comp += p.CA[k, l] * A[l]
        end
        benefit = (m == 0.0) ? 0.0 : m / (1.0 + p.hA[k] * m)
        dA_vec[k] = (p.rA[k] - p.dA) * A[k] + benefit * A[k] - comp * A[k] + p.muA
    end
    return nothing
end

function integrate_to_steady(u0, p::ModelParams; tmax=4_000.0, steady_tol=1e-8)
    prob = ODEProblem(lever_mutualism!, u0, (0.0, tmax), p)
    condition(u, t, integrator) = begin
        du = integrator.du
        du === nothing && return false
        return maximum(abs, du) < steady_tol
    end
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!; save_positions=(false, false))
    sol = DifferentialEquations.solve(prob, Tsit5();
        callback=cb, reltol=1e-9, abstol=1e-11, save_everystep=false, save_start=false
    )
    uend = copy(sol.u[end])
    @. uend = max(uend, 0.0)
    return uend
end

function sample_feasible_params(adj::BitMatrix; t=0.5, μ=1e-4, rng=Random.default_rng(),
                                max_tries=5000, thr=1e-3, tmax=4_000.0)
    for _ in 1:max_tries
        p = sample_parameters(adj; t=t, μ=μ, rng=rng)
        p.dA = 0.0
        u0 = ones(Float64, p.Sp + p.Sa)
        ueq = integrate_to_steady(u0, p; tmax=tmax)
        minimum(ueq) > thr && return (params=p, ueq=ueq)
    end
    error("Failed to find feasible baseline after $max_tries tries.")
end

function modelparams_to_vector(p::ModelParams)
    return vcat(
        p.rP, p.rA, p.hP, p.hA, vec(p.CP), vec(p.CA), vec(p.GP), vec(p.GA), [p.muP, p.muA, p.dA]
    )
end

# ---------------------------------------------------------------------------
# HC utilities
# ---------------------------------------------------------------------------

function corrected_equilibrium_at_t(F::System, vars, p0, u, t, x_guess)
    pt = t .* p0 + (1 - t) .* (p0 + u)
    Ft = System(F(vars, pt), variables=vars)
    N = newton(Ft, x_guess)
    is_success(N) || error("Newton correction failed at t=$t with return_code=$(N.return_code)")
    return ComplexF64.(solution(N))
end

function fold_direction(Ft::System, x0::AbstractVector)
    Fi = InterpretedSystem(Ft)
    n = length(x0)
    J = zeros(ComplexF64, n, n)
    jacobian!(J, Fi, ComplexF64.(x0))
    S = svd(J)
    v = S.V[:, end]
    v_re = real.(v)
    v_im = imag.(v)
    d = norm(v_re) >= norm(v_im) ? v_re : v_im
    norm(d) > eps() || return nothing
    return ComplexF64.(d ./ norm(d))
end

function newton_with_retries(Ft::System, seed_eq::AbstractVector, rng;
                             local_scales=(1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5),
                             broad_scales=(5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0, 5.0),
                             local_random_tries_per_scale=96,
                             broad_random_tries_per_scale=128,
                             min_distinct_distance=1e-3)
    x0 = ComplexF64.(seed_eq)
    n = length(x0)
    v_fold = fold_direction(Ft, x0)
    best_dist = 0.0
    converged_non_distinct = false

    function try_start(start)
        N = newton(Ft, start)
        if is_success(N)
            x = ComplexF64.(solution(N))
            d = norm(x - x0)
            best_dist = max(best_dist, d)
            if d > min_distinct_distance
                return N
            end
            converged_non_distinct = true
        end
        return nothing
    end

    # Stage 1: local search around probe_eq, biased by the fold direction.
    for scale in local_scales
        if v_fold !== nothing
            N = try_start(x0 + scale .* v_fold)
            N !== nothing && return N
            N = try_start(x0 - scale .* v_fold)
            N !== nothing && return N
        end

        for i in 1:n
            start = copy(x0)
            start[i] += scale
            N = try_start(start)
            N !== nothing && return N
            start = copy(x0)
            start[i] -= scale
            N = try_start(start)
            N !== nothing && return N
        end

        for _ in 1:local_random_tries_per_scale
            dir = randn(rng, n)
            if v_fold !== nothing
                dir .+= 0.5 .* real.(v_fold) .* randn(rng)
            end
            dir ./= max(norm(dir), eps())
            N = try_start(x0 + scale .* dir)
            N !== nothing && return N
        end
    end

    # Stage 2: broader jumps if local kicks keep converging to the same root.
    for scale in broad_scales
        for _ in 1:broad_random_tries_per_scale
            dir = randn(rng, n)
            dir ./= max(norm(dir), eps())
            N = try_start(x0 + scale .* dir)
            N !== nothing && return N
        end
    end

    if converged_non_distinct
        error("Newton converged repeatedly but only to the same equilibrium (max distance from seed=$best_dist, required>$min_distinct_distance).")
    end
    error("Newton failed for all perturbation attempts near probe_eq.")
end

function sample_branch_points(T::Tracker, F::System, vars, p0, u,
                              x_start::AbstractVector, t_start, t_end; n_points=100)
    t_values = [(1 - s) * t_start + s * t_end for s in range(0.0, 1.0, length=n_points)]
    x_values = Matrix{ComplexF64}(undef, length(x_start), n_points)
    x_curr = ComplexF64.(x_start)
    x_values[:, 1] = x_curr
    t_curr = t_values[1]
    last_idx = 1
    for idx in 2:n_points
        t_next = t_values[idx]
        step_res = track(T, x_curr, t_curr, t_next)
        if !is_success(step_res) && step_res.return_code == :terminated_invalid_startvalue
            x_curr = corrected_equilibrium_at_t(F, vars, p0, u, t_curr, x_curr)
            step_res = track(T, x_curr, t_curr, t_next)
        end
        if !is_success(step_res)
            @warn "Branch tracking stopped at point $idx/$n_points (return_code=$(step_res.return_code))"
            break
        end
        x_curr = ComplexF64.(solution(step_res))
        x_values[:, idx] = x_curr
        t_curr = t_next
        last_idx = idx
    end
    return t_values[1:last_idx], x_values[:, 1:last_idx]
end

function dA_values_from_t(t_values, p0, u, dA_idx::Int)
    return real.(t_values .* p0[dA_idx] .+ (1 .- t_values) .* (p0[dA_idx] + u[dA_idx]))
end

function integrate_equilibria_across_dA(base_params::ModelParams, dA_values::AbstractVector, u0::AbstractVector; tmax=4_000.0)
    p = deepcopy(base_params)
    n = p.Sp + p.Sa
    U = Matrix{Float64}(undef, n, length(dA_values))
    u_curr = Float64.(u0)
    for (j, dA_val) in enumerate(dA_values)
        p.dA = dA_val
        u_curr = integrate_to_steady(u_curr, p; tmax=tmax)
        U[:, j] = u_curr
    end
    return U
end

# ---------------------------------------------------------------------------
# DataFrame assembly + CSV export
# ---------------------------------------------------------------------------

function build_branches_dataframe(
    dA_probe_branch, x_probe_branch,
    dA_newton_branch, x_newton_branch,
    dA_integrated, x_integrated,
    dA_post_fold, x_integrated_post_fold,
    dA_backward, x_backward,
    Sp::Int, Sa::Int,
    collapse_dA::Float64, delta_event_val::Float64
)
    delta_col          = Float64[]
    species_id_col     = Int[]
    abundance_col      = Float64[]
    source_type_col    = String[]
    pass_direction_col = String[]
    branch_id_col      = Union{String, Missing}[]
    scrit_col          = Float64[]
    delta_event_col    = Float64[]

    function add_rows!(dA_vals, x_mat, src_type::String, pass_dir::String, bid::Union{String,Missing})
        for (j, dA) in enumerate(dA_vals)
            for k in 1:Sa
                idx = Sp + k
                abund = real(x_mat[idx, j])
                push!(delta_col,          dA)
                push!(species_id_col,     k)
                push!(abundance_col,      abund)
                push!(source_type_col,    src_type)
                push!(pass_direction_col, pass_dir)
                push!(branch_id_col,      bid)
                push!(scrit_col,          collapse_dA)
                push!(delta_event_col,    delta_event_val)
            end
        end
    end

    # Algebraic: stable probe branch
    add_rows!(dA_probe_branch, x_probe_branch, "algebraic", "none", "pre_fold_1")

    # Algebraic: unstable Newton branch
    add_rows!(dA_newton_branch, x_newton_branch, "algebraic", "none", "pre_fold_2")

    # Algebraic: zero equilibrium line from delta_event to end of post-fold range
    dA_zero_range = collect(range(delta_event_val, dA_post_fold[end], length=40))
    x_zero_mat    = zeros(Float64, Sp + Sa, length(dA_zero_range))
    add_rows!(dA_zero_range, x_zero_mat, "algebraic", "none", "post_forward")

    # Dynamic: forward pre-fold sweep
    add_rows!(dA_integrated, x_integrated, "dynamic", "forward", missing)

    # Dynamic: forward post-fold
    add_rows!(dA_post_fold, x_integrated_post_fold, "dynamic", "forward", missing)

    # Dynamic: backward sweep
    add_rows!(dA_backward, x_backward, "dynamic", "backward", missing)

    return DataFrame(
        delta          = delta_col,
        species_id     = species_id_col,
        abundance      = abundance_col,
        source_type    = source_type_col,
        pass_direction = pass_direction_col,
        branch_id      = branch_id_col,
        scrit          = scrit_col,
        delta_event    = delta_event_col,
    )
end

function save_branches_table(df::DataFrame, outdir::AbstractString, stem::AbstractString)
    mkpath(outdir)
    outpath = joinpath(outdir, stem * ".csv")
    CSV.write(outpath, df)
    println("Saved branches table to $outpath")
    return outpath
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Sp, Sa = 10, 10
F    = lever_steady_state_system(Sp, Sa)
vars = variables(F)
rng  = MersenneTwister(1)
adj  = trues(Sp, Sa)

println("Sampling feasible baseline parameters...")
base   = sample_feasible_params(adj; rng=rng)
start0 = base.ueq
p0     = modelparams_to_vector(base.params)

# Vary only dA from 0 to 10, keep everything else fixed.
u      = zeros(length(p0))
dA_idx = length(p0)
u[dA_idx] = 10.0

println("Running parameter homotopy to find fold...")
H   = ParameterHomotopy(F; start_parameters=p0, target_parameters=p0 + u)
T   = Tracker(H)
start_t, target_t = 1.0, 0.0
res = track(T, start0, start_t, target_t)

collapse_dA    = real(res.t * p0[dA_idx] + (1 - res.t) * (p0[dA_idx] + u[dA_idx]))
tracked_length = abs(res.t - start_t)
direction      = sign(res.t - start_t)
next_t         = res.t - direction * (0.001 * tracked_length)
track_eq       = solution(res)
println("Fold (collapse) at dA = $collapse_dA")

probe_res = track(T, start0, start_t, next_t)
is_success(probe_res) || error("Failed to compute probe equilibrium: return_code=$(probe_res.return_code)")
probe_eq = solution(probe_res)

ft = F(vars, next_t .* p0 + (1 - next_t) .* (p0 + u))
Ft = System(ft, variables=vars)
println("Searching for second branch via Newton...")
N        = newton_with_retries(Ft, probe_eq, rng)
newton_eq = solution(N)

# Sample branch curves
n_branch_points = 100
println("Sampling probe branch (100 points)...")
t_probe_branch,  x_probe_branch  = sample_branch_points(T, F, vars, p0, u, probe_eq,  next_t, start_t; n_points=n_branch_points)
println("Sampling Newton branch (100 points)...")
t_newton_branch, x_newton_branch = sample_branch_points(T, F, vars, p0, u, newton_eq, next_t, start_t; n_points=n_branch_points)

dA_probe_branch  = dA_values_from_t(t_probe_branch,  p0, u, dA_idx)
dA_newton_branch = dA_values_from_t(t_newton_branch, p0, u, dA_idx)

dA_min = min(minimum(dA_probe_branch), minimum(dA_newton_branch))
dA_max = max(maximum(dA_probe_branch), maximum(dA_newton_branch))

# Forward ODE: 10 pre-fold points + 5 post-fold points = 15 total
dA_integrated = collect(range(dA_min, dA_max, length=10))
println("Forward ODE integration (pre-fold, 10 points)...")
x_integrated  = integrate_equilibria_across_dA(base.params, dA_integrated, start0)

post_fold_scale = 1.5
post_fold_start = collapse_dA * 1.001
post_fold_end   = post_fold_scale * collapse_dA
post_fold_end > post_fold_start || error("Invalid post-fold range: [$post_fold_start, $post_fold_end]")
dA_post_fold = collect(range(post_fold_start, post_fold_end, length=5))
println("Forward ODE integration (post-fold, 5 points)...")
x_integrated_post_fold = integrate_equilibria_across_dA(base.params, dA_post_fold, x_integrated[:, end])
integration_endpoint   = x_integrated_post_fold[:, end]

# Backward ODE: reverse all forward points
dA_backward = reverse(vcat(dA_integrated, dA_post_fold))
println("Backward ODE integration (15 points)...")
x_backward  = integrate_equilibria_across_dA(base.params, dA_backward, integration_endpoint)

# The recovery marker = end of Newton branch (where it terminates at low dA)
delta_event_val = dA_newton_branch[end]
println("delta_event (recovery marker) = $delta_event_val")

println("Building branches DataFrame...")
df = build_branches_dataframe(
    dA_probe_branch, x_probe_branch,
    dA_newton_branch, x_newton_branch,
    dA_integrated, x_integrated,
    dA_post_fold, x_integrated_post_fold,
    dA_backward, x_backward,
    Sp, Sa,
    collapse_dA, delta_event_val
)

outdir = joinpath(@__DIR__, "data")
save_branches_table(df, outdir, "lever_bifurcation_branches")
println("Done. Rows: $(nrow(df)), Columns: $(names(df))")
