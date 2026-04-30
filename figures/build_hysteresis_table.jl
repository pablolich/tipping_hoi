#!/usr/bin/env julia
# Build tidy hysteresis table for later plotting in Python (no plotting here).

using LinearAlgebra
using Random
using JSON3
using SHA
using JLD2
using Arrow
using CSV
using DataFrames
using HomotopyContinuation
using DifferentialEquations
using SciMLBase

# ------------------------------ JSON/config utils ------------------------------

function to_dict(x)
    if x isa JSON3.Object
        d = Dict{String,Any}()
        for (k, v) in pairs(x)
            d[String(k)] = to_dict(v)
        end
        return d
    elseif x isa JSON3.Array
        return [to_dict(v) for v in x]
    else
        return x
    end
end

function canonicalize(x)
    if x isa AbstractDict
        ks = sort(collect(keys(x)))
        vals = [canonicalize(x[k]) for k in ks]
        syms = Tuple(Symbol.(ks))
        return NamedTuple{syms}(vals)
    elseif x isa AbstractVector
        return [canonicalize(v) for v in x]
    else
        return x
    end
end

function normalize_sampler(raw::Dict{String,Any})
    name = get(raw, "name", "stable_section2")
    out = Dict{String,Any}("name" => String(name))

    if name == "stable_section2"
        out["sigmaA"] = Float64(get(raw, "sigmaA", 1.0))
        out["safety"] = Float64(get(raw, "safety", 1.2))
        out["eps"] = Float64(get(raw, "eps", 1e-12))
        out["enforce_r_positive"] = Bool(get(raw, "enforce_r_positive", true))
        out["max_tries"] = Int(get(raw, "max_tries", 200000))
        out["check_alpha_stability"] = Bool(get(raw, "check_alpha_stability", true))
        out["verify_x1_equilibrium"] = Bool(get(raw, "verify_x1_equilibrium", true))
        alphas_check = get(raw, "alphas_check", get(raw, "alpha_check_grid", [0.0, 0.5, 1.0]))
        out["alphas_check"] = Float64.(alphas_check)
        out["x1_tol"] = Float64(get(raw, "x1_tol", 1e-8))
    elseif name == "stable_simple"
        out["sigmaA"] = Float64(get(raw, "sigmaA", 1.0))
        out["eps"] = Float64(get(raw, "eps", 1e-12))
    elseif name == "hit_and_run"
        out["lower"] = Float64(get(raw, "lower", -10.0))
        out["upper"] = Float64(get(raw, "upper", 10.0))
    else
        error("Unknown sampler name: $name")
    end
    return out
end

function normalize_config(cfg::Dict{String,Any})
    out = copy(cfg)

    function require_key(d::AbstractDict, key::AbstractString)
        if haskey(d, key)
            return d[key]
        elseif haskey(d, Symbol(key))
            return d[Symbol(key)]
        else
            error("Missing required key: $(key)")
        end
    end

    n_values = get(out, "n_values", nothing)
    n_values === nothing && error("Missing required key: n_values")
    out["n_values"] = [Int(v) for v in n_values]

    nsys_raw = get(out, "N_systems_per_n", nothing)
    nsys_raw === nothing && error("Missing required key: N_systems_per_n")
    nsys_raw isa Number || error("N_systems_per_n must be a scalar number")
    out["N_systems_per_n"] = Int(nsys_raw)

    out["M_dirs"] = Int(require_key(out, "M_dirs"))

    alpha_grid = get(out, "alpha_grid", get(out, "alphas", nothing))
    alpha_grid === nothing && error("Missing required key: alpha_grid")
    out["alpha_grid"] = Float64.(alpha_grid)

    out["max_pert"] = Float64(require_key(out, "max_pert"))

    seeds_raw = get(out, "seeds", Dict{String,Any}())
    seeds = Dict{String,Any}()
    seeds["params"] = get(seeds_raw, "params", get(out, "seed_params", nothing))
    seeds["dirs"] = get(seeds_raw, "dirs", get(out, "seed_dirs", nothing))
    out["seeds"] = seeds

    sampler_raw = get(out, "sampler", Dict{String,Any}())
    out["sampler"] = normalize_sampler(sampler_raw)

    dynamics_raw = get(out, "dynamics", Dict{String,Any}())
    dyn = Dict{String,Any}()
    dyn["tspan"] = Tuple(Float64.(get(dynamics_raw, "tspan", [0.0, 200.0])))
    dyn["reltol"] = Float64(get(dynamics_raw, "reltol", 1e-8))
    dyn["abstol"] = Float64(get(dynamics_raw, "abstol", 1e-10))
    dyn["saveat"] = get(dynamics_raw, "saveat", nothing)
    dyn["eps_extinct"] = Float64(get(dynamics_raw, "eps_extinct", 1e-9))
    dyn["post_delta_frac"] = Float64(get(dynamics_raw, "post_delta_frac", 0.001))
    dyn["post_delta_abs"] = get(dynamics_raw, "post_delta_abs", nothing)
    dyn["post_delta_abs_default"] = Float64(get(dynamics_raw, "post_delta_abs_default", 1e-6))
    out["dynamics"] = dyn

    seq_raw = get(out, "sequential", Dict{String,Any}())
    seq = Dict{String,Any}()
    seq["tol_pos"] = Float64(get(seq_raw, "tol_pos", 1e-9))
    seq["tol_neg"] = Float64(get(seq_raw, "tol_neg", 1e-9))
    seq["imag_tol"] = Float64(get(seq_raw, "imag_tol", 1e-8))
    seq["lambda_tol"] = Float64(get(seq_raw, "lambda_tol", 1e-9))
    out["sequential"] = seq

    back_raw = get(out, "backtrack", Dict{String,Any}())
    back = Dict{String,Any}()
    back["check_stability"] = Bool(get(back_raw, "check_stability", true))
    back["check_invasibility"] = Bool(get(back_raw, "check_invasibility", true))
    back["lambda_tol"] = Float64(get(back_raw, "lambda_tol", 1e-9))
    back["invasion_tol"] = Float64(get(back_raw, "invasion_tol", 1e-10))
    back["max_step_ratio"] = Float64(get(back_raw, "max_step_ratio", 0.3))
    back["tol_pos"] = Float64(get(back_raw, "tol_pos", seq["tol_pos"]))
    back["tol_neg"] = Float64(get(back_raw, "tol_neg", seq["tol_neg"]))
    back["eps_seed_extinct"] = Float64(get(back_raw, "eps_seed_extinct", 10.0 * dyn["eps_extinct"]))
    out["backtrack"] = back

    return out
end

function model_id_from_config(cfg::Dict{String,Any})
    hcfg = Dict(
        "n_values" => cfg["n_values"],
        "N_systems_per_n" => cfg["N_systems_per_n"],
        "M_dirs" => cfg["M_dirs"],
        "alpha_grid" => cfg["alpha_grid"],
        "max_pert" => cfg["max_pert"],
        "seeds" => cfg["seeds"],
        "sampler" => cfg["sampler"],
    )
    json_str = JSON3.write(canonicalize(hcfg))
    return bytes2hex(sha1(json_str))
end

# --------------------------- system building (HC) ------------------------------

function build_system_from_params(r0::AbstractVector{<:Real},
                                  A::AbstractMatrix{<:Real},
                                  B::Array{<:Real,3})
    n = length(r0)
    @var x[1:n] dr[1:n] alpha
    eqs = Vector{Expression}(undef, n)
    @inbounds for i in 1:n
        lin = sum(A[i, j] * x[j] for j in 1:n)
        quad = sum(B[j, k, i] * x[j] * x[k] for j in 1:n, k in 1:n)
        eqs[i] = (r0[i] + dr[i]) + (1 - alpha) * lin + alpha * quad
    end
    return System(eqs; variables=x, parameters=vcat(dr, alpha)), x
end

function make_glvhoi_rhs(A::AbstractMatrix{<:Real},
                         B::Array{<:Real,3},
                         r::AbstractVector{<:Real},
                         alpha::Real)
    n = length(r)
    quad = zeros(Float64, n)
    function f!(dx, x, p, t)
        mul!(dx, A, x)
        fill!(quad, 0.0)
        @inbounds for i in 1:n
            s = 0.0
            for j in 1:n, k in 1:n
                s += B[j, k, i] * x[j] * x[k]
            end
            quad[i] = s
        end
        @inbounds for i in 1:n
            f = r[i] + (1 - alpha) * dx[i] + alpha * quad[i]
            dx[i] = x[i] * f
        end
        return nothing
    end
    return f!
end

function jacobian_F(A::AbstractMatrix{<:Real},
                    B::Array{<:Real,3},
                    x::AbstractVector{<:Real},
                    alpha::Real)
    n = length(x)
    JF = Matrix{Float64}(undef, n, n)
    @inbounds @views for i in 1:n
        Bi = B[:, :, i]
        vL = Bi' * x
        vR = Bi * x
        @inbounds @simd for m in 1:n
            JF[i, m] = (1 - alpha) * A[i, m] + alpha * (vL[m] + vR[m])
        end
    end
    return JF
end

function per_capita_growth(A::AbstractMatrix{<:Real},
                           B::Array{<:Real,3},
                           r_eff::AbstractVector{<:Real},
                           x::AbstractVector{<:Real},
                           alpha::Real)
    n = length(x)
    fx = Vector{Float64}(undef, n)
    lin = similar(x, Float64)
    mul!(lin, A, x)
    tmp = similar(x, Float64)
    @inbounds @views for i in 1:n
        Bi = B[:, :, i]
        mul!(tmp, Bi, x)
        quad = dot(x, tmp)
        fx[i] = r_eff[i] + (1 - alpha) * lin[i] + alpha * quad
    end
    return fx
end

# ------------------------------- math helpers ----------------------------------

const X_TOL = 1e-12
const PARAM_TOL = 1e-9
const MAX_ITERS = 8
const MAX_STEPS_PT = 1_000_000

get_parameters_at_t(t, p_start, p_target) = t .* p_start .+ (1 - t) .* p_target

function delta_from_dr(dr::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    denom = dot(u, u)
    return denom == 0.0 ? 0.0 : dot(dr, u) / denom
end

function clamp_small_negatives!(x::Vector{Float64}, tol_neg::Real)
    @inbounds for i in eachindex(x)
        if x[i] < 0 && abs(x[i]) <= tol_neg
            x[i] = 0.0
        end
    end
    return x
end

support_indices(x::AbstractVector{<:Real}, tol_pos::Real) = findall(x .> tol_pos)

function restrict_params(A::AbstractMatrix{<:Real},
                         B::Array{<:Real,3},
                         r0::AbstractVector{<:Real},
                         u::AbstractVector{<:Real},
                         idx::Vector{Int},
                         x_cur::Union{Nothing,AbstractVector}=nothing)
    A2 = A[idx, idx]
    B2 = B[idx, idx, idx]
    r2 = r0[idx]
    u2 = u[idx]
    if x_cur === nothing
        return A2, B2, r2, u2
    else
        x2 = x_cur[idx]
        return A2, B2, r2, u2, x2
    end
end

function lift_active_state(x_active::AbstractVector{<:Real},
                           active_idx::Vector{Int},
                           n::Int)
    x_full = zeros(Float64, n)
    x_full[active_idx] .= x_active
    return x_full
end

function lambda_max_equilibrium(A::AbstractMatrix{<:Real},
                                B::Array{<:Real,3},
                                x::AbstractVector{<:Real},
                                alpha::Real)
    JF = jacobian_F(A, B, x, alpha)
    J = Matrix{Float64}(JF)
    @inbounds for i in 1:length(x)
        @views J[i, :] .*= x[i]
    end
    return maximum(real.(eigvals(J)))
end

function stability_at_equilibrium(A::AbstractMatrix{<:Real},
                                  B::Array{<:Real,3},
                                  x::AbstractVector{<:Real},
                                  alpha::Real;
                                  tol_pos::Real,
                                  lambda_tol::Real)
    support = support_indices(x, tol_pos)
    if isempty(support)
        return (lambda_max_real=-Inf, is_stable=true)
    end
    JF = jacobian_F(A, B, x, alpha)
    J = Diagonal(x[support]) * JF[support, support]
    lambda_max_real = maximum(real.(eigvals(J)))
    is_stable = lambda_max_real < -lambda_tol
    return (lambda_max_real=lambda_max_real, is_stable=is_stable)
end

# --------------------------- pass4-style event tracking ------------------------

function invasion_max_at_state(r0_full::AbstractVector{<:Real},
                               A_full::AbstractMatrix{<:Real},
                               B_full::Array{<:Real,3},
                               u_full::AbstractVector{<:Real},
                               x_active::AbstractVector{<:Real},
                               active_idx::Vector{Int},
                               inactive_idx::Vector{Int},
                               delta::Real,
                               alpha::Real)
    isempty(inactive_idx) && return -Inf
    n = length(r0_full)
    x_full = zeros(Float64, n)
    x_full[active_idx] .= x_active
    r_eff = r0_full .+ delta .* u_full
    growth = per_capita_growth(A_full, B_full, r_eff, x_full, alpha)
    return maximum(growth[inactive_idx])
end

function track_segment_until_event!(syst::System,
                                    A_active::AbstractMatrix{<:Real},
                                    B_active::Array{<:Real,3},
                                    r0_full::AbstractVector{<:Real},
                                    A_full::AbstractMatrix{<:Real},
                                    B_full::Array{<:Real,3},
                                    alpha::Real,
                                    u_active::AbstractVector{<:Real},
                                    u_full::AbstractVector{<:Real},
                                    active_idx::Vector{Int},
                                    inactive_idx::Vector{Int},
                                    x_start::AbstractVector,
                                    p_start::Vector{Float64},
                                    p_tgt::Vector{Float64};
                                    max_step_ratio::Float64 = 0.3,
                                    check_stability::Bool = true,
                                    check_invasibility::Bool = true,
                                    lambda_tol::Float64 = 1e-9,
                                    invasion_tol::Float64 = 1e-10)
    par_dist = norm(p_tgt .- p_start)
    tracker = Tracker(
        CoefficientHomotopy(syst; start_coefficients=p_start, target_coefficients=p_tgt);
        options=TrackerOptions(max_steps=MAX_STEPS_PT)
    )
    init!(tracker, x_start, 1.0, 0.0; max_initial_step_size=par_dist * max_step_ratio)

    t0 = real(tracker.state.t)
    x0 = real.(tracker.state.x)
    if minimum(x0) <= X_TOL
        return :negative, t0, x0, t0, x0, :initial_negative
    end

    lambda0 = check_stability ? lambda_max_equilibrium(A_active, B_active, x0, alpha) : -Inf
    if check_stability && lambda0 > lambda_tol
        return :unstable, t0, x0, t0, x0, :initial_unstable
    end

    inv0 = -Inf
    if check_invasibility
        dr0 = get_parameters_at_t(t0, p_start, p_tgt)[1:length(u_active)]
        delta0 = delta_from_dr(dr0, u_active)
        inv0 = invasion_max_at_state(r0_full, A_full, B_full, u_full, x0, active_idx, inactive_idx, delta0, alpha)
        if inv0 > invasion_tol
            return :invasion, t0, x0, t0, x0, :initial_invasion
        end
    end

    while is_tracking(tracker.state.code)
        HomotopyContinuation.step!(tracker)
        t1 = real(tracker.state.t)
        x1 = real.(tracker.state.x)

        if is_terminated(tracker.state.code)
            return :fold, t0, x0, t1, x1, tracker.state.code
        end

        if minimum(x1) <= X_TOL
            return :negative, t0, x0, t1, x1, tracker.state.code
        end

        if check_stability
            lambda1 = lambda_max_equilibrium(A_active, B_active, x1, alpha)
            if (lambda0 <= lambda_tol) && (lambda1 > lambda_tol)
                return :unstable, t0, x0, t1, x1, tracker.state.code
            end
            lambda0 = lambda1
        end

        if check_invasibility
            dr1 = get_parameters_at_t(t1, p_start, p_tgt)[1:length(u_active)]
            delta1 = delta_from_dr(dr1, u_active)
            inv1 = invasion_max_at_state(r0_full, A_full, B_full, u_full, x1, active_idx, inactive_idx, delta1, alpha)
            if (inv0 <= invasion_tol) && (inv1 > invasion_tol)
                return :invasion, t0, x0, t1, x1, tracker.state.code
            end
            inv0 = inv1
        end

        t0 = t1
        x0 = x1
    end

    return :success, t0, x0, real(tracker.state.t), real.(tracker.state.x), tracker.state.code
end

function findparscrit_recursive_event(syst::System,
                                      A_active::AbstractMatrix{<:Real},
                                      B_active::Array{<:Real,3},
                                      r0_full::AbstractVector{<:Real},
                                      A_full::AbstractMatrix{<:Real},
                                      B_full::Array{<:Real,3},
                                      alpha::Real,
                                      u_active::AbstractVector{<:Real},
                                      u_full::AbstractVector{<:Real},
                                      active_idx::Vector{Int},
                                      inactive_idx::Vector{Int},
                                      x_start::AbstractVector,
                                      p_start::Vector{Float64},
                                      p_target::Vector{Float64};
                                      max_step_ratio::Float64 = 0.3,
                                      check_stability::Bool = true,
                                      check_invasibility::Bool = true,
                                      lambda_tol::Float64 = 1e-9,
                                      invasion_tol::Float64 = 1e-10)
    event, t0, x0, t1, x1, rc =
        track_segment_until_event!(
            syst, A_active, B_active, r0_full, A_full, B_full,
            alpha, u_active, u_full, active_idx, inactive_idx,
            x_start, p_start, p_target;
            max_step_ratio=max_step_ratio,
            check_stability=check_stability,
            check_invasibility=check_invasibility,
            lambda_tol=lambda_tol,
            invasion_tol=invasion_tol,
        )

    if event == :success
        return (flag=:success, pars_crit=p_target, xstar_crit=x1,
                p0=p_target, x0=x1, p_pos=p_target, x_pos=x1,
                depth=0, status=:success)
    elseif event == :fold
        p_fail = get_parameters_at_t(t1, p_start, p_target)
        return (flag=:fold, pars_crit=p_fail, xstar_crit=x1,
                p0=get_parameters_at_t(t0, p_start, p_target), x0=x0,
                p_pos=get_parameters_at_t(t0, p_start, p_target), x_pos=x0,
                depth=0, status=rc)
    end

    p0 = get_parameters_at_t(t0, p_start, p_target)
    p1 = get_parameters_at_t(t1, p_start, p_target)
    x_bad = x1
    target_event = event

    depth = 0
    while depth < MAX_ITERS
        if norm(p1 .- p0) < PARAM_TOL
            return (flag=target_event, pars_crit=p1, xstar_crit=x_bad,
                    p0=p0, x0=x0, p_pos=p0, x_pos=x0,
                    depth=depth, status=:converged_param)
        end

        p_mid = 0.5 .* (p0 .+ p1)
        event_mid, t0m, x0m, t1m, x1m, _ =
            track_segment_until_event!(
                syst, A_active, B_active, r0_full, A_full, B_full,
                alpha, u_active, u_full, active_idx, inactive_idx,
                x0, p0, p_mid;
                max_step_ratio=max_step_ratio,
                check_stability=check_stability,
                check_invasibility=check_invasibility,
                lambda_tol=lambda_tol,
                invasion_tol=invasion_tol,
            )

        if event_mid == target_event
            p1 = get_parameters_at_t(t1m, p0, p_mid)
            x_bad = x1m
        elseif event_mid == :success
            p0 = p_mid
            x0 = x1m
        else
            return (flag=Symbol(event_mid),
                    pars_crit=get_parameters_at_t(t1m, p0, p_mid),
                    xstar_crit=x1m,
                    p0=p0, x0=x0, p_pos=p0, x_pos=x0,
                    depth=depth, status=:other_event_precedes_target)
        end
        depth += 1
    end

    return (flag=target_event, pars_crit=p1, xstar_crit=x_bad,
            p0=p0, x0=x0, p_pos=p0, x_pos=x0,
            depth=depth, status=:max_iters)
end

# ---------------------- continuation helpers (pauls_idea style) ----------------

function corrected_equilibrium_at_t(F::System, vars, p_start, p_target, t, x_guess)
    pt = get_parameters_at_t(t, p_start, p_target)
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

function sample_branch_points(T::Tracker, F::System, vars, p_start, p_target,
                              x_start::AbstractVector, t_start, t_end;
                              n_points=100, endpoint_override=nothing)
    t_values = [(1 - s) * t_start + s * t_end for s in range(0.0, 1.0, length=n_points)]
    x_values = Matrix{ComplexF64}(undef, length(x_start), n_points)
    x_curr = ComplexF64.(x_start)
    x_values[:, 1] = x_curr
    t_curr = t_values[1]

    for idx in 2:n_points
        t_next = t_values[idx]
        step_res = track(T, x_curr, t_curr, t_next)
        if !is_success(step_res) && step_res.return_code == :terminated_invalid_startvalue
            x_curr = corrected_equilibrium_at_t(F, vars, p_start, p_target, t_curr, x_curr)
            step_res = track(T, x_curr, t_curr, t_next)
        end
        if !is_success(step_res)
            if idx == n_points && endpoint_override !== nothing
                x_curr = ComplexF64.(endpoint_override)
                x_values[:, idx] = x_curr
                t_curr = t_next
                continue
            end
            error("Branch tracking failed at point $idx with return_code=$(step_res.return_code)")
        end
        x_curr = ComplexF64.(solution(step_res))
        x_values[:, idx] = x_curr
        t_curr = t_next
    end
    return t_values, x_values
end

function delta_values_from_t(t_values::AbstractVector{<:Real},
                             p_start::AbstractVector{<:Real},
                             p_target::AbstractVector{<:Real},
                             u::AbstractVector{<:Real})
    out = Vector{Float64}(undef, length(t_values))
    for (i, t) in enumerate(t_values)
        p = get_parameters_at_t(t, p_start, p_target)
        dr = p[1:length(u)]
        out[i] = delta_from_dr(dr, u)
    end
    return out
end

function sample_branch_between_deltas(syst::System,
                                      vars,
                                      x_start::AbstractVector,
                                      p_start::Vector{Float64},
                                      p_target::Vector{Float64},
                                      u::AbstractVector{<:Real};
                                      n_points::Int=100,
                                      endpoint_override=nothing)
    if norm(p_target .- p_start) == 0.0
        t_values = collect(range(1.0, 0.0, length=n_points))
        x_values = repeat(ComplexF64.(reshape(x_start, :, 1)), 1, n_points)
        delta_vals = fill(delta_from_dr(p_start[1:length(u)], u), n_points)
        return delta_vals, x_values
    end
    H = ParameterHomotopy(syst; start_parameters=p_start, target_parameters=p_target)
    T = Tracker(H)
    t_values, x_values = sample_branch_points(
        T, syst, vars, p_start, p_target, x_start, 1.0, 0.0;
        n_points=n_points, endpoint_override=endpoint_override,
    )
    delta_vals = delta_values_from_t(t_values, p_start, p_target, u)
    return delta_vals, x_values
end

# ------------------------------- candidate scan --------------------------------

function parse_sys_id(path::AbstractString)
    m = match(r"sys_(\d+)_", basename(path))
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function parse_n_from_dir(path::AbstractString)
    m = match(r"n_(\d+)$", basename(path))
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function to_float_or_nan(x)
    if x === nothing || ismissing(x)
        return NaN
    end
    v = try
        Float64(x)
    catch
        return NaN
    end
    return isfinite(v) ? v : NaN
end

function to_int_or_default(x, default::Int)
    if x === nothing || ismissing(x)
        return default
    end
    return try
        Int(x)
    catch
        default
    end
end

function collect_pass4_invasion_candidates(results_root::AbstractString)
    isdir(results_root) || error("Missing results directory: $results_root")
    out = NamedTuple[]
    n_dirs = sort(filter(d -> isdir(d) && startswith(basename(d), "n_"), readdir(results_root; join=true)))
    for n_dir in n_dirs
        n_val = parse_n_from_dir(n_dir)
        n_val === nothing && continue
        files = sort(filter(f -> endswith(f, "_pass4_fold_backtrack.arrow"), readdir(n_dir; join=true)))
        for path in files
            sys_id = parse_sys_id(path)
            sys_id === nothing && continue
            df = DataFrame(Arrow.Table(path))
            nrow(df) == 0 && continue
            needed = ["row_idx", "alpha", "alpha_idx", "ray_id", "hc_event", "reversal_frac", "delta_event"]
            all(c -> c in names(df), needed) || continue
            for row in eachrow(df)
                hc_event = lowercase(strip(String(row.hc_event)))
                rev = to_float_or_nan(row.reversal_frac)
                if hc_event == "invasion" && isfinite(rev) && rev > 0.25
                    push!(out, (
                        n=n_val,
                        sys_id=sys_id,
                        row_idx=to_int_or_default(row.row_idx, -1),
                        alpha=to_float_or_nan(row.alpha),
                        alpha_idx=to_int_or_default(row.alpha_idx, -1),
                        ray_id=to_int_or_default(row.ray_id, -1),
                        hc_event=hc_event,
                        reversal_frac=rev,
                        delta_event=to_float_or_nan(row.delta_event),
                        pass4_path=path,
                    ))
                end
            end
        end
    end
    return out
end

function validate_candidate_pass2(candidate::NamedTuple, results_root::AbstractString)
    n_dir = joinpath(results_root, "n_$(candidate.n)")
    pass2_arrow = joinpath(n_dir, "sys_$(candidate.sys_id)_pass2.arrow")
    pass2_jld = joinpath(n_dir, "sys_$(candidate.sys_id)_pass2.jld2")
    if !isfile(pass2_arrow)
        return (ok=false, reason="missing pass2 arrow: $pass2_arrow")
    end
    if !isfile(pass2_jld)
        return (ok=false, reason="missing pass2 jld2: $pass2_jld")
    end

    pass2_df = DataFrame(Arrow.Table(pass2_arrow))
    if !(1 <= candidate.row_idx <= nrow(pass2_df))
        return (ok=false, reason="row_idx $(candidate.row_idx) out of bounds in pass2")
    end
    row2 = pass2_df[candidate.row_idx, :]
    if to_int_or_default(getproperty(row2, :alpha_idx), -1) != candidate.alpha_idx ||
       to_int_or_default(getproperty(row2, :ray_id), -1) != candidate.ray_id
        return (ok=false, reason="pass2 row key mismatch for (alpha_idx, ray_id)")
    end

    delta_post = hasproperty(row2, :s_plus) ? to_float_or_nan(getproperty(row2, :s_plus)) : NaN
    if !(isfinite(delta_post) && delta_post >= 0.0)
        return (ok=false, reason="invalid pass2 s_plus (delta_post)")
    end

    pass2_data = JLD2.load(pass2_jld)
    if !haskey(pass2_data, "x_post")
        return (ok=false, reason="pass2 jld2 missing x_post")
    end
    x_post_arr = pass2_data["x_post"]
    if ndims(x_post_arr) != 2
        return (ok=false, reason="x_post has unsupported dims=$(ndims(x_post_arr))")
    end
    if candidate.row_idx > size(x_post_arr, 2)
        return (ok=false, reason="row_idx $(candidate.row_idx) out of bounds in x_post")
    end

    x_post = Vector{Float64}(x_post_arr[:, candidate.row_idx])
    if haskey(pass2_data, "x_post_valid")
        x_post_valid = pass2_data["x_post_valid"]
        if candidate.row_idx > length(x_post_valid) || !Bool(x_post_valid[candidate.row_idx])
            return (ok=false, reason="x_post_valid is false for row_idx $(candidate.row_idx)")
        end
    end
    if !all(isfinite.(x_post))
        return (ok=false, reason="x_post has non-finite values")
    end

    return (
        ok=true,
        reason="ok",
        pass2_df=pass2_df,
        pass2_row=row2,
        delta_post=delta_post,
        x_post=x_post,
    )
end

function validate_candidate_boundary_mix(candidate::NamedTuple, results_root::AbstractString)
    n_dir = joinpath(results_root, "n_$(candidate.n)")
    pass1_arrow = joinpath(n_dir, "sys_$(candidate.sys_id)_pass1.arrow")
    if !isfile(pass1_arrow)
        return (ok=false, reason="missing pass1 arrow: $pass1_arrow")
    end

    pass1_df = DataFrame(Arrow.Table(pass1_arrow))
    n_fold = 0
    n_negative = 0
    for row in eachrow(pass1_df)
        row_alpha_idx = hasproperty(row, :alpha_idx) ? to_int_or_default(getproperty(row, :alpha_idx), -1) : -1
        row_alpha_idx == candidate.alpha_idx || continue

        row_alpha = hasproperty(row, :alpha) ? to_float_or_nan(getproperty(row, :alpha)) : NaN
        if isfinite(row_alpha) && isfinite(candidate.alpha)
            isapprox(row_alpha, candidate.alpha; atol=1e-12, rtol=0.0) || continue
        end

        flag = hasproperty(row, :flag) ? lowercase(strip(String(getproperty(row, :flag)))) : ""
        if flag == "fold"
            n_fold += 1
        elseif flag == "negative"
            n_negative += 1
        end
    end

    if n_fold == 0 || n_negative == 0
        return (
            ok=false,
            reason="pass1 mix filter failed for (sys=$(candidate.sys_id), alpha_idx=$(candidate.alpha_idx)): fold_dirs=$(n_fold), extinction_dirs=$(n_negative)",
        )
    end

    return (
        ok=true,
        reason="ok",
        fold_dirs=n_fold,
        extinction_dirs=n_negative,
    )
end

function select_candidate(results_root::AbstractString)
    all_candidates = collect_pass4_invasion_candidates(results_root)
    isempty(all_candidates) && error("No pass4 candidates satisfy hc_event==\"invasion\" and reversal_frac>0.25.")

    n_min = minimum(c.n for c in all_candidates)
    ranked = [c for c in all_candidates if c.n == n_min]
    sort!(ranked, by=c -> (c.reversal_frac, c.sys_id, c.alpha_idx, c.ray_id, c.row_idx))

    println("Candidate scan: found $(length(all_candidates)) valid pass4 rows across all n; lowest n = $n_min.")
    println("Ranked candidates at n=$n_min (by reversal_frac, sys_id, alpha_idx, ray_id, row_idx):")
    for (k, c) in enumerate(ranked)
        println("  [$k] n=$(c.n) sys=$(c.sys_id) alpha_idx=$(c.alpha_idx) ray=$(c.ray_id) row=$(c.row_idx) reversal_frac=$(round(c.reversal_frac, digits=6))")
    end

    for (k, c) in enumerate(ranked)
        chk_pass2 = validate_candidate_pass2(c, results_root)
        if !chk_pass2.ok
            println("  skip rank [$k]: $(chk_pass2.reason)")
            continue
        end

        chk_mix = validate_candidate_boundary_mix(c, results_root)
        if chk_mix.ok
            println("Selected candidate rank [$k]: n=$(c.n), sys=$(c.sys_id), row=$(c.row_idx), alpha_idx=$(c.alpha_idx), ray=$(c.ray_id)")
            println("  pass1 boundary mix at selected (sys, alpha): fold_dirs=$(chk_mix.fold_dirs), extinction_dirs=$(chk_mix.extinction_dirs)")
            return merge(c, chk_pass2, chk_mix)
        end
        println("  skip rank [$k]: $(chk_mix.reason)")
    end

    error("No candidate passed pass2 + mixed-boundary validation for lowest n=$n_min.")
end

function select_extinction_candidate(pass1_df::DataFrame,
                                     n::Int,
                                     sys_id::Int,
                                     alpha_idx::Int,
                                     alpha::Float64)
    candidates = NamedTuple[]
    for (row_pos, row) in enumerate(eachrow(pass1_df))
        row_alpha_idx = hasproperty(row, :alpha_idx) ? to_int_or_default(getproperty(row, :alpha_idx), -1) : -1
        row_alpha_idx == alpha_idx || continue

        row_alpha = hasproperty(row, :alpha) ? to_float_or_nan(getproperty(row, :alpha)) : NaN
        if isfinite(row_alpha) && !isapprox(row_alpha, alpha; atol=1e-12, rtol=0.0)
            continue
        end

        flag = hasproperty(row, :flag) ? lowercase(strip(String(getproperty(row, :flag)))) : ""
        flag == "negative" || continue

        scrit = hasproperty(row, :scrit) ? to_float_or_nan(getproperty(row, :scrit)) : NaN
        isfinite(scrit) && scrit > 0.0 || continue

        ray_id = hasproperty(row, :ray_id) ? to_int_or_default(getproperty(row, :ray_id), -1) : -1
        row_idx = hasproperty(row, :row_idx) ? to_int_or_default(getproperty(row, :row_idx), row_pos) : row_pos

        push!(candidates, (
            row_pos=row_pos,
            row_idx=row_idx,
            ray_id=ray_id,
            alpha=row_alpha,
            alpha_idx=row_alpha_idx,
            flag=flag,
            scrit=scrit,
        ))
    end

    isempty(candidates) &&
        error("No pass1 rows with flag=\"negative\" for (n=$(n), sys_id=$(sys_id), alpha_idx=$(alpha_idx), alpha=$(alpha)).")

    sort!(candidates, by=c -> (c.scrit, c.ray_id, c.row_idx))

    println("Extinction candidate scan for (n=$(n), sys_id=$(sys_id), alpha_idx=$(alpha_idx), alpha=$(alpha)):")
    println("  found $(length(candidates)) pass1 rows with flag=\"negative\"")
    println("  ranked by (scrit, ray_id, row_idx):")
    for (k, c) in enumerate(candidates)
        println("    [$k] ray=$(c.ray_id) row=$(c.row_idx) scrit=$(round(c.scrit, digits=6))")
    end

    chosen = candidates[1]
    println("Selected extinction candidate rank [1]: ray=$(chosen.ray_id), row=$(chosen.row_idx), scrit=$(chosen.scrit)")
    return chosen
end

# ----------------------------------- IO ---------------------------------------

function safe_write_arrow(path::AbstractString, df::DataFrame)
    tmp = path * ".tmp"
    Arrow.write(tmp, df)
    mv(tmp, path; force=true)
end

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
end

function slice_arr(arr, ray_id::Int, alpha_idx::Int, pass1_data::Dict{String,Any})
    if ndims(arr) == 3
        return Vector{Float64}(arr[:, ray_id, alpha_idx])
    elseif ndims(arr) == 2
        ray_ids = haskey(pass1_data, "ray_ids") ? Vector{Int}(pass1_data["ray_ids"]) : nothing
        alpha_idx_chunk = haskey(pass1_data, "alpha_idx") ? Int(pass1_data["alpha_idx"]) : nothing
        alpha_idx_chunk !== nothing && alpha_idx_chunk != alpha_idx &&
            error("Chunk alpha_idx=$(alpha_idx_chunk) does not match requested alpha_idx=$(alpha_idx)")
        ray_ids === nothing && error("Chunk missing ray_ids; cannot map ray_id=$(ray_id)")
        local_idx = findfirst(==(ray_id), ray_ids)
        local_idx === nothing && error("ray_id=$(ray_id) not found in chunk")
        return Vector{Float64}(arr[:, local_idx])
    else
        error("Unsupported drcrit dimensions=$(ndims(arr))")
    end
end

function boundary_state_from_pass1(pass1_data::Dict{String,Any}, ray_id::Int, alpha_idx::Int)
    if haskey(pass1_data, "x_boundary")
        return slice_arr(pass1_data["x_boundary"], ray_id, alpha_idx, pass1_data)
    elseif haskey(pass1_data, "x_minus")
        return slice_arr(pass1_data["x_minus"], ray_id, alpha_idx, pass1_data)
    else
        error("pass1 jld2 missing x_boundary/x_minus arrays.")
    end
end

# ------------------------------- ODE trajectory --------------------------------

function integrate_to_endpoint(A::AbstractMatrix{<:Real},
                               B::Array{<:Real,3},
                               r0::AbstractVector{<:Real},
                               u::AbstractVector{<:Real},
                               alpha::Real,
                               delta::Real,
                               x0::AbstractVector{<:Real},
                               dyn::Dict{String,Any},
                               tol_neg::Real)
    r_eff = r0 .+ delta .* u
    f! = make_glvhoi_rhs(A, B, r_eff, alpha)
    prob = ODEProblem(f!, Vector{Float64}(x0), dyn["tspan"])
    sol = if dyn["saveat"] === nothing
        DifferentialEquations.solve(prob, Tsit5(); reltol=dyn["reltol"], abstol=dyn["abstol"])
    else
        DifferentialEquations.solve(prob, Tsit5(); reltol=dyn["reltol"], abstol=dyn["abstol"], saveat=dyn["saveat"])
    end
    SciMLBase.successful_retcode(sol) || error("ODE solve failed at delta=$(delta), retcode=$(sol.retcode)")
    x_end = Vector{Float64}(sol.u[end])
    clamp_small_negatives!(x_end, tol_neg)
    return x_end
end

function build_ode_seed(x_good_full::AbstractVector{<:Real},
                        inactive_idx::Vector{Int};
                        seed_floor::Float64=1e-3)
    x_seed = max.(Vector{Float64}(x_good_full), 0.0)
    n = length(x_seed)
    inactive_mask = falses(n)
    inactive_mask[inactive_idx] .= true
    active_idx = findall(x -> !x, inactive_mask)
    min_active = isempty(active_idx) ? 0.0 : minimum(x_seed[active_idx])
    seed_amp = max(0.01 * min_active, seed_floor)
    for idx in inactive_idx
        x_seed[idx] += seed_amp
    end
    return x_seed
end

# ---------------------------------- main --------------------------------------

function usage_text()
    return """
    Build a tidy hysteresis table for panel plotting.

    Usage:
      julia --startup-file=no new_code/figures/build_hysteresis_table.jl [options] [config_path]

    Options:
      --config PATH         JSON config path (default: config_mini.json)
      --model-id ID         Override model ID used in output rows/path fallback
      --results-root PATH   Legacy results model root (contains n_*/sys_*_pass*.arrow)
      --bank-root PATH      Legacy bank model root (contains n_*/sys_*_bank.jld2)
      --output-dir PATH     Output folder for *.arrow/*.csv (default: data/figure_inputs/fig1_hysteresis, relative to repo root)
      --help, -h            Show this message and exit

    Notes:
      - --results-root and --bank-root must be provided together.
      - If neither is provided, roots are derived from model_id:
          results_boundary_dynamics/model_<model_id>
          parameters_bank/model_<model_id>
    """
end

function parse_args(args::Vector{String})
    cfg_path = "config_mini.json"
    model_id_override = nothing
    results_root_override = nothing
    bank_root_override = nothing
    output_dir = normpath(joinpath(@__DIR__, "..", "data", "figure_inputs", "fig1_hysteresis"))
    show_help = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            show_help = true
            i += 1
        elseif arg == "--config" && i < length(args)
            cfg_path = args[i + 1]
            i += 2
        elseif arg == "--model-id" && i < length(args)
            model_id_override = String(args[i + 1])
            i += 2
        elseif arg == "--results-root" && i < length(args)
            results_root_override = String(args[i + 1])
            i += 2
        elseif arg == "--bank-root" && i < length(args)
            bank_root_override = String(args[i + 1])
            i += 2
        elseif arg == "--output-dir" && i < length(args)
            output_dir = String(args[i + 1])
            i += 2
        elseif arg == "--config" || arg == "--model-id" || arg == "--results-root" ||
               arg == "--bank-root" || arg == "--output-dir"
            error("Missing value for flag: $arg")
        elseif startswith(arg, "--")
            error("Unknown flag: $arg")
        elseif cfg_path == "config_mini.json"
            cfg_path = arg
            i += 1
        else
            error("Unexpected argument: $arg")
        end
    end

    has_results = results_root_override !== nothing
    has_bank = bank_root_override !== nothing
    has_results == has_bank ||
        error("Flags --results-root and --bank-root must be provided together.")

    return (
        cfg_path=cfg_path,
        model_id_override=model_id_override,
        results_root_override=results_root_override,
        bank_root_override=bank_root_override,
        output_dir=output_dir,
        show_help=show_help,
    )
end

function main()
    opts = parse_args(ARGS)
    if opts.show_help
        println(usage_text())
        return
    end

    cfg = normalize_config(to_dict(JSON3.read(read(opts.cfg_path, String))))
    use_explicit_roots = opts.results_root_override !== nothing

    model_id = ""
    results_root = ""
    bank_root = ""
    if use_explicit_roots
        results_root = String(opts.results_root_override)
        bank_root = String(opts.bank_root_override)
        inferred_model_id = let base = basename(normpath(results_root))
            startswith(base, "model_") ? base[7:end] : base
        end
        model_id = opts.model_id_override === nothing ? inferred_model_id : String(opts.model_id_override)
    else
        model_id = opts.model_id_override === nothing ? model_id_from_config(cfg) : String(opts.model_id_override)
        results_root = joinpath("results_boundary_dynamics", "model_$(model_id)")
        bank_root = joinpath("parameters_bank", "model_$(model_id)")
    end

    output_dir = String(opts.output_dir)
    isdir(results_root) || error("Missing results directory: $results_root")
    isdir(bank_root) || error("Missing bank directory: $bank_root")

    println("build_hysteresis_table")
    println("  config:   $(opts.cfg_path)")
    println("  model_id: $(model_id)")
    println("  results_root: $(results_root)")
    println("  bank_root:    $(bank_root)")
    println("  output_dir:   $(output_dir)")

    candidate = select_candidate(results_root)
    n = candidate.n
    sys_id = candidate.sys_id
    row_idx = candidate.row_idx
    alpha_idx = candidate.alpha_idx
    ray_id = candidate.ray_id
    alpha = candidate.alpha
    hc_event = candidate.hc_event
    reversal_frac = candidate.reversal_frac
    delta_event = candidate.delta_event
    delta_post = candidate.delta_post
    x_post = candidate.x_post

    n_dir_res = joinpath(results_root, "n_$(n)")
    pass1_arrow = joinpath(n_dir_res, "sys_$(sys_id)_pass1.arrow")
    pass1_jld = joinpath(n_dir_res, "sys_$(sys_id)_pass1.jld2")
    pass2_arrow = joinpath(n_dir_res, "sys_$(sys_id)_pass2.arrow")
    bank_jld = joinpath(bank_root, "n_$(n)", "sys_$(sys_id)_bank.jld2")
    isfile(pass1_arrow) || error("Missing pass1 arrow: $pass1_arrow")
    isfile(pass1_jld) || error("Missing pass1 jld2: $pass1_jld")
    isfile(pass2_arrow) || error("Missing pass2 arrow: $pass2_arrow")
    isfile(bank_jld) || error("Missing bank file: $bank_jld")

    pass1_df = DataFrame(Arrow.Table(pass1_arrow))
    pass1_data = JLD2.load(pass1_jld)
    pass2_df = candidate.pass2_df
    bank = JLD2.load(bank_jld)

    1 <= row_idx <= nrow(pass1_df) || error("row_idx $(row_idx) out of bounds for pass1.")
    1 <= row_idx <= nrow(pass2_df) || error("row_idx $(row_idx) out of bounds for pass2.")

    row1 = pass1_df[row_idx, :]
    row2 = pass2_df[row_idx, :]
    if to_int_or_default(getproperty(row1, :alpha_idx), -1) != alpha_idx ||
       to_int_or_default(getproperty(row1, :ray_id), -1) != ray_id
        error("pass1 row key mismatch for selected candidate.")
    end
    if to_int_or_default(getproperty(row2, :alpha_idx), -1) != alpha_idx ||
       to_int_or_default(getproperty(row2, :ray_id), -1) != ray_id
        error("pass2 row key mismatch for selected candidate.")
    end

    scrit = to_float_or_nan(getproperty(row1, :scrit))
    isfinite(scrit) && scrit > 0.0 || error("Invalid scrit=$(scrit) in selected pass1 row.")
    delta_pre = 0.999 * scrit
    if !isfinite(delta_event)
        error("Selected pass4 row has non-finite delta_event.")
    end
    if delta_event > delta_post + 1e-10
        error("delta_event=$(delta_event) exceeds delta_post=$(delta_post).")
    end
    delta_event = clamp(delta_event, 0.0, delta_post)

    drcrit_arr = pass1_data["drcrit"]
    drcrit = slice_arr(drcrit_arr, ray_id, alpha_idx, pass1_data)
    u_full = drcrit ./ scrit

    r0 = Vector{Float64}(bank["r0"])
    A = Matrix{Float64}(bank["A"])
    B = Array{Float64,3}(bank["B"])
    n_full = length(r0)
    length(x_post) == n_full || error("x_post length mismatch: got $(length(x_post)), expected $(n_full)")

    dyn = cfg["dynamics"]
    seq = cfg["sequential"]
    back = cfg["backtrack"]

    clamp_small_negatives!(x_post, back["tol_neg"])
    active_idx = support_indices(x_post, back["tol_pos"])
    isempty(active_idx) && error("Selected x_post has empty active support.")
    inactive_idx = setdiff(collect(1:n_full), active_idx)

    A_act, B_act, r0_act, u_act, x_post_act = restrict_params(A, B, r0, u_full, active_idx, x_post)
    syst_full, vars_full = build_system_from_params(r0, A, B)
    syst_act, vars_act = build_system_from_params(r0_act, A_act, B_act)

    println("Selected candidate key: (n=$(n), sys_id=$(sys_id), row_idx=$(row_idx), alpha_idx=$(alpha_idx), ray_id=$(ray_id))")
    println("  alpha=$(alpha), scrit=$(scrit), delta_pre=$(delta_pre), delta_post=$(delta_post), delta_event=$(delta_event)")
    println("  active_start=$(length(active_idx)), inactive_start=$(length(inactive_idx)), hc_event=$(hc_event), reversal_frac=$(reversal_frac)")

    # A1) Pre-fold two branches.
    p_zero_full = vcat(zeros(n_full), Float64(alpha))
    p_pre_full = vcat(delta_pre .* u_full, Float64(alpha))

    H_pre = ParameterHomotopy(syst_full; start_parameters=p_zero_full, target_parameters=p_pre_full)
    T_pre = Tracker(H_pre)
    res_pre = track(T_pre, ComplexF64.(ones(n_full)), 1.0, 0.0)
    is_success(res_pre) || error("Failed to track baseline equilibrium to delta_pre; return_code=$(res_pre.return_code)")
    eq_pre = ComplexF64.(solution(res_pre))

    Ft_pre = System(syst_full(vars_full, p_pre_full), variables=vars_full)
    rng = MersenneTwister(1)
    N_alt = newton_with_retries(Ft_pre, eq_pre, rng; min_distinct_distance=1e-4)
    eq_alt = ComplexF64.(solution(N_alt))

    pre1_deltas, pre1_x = sample_branch_between_deltas(
        syst_full, vars_full, eq_pre, p_pre_full, p_zero_full, u_full; n_points=100,
    )
    pre2_deltas, pre2_x = sample_branch_between_deltas(
        syst_full, vars_full, eq_alt, p_pre_full, p_zero_full, u_full; n_points=100,
    )

    # A2) Post-fold realized branch.
    p_post_act = vcat(delta_post .* u_act, Float64(alpha))
    delta_cap = 1.5 * scrit
    p_cap_act = vcat(delta_cap .* u_act, Float64(alpha))

    out_forward = findparscrit_recursive_event(
        syst_act, A_act, B_act, r0, A, B, alpha,
        u_act, u_full, active_idx, inactive_idx,
        x_post_act, p_post_act, p_cap_act;
        max_step_ratio=back["max_step_ratio"],
        check_stability=back["check_stability"],
        check_invasibility=back["check_invasibility"],
        lambda_tol=back["lambda_tol"],
        invasion_tol=back["invasion_tol"],
    )
    event_forward = String(out_forward.flag)
    delta_stop_forward = delta_from_dr(out_forward.pars_crit[1:length(u_act)], u_act)
    delta_stop_forward = clamp(delta_stop_forward, min(delta_post, delta_cap), max(delta_post, delta_cap))
    p_stop_forward = vcat(delta_stop_forward .* u_act, Float64(alpha))

    postf_deltas_act, postf_x_act = sample_branch_between_deltas(
        syst_act, vars_act, x_post_act, p_post_act, p_stop_forward, u_act;
        n_points=100, endpoint_override=out_forward.xstar_crit,
    )

    p_event_act = vcat(delta_event .* u_act, Float64(alpha))
    postb_deltas_act, postb_x_act = sample_branch_between_deltas(
        syst_act, vars_act, x_post_act, p_post_act, p_event_act, u_act;
        n_points=100,
    )

    postf_x = Matrix{Float64}(undef, n_full, size(postf_x_act, 2))
    postb_x = Matrix{Float64}(undef, n_full, size(postb_x_act, 2))
    for j in 1:size(postf_x_act, 2)
        postf_x[:, j] .= lift_active_state(real.(postf_x_act[:, j]), active_idx, n_full)
    end
    for j in 1:size(postb_x_act, 2)
        postb_x[:, j] .= lift_active_state(real.(postb_x_act[:, j]), active_idx, n_full)
    end

    println("Algebraic ranges:")
    println("  pre_fold_1: [$(minimum(pre1_deltas)), $(maximum(pre1_deltas))], points=$(length(pre1_deltas))")
    println("  pre_fold_2: [$(minimum(pre2_deltas)), $(maximum(pre2_deltas))], points=$(length(pre2_deltas))")
    println("  post_forward: [$(minimum(postf_deltas_act)), $(maximum(postf_deltas_act))], points=$(length(postf_deltas_act)), stop_event=$(event_forward)")
    println("  post_backward: [$(minimum(postb_deltas_act)), $(maximum(postb_deltas_act))], points=$(length(postb_deltas_act))")

    # B) Dynamic trajectories.
    delta_max_dyn = maximum(vcat(pre1_deltas, pre2_deltas, postf_deltas_act, postb_deltas_act))
    n_dyn_points = 15
    delta_grid_dyn = collect(range(0.0, delta_max_dyn, length=n_dyn_points))
    println("Dynamic delta range: [0.0, $(delta_max_dyn)] with $(length(delta_grid_dyn)) points.")

    x_fwd = Matrix{Float64}(undef, n_full, length(delta_grid_dyn))
    x_curr = ones(Float64, n_full)
    for (j, delta) in enumerate(delta_grid_dyn)
        x_curr = integrate_to_endpoint(A, B, r0, u_full, alpha, delta, x_curr, dyn, back["tol_neg"])
        x_fwd[:, j] .= x_curr
    end

    seed_floor = max(back["eps_seed_extinct"], 10.0 * dyn["eps_extinct"])
    delta_grid_bwd = reverse(delta_grid_dyn)
    x_bwd = Matrix{Float64}(undef, n_full, length(delta_grid_bwd))
    x_curr = copy(x_fwd[:, end])
    for (j, delta) in enumerate(delta_grid_bwd)
        absent_idx = findall(x_curr .<= dyn["eps_extinct"])
        x_seed = build_ode_seed(x_curr, absent_idx; seed_floor=seed_floor)
        x_curr = integrate_to_endpoint(A, B, r0, u_full, alpha, delta, x_seed, dyn, back["tol_neg"])
        x_bwd[:, j] .= x_curr
    end

    # Build tidy table.
    rows = NamedTuple[]

    function push_algebraic_rows!(branch_id::String, deltas::Vector{Float64}, x_mat::Matrix{<:Real};
                                  boundary_type::String="fold",
                                  row_idx_local::Int=row_idx,
                                  ray_id_local::Int=ray_id,
                                  scrit_local::Float64=Float64(scrit),
                                  delta_post_local::Float64=Float64(delta_post),
                                  delta_event_local::Float64=Float64(delta_event),
                                  hc_event_local::String=String(hc_event),
                                  reversal_frac_local::Float64=Float64(reversal_frac))
        for j in eachindex(deltas)
            delta = Float64(deltas[j])
            x = Vector{Float64}(real.(x_mat[:, j]))
            clamp_small_negatives!(x, back["tol_neg"])
            stab = stability_at_equilibrium(A, B, x, alpha; tol_pos=seq["tol_pos"], lambda_tol=seq["lambda_tol"])
            for species_id in 1:n_full
                push!(rows, (
                    model_id=model_id,
                    n=n,
                    sys_id=sys_id,
                    row_idx=row_idx_local,
                    alpha=Float64(alpha),
                    alpha_idx=alpha_idx,
                    ray_id=ray_id_local,
                    delta=delta,
                    species_id=species_id,
                    abundance=Float64(x[species_id]),
                    source_type="algebraic",
                    boundary_type=boundary_type,
                    pass_direction="none",
                    branch_id=branch_id,
                    is_stable=Bool(stab.is_stable),
                    lambda_max_real=Float64(stab.lambda_max_real),
                    scrit=Float64(scrit_local),
                    delta_post=Float64(delta_post_local),
                    delta_event=Float64(delta_event_local),
                    hc_event=String(hc_event_local),
                    reversal_frac=Float64(reversal_frac_local),
                ))
            end
        end
    end

    function push_dynamic_rows!(pass_direction::String, deltas::Vector{Float64}, x_mat::Matrix{Float64};
                                boundary_type::String="fold",
                                row_idx_local::Int=row_idx,
                                ray_id_local::Int=ray_id,
                                scrit_local::Float64=Float64(scrit),
                                delta_post_local::Float64=Float64(delta_post),
                                delta_event_local::Float64=Float64(delta_event),
                                hc_event_local::String=String(hc_event),
                                reversal_frac_local::Float64=Float64(reversal_frac))
        for j in eachindex(deltas)
            delta = Float64(deltas[j])
            x = Vector{Float64}(x_mat[:, j])
            for species_id in 1:n_full
                push!(rows, (
                    model_id=model_id,
                    n=n,
                    sys_id=sys_id,
                    row_idx=row_idx_local,
                    alpha=Float64(alpha),
                    alpha_idx=alpha_idx,
                    ray_id=ray_id_local,
                    delta=delta,
                    species_id=species_id,
                    abundance=Float64(x[species_id]),
                    source_type="dynamic",
                    boundary_type=boundary_type,
                    pass_direction=pass_direction,
                    branch_id=missing,
                    is_stable=missing,
                    lambda_max_real=missing,
                    scrit=Float64(scrit_local),
                    delta_post=Float64(delta_post_local),
                    delta_event=Float64(delta_event_local),
                    hc_event=String(hc_event_local),
                    reversal_frac=Float64(reversal_frac_local),
                ))
            end
        end
    end

    push_algebraic_rows!("pre_fold_1", pre1_deltas, real.(pre1_x))
    push_algebraic_rows!("pre_fold_2", pre2_deltas, real.(pre2_x))
    push_algebraic_rows!("post_forward", postf_deltas_act, postf_x)
    push_algebraic_rows!("post_backward", postb_deltas_act, postb_x)
    push_dynamic_rows!("forward", delta_grid_dyn, x_fwd)
    push_dynamic_rows!("backward", delta_grid_bwd, x_bwd)

    # C) Extinction-boundary reduced continuation for the same (n, sys_id, alpha).
    ext_candidate = select_extinction_candidate(pass1_df, n, sys_id, alpha_idx, Float64(alpha))
    ext_row_idx = ext_candidate.row_idx
    ext_ray_id = ext_candidate.ray_id
    delta_crit_ext = Float64(ext_candidate.scrit)
    ext_ray_id > 0 || error("Invalid extinction candidate ray_id=$(ext_ray_id).")

    drcrit_ext = slice_arr(drcrit_arr, ext_ray_id, alpha_idx, pass1_data)
    u_ext_full = drcrit_ext ./ delta_crit_ext
    all(isfinite.(u_ext_full)) || error("Extinction direction has non-finite entries for ray=$(ext_ray_id).")

    # Full-system branch from baseline (x=1, delta=0) up to the extinction scrit along u_ext_full.
    p_crit_ext_full = vcat(delta_crit_ext .* u_ext_full, Float64(alpha))
    extpre_deltas, extpre_x = sample_branch_between_deltas(
        syst_full, vars_full, ComplexF64.(ones(n_full)), p_zero_full, p_crit_ext_full, u_ext_full; n_points=100,
    )

    x_boundary_full = boundary_state_from_pass1(pass1_data, ext_ray_id, alpha_idx)
    all(isfinite.(x_boundary_full)) || error("Extinction boundary state has non-finite entries for ray=$(ext_ray_id).")
    clamp_small_negatives!(x_boundary_full, back["tol_neg"])

    extinct_idx = findall(x_boundary_full .<= seq["tol_pos"])
    active_idx_ext = setdiff(collect(1:n_full), extinct_idx)
    isempty(extinct_idx) && error("Selected extinction candidate has no extinct species at delta_crit.")
    isempty(active_idx_ext) && error("Selected extinction candidate has empty active support.")

    A_ext, B_ext, r0_ext, u_ext, x_ext_guess = restrict_params(
        A, B, r0, u_ext_full, active_idx_ext, x_boundary_full,
    )
    syst_ext, vars_ext = build_system_from_params(r0_ext, A_ext, B_ext)

    p_crit_ext = vcat(delta_crit_ext .* u_ext, Float64(alpha))
    Ft_ext = System(syst_ext(vars_ext, p_crit_ext), variables=vars_ext)
    N_ext = newton(Ft_ext, ComplexF64.(x_ext_guess))
    is_success(N_ext) ||
        error("Reduced Newton correction failed at extinction delta_crit=$(delta_crit_ext), return_code=$(N_ext.return_code)")
    x_crit_ext = ComplexF64.(solution(N_ext))

    delta_cap_ext = max(delta_crit_ext, min(Float64(cfg["max_pert"]), 1.5 * delta_crit_ext))
    p_ext_fwd = vcat(delta_cap_ext .* u_ext, Float64(alpha))

    extf_deltas_act, extf_x_act = sample_branch_between_deltas(
        syst_ext, vars_ext, x_crit_ext, p_crit_ext, p_ext_fwd, u_ext; n_points=100,
    )
    x_cap_ext = ComplexF64.(extf_x_act[:, end])
    extb_deltas_act, extb_x_act = sample_branch_between_deltas(
        syst_ext, vars_ext, x_cap_ext, p_ext_fwd, p_crit_ext, u_ext; n_points=100,
    )

    extf_x = Matrix{Float64}(undef, n_full, size(extf_x_act, 2))
    extb_x = Matrix{Float64}(undef, n_full, size(extb_x_act, 2))
    for j in 1:size(extf_x_act, 2)
        extf_x[:, j] .= lift_active_state(real.(extf_x_act[:, j]), active_idx_ext, n_full)
    end
    for j in 1:size(extb_x_act, 2)
        extb_x[:, j] .= lift_active_state(real.(extb_x_act[:, j]), active_idx_ext, n_full)
    end

    println("Extinction reduced continuation:")
    println("  selected key: row=$(ext_row_idx), ray=$(ext_ray_id), delta_crit=$(delta_crit_ext)")
    println("  support at delta_crit: active=$(length(active_idx_ext)), extinct=$(length(extinct_idx))")
    println("  extinction_full_to_scrit: [$(minimum(extpre_deltas)), $(maximum(extpre_deltas))], points=$(length(extpre_deltas))")
    println("  extinction_forward: [$(minimum(extf_deltas_act)), $(maximum(extf_deltas_act))], points=$(length(extf_deltas_act))")
    println("  extinction_backward: [$(minimum(extb_deltas_act)), $(maximum(extb_deltas_act))], points=$(length(extb_deltas_act))")

    # Dynamic trajectories for extinction boundary over [0, max(extinction backward branch)].
    n_dyn_ext = length(delta_grid_dyn)
    delta_ext_min = 0.0
    delta_ext_max = maximum(extb_deltas_act)
    delta_grid_ext_fwd = collect(range(delta_ext_min, delta_ext_max, length=n_dyn_ext))
    delta_grid_ext_bwd = reverse(delta_grid_ext_fwd)
    println("Extinction dynamic delta range: [$(delta_ext_min), $(delta_ext_max)] with $(n_dyn_ext) points.")

    x_fwd_ext = Matrix{Float64}(undef, n_full, length(delta_grid_ext_fwd))
    x_curr = ones(Float64, n_full)
    for (j, delta) in enumerate(delta_grid_ext_fwd)
        x_curr = integrate_to_endpoint(A, B, r0, u_ext_full, alpha, delta, x_curr, dyn, back["tol_neg"])
        x_fwd_ext[:, j] .= x_curr
    end

    x_bwd_ext = Matrix{Float64}(undef, n_full, length(delta_grid_ext_bwd))
    x_curr = copy(x_fwd_ext[:, end])
    for (j, delta) in enumerate(delta_grid_ext_bwd)
        absent_idx = findall(x_curr .<= dyn["eps_extinct"])
        x_seed = build_ode_seed(x_curr, absent_idx; seed_floor=seed_floor)
        x_curr = integrate_to_endpoint(A, B, r0, u_ext_full, alpha, delta, x_seed, dyn, back["tol_neg"])
        x_bwd_ext[:, j] .= x_curr
    end

    push_algebraic_rows!(
        "extinction_forward", extpre_deltas, real.(extpre_x);
        boundary_type="extinction",
        row_idx_local=ext_row_idx,
        ray_id_local=ext_ray_id,
        scrit_local=delta_crit_ext,
        delta_post_local=delta_crit_ext,
        delta_event_local=delta_crit_ext,
        hc_event_local="negative",
        reversal_frac_local=NaN,
    )
    push_algebraic_rows!(
        "extinction_forward", extf_deltas_act, extf_x;
        boundary_type="extinction",
        row_idx_local=ext_row_idx,
        ray_id_local=ext_ray_id,
        scrit_local=delta_crit_ext,
        delta_post_local=delta_crit_ext,
        delta_event_local=delta_crit_ext,
        hc_event_local="negative",
        reversal_frac_local=NaN,
    )
    push_algebraic_rows!(
        "extinction_backward", extb_deltas_act, extb_x;
        boundary_type="extinction",
        row_idx_local=ext_row_idx,
        ray_id_local=ext_ray_id,
        scrit_local=delta_crit_ext,
        delta_post_local=delta_crit_ext,
        delta_event_local=delta_crit_ext,
        hc_event_local="negative",
        reversal_frac_local=NaN,
    )
    push_dynamic_rows!(
        "forward", delta_grid_ext_fwd, x_fwd_ext;
        boundary_type="extinction",
        row_idx_local=ext_row_idx,
        ray_id_local=ext_ray_id,
        scrit_local=delta_crit_ext,
        delta_post_local=delta_crit_ext,
        delta_event_local=delta_crit_ext,
        hc_event_local="negative",
        reversal_frac_local=NaN,
    )
    push_dynamic_rows!(
        "backward", delta_grid_ext_bwd, x_bwd_ext;
        boundary_type="extinction",
        row_idx_local=ext_row_idx,
        ray_id_local=ext_ray_id,
        scrit_local=delta_crit_ext,
        delta_post_local=delta_crit_ext,
        delta_event_local=delta_crit_ext,
        hc_event_local="negative",
        reversal_frac_local=NaN,
    )

    df_out = DataFrame(rows)

    out_dir = output_dir
    ensure_dir(out_dir)
    stem = "sys_$(sys_id)_alphaidx_$(alpha_idx)_ray_$(ray_id)_row_$(row_idx)_hysteresis_table"
    arrow_path = joinpath(out_dir, stem * ".arrow")
    csv_path = joinpath(out_dir, stem * ".csv")

    safe_write_arrow(arrow_path, df_out)
    CSV.write(csv_path, df_out)

    println("Wrote hysteresis table:")
    println("  Arrow: $arrow_path")
    println("  CSV:   $csv_path")
    println("  rows:  $(nrow(df_out))")
end

main()
