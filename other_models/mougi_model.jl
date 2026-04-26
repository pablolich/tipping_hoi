# Utilities for the Mougi-style ecosystem-engineering food-web model.
# Used by generate_bank_mougi.jl (bank generation) and glvhoi_utils.jl (HC dispatch).
#
# Species ordering: arbitrary species ids 1:n.
# Directed trophic links use a[i, j] > 0 to mean species i consumes species j.
# Perturbations act on the baseline intrinsic growth rates r0 via r0 + dr.

using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using SciMLBase
using HomotopyContinuation

if !@isdefined(LeverOriginalParams)
    include(joinpath(@__DIR__, "lever_model.jl"))
end

struct MougiParams
    n::Int
    connectance::Float64
    topology_mode::String
    r0::Vector{Float64}
    s::Vector{Float64}
    e::Matrix{Float64}
    a::Matrix{Float64}
    h::Matrix{Float64}
    E0::Vector{Float64}
    engineer_mask::Vector{Bool}
    receiver_mask::Vector{Bool}
    betaE::Matrix{Float64}
    gammaE::Matrix{Float64}
    q_r::Float64
    q_a::Float64
end

@inline mougi_n_species(p::MougiParams) = p.n

function _dict_or_prop_get_with_default(x, key::AbstractString, default)
    if x isa AbstractDict
        if haskey(x, key)
            return x[key]
        elseif haskey(x, Symbol(key))
            return x[Symbol(key)]
        else
            return default
        end
    end
    sym = Symbol(key)
    return hasproperty(x, sym) ? getproperty(x, sym) : default
end

function mougi_params_payload(p::MougiParams)
    return (
        mougi_n              = p.n,
        mougi_connectance    = p.connectance,
        mougi_topology_mode  = p.topology_mode,
        mougi_r0             = copy(p.r0),
        mougi_s              = copy(p.s),
        mougi_e              = [collect(@view p.e[i, :]) for i in 1:p.n],
        mougi_a              = [collect(@view p.a[i, :]) for i in 1:p.n],
        mougi_h              = [collect(@view p.h[i, :]) for i in 1:p.n],
        mougi_E0             = copy(p.E0),
        mougi_engineer_mask  = collect(p.engineer_mask),
        mougi_receiver_mask  = collect(p.receiver_mask),
        mougi_betaE          = [collect(@view p.betaE[i, :]) for i in 1:p.n],
        mougi_gammaE         = [collect(@view p.gammaE[i, :]) for i in 1:p.n],
        mougi_q_r            = p.q_r,
        mougi_q_a            = p.q_a,
    )
end

function mougi_params_from_payload(payload)
    n = Int(_dict_or_prop_get(payload, "mougi_n"))
    return MougiParams(
        n,
        Float64(_dict_or_prop_get(payload, "mougi_connectance")),
        String(_dict_or_prop_get_with_default(payload, "mougi_topology_mode", "random")),
        Vector{Float64}(_dict_or_prop_get(payload, "mougi_r0")),
        Vector{Float64}(_dict_or_prop_get(payload, "mougi_s")),
        _to_matrix_f64(_dict_or_prop_get(payload, "mougi_e")),
        _to_matrix_f64(_dict_or_prop_get(payload, "mougi_a")),
        _to_matrix_f64(_dict_or_prop_get(payload, "mougi_h")),
        Vector{Float64}(_dict_or_prop_get(payload, "mougi_E0")),
        Bool.(_dict_or_prop_get(payload, "mougi_engineer_mask")),
        Bool.(_dict_or_prop_get(payload, "mougi_receiver_mask")),
        _to_matrix_f64(_dict_or_prop_get(payload, "mougi_betaE")),
        _to_matrix_f64(_dict_or_prop_get(payload, "mougi_gammaE")),
        Float64(_dict_or_prop_get(payload, "mougi_q_r")),
        Float64(_dict_or_prop_get(payload, "mougi_q_a")),
    )
end

function _sample_mask(n::Int, count::Int; rng::AbstractRNG)
    count <= n || error("mask count=$(count) exceeds n=$(n)")
    mask = falses(n)
    count == 0 && return collect(mask)
    idx = randperm(rng, n)[1:count]
    mask[idx] .= true
    return collect(mask)
end

function _sample_engineer_receiver_counts(n::Int; rng::AbstractRNG)
    valid = Tuple{Int, Int}[]
    for n_engineers in 1:n, n_receivers in 1:n
        dominance = (n_engineers / n) * (n_receivers / n)
        if 0.08 <= dominance <= 0.20
            push!(valid, (n_engineers, n_receivers))
        end
    end
    isempty(valid) && error("No valid engineer/receiver count pairs for n=$(n)")
    return rand(rng, valid)
end

function _sample_random_food_web(n::Int, connectance::Float64; rng::AbstractRNG)
    a = zeros(Float64, n, n)
    e = zeros(Float64, n, n)
    h = zeros(Float64, n, n)

    @inbounds for i in 1:(n - 1), j in (i + 1):n
        rand(rng) <= connectance || continue
        if rand(rng) < 0.5
            predator, prey = i, j
        else
            predator, prey = j, i
        end
        a[predator, prey] = rand(rng, Uniform(0.0, 0.01))
        e[predator, prey] = rand(rng, Uniform(0.1, 0.25))
        h[predator, prey] = rand(rng, Uniform(1.0, 10.0))
    end

    return a, e, h
end

"""
    sample_mougi_params(n=6; connectance=1.0, topology_mode="random", rng,
                        q_r_range=(0.1, 0.3), q_a_range=(0.7, 0.9))

Sample a Mougi-style ecosystem-engineering food web using the paper's random-web
construction with the project's persistence-biased defaults.
"""
function sample_mougi_params(n::Int = 6;
                             connectance::Real      = 1.0,
                             topology_mode::String  = "random",
                             rng::AbstractRNG       = Random.default_rng(),
                             q_r_range::Tuple{<:Real, <:Real} = (0.1, 0.3),
                             q_a_range::Tuple{<:Real, <:Real} = (0.7, 0.9))
    topology_mode == "random" || error("Only topology_mode=\"random\" is supported")
    0.0 < connectance <= 1.0 || error("connectance must lie in (0, 1], got $(connectance)")

    r0 = rand(rng, Uniform(0.2, 1.0), n)
    s  = ones(Float64, n)
    E0 = rand(rng, Uniform(0.0, 0.1), n)
    a, e, h = _sample_random_food_web(n, Float64(connectance); rng=rng)

    n_engineers, n_receivers = _sample_engineer_receiver_counts(n; rng=rng)
    engineer_mask = _sample_mask(n, n_engineers; rng=rng)
    receiver_mask = _sample_mask(n, n_receivers; rng=rng)

    q_r = rand(rng, Uniform(Float64(q_r_range[1]), Float64(q_r_range[2])))
    q_a = rand(rng, Uniform(Float64(q_a_range[1]), Float64(q_a_range[2])))

    betaE  = ones(Float64, n, n)
    gammaE = ones(Float64, n, n)

    @inbounds for engineer in 1:n
        engineer_mask[engineer] || continue
        for receiver in 1:n
            receiver_mask[receiver] || continue

            if rand(rng) < q_r
                betaE[engineer, receiver] = rand(rng, Uniform(0.0, 1.0))
            else
                betaE[engineer, receiver] = rand(rng, Uniform(1.0, 2.0))
            end

            if rand(rng) < q_a
                gammaE[engineer, receiver] = rand(rng, Uniform(0.0, 1.0))
            else
                gammaE[engineer, receiver] = rand(rng, Uniform(1.0, 2.0))
            end
        end
    end

    return MougiParams(
        n,
        Float64(connectance),
        topology_mode,
        r0,
        s,
        e,
        a,
        h,
        E0,
        engineer_mask,
        receiver_mask,
        betaE,
        gammaE,
        q_r,
        q_a,
    )
end

function _fill_mougi_rate_terms!(B::Vector{Float64},
                                 G::Vector{Float64},
                                 D::Vector{Float64},
                                 prey_gain::Vector{Float64},
                                 p::MougiParams,
                                 x::AbstractVector{<:Real})
    n = p.n
    fill!(B, 1.0)
    fill!(G, 1.0)
    fill!(D, 1.0)
    fill!(prey_gain, 0.0)

    @inbounds for i in 1:n
        if p.receiver_mask[i]
            for engineer in 1:n
                p.engineer_mask[engineer] || continue
                denom = x[engineer] + p.E0[engineer]
                B[i] += (p.betaE[engineer, i] - 1.0) * x[engineer] / denom
                G[i] += (p.gammaE[engineer, i] - 1.0) * x[engineer] / denom
            end
        end

        for prey in 1:n
            a_ip = p.a[i, prey]
            a_ip == 0.0 && continue
            D[i] += p.h[i, prey] * a_ip * x[prey]
            prey_gain[i] += p.e[i, prey] * a_ip * x[prey]
        end
    end

    return nothing
end

function mougi_engineering_factors(p::MougiParams, x::AbstractVector{<:Real})
    B = ones(Float64, p.n)
    G = ones(Float64, p.n)
    D = ones(Float64, p.n)
    prey_gain = zeros(Float64, p.n)
    _fill_mougi_rate_terms!(B, G, D, prey_gain, p, x)
    return B, G
end

function _make_mougi_rhs_from_dr(p::MougiParams,
                                 dr_full::AbstractVector{<:Real})
    n = p.n
    length(dr_full) == n ||
        error("_make_mougi_rhs_from_dr expected dr_full of length $(n), got $(length(dr_full))")

    B = ones(Float64, n)
    G = ones(Float64, n)
    D = ones(Float64, n)
    prey_gain = zeros(Float64, n)

    function f!(dx, x, _, _)
        _fill_mougi_rate_terms!(B, G, D, prey_gain, p, x)

        @inbounds for i in 1:n
            gain = G[i] * prey_gain[i] / D[i]
            loss = 0.0
            for predator in 1:n
                a_pi = p.a[predator, i]
                a_pi == 0.0 && continue
                loss += G[predator] * a_pi * x[predator] / D[predator]
            end
            fi = (p.r0[i] + dr_full[i]) * B[i] - p.s[i] * x[i] + gain - loss
            dx[i] = x[i] * fi
        end
        return nothing
    end

    return f!
end

function make_mougi_rhs(p::MougiParams,
                        u_full::AbstractVector{<:Real},
                        delta::Real)
    dr = Float64(delta) .* Float64.(u_full)
    return _make_mougi_rhs_from_dr(p, dr)
end

function integrate_mougi_to_steady(p::MougiParams,
                                   x0::AbstractVector{<:Real};
                                   tmax::Real       = 4000.0,
                                   steady_tol::Real = 1e-8)
    n      = p.n
    dr_z   = zeros(Float64, n)
    f!     = _make_mougi_rhs_from_dr(p, dr_z)
    u_init = Float64.(x0)
    prob   = ODEProblem(f!, u_init, (0.0, Float64(tmax)))

    condition(_, _, integrator) = begin
        du = integrator.du
        du === nothing && return false
        return maximum(abs, du) < steady_tol
    end
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!; save_positions=(false, false))

    sol = DifferentialEquations.solve(prob, Tsit5();
        callback       = cb,
        reltol         = 1e-9,
        abstol         = 1e-11,
        save_everystep = false,
        save_start     = false,
    )

    if !SciMLBase.successful_retcode(sol)
        return (success=false, retcode=string(sol.retcode), x_eq=fill(NaN, n), du_max=Inf)
    end

    x_eq = Vector{Float64}(sol.u[end])
    @. x_eq = max(x_eq, 0.0)
    du_end = similar(x_eq)
    f!(du_end, x_eq, nothing, 0.0)
    return (success=true, retcode=string(sol.retcode), x_eq=x_eq, du_max=maximum(abs.(du_end)))
end

function mougi_jacobian_fd(p::MougiParams,
                           x::AbstractVector{<:Real},
                           dr_full::AbstractVector{<:Real})
    n = p.n
    f! = _make_mougi_rhs_from_dr(p, dr_full)
    J  = zeros(Float64, n, n)
    fp = zeros(Float64, n)
    fm = zeros(Float64, n)
    xp = Float64.(x)
    xm = Float64.(x)

    @inbounds for j in 1:n
        copyto!(xp, x)
        copyto!(xm, x)
        step = 1e-6 * max(1.0, abs(x[j]))
        xp[j] += step
        xm[j] = max(0.0, xm[j] - step)

        f!(fp, xp, nothing, 0.0)
        f!(fm, xm, nothing, 0.0)
        denom = xp[j] - xm[j]
        @views J[:, j] .= (fp .- fm) ./ denom
    end

    return J
end

function mougi_jacobian_fd(p::MougiParams, x::AbstractVector{<:Real})
    return mougi_jacobian_fd(p, x, zeros(Float64, p.n))
end

function mougi_lambda_max_equilibrium(p::MougiParams,
                                      x::AbstractVector{<:Real},
                                      dr_full::AbstractVector{<:Real})
    J = mougi_jacobian_fd(p, x, dr_full)
    return maximum(real.(eigvals(J)))
end

function mougi_lambda_max_equilibrium(p::MougiParams,
                                      x::AbstractVector{<:Real})
    return mougi_lambda_max_equilibrium(p, x, zeros(Float64, p.n))
end

function _mougi_engineering_numden(p::MougiParams,
                                   receiver::Int,
                                   coeffs::AbstractMatrix{<:Real},
                                   x)
    if !p.receiver_mask[receiver]
        return 1.0, 1.0
    end

    engineers = [idx for idx in 1:p.n if p.engineer_mask[idx]]
    isempty(engineers) && return 1.0, 1.0

    den = 1.0
    for engineer in engineers
        den *= x[engineer] + p.E0[engineer]
    end

    num = den
    for engineer in engineers
        other = 1.0
        for other_engineer in engineers
            other_engineer == engineer && continue
            other *= x[other_engineer] + p.E0[other_engineer]
        end
        num += (coeffs[engineer, receiver] - 1.0) * x[engineer] * other
    end

    return num, den
end

function _mougi_predator_denominator(p::MougiParams,
                                     predator::Int,
                                     x)
    den = 1.0
    for prey in 1:p.n
        a_ip = p.a[predator, prey]
        a_ip == 0.0 && continue
        den += p.h[predator, prey] * a_ip * x[prey]
    end
    return den
end

function _mougi_prey_gain_sum(p::MougiParams,
                              predator::Int,
                              x)
    gain = 0.0
    for prey in 1:p.n
        a_ip = p.a[predator, prey]
        a_ip == 0.0 && continue
        gain += p.e[predator, prey] * a_ip * x[prey]
    end
    return gain
end

function build_mougi_cleared_system(p::MougiParams)
    n = p.n
    @var x[1:n] dr[1:n]
    eqs = Vector{Expression}(undef, n)

    B_num = Vector{Any}(undef, n)
    G_num = Vector{Any}(undef, n)
    E_den = Vector{Any}(undef, n)
    D_den = Vector{Any}(undef, n)
    prey_sum = Vector{Any}(undef, n)

    for i in 1:n
        B_num[i], E_den[i] = _mougi_engineering_numden(p, i, p.betaE, x)
        G_num[i], _ = _mougi_engineering_numden(p, i, p.gammaE, x)
        D_den[i] = _mougi_predator_denominator(p, i, x)
        prey_sum[i] = _mougi_prey_gain_sum(p, i, x)
    end

    for i in 1:n
        predators = [pred for pred in 1:n if p.a[pred, i] > 0.0]

        pred_total = 1.0
        for pred in predators
            pred_total *= E_den[pred] * D_den[pred]
        end

        self_factor = E_den[i] * D_den[i]
        growth_term = (p.r0[i] + dr[i]) * B_num[i] * D_den[i] * pred_total
        self_term   = -p.s[i] * x[i] * self_factor * pred_total
        gain_term   = G_num[i] * prey_sum[i] * pred_total

        loss_term = 0.0
        for pred in predators
            other_pred_total = 1.0
            for other_pred in predators
                other_pred == pred && continue
                other_pred_total *= E_den[other_pred] * D_den[other_pred]
            end
            loss_term += G_num[pred] * p.a[pred, i] * x[pred] *
                         self_factor * other_pred_total
        end

        eqs[i] = growth_term + self_term + gain_term - loss_term
    end

    return System(eqs; variables=x, parameters=dr), x
end

"""
    mougi_alpha_eff_symbolic(p, x_star, dr_full=zeros(n)) → Float64

Compute `alpha_eff` from the denominator-cleared Mougi HC system using the
shared symbolic grouped metric.
"""
function mougi_alpha_eff_symbolic(p::MougiParams,
                                  x_star::AbstractVector{<:Real},
                                  dr_full::AbstractVector{<:Real})
    length(dr_full) == p.n ||
        error("mougi_alpha_eff_symbolic expected dr_full of length $(p.n), got $(length(dr_full))")
    syst, _ = build_mougi_cleared_system(p)
    return symbolic_alpha_eff(syst, x_star; parameter_values=Float64.(dr_full)).alpha_eff
end

function mougi_alpha_eff_symbolic(p::MougiParams,
                                  x_star::AbstractVector{<:Real})
    return mougi_alpha_eff_symbolic(p, x_star, zeros(Float64, p.n))
end
