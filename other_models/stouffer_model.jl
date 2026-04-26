using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using SciMLBase
using HomotopyContinuation

if !@isdefined(LeverOriginalParams)
    include(joinpath(@__DIR__, "lever_model.jl"))
end

const STOUFFER_LOGRATIO_MEAN = 6.1
const STOUFFER_LOGRATIO_SD = 5.75
const STOUFFER_WEIGHT_LOGMEAN = -3.0
const STOUFFER_WEIGHT_LOGSD = 1.5
const STOUFFER_ASSIMILATION = 0.85
const STOUFFER_DEFAULT_B0 = 0.5
const STOUFFER_DEFAULT_K = 1.0
const STOUFFER_DEFAULT_Mb = 1.0
const STOUFFER_DEFAULT_AR = 1.0
const STOUFFER_DEFAULT_AX = 0.314
const STOUFFER_DEFAULT_AY = 8.0 * STOUFFER_DEFAULT_AX

struct StoufferParams
    n::Int
    connectance::Float64
    adj::Matrix{Bool}
    basal_mask::Vector{Bool}
    prey_lists::Vector{Vector{Int}}
    pred_lists::Vector{Vector{Int}}
    w::Matrix{Float64}
    M::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    e::Matrix{Float64}
    K::Float64
    B0::Float64
    Mb::Float64
    ar::Float64
    ax::Float64
    ay::Float64
end

function _stouffer_matrix_rows(M::AbstractMatrix)
    return [collect(@view M[i, :]) for i in 1:size(M, 1)]
end

function _to_matrix_bool(x)
    if x isa AbstractMatrix
        return Matrix{Bool}(x)
    end

    rows = x
    m = length(rows)
    m == 0 && return falses(0, 0)
    n = length(rows[1])
    M = Matrix{Bool}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = Bool(rows[i][j])
    end
    return M
end

function _stouffer_get_with_default(x, key::AbstractString, default)
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

function _stouffer_prey_lists(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    return [findall(@view adj[i, :]) for i in 1:n]
end

function _stouffer_pred_lists(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    return [findall(@view adj[:, i]) for i in 1:n]
end

function StoufferParams(connectance::Real,
                        adj::AbstractMatrix{Bool},
                        basal_mask::AbstractVector{Bool},
                        w::AbstractMatrix{<:Real},
                        M::AbstractVector{<:Real},
                        x::AbstractVector{<:Real},
                        y::AbstractVector{<:Real},
                        e::AbstractMatrix{<:Real};
                        K::Real = STOUFFER_DEFAULT_K,
                        B0::Real = STOUFFER_DEFAULT_B0,
                        Mb::Real = STOUFFER_DEFAULT_Mb,
                        ar::Real = STOUFFER_DEFAULT_AR,
                        ax::Real = STOUFFER_DEFAULT_AX,
                        ay::Real = STOUFFER_DEFAULT_AY)
    n = size(adj, 1)
    size(adj, 2) == n || error("Stouffer adjacency must be square")
    length(basal_mask) == n || error("basal_mask length mismatch")
    size(w) == (n, n) || error("w size mismatch")
    size(e) == (n, n) || error("e size mismatch")
    length(M) == n || error("M length mismatch")
    length(x) == n || error("x length mismatch")
    length(y) == n || error("y length mismatch")

    adj_m = Matrix{Bool}(adj)
    prey_lists = _stouffer_prey_lists(adj_m)
    pred_lists = _stouffer_pred_lists(adj_m)

    return StoufferParams(
        n,
        Float64(connectance),
        adj_m,
        Bool.(basal_mask),
        prey_lists,
        pred_lists,
        Matrix{Float64}(w),
        Float64.(M),
        Float64.(x),
        Float64.(y),
        Matrix{Float64}(e),
        Float64(K),
        Float64(B0),
        Float64(Mb),
        Float64(ar),
        Float64(ax),
        Float64(ay),
    )
end

@inline stouffer_n_species(p::StoufferParams) = p.n
@inline stouffer_is_consumer(p::StoufferParams, i::Int) = !p.basal_mask[i]

function stouffer_params_payload(p::StoufferParams)
    return (
        stouffer_n           = p.n,
        stouffer_connectance = p.connectance,
        stouffer_adj         = _stouffer_matrix_rows(p.adj),
        stouffer_basal_mask  = collect(p.basal_mask),
        stouffer_w           = _stouffer_matrix_rows(p.w),
        stouffer_M           = copy(p.M),
        stouffer_x           = copy(p.x),
        stouffer_y           = copy(p.y),
        stouffer_e           = _stouffer_matrix_rows(p.e),
        stouffer_K           = p.K,
        stouffer_B0          = p.B0,
        stouffer_Mb          = p.Mb,
        stouffer_ar          = p.ar,
        stouffer_ax          = p.ax,
        stouffer_ay          = p.ay,
    )
end

function stouffer_params_from_payload(payload)
    n_raw = _stouffer_get_with_default(payload, "stouffer_n", nothing)
    n_raw === nothing && (n_raw = _stouffer_get_with_default(payload, "n", nothing))
    n_raw === nothing && error("stouffer_params_from_payload requires stouffer_n or n")
    n = Int(n_raw)
    adj = _to_matrix_bool(_dict_or_prop_get(payload, "stouffer_adj"))
    size(adj) == (n, n) || error("stouffer_adj size mismatch")

    return StoufferParams(
        Float64(_dict_or_prop_get(payload, "stouffer_connectance")),
        adj,
        Bool.(_dict_or_prop_get(payload, "stouffer_basal_mask")),
        _to_matrix_f64(_dict_or_prop_get(payload, "stouffer_w")),
        Vector{Float64}(_dict_or_prop_get(payload, "stouffer_M")),
        Vector{Float64}(_dict_or_prop_get(payload, "stouffer_x")),
        Vector{Float64}(_dict_or_prop_get(payload, "stouffer_y")),
        _to_matrix_f64(_dict_or_prop_get(payload, "stouffer_e"));
        K  = Float64(_stouffer_get_with_default(payload, "stouffer_K", STOUFFER_DEFAULT_K)),
        B0 = Float64(_stouffer_get_with_default(payload, "stouffer_B0", STOUFFER_DEFAULT_B0)),
        Mb = Float64(_stouffer_get_with_default(payload, "stouffer_Mb", STOUFFER_DEFAULT_Mb)),
        ar = Float64(_stouffer_get_with_default(payload, "stouffer_ar", STOUFFER_DEFAULT_AR)),
        ax = Float64(_stouffer_get_with_default(payload, "stouffer_ax", STOUFFER_DEFAULT_AX)),
        ay = Float64(_stouffer_get_with_default(payload, "stouffer_ay", STOUFFER_DEFAULT_AY)),
    )
end

function stouffer_baseline_r(p::StoufferParams)
    r = zeros(Float64, p.n)
    @inbounds for i in 1:p.n
        r[i] = p.basal_mask[i] ? 1.0 : -p.x[i]
    end
    return r
end

function _stouffer_consumer_has_basal_path(adj::AbstractMatrix{Bool},
                                           basal_mask::AbstractVector{Bool},
                                           start::Int)
    basal_mask[start] && return true

    seen = falses(size(adj, 1))
    stack = Int[start]
    while !isempty(stack)
        node = pop!(stack)
        seen[node] && continue
        seen[node] = true
        for prey in findall(@view adj[node, :])
            basal_mask[prey] && return true
            seen[prey] || push!(stack, prey)
        end
    end
    return false
end

function _stouffer_niche_web_valid(adj::AbstractMatrix{Bool},
                                   basal_mask::AbstractVector{Bool})
    any(basal_mask) || return false
    any(.!basal_mask) || return false

    n = size(adj, 1)
    @inbounds for i in 1:n
        if !basal_mask[i]
            any(@view adj[i, :]) || return false
            _stouffer_consumer_has_basal_path(adj, basal_mask, i) || return false
        end
    end
    return true
end

function _sample_stouffer_niche_web(n::Int,
                                    connectance::Float64;
                                    rng::AbstractRNG,
                                    max_tries::Int = 500)
    0.0 < connectance < 0.5 || error("stouffer connectance must lie in (0, 0.5), got $(connectance)")
    beta_param = 1.0 / (2.0 * connectance) - 1.0
    beta_dist = Beta(1.0, beta_param)

    for _ in 1:max_tries
        niche = sort(rand(rng, n))
        adj = falses(n, n)

        @inbounds for i in 1:n
            range_len = niche[i] * rand(rng, beta_dist)
            hi = min(niche[i], 1.0 - range_len / 2.0)
            lo = range_len / 2.0
            hi < lo && continue
            center = lo == hi ? lo : rand(rng, Uniform(lo, hi))
            left = center - range_len / 2.0
            right = center + range_len / 2.0

            for j in 1:n
                i == j && continue
                if left <= niche[j] <= right
                    adj[i, j] = true
                end
            end
        end

        basal_mask = .!vec(any(adj; dims=2))
        _stouffer_niche_web_valid(adj, basal_mask) || continue
        return adj, basal_mask
    end

    error("failed to sample a valid Stouffer niche-model web after $(max_tries) attempts")
end

function _stouffer_trophic_levels(adj::AbstractMatrix{Bool},
                                  basal_mask::AbstractVector{Bool})
    n = size(adj, 1)
    A = Matrix{Float64}(I, n, n)
    b = ones(Float64, n)

    @inbounds for i in 1:n
        if basal_mask[i]
            A[i, :] .= 0.0
            A[i, i] = 1.0
            b[i] = 1.0
            continue
        end

        prey = findall(@view adj[i, :])
        isempty(prey) && error("consumer $(i) has no prey")
        w = 1.0 / length(prey)
        for j in prey
            A[i, j] -= w
        end
    end

    tl = A \ b
    all(isfinite, tl) || error("non-finite trophic level solution")
    return tl
end

function _stouffer_mass_logtarget(z::AbstractVector{<:Real},
                                  adj::AbstractMatrix{Bool})
    dist = Normal(STOUFFER_LOGRATIO_MEAN, STOUFFER_LOGRATIO_SD)
    total = 0.0
    @inbounds for i in 1:size(adj, 1), j in 1:size(adj, 2)
        adj[i, j] || continue
        total += logpdf(dist, Float64(z[i]) - Float64(z[j]))
    end
    return total
end

function _sample_stouffer_body_masses(adj::AbstractMatrix{Bool},
                                      basal_mask::AbstractVector{Bool};
                                      rng::AbstractRNG,
                                      Mb::Float64,
                                      mcmc_burnin::Int,
                                      mcmc_steps::Int,
                                      proposal_sd::Float64)
    n = size(adj, 1)
    z = fill(log(Mb), n)
    consumer_idx = findall(.!basal_mask)
    isempty(consumer_idx) && return exp.(z)

    trophic_levels = _stouffer_trophic_levels(adj, basal_mask)
    @inbounds for i in consumer_idx
        z[i] = log(Mb) + max(trophic_levels[i] - 1.0, 1.0) * STOUFFER_LOGRATIO_MEAN
    end

    current = _stouffer_mass_logtarget(z, adj)
    total_steps = mcmc_burnin + mcmc_steps
    total_steps >= 1 || error("body-mass MCMC requires at least one step")

    for _ in 1:total_steps
        idx = rand(rng, consumer_idx)
        proposal = copy(z)
        proposal[idx] += randn(rng) * proposal_sd
        proposal[idx] > log(eps(Float64)) || continue

        proposed = _stouffer_mass_logtarget(proposal, adj)
        if log(rand(rng)) <= proposed - current
            z = proposal
            current = proposed
        end
    end

    M = exp.(z)
    M[basal_mask] .= Mb
    return M
end

function _stouffer_weight_matrix(adj::AbstractMatrix{Bool};
                                 rng::AbstractRNG,
                                 logmean::Float64,
                                 logsd::Float64)
    n = size(adj, 1)
    W = zeros(Float64, n, n)
    dist = LogNormal(logmean, logsd)
    @inbounds for i in 1:n, j in 1:n
        adj[i, j] || continue
        W[i, j] = rand(rng, dist)
    end
    return W
end

function _stouffer_assimilation_matrix(adj::AbstractMatrix{Bool})
    E = zeros(Float64, size(adj))
    @inbounds for i in 1:size(adj, 1), j in 1:size(adj, 2)
        adj[i, j] || continue
        E[i, j] = STOUFFER_ASSIMILATION
    end
    return E
end

function sample_stouffer_params(n::Int = 6;
                                rng::AbstractRNG = Random.default_rng(),
                                connectance_set = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
                                mcmc_burnin::Int = 1000,
                                mcmc_steps::Int = 4000,
                                proposal_sd::Real = 1.0,
                                K::Real = STOUFFER_DEFAULT_K,
                                B0::Real = STOUFFER_DEFAULT_B0,
                                Mb::Real = STOUFFER_DEFAULT_Mb,
                                ar::Real = STOUFFER_DEFAULT_AR,
                                ax::Real = STOUFFER_DEFAULT_AX,
                                ay::Real = STOUFFER_DEFAULT_AY)
    n >= 2 || error("stouffer n must be >= 2, got $(n)")

    connectance_choices = collect(Float64.(connectance_set))
    isempty(connectance_choices) && error("connectance_set must be non-empty")
    connectance = rand(rng, connectance_choices)

    adj, basal_mask = _sample_stouffer_niche_web(n, connectance; rng=rng)
    w = _stouffer_weight_matrix(adj; rng=rng,
        logmean=STOUFFER_WEIGHT_LOGMEAN,
        logsd=STOUFFER_WEIGHT_LOGSD,
    )
    M = _sample_stouffer_body_masses(adj, basal_mask;
        rng=rng,
        Mb=Float64(Mb),
        mcmc_burnin=mcmc_burnin,
        mcmc_steps=mcmc_steps,
        proposal_sd=Float64(proposal_sd),
    )

    x = zeros(Float64, n)
    y = zeros(Float64, n)
    @inbounds for i in 1:n
        if basal_mask[i]
            x[i] = 0.0
            y[i] = 0.0
        else
            x[i] = (Float64(ax) / Float64(ar)) * (M[i] / Float64(Mb))^(-0.25)
            y[i] = Float64(ay) / Float64(ax)
        end
    end

    e = _stouffer_assimilation_matrix(adj)
    return StoufferParams(connectance, adj, basal_mask, w, M, x, y, e;
        K=K, B0=B0, Mb=Mb, ar=ar, ax=ax, ay=ay)
end

function _stouffer_denominators(p::StoufferParams,
                                B::AbstractVector{<:Real})
    n = p.n
    length(B) == n || error("_stouffer_denominators expected B of length $(n)")
    D = fill(p.B0, n)
    @inbounds for i in 1:n
        denom = p.B0
        for prey in p.prey_lists[i]
            denom += p.w[i, prey] * Float64(B[prey])
        end
        D[i] = denom
    end
    return D
end

function stouffer_percapita_growth(p::StoufferParams,
                                   B::AbstractVector{<:Real},
                                   dr_full::AbstractVector{<:Real})
    n = p.n
    length(B) == n || error("stouffer_percapita_growth expected B of length $(n), got $(length(B))")
    length(dr_full) == n || error("stouffer_percapita_growth expected dr_full of length $(n), got $(length(dr_full))")

    D = _stouffer_denominators(p, B)
    F = zeros(Float64, n)
    basal_sum = 0.0
    @inbounds for j in 1:n
        p.basal_mask[j] || continue
        basal_sum += Float64(B[j])
    end

    @inbounds for i in 1:n
        if p.basal_mask[i]
            F[i] = (1.0 + Float64(dr_full[i])) * (1.0 - basal_sum / p.K)
        else
            gain_num = 0.0
            for prey in p.prey_lists[i]
                gain_num += p.w[i, prey] * Float64(B[prey])
            end
            F[i] = (-p.x[i] + Float64(dr_full[i])) + p.x[i] * p.y[i] * gain_num / D[i]
        end
    end

    @inbounds for predator in 1:n
        Dk = D[predator]
        Dk == 0.0 && continue
        for prey in p.prey_lists[predator]
            loss = p.x[predator] * p.y[predator] * Float64(B[predator]) *
                   p.w[predator, prey] * Float64(B[prey]) / (p.e[predator, prey] * Dk)
            F[prey] -= loss
        end
    end

    return F
end

function _make_stouffer_rhs_from_dr(p::StoufferParams,
                                    dr_full::AbstractVector{<:Real})
    n = p.n
    length(dr_full) == n ||
        error("_make_stouffer_rhs_from_dr expected dr_full of length $(n), got $(length(dr_full))")

    function f!(dx, B, _, _)
        F = stouffer_percapita_growth(p, B, dr_full)
        @inbounds for i in 1:n
            dx[i] = Float64(B[i]) * F[i]
        end
        return nothing
    end

    return f!
end

function make_stouffer_rhs(p::StoufferParams,
                           u_full::AbstractVector{<:Real},
                           delta::Real)
    n = p.n
    length(u_full) == n || error("make_stouffer_rhs expected u_full of length $(n), got $(length(u_full))")
    dr = Float64(delta) .* Float64.(u_full)
    return _make_stouffer_rhs_from_dr(p, dr)
end

function integrate_stouffer_to_steady(p::StoufferParams,
                                      x0::AbstractVector{<:Real};
                                      tmax::Real = 10_000.0,
                                      steady_tol::Real = 1e-8)
    n = p.n
    f! = _make_stouffer_rhs_from_dr(p, zeros(Float64, n))
    u_init = Float64.(x0)
    prob = ODEProblem(f!, u_init, (0.0, Float64(tmax)))

    condition(_, _, integrator) = begin
        du = integrator.du
        du === nothing && return false
        return maximum(abs, du) < steady_tol
    end
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!; save_positions=(false, false))

    sol = DifferentialEquations.solve(prob, Tsit5();
        callback=cb,
        reltol=1e-9,
        abstol=1e-11,
        save_everystep=false,
        save_start=false,
    )

    if !SciMLBase.successful_retcode(sol)
        return (success=false, retcode=string(sol.retcode),
                x_eq=fill(NaN, n), du_max=Inf)
    end

    x_eq = Vector{Float64}(sol.u[end])
    @. x_eq = max(x_eq, 0.0)
    du_end = similar(x_eq)
    f!(du_end, x_eq, nothing, 0.0)
    return (success=true, retcode=string(sol.retcode),
            x_eq=x_eq, du_max=maximum(abs.(du_end)))
end

function stouffer_jacobian_fd(p::StoufferParams,
                              B::AbstractVector{<:Real},
                              dr_full::AbstractVector{<:Real})
    n = p.n
    length(B) == n || error("stouffer_jacobian_fd expected B of length $(n), got $(length(B))")
    length(dr_full) == n || error("stouffer_jacobian_fd expected dr_full of length $(n), got $(length(dr_full))")

    f! = _make_stouffer_rhs_from_dr(p, dr_full)
    J = zeros(Float64, n, n)
    fp = zeros(Float64, n)
    fm = zeros(Float64, n)
    xp = Float64.(B)
    xm = Float64.(B)

    @inbounds for j in 1:n
        copyto!(xp, B)
        copyto!(xm, B)
        step = 1e-6 * max(1.0, abs(B[j]))
        xp[j] += step
        xm[j] = max(0.0, xm[j] - step)

        f!(fp, xp, nothing, 0.0)
        f!(fm, xm, nothing, 0.0)
        denom = xp[j] - xm[j]
        @views J[:, j] .= (fp .- fm) ./ denom
    end

    return J
end

function stouffer_jacobian_fd(p::StoufferParams,
                              B::AbstractVector{<:Real})
    return stouffer_jacobian_fd(p, B, zeros(Float64, p.n))
end

function stouffer_lambda_max_equilibrium(p::StoufferParams,
                                         B::AbstractVector{<:Real},
                                         dr_full::AbstractVector{<:Real})
    J = stouffer_jacobian_fd(p, B, dr_full)
    return maximum(real.(eigvals(J)))
end

function stouffer_lambda_max_equilibrium(p::StoufferParams,
                                         B::AbstractVector{<:Real})
    return stouffer_lambda_max_equilibrium(p, B, zeros(Float64, p.n))
end

function build_stouffer_cleared_system(p::StoufferParams)
    n = p.n
    @var x[1:n] dr[1:n]

    D = Vector{Any}(undef, n)
    @inbounds for i in 1:n
        denom = p.B0
        for prey in p.prey_lists[i]
            denom += p.w[i, prey] * x[prey]
        end
        D[i] = denom
    end

    basal_sum = sum(x[j] for j in 1:n if p.basal_mask[j])
    eqs = Vector{Expression}(undef, n)

    @inbounds for i in 1:n
        pred_prod = 1.0
        for predator in p.pred_lists[i]
            pred_prod *= D[predator]
        end

        eq = if p.basal_mask[i]
            (1.0 + dr[i]) * (1.0 - basal_sum / p.K) * pred_prod
        else
            gain_num = sum(p.w[i, prey] * x[prey] for prey in p.prey_lists[i])
            ((-p.x[i] + dr[i]) * D[i] + p.x[i] * p.y[i] * gain_num) * pred_prod
        end

        for predator in p.pred_lists[i]
            coeff = p.x[predator] * p.y[predator] * p.w[predator, i] / p.e[predator, i]
            other_prod = stouffer_is_consumer(p, i) ? D[i] : 1.0
            for other in p.pred_lists[i]
                other == predator && continue
                other_prod *= D[other]
            end
            eq -= coeff * x[predator] * x[i] * other_prod
        end

        eqs[i] = expand(eq)
    end

    return System(eqs; variables=x, parameters=dr), x
end

function stouffer_alpha_eff_symbolic(p::StoufferParams,
                                     x_star::AbstractVector{<:Real},
                                     dr_full::AbstractVector{<:Real})
    length(dr_full) == p.n ||
        error("stouffer_alpha_eff_symbolic expected dr_full of length $(p.n), got $(length(dr_full))")
    syst, _ = build_stouffer_cleared_system(p)
    return symbolic_alpha_eff(syst, x_star; parameter_values=Float64.(dr_full)).alpha_eff
end

function stouffer_alpha_eff_symbolic(p::StoufferParams,
                                     x_star::AbstractVector{<:Real})
    return stouffer_alpha_eff_symbolic(p, x_star, zeros(Float64, p.n))
end

function stouffer_alpha_eff_monomial(p::StoufferParams,
                                     x_star::AbstractVector{<:Real},
                                     dr_full::AbstractVector{<:Real})
    length(dr_full) == p.n ||
        error("stouffer_alpha_eff_monomial expected dr_full of length $(p.n), got $(length(dr_full))")
    syst, _ = build_stouffer_cleared_system(p)
    return symbolic_alpha_eff_monomial_abs(syst, x_star; parameter_values=Float64.(dr_full)).alpha_eff
end

function stouffer_alpha_eff_monomial(p::StoufferParams,
                                     x_star::AbstractVector{<:Real})
    return stouffer_alpha_eff_monomial(p, x_star, zeros(Float64, p.n))
end
