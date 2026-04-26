# Utilities for the Karatayev-style food-web model.
# Used by generate_bank_karatayev.jl (bank generation) and glvhoi_utils.jl (HC dispatch).
#
# Species ordering: [resources (N_1..N_nR), consumers (C_1..C_nC)].
# Consumer mortality m is the perturbation parameter (dr[n_R+k] = delta_m_k).
# Resource rows in U are zeroed out so dr[1:n_R] = 0 in practice.

# Include lever_model.jl for shared utilities (_dict_or_prop_get, _to_matrix_f64,
# effective_interaction_metrics) if not already loaded.
if !@isdefined(LeverOriginalParams)
    include(joinpath(@__DIR__, "lever_model.jl"))
end

struct KaratayevParams
    n_R::Int
    n_C::Int
    r::Vector{Float64}        # resource intrinsic growth rates (length n_R)
    K::Vector{Float64}        # resource carrying capacities (length n_R)
    f::Vector{Float64}        # edibility / feedback strength per resource (length n_R)
    delta::Vector{Float64}    # consumer grazing rates (length n_C)
    b::Vector{Float64}        # consumer conversion efficiencies (length n_C)
    beta::Vector{Float64}     # consumer intraspecific density dependence (length n_C)
    m0::Float64               # baseline consumer mortality (baked in)
    a::Float64                # interspecific resource competition (fixed = 0.025)
    specials::Matrix{Float64} # n_R × n_C, column-normalised preference matrix
    comps::Matrix{Float64}    # n_R × n_C, row-normalised recruitment weights
    feedback_mode::String     # "FMI" or "RMI"
end

@inline karatayev_n_species(p::KaratayevParams) = p.n_R + p.n_C

# ──────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────

function karatayev_params_payload(p::KaratayevParams)
    return (
        karat_n_R           = p.n_R,
        karat_n_C           = p.n_C,
        karat_r             = copy(p.r),
        karat_K             = copy(p.K),
        karat_f             = copy(p.f),
        karat_delta         = copy(p.delta),
        karat_b             = copy(p.b),
        karat_beta          = copy(p.beta),
        karat_m0            = p.m0,
        karat_a             = p.a,
        karat_specials      = copy(p.specials),
        karat_comps         = copy(p.comps),
        karat_feedback_mode = p.feedback_mode,
    )
end

function karatayev_params_from_payload(payload)
    return KaratayevParams(
        Int(_dict_or_prop_get(payload, "karat_n_R")),
        Int(_dict_or_prop_get(payload, "karat_n_C")),
        Vector{Float64}(_dict_or_prop_get(payload, "karat_r")),
        Vector{Float64}(_dict_or_prop_get(payload, "karat_K")),
        Vector{Float64}(_dict_or_prop_get(payload, "karat_f")),
        Vector{Float64}(_dict_or_prop_get(payload, "karat_delta")),
        Vector{Float64}(_dict_or_prop_get(payload, "karat_b")),
        Vector{Float64}(_dict_or_prop_get(payload, "karat_beta")),
        Float64(_dict_or_prop_get(payload, "karat_m0")),
        Float64(_dict_or_prop_get(payload, "karat_a")),
        _to_matrix_f64(_dict_or_prop_get(payload, "karat_specials")),
        _to_matrix_f64(_dict_or_prop_get(payload, "karat_comps")),
        String(_dict_or_prop_get(payload, "karat_feedback_mode")),
    )
end

# ──────────────────────────────────────────────────────────────────
# Parameter sampling  (translated from R parmgen / VarWeb / VarWeb3)
# ──────────────────────────────────────────────────────────────────

# column-normalised preference matrix with lognormal entries (VarWeb3 spirit).
function _sample_specials(n_R::Int, n_C::Int;
                          sigma::Float64 = 0.5,
                          rng::AbstractRNG = Random.default_rng())
    mat = exp.(sigma .* randn(rng, n_R, n_C))
    for k in 1:n_C
        s = sum(mat[:, k])
        mat[:, k] ./= s
    end
    return mat
end

# row-normalised recruitment weight matrix (VarWeb spirit).
function _sample_comps(n_R::Int, n_C::Int;
                       rng::AbstractRNG = Random.default_rng())
    comps = zeros(Float64, n_R, n_C)
    for i in 1:n_R
        row = rand(rng, Gamma(1.0), n_C)
        comps[i, :] = row ./ sum(row)
    end
    return comps
end

"""
    sample_karatayev_params(n_R, n_C; feedback_mode, rng)

Sample a random `KaratayevParams` using the R parmgen defaults:
  K̄=1.35, r̄=1.0, δ̄=1.1, b̄=1.0, m₀=0.05, β̄=0.15, f̄=1.2
  fvar=0.15 (trait variability), QvarTrans=0.125 (edibility variability).
"""
function sample_karatayev_params(n_R::Int, n_C::Int;
                                 feedback_mode::String    = "FMI",
                                 rng::AbstractRNG         = Random.default_rng())
    fvar      = 0.15
    QvarTrans = 0.125

    K_bar     = 1.35
    r_bar     = 1.0
    delta_bar = 1.1
    b_bar     = 1.0
    m0        = 0.025
    beta_bar  = 0.15
    f_bar     = 0.8
    a         = 0.025

    r     = [r_bar     * (1 + fvar      * (2*rand(rng) - 1)) for _ in 1:n_R]
    K     = [K_bar     * (1 + fvar      * (2*rand(rng) - 1)) for _ in 1:n_R]
    f     = [f_bar     * (1 + QvarTrans * (2*rand(rng) - 1)) for _ in 1:n_R]
    delta = [delta_bar * (1 + fvar      * (2*rand(rng) - 1)) for _ in 1:n_C]
    b     = [b_bar     * (1 + fvar      * (2*rand(rng) - 1)) for _ in 1:n_C]
    beta  = [beta_bar  * (1 + fvar      * (2*rand(rng) - 1)) for _ in 1:n_C]

    specials = _sample_specials(n_R, n_C; rng=rng)
    comps    = _sample_comps(n_R, n_C; rng=rng)

    return KaratayevParams(n_R, n_C, r, K, f, delta, b, beta, m0, a,
                           specials, comps, feedback_mode)
end

# ──────────────────────────────────────────────────────────────────
# ODE right-hand side
# ──────────────────────────────────────────────────────────────────

"""
    make_karatayev_rhs(p, dr_full)

Return an in-place ODE function `f!(dx, x, _, _)` for the Karatayev food-web.

State: x = [N_1..N_nR, C_1..C_nC].

Resource per-capita growth:
  F_i = r_i_eff*(1 - ((1-a)*N_i + a*ΣN_j)/K_i) - Σ_k Eats[i,k]*δ_k*C_k

Consumer full-state rate:
  dC_k/dt = C_k*G_k - β_k*C_k²
  G_k = δ_k*b_k*(Σ_i N_i*Eats[i,k])*(1 - RMIlevs_k) - m_eff

FMI: Eats[i,k] = (1 - f_i*N_i/K_i)*specials[i,k],  RMIlevs_k = 0
RMI: Eats[i,k] = specials[i,k],  RMIlevs_k = Σ_i (f_i*N_i/K_i)*comps[i,k]
"""
function make_karatayev_rhs(p::KaratayevParams,
                             dr_full::AbstractVector{<:Real})
    n_R = p.n_R
    n_C = p.n_C
    a   = p.a

    function f!(dx, x, _, _)
        @inbounds for i in 1:n_R
            r_eff = p.r[i] + dr_full[i]
            comp = (1 - a) * x[i]
            for j in 1:n_R; comp += a * x[j]; end
            # result: (1-a)*N_i + a*Σ_j N_j  (sum includes i)

            eat_sum = 0.0
            if p.feedback_mode == "FMI"
                for k in 1:n_C
                    eat_sum += p.specials[i, k] * (1 - p.f[i] * x[i] / p.K[i]) *
                               p.delta[k] * x[n_R + k]
                end
            else  # RMI
                for k in 1:n_C
                    eat_sum += p.specials[i, k] * p.delta[k] * x[n_R + k]
                end
            end

            Fi     = r_eff * (1 - comp / p.K[i]) - eat_sum
            dx[i]  = x[i] * Fi
        end

        @inbounds for k in 1:n_C
            m_eff = p.m0 + dr_full[n_R + k]

            if p.feedback_mode == "FMI"
                Gk = 0.0
                for i in 1:n_R
                    Gk += p.delta[k] * p.b[k] * p.specials[i, k] *
                          (1 - p.f[i] * x[i] / p.K[i]) * x[i]
                end
                Gk -= m_eff
            else  # RMI
                S_k    = 0.0
                RMIlev = 0.0
                for i in 1:n_R
                    S_k    += p.specials[i, k] * x[i]
                    RMIlev += p.comps[i, k] * p.f[i] * x[i] / p.K[i]
                end
                Gk = p.delta[k] * p.b[k] * S_k * (1 - RMIlev) - m_eff
            end

            dx[n_R + k] = x[n_R + k] * Gk - p.beta[k] * x[n_R + k]^2
        end
        return nothing
    end
    return f!
end

# ──────────────────────────────────────────────────────────────────
# ODE integration to steady state
# ──────────────────────────────────────────────────────────────────

function integrate_karatayev_to_steady(x0::AbstractVector{<:Real},
                                       p::KaratayevParams;
                                       tmax::Real       = 2000.0,
                                       steady_tol::Real = 1e-8)
    n      = karatayev_n_species(p)
    dr_z   = zeros(Float64, n)
    f!     = make_karatayev_rhs(p, dr_z)
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

# ──────────────────────────────────────────────────────────────────
# Jacobian of per-capita growth
# ──────────────────────────────────────────────────────────────────

"""
    karatayev_jacobian_F(p, x, dr_full)

Jacobian of the per-capita growth vector F (not of dx/dt) evaluated at x.
Row i = species i, column j = ∂F_i/∂x[j].

The community (stability) matrix is diag(x) * JF evaluated at equilibrium.
"""
function karatayev_jacobian_F(p::KaratayevParams,
                               x::AbstractVector{<:Real},
                               dr_full::AbstractVector{<:Real})
    n_R = p.n_R
    n_C = p.n_C
    n   = n_R + n_C
    a   = p.a
    JF  = zeros(Float64, n, n)

    @inbounds for i in 1:n_R
        r_eff = p.r[i] + dr_full[i]

        if p.feedback_mode == "FMI"
            C_sum = 0.0
            for k in 1:n_C
                C_sum += p.specials[i, k] * p.delta[k] * x[n_R + k]
            end
            # ∂F_i/∂N_i (competition + feedback relief)
            JF[i, i] = -r_eff / p.K[i] + p.f[i] / p.K[i] * C_sum
            # ∂F_i/∂N_j (j ≠ i, interspecific competition)
            for j in 1:n_R
                j == i && continue
                JF[i, j] = -r_eff * a / p.K[i]
            end
            # ∂F_i/∂C_k
            for k in 1:n_C
                JF[i, n_R + k] = -p.specials[i, k] * p.delta[k] *
                                  (1 - p.f[i] * x[i] / p.K[i])
            end
        else  # RMI
            JF[i, i] = -r_eff / p.K[i]
            for j in 1:n_R
                j == i && continue
                JF[i, j] = -r_eff * a / p.K[i]
            end
            for k in 1:n_C
                JF[i, n_R + k] = -p.specials[i, k] * p.delta[k]
            end
        end
    end

    @inbounds for k in 1:n_C
        idx = n_R + k

        if p.feedback_mode == "FMI"
            for i in 1:n_R
                JF[idx, i] = p.delta[k] * p.b[k] * p.specials[i, k] *
                              (1 - 2 * p.f[i] * x[i] / p.K[i])
            end
            JF[idx, idx] = -p.beta[k]
        else  # RMI
            S_k    = 0.0
            RMIlev = 0.0
            for i in 1:n_R
                S_k    += p.specials[i, k] * x[i]
                RMIlev += p.comps[i, k] * p.f[i] * x[i] / p.K[i]
            end
            for i in 1:n_R
                JF[idx, i] = p.delta[k] * p.b[k] *
                              (p.specials[i, k] * (1 - RMIlev) -
                               S_k * p.comps[i, k] * p.f[i] / p.K[i])
            end
            JF[idx, idx] = -p.beta[k]
        end
    end

    return JF
end

function karatayev_jacobian_F(p::KaratayevParams, x::AbstractVector{<:Real})
    return karatayev_jacobian_F(p, x, zeros(Float64, karatayev_n_species(p)))
end

function karatayev_lambda_max_equilibrium(p::KaratayevParams,
                                          x::AbstractVector{<:Real},
                                          dr_full::AbstractVector{<:Real})
    JF = karatayev_jacobian_F(p, x, dr_full)
    J  = Matrix{Float64}(JF)
    @inbounds for i in 1:length(x)
        @views J[i, :] .*= x[i]
    end
    return maximum(real.(eigvals(J)))
end

function karatayev_lambda_max_equilibrium(p::KaratayevParams,
                                          x::AbstractVector{<:Real})
    return karatayev_lambda_max_equilibrium(p, x,
               zeros(Float64, karatayev_n_species(p)))
end

# ──────────────────────────────────────────────────────────────────
# Effective polynomial coefficients → α_eff
# ──────────────────────────────────────────────────────────────────

"""
    karatayev_effective_coefficients(p, dr_full) → (r_eff, Aeff, Beff)

Express the per-capita growth F_i(x) in GLV+HOI form:
  F_i = r_eff[i] + Σ_j Aeff[i,j]*x[j] + Σ_{j,k} Beff[j,k,i]*x[j]*x[k]

Legacy grouped coefficient representation retained for comparison with the
older `effective_interaction_metrics`-based `alpha_eff`.
"""
function karatayev_effective_coefficients(p::KaratayevParams,
                                          dr_full::AbstractVector{<:Real})
    n_R = p.n_R
    n_C = p.n_C
    n   = n_R + n_C

    r_eff = zeros(Float64, n)
    Aeff  = zeros(Float64, n, n)
    Beff  = zeros(Float64, n, n, n)

    # ── Resource species ───────────────────────────────────────────
    @inbounds for i in 1:n_R
        ri       = p.r[i] + dr_full[i]
        r_eff[i] = ri

        # Intraspecific competition: coefficient of N_i is -(r_i/K_i)*(1-a+a) = -ri/K_i
        Aeff[i, i] += -ri / p.K[i]
        # Interspecific competition: coefficient of N_j (j≠i) is -ri*a/K_i
        for j in 1:n_R
            j == i && continue
            Aeff[i, j] += -ri * p.a / p.K[i]
        end
        # Predation loss: -specials[i,k]*delta_k*C_k (linear)
        for k in 1:n_C
            Aeff[i, n_R + k] += -p.specials[i, k] * p.delta[k]
        end
        # FMI: feedback relief bilinear N_i*C_k term
        if p.feedback_mode == "FMI"
            for k in 1:n_C
                Beff[i, n_R + k, i] += p.f[i] / p.K[i] * p.specials[i, k] * p.delta[k]
            end
        end
    end

    # ── Consumer species ───────────────────────────────────────────
    @inbounds for k in 1:n_C
        idx       = n_R + k
        m_eff     = p.m0 + dr_full[idx]
        r_eff[idx] = -m_eff

        # Linear N_i terms from foraging
        for i in 1:n_R
            Aeff[idx, i] += p.delta[k] * p.b[k] * p.specials[i, k]
        end
        # Density dependence
        Aeff[idx, idx] += -p.beta[k]

        if p.feedback_mode == "FMI"
            # -delta_k*b_k*specials[i,k]*f_i/K_i * N_i^2
            for i in 1:n_R
                Beff[i, i, idx] += -p.delta[k] * p.b[k] *
                                    p.specials[i, k] * p.f[i] / p.K[i]
            end
        else  # RMI
            # -delta_k*b_k*specials[i,k]*comps[j,k]*f_j/K_j * N_i*N_j
            for i in 1:n_R
                for j in 1:n_R
                    Beff[i, j, idx] += -p.delta[k] * p.b[k] *
                                        p.specials[i, k] * p.comps[j, k] *
                                        p.f[j] / p.K[j]
                end
            end
        end
    end

    return r_eff, Aeff, Beff
end

function karatayev_effective_coefficients(p::KaratayevParams)
    return karatayev_effective_coefficients(p,
               zeros(Float64, karatayev_n_species(p)))
end

# ──────────────────────────────────────────────────────────────────
# Denominator-cleared HC polynomial system
# ──────────────────────────────────────────────────────────────────

"""
    build_karatayev_cleared_system(p) → (System, x)

Return a HomotopyContinuation polynomial system for the Karatayev food-web.
Variables: x[1:n] = [N_1..N_nR, C_1..C_nC].
Parameters: dr[1:n] (consumer mortality perturbations in dr[n_R+1:n]).

Resource equations are multiplied by K_i to clear the denominator.
Consumer equations are already polynomial.
"""
function build_karatayev_cleared_system(p::KaratayevParams)
    n_R = p.n_R
    n_C = p.n_C
    n   = n_R + n_C
    a   = p.a

    @var x[1:n] dr[1:n]
    eqs = Vector{Expression}(undef, n)

    # ── Resource equations (K_i × per-capita = 0) ─────────────────
    for i in 1:n_R
        r_eff = p.r[i] + dr[i]
        # competition term: (1-a)*N_i + a*Σ_j N_j
        comp  = (1 - a) * x[i] + a * sum(x[j] for j in 1:n_R)

        pred = sum(p.specials[i, k] * p.delta[k] * x[n_R + k] for k in 1:n_C)

        if p.feedback_mode == "FMI"
            # K_i*F_i = r_eff*K_i - r_eff*comp - K_i*pred + f_i*N_i*pred
            fi_term = p.f[i] * x[i] * pred
            eqs[i]  = r_eff * p.K[i] - r_eff * comp - p.K[i] * pred + fi_term
        else  # RMI
            eqs[i]  = r_eff * p.K[i] - r_eff * comp - p.K[i] * pred
        end
    end

    # ── Consumer equations (per-capita G_k = β_k*C_k) ─────────────
    for k in 1:n_C
        idx   = n_R + k
        m_eff = p.m0 + dr[idx]
        S_k   = sum(p.specials[i, k] * x[i] for i in 1:n_R)

        if p.feedback_mode == "FMI"
            quad = sum(p.specials[i, k] * p.f[i] / p.K[i] * x[i]^2 for i in 1:n_R)
            eqs[idx] = p.delta[k] * p.b[k] * (S_k - quad) - m_eff - p.beta[k] * x[idx]
        else  # RMI
            R_k = sum(p.specials[i, k] * p.comps[j, k] * p.f[j] / p.K[j] *
                      x[i] * x[j]
                      for i in 1:n_R, j in 1:n_R)
            eqs[idx] = p.delta[k] * p.b[k] * (S_k - R_k) - m_eff - p.beta[k] * x[idx]
        end
    end

    return System(eqs; variables=x, parameters=dr), x
end

"""
    build_karatayev_alpha_eff_system(p) → (System, x)

Return the polynomial per-capita growth system used for `alpha_eff` extraction.
Unlike `build_karatayev_cleared_system`, resource equations are not rescaled by
`K_i`, so grouped symbolic `alpha_eff` matches the legacy per-capita metric.
"""
function build_karatayev_alpha_eff_system(p::KaratayevParams)
    n_R = p.n_R
    n_C = p.n_C
    n   = n_R + n_C
    a   = p.a

    @var x[1:n] dr[1:n]
    eqs = Vector{Expression}(undef, n)

    for i in 1:n_R
        r_eff = p.r[i] + dr[i]
        comp  = (1 - a) * x[i] + a * sum(x[j] for j in 1:n_R)

        if p.feedback_mode == "FMI"
            eat_sum = sum(p.specials[i, k] * (1 - p.f[i] * x[i] / p.K[i]) *
                          p.delta[k] * x[n_R + k]
                          for k in 1:n_C)
        else
            eat_sum = sum(p.specials[i, k] * p.delta[k] * x[n_R + k]
                          for k in 1:n_C)
        end

        eqs[i] = r_eff * (1 - comp / p.K[i]) - eat_sum
    end

    for k in 1:n_C
        idx   = n_R + k
        m_eff = p.m0 + dr[idx]

        if p.feedback_mode == "FMI"
            eqs[idx] = sum(p.delta[k] * p.b[k] * p.specials[i, k] *
                           (1 - p.f[i] * x[i] / p.K[i]) * x[i]
                           for i in 1:n_R) - m_eff - p.beta[k] * x[idx]
        else
            S_k    = sum(p.specials[i, k] * x[i] for i in 1:n_R)
            RMIlev = sum(p.comps[i, k] * p.f[i] * x[i] / p.K[i]
                         for i in 1:n_R)
            eqs[idx] = p.delta[k] * p.b[k] * S_k * (1 - RMIlev) -
                       m_eff - p.beta[k] * x[idx]
        end
    end

    return System(eqs; variables=x, parameters=dr), x
end

"""
    karatayev_alpha_eff_symbolic(p, x_star, dr_full=zeros(n)) → Float64

Compute `alpha_eff` from the polynomial per-capita HC system using the shared
symbolic grouped metric.
"""
function karatayev_alpha_eff_symbolic(p::KaratayevParams,
                                      x_star::AbstractVector{<:Real},
                                      dr_full::AbstractVector{<:Real})
    length(dr_full) == karatayev_n_species(p) ||
        error("karatayev_alpha_eff_symbolic expected dr_full of length $(karatayev_n_species(p)), got $(length(dr_full))")
    syst, _ = build_karatayev_alpha_eff_system(p)
    return symbolic_alpha_eff(syst, x_star; parameter_values=Float64.(dr_full)).alpha_eff
end

function karatayev_alpha_eff_symbolic(p::KaratayevParams,
                                      x_star::AbstractVector{<:Real})
    return karatayev_alpha_eff_symbolic(p, x_star,
               zeros(Float64, karatayev_n_species(p)))
end

"""
    karatayev_alpha_eff_monomial(p, x_star, dr_full=zeros(n)) → Float64

Comparison-only Karatayev `alpha_eff` that sums absolute monomial contributions
on the polynomial per-capita HC system before any cancellation.
"""
function karatayev_alpha_eff_monomial(p::KaratayevParams,
                                      x_star::AbstractVector{<:Real},
                                      dr_full::AbstractVector{<:Real})
    length(dr_full) == karatayev_n_species(p) ||
        error("karatayev_alpha_eff_monomial expected dr_full of length $(karatayev_n_species(p)), got $(length(dr_full))")
    syst, _ = build_karatayev_alpha_eff_system(p)
    return symbolic_alpha_eff_monomial_abs(syst, x_star; parameter_values=Float64.(dr_full)).alpha_eff
end

function karatayev_alpha_eff_monomial(p::KaratayevParams,
                                      x_star::AbstractVector{<:Real})
    return karatayev_alpha_eff_monomial(p, x_star,
               zeros(Float64, karatayev_n_species(p)))
end
