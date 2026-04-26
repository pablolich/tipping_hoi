# Utilities for the Aguadé-Gorgorió (2024) Allee-effect community model.
# Used by generate_bank_aguade.jl (bank generation) and glvhoi_utils.jl (HC dispatch).
#
# ODE: dx_i/dt = x_i * (Σ_j A_ij * x_j/(γ_j + x_j)  -  d_i  -  Σ_j B_ij * x_j)
#
# A_ij ≥ 0: facilitation (diagonal = Allee self-feedback, off-diagonal = mutualism)
# B_ij ≥ 0: self-regulation (diagonal) and competition (off-diagonal)
# γ_i > 0:  half-saturation constants
# d_i > 0:  per-species death rates (perturbation parameter for HC scan)

using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using SciMLBase
using HomotopyContinuation

# Load shared helpers (_dict_or_prop_get, _to_matrix_f64, effective_interaction_metrics)
if !@isdefined(LeverOriginalParams)
    include(joinpath(@__DIR__, "lever_model.jl"))
end

struct AguadeParams
    n::Int
    A::Matrix{Float64}      # n×n facilitation matrix (A_ij ≥ 0)
    B::Matrix{Float64}      # n×n competition matrix  (B_ij ≥ 0)
    d::Vector{Float64}      # n   death rates          (d_i > 0)
    gamma::Vector{Float64}  # n   half-saturation      (γ_i > 0)
end

# ──────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────

function aguade_params_payload(p::AguadeParams)
    return (
        aguade_n     = p.n,
        aguade_A     = [collect(p.A[i, :]) for i in 1:p.n],
        aguade_B     = [collect(p.B[i, :]) for i in 1:p.n],
        aguade_d     = copy(p.d),
        aguade_gamma = copy(p.gamma),
    )
end

function aguade_params_from_payload(payload)
    n = Int(_dict_or_prop_get(payload, "aguade_n"))
    return AguadeParams(
        n,
        _to_matrix_f64(_dict_or_prop_get(payload, "aguade_A")),
        _to_matrix_f64(_dict_or_prop_get(payload, "aguade_B")),
        Vector{Float64}(_dict_or_prop_get(payload, "aguade_d")),
        Vector{Float64}(_dict_or_prop_get(payload, "aguade_gamma")),
    )
end

# ──────────────────────────────────────────────────────────────────
# Parameter sampling (LogNormal, dense matrices)
# ──────────────────────────────────────────────────────────────────

"""
    sample_aguade_params(n=6; rng, mean_gamma, mean_d, mean_A_diag, mean_B_diag,
                         sigma_intra_scale, mean_A_range, mean_B_range,
                         sigma_inter_scale_range)

Sample a random `AguadeParams` following the broad Aguadé-Gorgorió AE
exploration used in the paper code.

Intra-species parameters:
  γ_i, d_i, A_ii, B_ii ~ LogNormal(log(mean), log(1.1))

Inter-species sampling:
  meanA      ~ Uniform(0, 0.5)
  meanB      ~ Uniform(0, 0.14)
  sigma_inter~ Uniform(1.00001, 3.0001)
  A_ij, B_ij ~ LogNormal(log(meanX), log(sigma_inter))

Dense matrices — no sparsity masking. The diagonals of A and B are overwritten
with the separately sampled A_ii and B_ii draws, matching the paper code.
"""
function sample_aguade_params(n::Int = 6;
                               rng::AbstractRNG = Random.default_rng(),
                               mean_gamma::Real = 1.0,
                               mean_d::Real = 0.1,
                               mean_A_diag::Real = 0.5,
                               mean_B_diag::Real = 0.1,
                               sigma_intra_scale::Real = 1.1,
                               mean_A_range::Tuple{<:Real,<:Real} = (nextfloat(0.0), 0.5),
                               mean_B_range::Tuple{<:Real,<:Real} = (nextfloat(0.0), 0.14),
                               sigma_inter_scale_range::Tuple{<:Real,<:Real} = (1.00001, 3.0001))
    gamma_mean = Float64(mean_gamma)
    d_mean = Float64(mean_d)
    A_diag_mean = Float64(mean_A_diag)
    B_diag_mean = Float64(mean_B_diag)

    sigma_intra_scale_f = Float64(sigma_intra_scale)
    sigma_intra_scale_f > 1.0 || error("sigma_intra_scale must be > 1, got $sigma_intra_scale_f")

    A_lo, A_hi = Float64.(mean_A_range)
    B_lo, B_hi = Float64.(mean_B_range)
    sigma_lo, sigma_hi = Float64.(sigma_inter_scale_range)

    0.0 < A_lo < A_hi || error("mean_A_range must satisfy 0 < lo < hi, got $(mean_A_range)")
    0.0 < B_lo < B_hi || error("mean_B_range must satisfy 0 < lo < hi, got $(mean_B_range)")
    1.0 < sigma_lo < sigma_hi || error("sigma_inter_scale_range must satisfy 1 < lo < hi, got $(sigma_inter_scale_range)")

    sigma_intra = log(sigma_intra_scale_f)
    gamma = rand(rng, LogNormal(log(gamma_mean), sigma_intra), n)
    d = rand(rng, LogNormal(log(d_mean), sigma_intra), n)
    Aii = rand(rng, LogNormal(log(A_diag_mean), sigma_intra), n)
    Bii = rand(rng, LogNormal(log(B_diag_mean), sigma_intra), n)

    mean_A = rand(rng, Uniform(A_lo, A_hi))
    mean_B = rand(rng, Uniform(B_lo, B_hi))
    sigma_inter_scale = rand(rng, Uniform(sigma_lo, sigma_hi))
    sigma_inter = log(sigma_inter_scale)

    A = rand(rng, LogNormal(log(mean_A), sigma_inter), n, n)
    B = rand(rng, LogNormal(log(mean_B), sigma_inter), n, n)
    for i in 1:n
        A[i, i] = Aii[i]
        B[i, i] = Bii[i]
    end
    return AguadeParams(n, A, B, d, gamma)
end

# ──────────────────────────────────────────────────────────────────
# ODE right-hand side
# ──────────────────────────────────────────────────────────────────

"""
    make_aguade_rhs(p, u_full, delta)

Return in-place ODE `f!(dx, x, _, _)` for the Aguadé model with perturbation
dr = delta * u_full added to the death rates.
"""
function make_aguade_rhs(p::AguadeParams,
                          u_full::AbstractVector{<:Real},
                          delta::Real)
    n  = p.n
    dr = Float64(delta) .* Float64.(u_full)

    function f!(dx, x, _, _)
        @inbounds for i in 1:n
            sat  = 0.0
            comp = 0.0
            for j in 1:n
                sat  += p.A[i, j] * x[j] / (p.gamma[j] + x[j])
                comp += p.B[i, j] * x[j]
            end
            dx[i] = x[i] * (sat - (p.d[i] + dr[i]) - comp)
        end
        return nothing
    end
    return f!
end

# ──────────────────────────────────────────────────────────────────
# ODE integration to steady state
# ──────────────────────────────────────────────────────────────────

function integrate_aguade_to_steady(p::AguadeParams,
                                     x0::AbstractVector{<:Real};
                                     tmax::Real       = 4000.0,
                                     steady_tol::Real = 1e-8)
    n      = p.n
    u_zero = zeros(Float64, n)
    f!     = make_aguade_rhs(p, u_zero, 0.0)
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
# Jacobian of per-capita growth and stability
# ──────────────────────────────────────────────────────────────────

"""
    aguade_jacobian_F(p, x)

Jacobian of the per-capita growth vector F (not of dx/dt) evaluated at x.
  ∂F_i/∂x_j = A_ij * γ_j / (γ_j + x_j)²  -  B_ij

Community (stability) matrix = diag(x) * JF at equilibrium.
"""
function aguade_jacobian_F(p::AguadeParams, x::AbstractVector{<:Real})
    n  = p.n
    JF = zeros(Float64, n, n)
    @inbounds for i in 1:n, j in 1:n
        gj        = p.gamma[j] + x[j]
        JF[i, j]  = p.A[i, j] * p.gamma[j] / (gj * gj) - p.B[i, j]
    end
    return JF
end

function aguade_lambda_max_equilibrium(p::AguadeParams, x::AbstractVector{<:Real})
    JF = aguade_jacobian_F(p, x)
    J  = Matrix{Float64}(JF)
    @inbounds for i in 1:length(x)
        @views J[i, :] .*= x[i]
    end
    return maximum(real.(eigvals(J)))
end

# ──────────────────────────────────────────────────────────────────
# Effective polynomial coefficients → α_eff
# ──────────────────────────────────────────────────────────────────

"""
    aguade_effective_coefficients(p) → (r_eff, Aeff, Beff)

Expand the denominator-cleared per-capita growth G_i(x) = F_i(x) * Q,
where Q = Π_k (γ_k + x_k), in degree-0/1/2 Taylor coefficients at x = 0.

  Γ      = Π_k γ_k
  Γ_{-l} = Γ / γ_l
  Γ_{-lm}= Γ / (γ_l γ_m)

Constant:   r_eff[i]    = -d_i * Γ
Linear:     Aeff[i,l]   = (A[i,l] - d_i) * Γ_{-l}  -  B[i,l] * Γ
Quadratic:  Beff[l,m,i] = (A[i,l] + A[i,m] - d_i) * Γ_{-lm}
                           - B[i,m]*Γ_{-l} - B[i,l]*Γ_{-m}   (l ≠ m)
            Beff[l,l,i] = -B[i,l] * Γ_{-l}

Legacy grouped coefficient representation retained for comparison with the
older `effective_interaction_metrics`-based `alpha_eff`.
"""
function aguade_effective_coefficients(p::AguadeParams)
    n    = p.n
    Γ    = prod(p.gamma)
    Γinv = [Γ / p.gamma[l] for l in 1:n]   # Γ_{-l}

    r_eff = zeros(Float64, n)
    Aeff  = zeros(Float64, n, n)
    Beff  = zeros(Float64, n, n, n)

    @inbounds for i in 1:n
        r_eff[i] = -p.d[i] * Γ

        for l in 1:n
            Aeff[i, l] = (p.A[i, l] - p.d[i]) * Γinv[l] - p.B[i, l] * Γ
        end

        for l in 1:n
            # diagonal
            Beff[l, l, i] = -p.B[i, l] * Γinv[l]
            # off-diagonal (l < m, filled symmetrically)
            for m in (l + 1):n
                Γ_lm = Γ / (p.gamma[l] * p.gamma[m])
                val  = (p.A[i, l] + p.A[i, m] - p.d[i]) * Γ_lm -
                        p.B[i, m] * Γinv[l] - p.B[i, l] * Γinv[m]
                Beff[l, m, i] = val
                Beff[m, l, i] = val
            end
        end
    end

    return r_eff, Aeff, Beff
end

# ──────────────────────────────────────────────────────────────────
# α_eff via symbolic extraction on the cleared polynomial
# ──────────────────────────────────────────────────────────────────

"""
    aguade_alpha_eff_monomial(p, x_star, dr_full=zeros(n)) → Float64

Comparison-only Aguadé `alpha_eff` that sums absolute monomial contributions on
the denominator-cleared HC system before any cancellation.
"""
function aguade_alpha_eff_monomial(p::AguadeParams,
                                   x_star::AbstractVector{<:Real},
                                   dr_full::AbstractVector{<:Real})
    length(dr_full) == p.n ||
        error("aguade_alpha_eff_monomial expected dr_full of length $(p.n), got $(length(dr_full))")
    syst, _ = build_aguade_cleared_system(p)
    return symbolic_alpha_eff_monomial_abs(syst, x_star;
        parameter_values=Float64.(dr_full)).alpha_eff
end

function aguade_alpha_eff_monomial(p::AguadeParams,
                                   x_star::AbstractVector{<:Real})
    return aguade_alpha_eff_monomial(p, x_star, zeros(Float64, p.n))
end

# ──────────────────────────────────────────────────────────────────
# Denominator-cleared HC polynomial system
# ──────────────────────────────────────────────────────────────────

"""
    build_aguade_cleared_system(p) → (System, x)

Return a HomotopyContinuation polynomial system for the Aguadé model.
Variables:  x[1:n]  (species abundances)
Parameters: dr[1:n] (perturbations to death rates d_i → d_i + dr_i)

Clearing denominators Q = Π_k (γ_k + x_k):
  G_i = Σ_j A_ij * x_j * Q_{-j}  -  (d_i + dr_i) * Q  -  Σ_j B_ij * x_j * Q  =  0

where Q_{-j} = Π_{k≠j} (γ_k + x_k) is degree n-1.
Total degree of G_i: n+1.
"""
function build_aguade_cleared_system(p::AguadeParams)
    n = p.n
    @var x[1:n] dr[1:n]

    # Q_{-j} = Π_{k≠j} (γ_k + x_k)  (degree n-1, polynomial in x)
    Q_mj = [prod(p.gamma[k] + x[k] for k in 1:n if k != j) for j in 1:n]

    # Q = (γ_j + x_j) * Q_{-j} for any j
    Q = (p.gamma[1] + x[1]) * Q_mj[1]

    eqs = Vector{Expression}(undef, n)
    for i in 1:n
        sat_term  = sum(p.A[i, j] * x[j] * Q_mj[j] for j in 1:n)
        comp_term = sum(p.B[i, j] * x[j] for j in 1:n) * Q
        eqs[i]    = sat_term - (p.d[i] + dr[i]) * Q - comp_term
    end

    return System(eqs; variables=x, parameters=dr), x
end

"""
    aguade_alpha_eff_symbolic(p, x_star, dr_full=zeros(n)) → Float64

Compute `alpha_eff` from the denominator-cleared HC system using the shared
symbolic grouped metric.
"""
function aguade_alpha_eff_symbolic(p::AguadeParams,
                                   x_star::AbstractVector{<:Real},
                                   dr_full::AbstractVector{<:Real})
    length(dr_full) == p.n ||
        error("aguade_alpha_eff_symbolic expected dr_full of length $(p.n), got $(length(dr_full))")
    syst, _ = build_aguade_cleared_system(p)
    return symbolic_alpha_eff(syst, x_star; parameter_values=Float64.(dr_full)).alpha_eff
end

function aguade_alpha_eff_symbolic(p::AguadeParams,
                                   x_star::AbstractVector{<:Real})
    return aguade_alpha_eff_symbolic(p, x_star, zeros(Float64, p.n))
end
