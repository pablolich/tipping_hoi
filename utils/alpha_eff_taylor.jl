# Taylor-expansion-based effective nonlinearity alpha_eff_taylor.
#
# Unified metric comparable across GLV+HOI, Gibbs, and published rational
# models.  For each system at its coexistence equilibrium x_star, compute
#   J[i,j]   = ∂f_i/∂x_j
#   H[i,j,k] = ∂²f_i/(∂x_j ∂x_k)
#   T1[i]    = Σ_j J[i,j] · x_star[j]
#   T2[i]    = 0.5 Σ_{jk} H[i,j,k] · x_star[j] · x_star[k]
#   P = mean_i|T1[i]|;  Q = mean_i|T2[i]|
#   alpha_eff_taylor = Q / (P + Q)
#
# Key constraint: the per-capita rate f here is the ORIGINAL rational f, not
# the denominator-cleared polynomial used by HomotopyContinuation.  See the
# README for the derivation.

using ForwardDiff
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "json_utils.jl"))

# Model param structs and *_params_from_payload helpers:
if !@isdefined(LeverOriginalParams)
    include(joinpath(@__DIR__, "..", "other_models", "lever_model.jl"))
end
if !@isdefined(KaratayevParams)
    include(joinpath(@__DIR__, "..", "other_models", "karatayev_model.jl"))
end
if !@isdefined(AguadeParams)
    include(joinpath(@__DIR__, "..", "other_models", "aguade_model.jl"))
end
if !@isdefined(MougiParams)
    include(joinpath(@__DIR__, "..", "other_models", "mougi_model.jl"))
end
if !@isdefined(StoufferParams)
    include(joinpath(@__DIR__, "..", "other_models", "stouffer_model.jl"))
end

# Inlined from glvhoi_utils.jl to avoid pulling in the full HC system builder.
# Standard: A_eff = (1-α)·A, B_eff = α·B.  Gibbs: A_eff = -A, B_eff = -B.
function _prescale_taylor(A::AbstractMatrix{<:Real},
                          B::AbstractArray{<:Real,3},
                          α::Real,
                          is_gibbs::Bool)
    if is_gibbs
        return -A, -B
    else
        return (1 - α) .* A, α .* B
    end
end

# Inlined from model_store_utils.jl:83.  For unique_equilibrium / all_negative
# banks, r is not stored — it's derived from A, B, α so that x* = ones(n) is
# an equilibrium: r_i(α) = −[(1−α) Σ_j A[i,j] + α Σ_{j,k} B[j,k,i]].
function _compute_r_unique_equilibrium_taylor(A::AbstractMatrix{<:Real},
                                              B::AbstractArray{<:Real,3},
                                              α::Real)
    n = size(A, 1)
    r = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        row_A = 0.0
        for j in 1:n
            row_A += A[i, j]
        end
        sum_B_i = 0.0
        for j in 1:n, k in 1:n
            sum_B_i += B[j, k, i]
        end
        r[i] = -((1 - α) * row_A + α * sum_B_i)
    end
    return r
end

# --------------------------------------------------------------------------
# Core math kernel
# --------------------------------------------------------------------------

"""
    compute_alpha_eff_taylor(f, x_star) -> NamedTuple

Taylor-expansion effective nonlinearity of the per-capita rate `f` around
`x_star`.  `f` must be a callable `x::AbstractVector -> AbstractVector` that
is generic in eltype (so ForwardDiff can run through it).

Returns `(alpha_eff_taylor, P, Q, T1, T2)`.  Index convention for the
internal Hessian is `H[i, j, k] = ∂²f_i / (∂x_j ∂x_k)`.
"""
function compute_alpha_eff_taylor(f, x_star::AbstractVector{<:Real})
    n  = length(x_star)
    xs = Float64.(x_star)

    J = ForwardDiff.jacobian(f, xs)                   # n × n
    # Nested jacobian: Hflat[(j-1)*n + i, k] = ∂J[i,j]/∂x_k = ∂²f_i/(∂x_j ∂x_k).
    Hflat = ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(f, y)), xs)
    H = Array{Float64,3}(undef, n, n, n)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        H[i, j, k] = Hflat[(j - 1) * n + i, k]
    end

    T1 = J * xs
    T2 = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n, k in 1:n
            s += H[i, j, k] * xs[j] * xs[k]
        end
        T2[i] = 0.5 * s
    end

    P = mean(abs, T1)
    Q = mean(abs, T2)
    α_taylor = (P + Q) > 0 ? Q / (P + Q) : 0.0
    return (alpha_eff_taylor = α_taylor, P = P, Q = Q, T1 = T1, T2 = T2)
end

# --------------------------------------------------------------------------
# AD-safe per-capita rate functions for the 5 published models.
# Each is eltype-generic and allocating; the existing model files' mutating
# helpers are left untouched to preserve HC/ODE numerics.
# --------------------------------------------------------------------------

# Lever plant-pollinator model.  Mu is zero in all stored banks; we assert
# that to avoid silently masking non-factorable dynamics.
function lever_per_capita_rates_ad(p::LeverOriginalParams,
                                   x::AbstractVector{T},
                                   dr_full::AbstractVector = zeros(T, p.Sp + p.Sa)
                                   ) where {T <: Real}
    (p.muP == 0 && p.muA == 0) || error(
        "lever_per_capita_rates_ad: nonzero immigration (muP=$(p.muP), muA=$(p.muA)) " *
        "breaks the x·f factoring; original f is only per-capita when muP=muA=0."
    )
    Sp = p.Sp; Sa = p.Sa
    n  = Sp + Sa
    F  = Vector{T}(undef, n)

    @inbounds for i in 1:Sp
        m = zero(T); comp = zero(T)
        for k in 1:Sa; m    += p.GP[i, k] * x[Sp + k]; end
        for j in 1:Sp; comp += p.CP[i, j] * x[j];      end
        benefit = m / (1 + p.hP[i] * m)
        F[i] = (p.rP[i] + dr_full[i]) + benefit - comp
    end
    @inbounds for k in 1:Sa
        idx = Sp + k
        q = zero(T); comp = zero(T)
        for i in 1:Sp; q    += p.GA[k, i] * x[i];        end
        for l in 1:Sa; comp += p.CA[k, l] * x[Sp + l];   end
        benefit = q / (1 + p.hA[k] * q)
        F[idx] = (p.rA[k] + dr_full[idx]) - p.dA + benefit - comp
    end
    return F
end

# Karatayev food-web.  Consumer per-capita rate = G_k − β_k·x_k (the −β·x²
# term in the ODE contributes −β·x to the per-capita rate).
function karatayev_per_capita_rates_ad(p::KaratayevParams,
                                       x::AbstractVector{T},
                                       dr_full::AbstractVector = zeros(T, p.n_R + p.n_C)
                                       ) where {T <: Real}
    n_R = p.n_R; n_C = p.n_C; a = p.a
    n   = n_R + n_C
    F   = Vector{T}(undef, n)

    @inbounds for i in 1:n_R
        r_eff = p.r[i] + dr_full[i]
        comp = zero(T)
        for j in 1:n_R
            comp += (j == i ? one(T) : a * one(T)) * x[j]
        end
        # (1-a)*x[i] + a*Σ_j x[j]  =  x[i] + a*(Σ_{j≠i} x[j])
        # Rewriting to match existing code exactly:
        comp_exact = (1 - a) * x[i]
        for j in 1:n_R; comp_exact += a * x[j]; end

        eat_sum = zero(T)
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

        F[i] = r_eff * (1 - comp_exact / p.K[i]) - eat_sum
    end

    @inbounds for k in 1:n_C
        idx   = n_R + k
        m_eff = p.m0 + dr_full[idx]

        if p.feedback_mode == "FMI"
            Gk = zero(T)
            for i in 1:n_R
                Gk += p.delta[k] * p.b[k] * p.specials[i, k] *
                      (1 - p.f[i] * x[i] / p.K[i]) * x[i]
            end
            Gk -= m_eff
        else  # RMI
            S_k    = zero(T)
            RMIlev = zero(T)
            for i in 1:n_R
                S_k    += p.specials[i, k] * x[i]
                RMIlev += p.comps[i, k] * p.f[i] * x[i] / p.K[i]
            end
            Gk = p.delta[k] * p.b[k] * S_k * (1 - RMIlev) - m_eff
        end

        F[idx] = Gk - p.beta[k] * x[idx]
    end

    return F
end

# Aguadé-Gorgorió Allee-effect model.
function aguade_per_capita_rates_ad(p::AguadeParams,
                                    x::AbstractVector{T},
                                    dr_full::AbstractVector = zeros(T, p.n)
                                    ) where {T <: Real}
    n = p.n
    F = Vector{T}(undef, n)
    @inbounds for i in 1:n
        sat  = zero(T)
        comp = zero(T)
        for j in 1:n
            sat  += p.A[i, j] * x[j] / (p.gamma[j] + x[j])
            comp += p.B[i, j] * x[j]
        end
        F[i] = sat - (p.d[i] + dr_full[i]) - comp
    end
    return F
end

# Mougi ecosystem-engineering food-web.  Inlines _fill_mougi_rate_terms! body
# with fresh allocations (the existing Float64 buffers break AD).
function mougi_per_capita_rates_ad(p::MougiParams,
                                   x::AbstractVector{T},
                                   dr_full::AbstractVector = zeros(T, p.n)
                                   ) where {T <: Real}
    n = p.n
    B_vec = ones(T, n)
    G_vec = ones(T, n)
    D_vec = ones(T, n)
    prey_gain = zeros(T, n)

    @inbounds for i in 1:n
        if p.receiver_mask[i]
            for engineer in 1:n
                p.engineer_mask[engineer] || continue
                denom = x[engineer] + p.E0[engineer]
                B_vec[i] += (p.betaE[engineer, i] - 1.0) * x[engineer] / denom
                G_vec[i] += (p.gammaE[engineer, i] - 1.0) * x[engineer] / denom
            end
        end
        for prey in 1:n
            a_ip = p.a[i, prey]
            a_ip == 0.0 && continue
            D_vec[i]     += p.h[i, prey] * a_ip * x[prey]
            prey_gain[i] += p.e[i, prey] * a_ip * x[prey]
        end
    end

    F = Vector{T}(undef, n)
    @inbounds for i in 1:n
        gain = G_vec[i] * prey_gain[i] / D_vec[i]
        loss = zero(T)
        for predator in 1:n
            a_pi = p.a[predator, i]
            a_pi == 0.0 && continue
            loss += G_vec[predator] * a_pi * x[predator] / D_vec[predator]
        end
        F[i] = (p.r0[i] + dr_full[i]) * B_vec[i] - p.s[i] * x[i] + gain - loss
    end
    return F
end

# Stouffer food-web.  Mirrors stouffer_percapita_growth but eltype-generic
# (existing function has Float64(...) casts that would throw on ForwardDiff.Dual).
function stouffer_per_capita_rates_ad(p::StoufferParams,
                                      x::AbstractVector{T},
                                      dr_full::AbstractVector = zeros(T, p.n)
                                      ) where {T <: Real}
    n = p.n
    D = Vector{T}(undef, n)
    @inbounds for i in 1:n
        denom = p.B0 * one(T)
        for prey in p.prey_lists[i]
            denom += p.w[i, prey] * x[prey]
        end
        D[i] = denom
    end

    F = zeros(T, n)
    basal_sum = zero(T)
    @inbounds for j in 1:n
        p.basal_mask[j] || continue
        basal_sum += x[j]
    end

    @inbounds for i in 1:n
        if p.basal_mask[i]
            F[i] = (one(T) + dr_full[i]) * (one(T) - basal_sum / p.K)
        else
            gain_num = zero(T)
            for prey in p.prey_lists[i]
                gain_num += p.w[i, prey] * x[prey]
            end
            F[i] = (-p.x[i] + dr_full[i]) + p.x[i] * p.y[i] * gain_num / D[i]
        end
    end

    @inbounds for predator in 1:n
        Dk = D[predator]
        # Keep the original model's behavior: skip predator rows with Dk == 0.
        # For positive x_star this branch is never taken.
        Dk == 0 && continue
        for prey in p.prey_lists[predator]
            loss = p.x[predator] * p.y[predator] * x[predator] *
                   p.w[predator, prey] * x[prey] / (p.e[predator, prey] * Dk)
            F[prey] -= loss
        end
    end

    return F
end

# --------------------------------------------------------------------------
# JSON bank dispatch
# --------------------------------------------------------------------------

const _GLVHOI_MODES = ("standard", "unique_equilibrium", "all_negative")

"""
    build_per_capita_rates(bank) -> (f, x_star, n)

Reconstruct the ORIGINAL per-capita rate `f: x -> f_vec` for a bank JSON
dict whose `dynamics_mode` is one of "gibbs", "lever", "karatayev",
"aguade", "mougi", "stouffer".  Errors on the GLV+HOI modes
(use `build_per_capita_rates_for_alpha` instead) and on unknown modes.
"""
function build_per_capita_rates(bank::AbstractDict)
    mode = String(get(bank, "dynamics_mode", ""))
    x_star = Float64.(bank["x_star"])
    n = length(x_star)

    if mode == "gibbs"
        # Gibbs bakes α into the stored A, B: F_i(x) = r_i − (A*x)_i − (B*x*x)_i.
        A = nested_to_matrix(bank["A"])
        B = nested_to_tensor3(bank["B"])
        r = Float64.(bank["r"])
        A_eff, B_eff = _prescale_taylor(A, B, 0.0, true)  # is_gibbs=true → A_eff=-A, B_eff=-B
        f = x -> _glvhoi_per_capita_rates(A_eff, B_eff, r, x)
        return (f, x_star, n)
    elseif mode == "lever"
        p = lever_params_from_payload(bank)
        f = x -> lever_per_capita_rates_ad(p, x)
        return (f, x_star, n)
    elseif mode == "karatayev"
        p = karatayev_params_from_payload(bank)
        f = x -> karatayev_per_capita_rates_ad(p, x)
        return (f, x_star, n)
    elseif mode == "aguade"
        p = aguade_params_from_payload(bank)
        f = x -> aguade_per_capita_rates_ad(p, x)
        return (f, x_star, n)
    elseif mode == "mougi"
        p = mougi_params_from_payload(bank)
        f = x -> mougi_per_capita_rates_ad(p, x)
        return (f, x_star, n)
    elseif mode == "stouffer"
        p = stouffer_params_from_payload(bank)
        f = x -> stouffer_per_capita_rates_ad(p, x)
        return (f, x_star, n)
    elseif mode in _GLVHOI_MODES
        error("build_per_capita_rates: mode=$mode needs an α; use build_per_capita_rates_for_alpha.")
    else
        error("build_per_capita_rates: unknown dynamics_mode=$(repr(mode))")
    end
end

"""
    build_per_capita_rates_for_alpha(bank, α) -> (f, x_star, n)

GLV+HOI dispatcher: for `dynamics_mode` in standard/balanced/…
or gibbs, reconstruct the per-capita rate at mixing parameter α.  For
gibbs, α is ignored (the bank stores A, B with α already baked in).
"""
function build_per_capita_rates_for_alpha(bank::AbstractDict, α::Real)
    mode = String(get(bank, "dynamics_mode", ""))
    x_star = Float64.(bank["x_star"])
    n = length(x_star)
    A = nested_to_matrix(bank["A"])
    B = nested_to_tensor3(bank["B"])
    α_f = Float64(α)

    if mode == "gibbs"
        A_eff, B_eff = _prescale_taylor(A, B, 0.0, true)
    elseif mode in _GLVHOI_MODES
        A_eff, B_eff = _prescale_taylor(A, B, α_f, false)
    else
        error("build_per_capita_rates_for_alpha: mode=$(repr(mode)) is not a GLV+HOI mode.")
    end

    # unique_equilibrium / all_negative banks don't store r — it's derived
    # from A, B, α on the fly to pin x* = ones(n) as equilibrium.
    r = if haskey(bank, "r")
        Float64.(bank["r"])
    elseif mode in ("unique_equilibrium", "all_negative")
        _compute_r_unique_equilibrium_taylor(A, B, α_f)
    else
        error("build_per_capita_rates_for_alpha: bank missing 'r' and mode=$(repr(mode)) has no r(α) derivation rule.")
    end

    f = x -> _glvhoi_per_capita_rates(A_eff, B_eff, r, x)
    return (f, x_star, n)
end

# Eltype-generic GLV+HOI per-capita rate
# F_i(x) = r[i] + (A_eff*x)[i] + Σ_{jk} B_eff[j,k,i] * x[j] * x[k].
function _glvhoi_per_capita_rates(A_eff::AbstractMatrix{<:Real},
                                  B_eff::AbstractArray{<:Real,3},
                                  r::AbstractVector{<:Real},
                                  x::AbstractVector{T}) where {T <: Real}
    n = length(x)
    F = Vector{T}(undef, n)
    @inbounds for i in 1:n
        lin = zero(T)
        for j in 1:n
            lin += A_eff[i, j] * x[j]
        end
        quad = zero(T)
        for j in 1:n, k in 1:n
            quad += B_eff[j, k, i] * x[j] * x[k]
        end
        F[i] = r[i] + lin + quad
    end
    return F
end

# GLV+HOI mode list helper (re-exported for the driver).
is_glvhoi_mode(mode::AbstractString) = String(mode) in _GLVHOI_MODES
