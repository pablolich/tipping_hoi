#!/usr/bin/env julia
#
# Brose–Williams allometric trophic network (ATN) — the food-web family for
# Fig. 4b, shipped at type III (`h = 2`, `c = 0`), split producers, uniform
# weights `w_ij = 1/n_i`, `M = Z^T` masses and a niche web at C = 0.15.
#
#     producers   dB_i/dt =  r_i G_i B_i          - sum_k x_k y B_k F_ki / e_ki
#     consumers   dB_i/dt = -x_i B_i + x_i y B_i sum_j F_ij
#                                                 - sum_k x_k y B_k F_ki / e_ki
#
#     F_ij = w_ij B_j^h / Q_i,   Q_i = B0^h + c B_i B0^h + sum_k w_ik B_k^h
#     x_i  = (a_x/a_r) (M_i/M_b)^(-1/4),  M_i = M_b Z^T_i
#     T_i  = shortest prey-path length to a basal species
#
# Citation chain: Yodzis & Innes 1992 (allometry) -> Williams, Brose & Martinez
# 2007 (n-species extension) -> Brose et al. 2006b (this parameterisation:
# w_ij = 1/n_i, B0 = 0.5, M = Z^T, the (h, c) switches).  See
# review-1_responses/atn_bank_plan.md §2 for what is convention and what is
# measurement.
#
# S&B 2010 (`other_models/stouffer_model.jl`) is the same family at h = 1, c = 0
# with a shared K, so this file mirrors that one's shape line for line — with one
# deliberate exception:
#
#   THE PREDATION LOSS IS LINEAR IN PREY BIOMASS.  In dB_i/dt the loss term is
#   B_i^h * sum_k (...), so the PER-CAPITA loss carries B_i^(h-1), not B_i^h.
#   stouffer_model.jl:458 is off by a factor of B_prey; that bug is the reason
#   this family exists.  Verified to 4.5e-16 in
#   review-1_responses/scratch/atn_coexistence/validate.py.
#
# The implementation is a PORT of review-1_responses/scratch/atn_coexistence/
# atn.py (+ atn_fold_probe/arm_model.py for `y` as a parameter and for the
# perturbed per-capita rate G), not a re-derivation.  The cleared system is a
# port of review-1_responses/scratch/atn_preflight/time_atn.jl:42-99, whose
# degrees were validated against sympy (12/12) and against
# HomotopyContinuation.degrees (5/5).

using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using SciMLBase
using Sundials                     # DifferentialEquations does not re-export CVODE_BDF
using HomotopyContinuation

if !@isdefined(LeverOriginalParams)
    include(joinpath(@__DIR__, "lever_model.jl"))
end

const ATN_DEFAULT_AR = 1.0
const ATN_DEFAULT_AX = 0.314       # invertebrate (Brose et al. 2006b p. 1231)
const ATN_DEFAULT_Y = 8.0          # a_y/a_x; f_J ~ 0.41 of Yodzis & Innes's ceiling
const ATN_DEFAULT_B0 = 0.5
const ATN_DEFAULT_K = 1.0
const ATN_DEFAULT_MB = 1.0
const ATN_DEFAULT_Z = 100.0
const ATN_E_HERB = 0.45            # efficiency on a basal prey
const ATN_E_CARN = 0.85            # efficiency on a consumer prey
const ATN_EXT_FLOOR = 1e-30        # Brose / S&B hard extinction floor
const ATN_DEFAULT_H = 2.0          # type III; h = 1 is type II and does not fold
const ATN_DEFAULT_C = 0.0          # c = 1 is Beddington-DeAngelis interference
const ATN_DEFAULT_CONNECTANCE = 0.15

struct ATNParams
    n::Int
    connectance::Float64
    adj::Matrix{Bool}
    basal_mask::Vector{Bool}
    prey_lists::Vector{Vector{Int}}
    pred_lists::Vector{Vector{Int}}
    w::Matrix{Float64}
    M::Vector{Float64}
    x::Vector{Float64}
    e::Matrix{Float64}
    h::Float64
    c::Float64
    y::Float64
    producers::String
    n_p::Int
    K::Float64
    B0::Float64
    Mb::Float64
    Z::Float64
    ar::Float64
    ax::Float64
end

# ------------------------------------------------------------------ payload --

function _atn_matrix_rows(M::AbstractMatrix)
    return [collect(@view M[i, :]) for i in 1:size(M, 1)]
end

function _atn_to_matrix_bool(x)
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

function _atn_get_with_default(x, key::AbstractString, default)
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

function _atn_prey_lists(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    return [findall(@view adj[i, :]) for i in 1:n]
end

function _atn_pred_lists(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    return [findall(@view adj[:, i]) for i in 1:n]
end

function ATNParams(connectance::Real,
                   adj::AbstractMatrix{Bool},
                   basal_mask::AbstractVector{Bool},
                   w::AbstractMatrix{<:Real},
                   M::AbstractVector{<:Real},
                   x::AbstractVector{<:Real},
                   e::AbstractMatrix{<:Real};
                   h::Real = ATN_DEFAULT_H,
                   c::Real = ATN_DEFAULT_C,
                   y::Real = ATN_DEFAULT_Y,
                   producers::AbstractString = "split",
                   n_p::Union{Nothing,Integer} = nothing,
                   K::Real = ATN_DEFAULT_K,
                   B0::Real = ATN_DEFAULT_B0,
                   Mb::Real = ATN_DEFAULT_MB,
                   Z::Real = ATN_DEFAULT_Z,
                   ar::Real = ATN_DEFAULT_AR,
                   ax::Real = ATN_DEFAULT_AX)
    n = size(adj, 1)
    size(adj, 2) == n || error("ATN adjacency must be square")
    length(basal_mask) == n || error("basal_mask length mismatch")
    size(w) == (n, n) || error("w size mismatch")
    size(e) == (n, n) || error("e size mismatch")
    length(M) == n || error("M length mismatch")
    length(x) == n || error("x length mismatch")
    producers in ("split", "shared") ||
        error("ATN producers must be \"split\" or \"shared\", got $(repr(producers))")

    adj_m = Matrix{Bool}(adj)
    mask = Bool.(basal_mask)
    prey_lists = _atn_prey_lists(adj_m)
    pred_lists = _atn_pred_lists(adj_m)

    # n_p parameterises the productivity model and is fixed at construction: it
    # must not follow extinctions, or a restricted sub-model would sit at a
    # different equilibrium from the run it was carved out of (atn.py:129).
    np = n_p === nothing ? count(mask) : Int(n_p)
    np >= 1 || error("ATN requires at least one basal species")

    return ATNParams(
        n,
        Float64(connectance),
        adj_m,
        mask,
        prey_lists,
        pred_lists,
        Matrix{Float64}(w),
        Float64.(M),
        Float64.(x),
        Matrix{Float64}(e),
        Float64(h),
        Float64(c),
        Float64(y),
        String(producers),
        np,
        Float64(K),
        Float64(B0),
        Float64(Mb),
        Float64(Z),
        Float64(ar),
        Float64(ax),
    )
end

@inline atn_n_species(p::ATNParams) = p.n
@inline atn_is_consumer(p::ATNParams, i::Int) = !p.basal_mask[i]

function atn_params_payload(p::ATNParams)
    return (
        atn_n           = p.n,
        atn_connectance = p.connectance,
        atn_adj         = _atn_matrix_rows(p.adj),
        atn_basal_mask  = collect(p.basal_mask),
        atn_w           = _atn_matrix_rows(p.w),
        atn_M           = copy(p.M),
        atn_x           = copy(p.x),
        atn_e           = _atn_matrix_rows(p.e),
        atn_h           = p.h,
        atn_c           = p.c,
        atn_y           = p.y,
        atn_producers   = p.producers,
        atn_np          = p.n_p,
        atn_K           = p.K,
        atn_B0          = p.B0,
        atn_Mb          = p.Mb,
        atn_Z           = p.Z,
        atn_ar          = p.ar,
        atn_ax          = p.ax,
    )
end

function atn_params_from_payload(payload)
    n_raw = _atn_get_with_default(payload, "atn_n", nothing)
    n_raw === nothing && (n_raw = _atn_get_with_default(payload, "n", nothing))
    n_raw === nothing && error("atn_params_from_payload requires atn_n or n")
    n = Int(n_raw)
    adj = _atn_to_matrix_bool(_dict_or_prop_get(payload, "atn_adj"))
    size(adj) == (n, n) || error("atn_adj size mismatch")

    return ATNParams(
        Float64(_dict_or_prop_get(payload, "atn_connectance")),
        adj,
        Bool.(_dict_or_prop_get(payload, "atn_basal_mask")),
        _to_matrix_f64(_dict_or_prop_get(payload, "atn_w")),
        Vector{Float64}(_dict_or_prop_get(payload, "atn_M")),
        Vector{Float64}(_dict_or_prop_get(payload, "atn_x")),
        _to_matrix_f64(_dict_or_prop_get(payload, "atn_e"));
        h         = Float64(_atn_get_with_default(payload, "atn_h", ATN_DEFAULT_H)),
        c         = Float64(_atn_get_with_default(payload, "atn_c", ATN_DEFAULT_C)),
        y         = Float64(_atn_get_with_default(payload, "atn_y", ATN_DEFAULT_Y)),
        producers = String(_atn_get_with_default(payload, "atn_producers", "split")),
        n_p       = Int(_atn_get_with_default(payload, "atn_np", count(Bool.(_dict_or_prop_get(payload, "atn_basal_mask"))))),
        K         = Float64(_atn_get_with_default(payload, "atn_K", ATN_DEFAULT_K)),
        B0        = Float64(_atn_get_with_default(payload, "atn_B0", ATN_DEFAULT_B0)),
        Mb        = Float64(_atn_get_with_default(payload, "atn_Mb", ATN_DEFAULT_MB)),
        Z         = Float64(_atn_get_with_default(payload, "atn_Z", ATN_DEFAULT_Z)),
        ar        = Float64(_atn_get_with_default(payload, "atn_ar", ATN_DEFAULT_AR)),
        ax        = Float64(_atn_get_with_default(payload, "atn_ax", ATN_DEFAULT_AX)),
    )
end

function atn_baseline_r(p::ATNParams)
    r = zeros(Float64, p.n)
    @inbounds for i in 1:p.n
        r[i] = p.basal_mask[i] ? 1.0 : -p.x[i]
    end
    return r
end

# ------------------------------------------------------------- niche  model --
#
# This is atn.py:niche_web, NOT _sample_stouffer_niche_web.  The two differ, and
# the differences are load-bearing:
#   * the smallest niche value is FORCED basal (r[1] = 0);
#   * the range centre is drawn on [r_i/2, n_i], not on [r_i/2, min(n_i, 1-r_i/2)];
#   * SELF-LINKS ARE ALLOWED and are not filtered out.  A species whose only prey
#     is itself then fails the reachability test below and the web is redrawn, so
#     the allowance costs nothing but must be preserved: filtering self-links out
#     here would change both the realised connectance and the basal mask.

function _atn_all_reach_basal(adj::AbstractMatrix{Bool},
                              basal_mask::AbstractVector{Bool})
    n = size(adj, 1)
    reach = copy(Bool.(basal_mask))
    for _ in 1:n
        changed = false
        @inbounds for i in 1:n
            reach[i] && continue
            for j in 1:n
                if adj[i, j] && reach[j]
                    reach[i] = true
                    changed = true
                    break
                end
            end
        end
        changed || break
    end
    return all(reach)
end

"""
    atn_niche_web(n, connectance; rng, max_tries) -> (adj::Matrix{Bool}, basal_mask::Vector{Bool})

Williams & Martinez (2000) niche model, energetically feasible, in atn.py's
parameterisation.  `adj[i, j] = true` means i eats j; species are ordered by
niche value.  Redraws on: no basal species, any species of total degree zero,
or any species without a directed prey-path down to a basal species.
"""
function atn_niche_web(n::Int,
                       connectance::Real = ATN_DEFAULT_CONNECTANCE;
                       rng::AbstractRNG = Random.default_rng(),
                       max_tries::Int = 5000)
    n >= 2 || error("atn n must be >= 2, got $(n)")
    C = Float64(connectance)
    0.0 < C < 0.5 || error("ATN connectance must lie in (0, 0.5), got $(C)")
    beta_dist = Beta(1.0, 1.0 / (2.0 * C) - 1.0)

    for _ in 1:max_tries
        niche = sort(rand(rng, n))
        r = niche .* rand(rng, beta_dist, n)
        r[1] = 0.0                                   # smallest niche forced basal

        adj = falses(n, n)
        @inbounds for i in 1:n
            lo_c = r[i] / 2.0
            hi_c = niche[i]
            # numpy's rng.uniform(low, high) returns low when low == high
            centre = hi_c > lo_c ? lo_c + (hi_c - lo_c) * rand(rng) : lo_c
            lo = centre - r[i] / 2.0
            hi = centre + r[i] / 2.0
            for j in 1:n
                if lo <= niche[j] <= hi
                    adj[i, j] = true                 # j == i permitted
                end
            end
        end

        has_prey = vec(any(adj; dims = 2))
        is_prey  = vec(any(adj; dims = 1))
        basal_mask = .!has_prey
        any(basal_mask) || continue
        all(has_prey .| is_prey) || continue         # no isolated species
        _atn_all_reach_basal(adj, basal_mask) || continue
        return Matrix{Bool}(adj), Vector{Bool}(basal_mask)
    end

    error("atn_niche_web failed at n=$(n), C=$(C) after $(max_tries) attempts")
end

"""
    atn_trophic_depth(adj, basal_mask) -> Vector{Float64}

`T_i` = length of the shortest prey-path to a basal species (Williams 2008);
`Inf` if no such path exists.  Self-loops are excluded from the relaxation — a
self-loop reaches nothing.
"""
function atn_trophic_depth(adj::AbstractMatrix{Bool},
                           basal_mask::AbstractVector{Bool})
    n = size(adj, 1)
    T = [basal_mask[i] ? 0.0 : Inf for i in 1:n]
    for _ in 1:n
        changed = false
        @inbounds for i in 1:n
            basal_mask[i] && continue
            best = Inf
            for j in 1:n
                (adj[i, j] && j != i) || continue    # a self-loop reaches nothing
                T[j] < best && (best = T[j])
            end
            cand = 1.0 + best
            if cand < T[i]
                T[i] = cand
                changed = true
            end
        end
        changed || break
    end
    return T
end

"""
    atn_masses(adj, basal_mask; Mb, Z) -> Vector{Float64}

`M_i = M_b Z^T_i` with `T` the trophic depth; a non-finite `T` maps to 0.
"""
function atn_masses(adj::AbstractMatrix{Bool},
                    basal_mask::AbstractVector{Bool};
                    Mb::Real = ATN_DEFAULT_MB,
                    Z::Real = ATN_DEFAULT_Z)
    T = atn_trophic_depth(adj, basal_mask)
    return [Float64(Mb) * Float64(Z)^(isfinite(t) ? t : 0.0) for t in T]
end

"""
    atn_weights(adj) -> Matrix{Float64}

Brose 2006b's weak generalist: `w_ij = 1/n_i` on links, 0 off links, where `n_i`
is the number of prey of i (self-links included in the count, as in atn.py).
"""
function atn_weights(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    W = zeros(Float64, n, n)
    @inbounds for i in 1:n
        ni = count(@view adj[i, :])
        ni == 0 && continue
        for j in 1:n
            adj[i, j] && (W[i, j] = 1.0 / ni)
        end
    end
    return W
end

function _atn_assimilation_matrix(basal_mask::AbstractVector{Bool})
    n = length(basal_mask)
    # e[k, j] is the efficiency of predator k on prey j -> herbivory iff j is basal.
    # Stored full (not masked by adj), as atn.py does.
    E = Matrix{Float64}(undef, n, n)
    @inbounds for k in 1:n, j in 1:n
        E[k, j] = basal_mask[j] ? ATN_E_HERB : ATN_E_CARN
    end
    return E
end

"""
    sample_atn_params(n; rng, connectance, h, c, y, producers, ...) -> ATNParams

Draw order matches `scratch/atn_fold_probe/arm_model.py:build`: web, masses,
weights.  `x0 ~ U[0.05, 1]` is drawn by the caller, after this returns.
"""
function sample_atn_params(n::Int = 6;
                           rng::AbstractRNG = Random.default_rng(),
                           connectance::Real = ATN_DEFAULT_CONNECTANCE,
                           h::Real = ATN_DEFAULT_H,
                           c::Real = ATN_DEFAULT_C,
                           y::Real = ATN_DEFAULT_Y,
                           producers::AbstractString = "split",
                           K::Real = ATN_DEFAULT_K,
                           B0::Real = ATN_DEFAULT_B0,
                           Mb::Real = ATN_DEFAULT_MB,
                           Z::Real = ATN_DEFAULT_Z,
                           ar::Real = ATN_DEFAULT_AR,
                           ax::Real = ATN_DEFAULT_AX,
                           max_tries::Int = 5000)
    adj, basal_mask = atn_niche_web(n, connectance; rng = rng, max_tries = max_tries)
    M = atn_masses(adj, basal_mask; Mb = Mb, Z = Z)
    w = atn_weights(adj)

    x = zeros(Float64, n)
    @inbounds for i in 1:n
        basal_mask[i] && continue
        x[i] = (Float64(ax) / Float64(ar)) * (M[i] / Float64(Mb))^(-0.25)
    end

    e = _atn_assimilation_matrix(basal_mask)
    return ATNParams(connectance, adj, basal_mask, w, M, x, e;
        h = h, c = c, y = y, producers = producers,
        K = K, B0 = B0, Mb = Mb, Z = Z, ar = ar, ax = ax)
end

# ------------------------------------------------------------------ dynamics --

# Integer exponents keep the symbolic path polynomial and the complex step exact;
# the general branch exists only so the RHS still runs at non-integer h (which
# build_atn_cleared_system refuses).
@inline function _atn_pow(z, h::Real)
    h == 1 && return z
    isinteger(h) && return z^Int(h)
    return z^h
end

# THE single expression for Q.  atn_percapita_growth, atn_row_scale and
# build_atn_cleared_system all route through here, so the identity
# G_i(x, dr) == P_i(x) * F_i(x, dr) that A3's row-scaled λ_max rests on cannot
# drift between the rational RHS and the cleared polynomial.
function _atn_denoms(p::ATNParams, X::AbstractVector, ::Type{T}) where {T}
    n = p.n
    B0h = p.B0^p.h
    Q = Vector{T}(undef, n)
    @inbounds for i in 1:n
        q = convert(T, B0h)
        if p.c != 0.0
            q += p.c * B0h * X[i]        # Beddington-DeAngelis interference
        end
        for j in p.prey_lists[i]
            q += p.w[i, j] * _atn_pow(X[j], p.h)
        end
        Q[i] = q
    end
    return Q
end

"""
    atn_denominators(p, B) -> Q

`Q_i = B0^h + c B_i B0^h + sum_{j in prey(i)} w_ij B_j^h`.  Eltype-generic:
`Q_i >= B0^h > 0` for every non-negative `B`, so nothing downstream divides by 0.

NON-NEGATIVE is load-bearing when h is odd.  At even h (the shipped h = 2) the
bound holds at any real `B`, since `B_j^h >= 0`; at odd h it does not — h = 1,
one prey, `w_ij = 1`, `B_j = -1` gives `Q_i = -0.5`.  utils/hc_lambda_utils.jl's
`check_row_scale` throws on that, and its docstring says what that costs.
"""
function atn_denominators(p::ATNParams, B::AbstractVector)
    length(B) == p.n || error("atn_denominators expected B of length $(p.n), got $(length(B))")
    return _atn_denoms(p, B, promote_type(eltype(B), Float64))
end

"""
    atn_percapita_growth(p, B, dr_full) -> F

`F_i = (dB_i/dt) / B_i`, with `dr` an additive shift on the intrinsic per-capita
rate — `(1 + dr_i)` on producer growth, `(-x_i + dr_i)` on consumer metabolism
(atn_bank_plan.md §2b; the same convention as stouffer_model.jl:578).

ELTYPE-GENERIC AND COMPLEX-SAFE by contract: no `Float64(...)` casts on `B`, no
`abs`/`max`/`where` on `B`, everything allocated at `zero(T)`.  The complex-step
Jacobian and the ForwardDiff path in `utils/alpha_eff_taylor.jl` both depend on
that; a single cast here would silently zero the derivative.
"""
function atn_percapita_growth(p::ATNParams,
                              B::AbstractVector,
                              dr_full::AbstractVector)
    n = p.n
    length(B) == n || error("atn_percapita_growth expected B of length $(n), got $(length(B))")
    length(dr_full) == n ||
        error("atn_percapita_growth expected dr_full of length $(n), got $(length(dr_full))")

    T = promote_type(eltype(B), eltype(dr_full), Float64)
    Q = _atn_denoms(p, B, T)
    F = zeros(T, n)

    # producer growth
    if p.producers == "split"
        Ki = p.K / p.n_p
        @inbounds for i in 1:n
            p.basal_mask[i] || continue
            F[i] = (one(T) + dr_full[i]) * (one(T) - B[i] / Ki)
        end
    else                                    # "shared": rank-1, S&B 2010
        basal_sum = zero(T)
        @inbounds for j in 1:n
            p.basal_mask[j] && (basal_sum += B[j])
        end
        @inbounds for i in 1:n
            p.basal_mask[i] || continue
            F[i] = (one(T) + dr_full[i]) * (one(T) - basal_sum / p.K)
        end
    end

    # consumer metabolism + ingestion
    @inbounds for i in 1:n
        p.basal_mask[i] && continue
        gain = zero(T)
        for j in p.prey_lists[i]
            gain += p.w[i, j] * _atn_pow(B[j], p.h)
        end
        F[i] = (-p.x[i] + dr_full[i]) + p.x[i] * p.y * gain / Q[i]
    end

    # Predation loss, applied to EVERY species.  In dB_i/dt this is
    #   B_i^h * sum_{k in pred(i)} x_k y B_k w_ki / (e_ki Q_k),
    # LINEAR in prey biomass, so per capita it carries B_i^(h-1).
    @inbounds for i in 1:n
        isempty(p.pred_lists[i]) && continue
        acc = zero(T)
        for k in p.pred_lists[i]
            acc += p.x[k] * p.y * B[k] * p.w[k, i] / (p.e[k, i] * Q[k])
        end
        F[i] -= _atn_pow(B[i], p.h - 1.0) * acc
    end

    return F
end

function atn_percapita_growth(p::ATNParams, B::AbstractVector)
    return atn_percapita_growth(p, B, zeros(Float64, p.n))
end

"""
    atn_row_scale(p, x) -> P

`P_i = Q_i^[i is a consumer] * prod_{k in pred(i)} Q_k`, the factor
`build_atn_cleared_system` multiplies row `i` by.  Group-A fix A3 takes
`λ_max` from `diag(x ./ P) * J_G` rather than from `diag(x) * J_G`.
"""
function atn_row_scale(p::ATNParams, x::AbstractVector)
    Q = atn_denominators(p, x)
    T = eltype(Q)
    P = Vector{T}(undef, p.n)
    @inbounds for i in 1:p.n
        s = atn_is_consumer(p, i) ? Q[i] : one(T)
        for k in p.pred_lists[i]
            s *= Q[k]
        end
        P[i] = s
    end
    return P
end

function _make_atn_rhs_from_dr(p::ATNParams, dr_full::AbstractVector{<:Real})
    n = p.n
    length(dr_full) == n ||
        error("_make_atn_rhs_from_dr expected dr_full of length $(n), got $(length(dr_full))")
    dr = Float64.(dr_full)

    function f!(dx, B, _, _)
        F = atn_percapita_growth(p, B, dr)
        @inbounds for i in 1:n
            dx[i] = B[i] * F[i]
        end
        return nothing
    end

    return f!
end

function make_atn_rhs(p::ATNParams,
                      u_full::AbstractVector{<:Real},
                      delta::Real)
    n = p.n
    length(u_full) == n || error("make_atn_rhs expected u_full of length $(n), got $(length(u_full))")
    dr = Float64(delta) .* Float64.(u_full)
    return _make_atn_rhs_from_dr(p, dr)
end

"""
    integrate_atn_to_steady(p, x0; tmax, ...) -> (success, retcode, x_eq, du_max, extinct)

Stiff integration to a FIXED horizon under the hard extinction floor.  Three
choices differ from `integrate_stouffer_to_steady` and each is deliberate:

  * CVODE_BDF, not Tsit5 — the mass span is `Z^T` with `Z = 100`, so the
    consumer timescales span decades and the system is stiff by construction.
  * NO steady-state callback.  `tmax = 1e4` is the horizon every measurement in
    `scratch/atn_preflight` and `scratch/atn_direct_richness` used; stopping
    early on `max|dB/dt| < tol` would make this port's acceptance rate
    incomparable with the Python it is cross-checked against (plan §5 P3).
  * `abort_on_extinction` terminates the moment any species is floored.  Zero is
    absorbing for this model, so a floored species can never come back and the
    draw can no longer satisfy full persistence — this is exact, not a heuristic.
"""
function integrate_atn_to_steady(p::ATNParams,
                                 x0::AbstractVector{<:Real};
                                 tmax::Real = 1.0e4,
                                 reltol::Real = 1e-10,
                                 abstol::Real = 1e-12,
                                 ext_floor::Real = ATN_EXT_FLOOR,
                                 abort_on_extinction::Bool = true,
                                 maxiters::Integer = 10_000_000)
    n = p.n
    length(x0) == n || error("integrate_atn_to_steady expected x0 of length $(n), got $(length(x0))")

    f! = _make_atn_rhs_from_dr(p, zeros(Float64, n))
    u_init = Float64.(x0)
    prob = ODEProblem(f!, u_init, (0.0, Float64(tmax)))

    floor_val = Float64(ext_floor)
    extinct = Ref(false)

    condition = function (u, _, _)
        @inbounds for i in eachindex(u)
            (u[i] != 0.0 && u[i] <= floor_val) && return true
        end
        return false
    end

    affect! = function (integrator)
        u = integrator.u
        hit = false
        @inbounds for i in eachindex(u)
            if u[i] != 0.0 && u[i] <= floor_val
                u[i] = 0.0
                hit = true
            end
        end
        if hit
            extinct[] = true
            u_modified!(integrator, true)
            abort_on_extinction && terminate!(integrator)
        end
        return nothing
    end

    cb = DiscreteCallback(condition, affect!; save_positions = (false, false))

    sol = DifferentialEquations.solve(prob, CVODE_BDF();
        callback = cb,
        reltol = Float64(reltol),
        abstol = Float64(abstol),
        maxiters = Int(maxiters),
        save_everystep = false,
        save_start = false,
    )

    if !SciMLBase.successful_retcode(sol)
        return (success = false, retcode = string(sol.retcode),
                x_eq = fill(NaN, n), du_max = Inf, extinct = extinct[])
    end

    x_eq = Vector{Float64}(sol.u[end])
    @. x_eq = max(x_eq, 0.0)
    du_end = similar(x_eq)
    f!(du_end, x_eq, nothing, 0.0)
    return (success = true, retcode = string(sol.retcode),
            x_eq = x_eq, du_max = maximum(abs.(du_end)), extinct = extinct[])
end

"""
    atn_jacobian_complex_step(p, B, dr_full = zeros) -> J

Exact Jacobian of `dB/dt` by the complex step (atn.py:jac): perturb `B_j` by
`1e-30 im` and read `imag(rhs)/1e-30`.

A FINITE DIFFERENCE IS FORBIDDEN HERE (plan §6 criterion 5).  Elsewhere in this
revision a forward finite difference accepted 154 of 154 degenerate producer
monocultures as stable, because the verdict was set by the sign of the
differencing step.  The complex step has no subtractive cancellation and no step
to choose.
"""
function atn_jacobian_complex_step(p::ATNParams,
                                   B::AbstractVector{<:Real},
                                   dr_full::AbstractVector{<:Real})
    n = p.n
    length(B) == n || error("atn_jacobian_complex_step expected B of length $(n), got $(length(B))")
    length(dr_full) == n ||
        error("atn_jacobian_complex_step expected dr_full of length $(n), got $(length(dr_full))")

    step = 1e-30
    dr = Float64.(dr_full)
    Bc = ComplexF64.(B)
    J = zeros(Float64, n, n)

    @inbounds for j in 1:n
        probe = copy(Bc)
        probe[j] += im * step
        F = atn_percapita_growth(p, probe, dr)
        for i in 1:n
            J[i, j] = imag(probe[i] * F[i]) / step
        end
    end

    return J
end

function atn_jacobian_complex_step(p::ATNParams, B::AbstractVector{<:Real})
    return atn_jacobian_complex_step(p, B, zeros(Float64, p.n))
end

function atn_lambda_max_equilibrium(p::ATNParams,
                                    B::AbstractVector{<:Real},
                                    dr_full::AbstractVector{<:Real})
    J = atn_jacobian_complex_step(p, B, dr_full)
    return maximum(real.(eigvals(J)))
end

function atn_lambda_max_equilibrium(p::ATNParams, B::AbstractVector{<:Real})
    return atn_lambda_max_equilibrium(p, B, zeros(Float64, p.n))
end

# --------------------------------------------------- cleared HC polynomial --

"""
    build_atn_cleared_system(p) -> (System, x)

Row `i` of the rational per-capita rate multiplied through by
`P_i = Q_i^[consumer] * prod_{k in pred(i)} Q_k`, so `G = P .* F` exactly.
Parameters are `dr[1:n]`.

Degrees are `1 + h|pred(i)|` for producers and `h(1 + |pred(i)|)` for consumers
(validated against sympy 12/12 and HomotopyContinuation.degrees 5/5 in
`scratch/atn_preflight/`), so Bézout inflates as `2^(consumers)` when h goes
1 -> 2.  That is a structural statement, not a cost model: `boundary_scan.jl`
tracks one known solution along a `ParameterHomotopy` and never solves from
scratch.
"""
function build_atn_cleared_system(p::ATNParams)
    (isinteger(p.h) && p.h >= 1) || error(
        "build_atn_cleared_system requires an integer h >= 1, got h = $(p.h). " *
        "A non-integer exponent (e.g. Williams's q = 0.2, h = 1.2) is not " *
        "algebraic: HomotopyContinuation would need the substitution x = u^5, " *
        "costing 5^n paths. This refusal is a documented SI point " *
        "(atn_bank_plan.md §12.4) — it is a solver constraint, not a judgement " *
        "about that parameterisation.")

    n = p.n
    hi = Int(p.h)
    @var x[1:n] dr[1:n]

    Q = _atn_denoms(p, x, Expression)

    Ki = p.K / p.n_p
    basal_sum = sum(x[j] for j in 1:n if p.basal_mask[j])
    eqs = Vector{Expression}(undef, n)

    @inbounds for i in 1:n
        pred_prod = 1.0
        for k in p.pred_lists[i]
            pred_prod *= Q[k]
        end

        eq = if p.basal_mask[i]
            if p.producers == "split"
                (1.0 + dr[i]) * (1.0 - x[i] / Ki) * pred_prod
            else
                (1.0 + dr[i]) * (1.0 - basal_sum / p.K) * pred_prod
            end
        else
            gain = sum(p.w[i, j] * x[j]^hi for j in p.prey_lists[i])
            ((-p.x[i] + dr[i]) * Q[i] + p.x[i] * p.y * gain) * pred_prod
        end

        # per-capita loss carries x_i^(h-1): the loss is LINEAR in prey biomass
        xih = hi == 1 ? 1.0 : x[i]^(hi - 1)
        for k in p.pred_lists[i]
            coeff = p.x[k] * p.y * p.w[k, i] / p.e[k, i]
            other = atn_is_consumer(p, i) ? Q[i] : Expression(1.0)
            for o in p.pred_lists[i]
                o == k && continue
                other *= Q[o]
            end
            eq -= coeff * x[k] * xih * other
        end

        eqs[i] = expand(eq)
    end

    return System(eqs; variables = x, parameters = dr), x
end

function atn_alpha_eff_symbolic(p::ATNParams,
                                x_star::AbstractVector{<:Real},
                                dr_full::AbstractVector{<:Real})
    length(dr_full) == p.n ||
        error("atn_alpha_eff_symbolic expected dr_full of length $(p.n), got $(length(dr_full))")
    syst, _ = build_atn_cleared_system(p)
    return symbolic_alpha_eff(syst, x_star; parameter_values = Float64.(dr_full)).alpha_eff
end

function atn_alpha_eff_symbolic(p::ATNParams, x_star::AbstractVector{<:Real})
    return atn_alpha_eff_symbolic(p, x_star, zeros(Float64, p.n))
end

function atn_alpha_eff_monomial(p::ATNParams,
                                x_star::AbstractVector{<:Real},
                                dr_full::AbstractVector{<:Real})
    length(dr_full) == p.n ||
        error("atn_alpha_eff_monomial expected dr_full of length $(p.n), got $(length(dr_full))")
    syst, _ = build_atn_cleared_system(p)
    return symbolic_alpha_eff_monomial_abs(syst, x_star; parameter_values = Float64.(dr_full)).alpha_eff
end

function atn_alpha_eff_monomial(p::ATNParams, x_star::AbstractVector{<:Real})
    return atn_alpha_eff_monomial(p, x_star, zeros(Float64, p.n))
end
