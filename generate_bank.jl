#!/usr/bin/env julia
# Example terminal calls:
#   julia --startup-file=no new_code/generate_bank.jl 3:5 10 8,16
#   julia --startup-file=no new_code/generate_bank.jl 3 50 100
# All other settings (seed, sigmaA, etc.) are set in pipeline_config.jl.

using Random
using LinearAlgebra
using JSON3

include(joinpath(@__DIR__, "pipeline_config.jl"))
include(joinpath(@__DIR__, "model_store_utils.jl"))

function parse_int_list(s::AbstractString)
    t = strip(s)
    if occursin(":", t)
        parts = split(t, ":")
        if length(parts) == 2
            a = parse(Int, strip(parts[1]))
            b = parse(Int, strip(parts[2]))
            return collect(a:b)
        elseif length(parts) == 3
            a = parse(Int, strip(parts[1]))
            step = parse(Int, strip(parts[2]))
            b = parse(Int, strip(parts[3]))
            return collect(a:step:b)
        else
            error("Invalid range: $s")
        end
    end
    return [parse(Int, strip(x)) for x in split(t, ",") if !isempty(strip(x))]
end

function usage_error()
    msg = """
    Usage:
      julia new_code/generate_bank.jl <n_values> <n_models> <n_dirs> [--alpha-grid VALUES]

    Args:
      n_values   Comma list or range, e.g. 3,4,5 or 3:6
      n_models   Number of models per (n, n_dirs)
      n_dirs     Comma list or range, e.g. 16 or 8,16 or 4:4:20

    Options:
      --alpha-grid VALUES  Comma-separated alpha values, e.g. 0.5 or 0.1,0.5,0.9
                           Defaults to SCAN_ALPHA_GRID from pipeline_config.jl

    All other settings (seed, sigmaA, b_sampler, etc.) are in pipeline_config.jl.
    """
    error(msg)
end

function parse_args(args::Vector{String})
    length(args) < 3 && usage_error()

    positional   = String[]
    alpha_grid   = nothing

    i = 1
    while i <= length(args)
        if args[i] == "--alpha-grid" && i < length(args)
            alpha_grid = [parse(Float64, strip(v)) for v in split(args[i+1], ",") if !isempty(strip(v))]
            i += 2
        elseif startswith(args[i], "--")
            error("Unknown flag: $(args[i])")
        else
            push!(positional, args[i])
            i += 1
        end
    end

    length(positional) != 3 && usage_error()
    n_values      = parse_int_list(positional[1])
    n_models      = parse(Int, positional[2])
    n_dirs_values = parse_int_list(positional[3])

    isempty(n_values)      && error("n_values cannot be empty")
    isempty(n_dirs_values) && error("n_dirs cannot be empty")
    n_models < 1           && error("n_models must be >= 1")
    any(n_values .< 1)     && error("All n values must be >= 1")
    any(n_dirs_values .< 1) && error("All n_dirs values must be >= 1")

    alpha_grid_out = alpha_grid === nothing ? collect(Float64, SCAN_ALPHA_GRID) : alpha_grid
    isempty(alpha_grid_out) && error("alpha_grid cannot be empty")

    return n_values, n_models, n_dirs_values, alpha_grid_out
end

function _rand_dirichlet_ones(rng::AbstractRNG, m::Int)
    w = randexp(rng, m)
    w ./= sum(w)
    return w
end

function _unique_pairs_excluding_i(n::Int, i::Int)
    pairs = Tuple{Int,Int}[]
    for j in 1:n-1, k in j+1:n
        (j == i || k == i) && continue
        push!(pairs, (j, k))
    end
    return pairs
end

function sample_B_dirichlet_sum_neg_r(r::AbstractVector{<:Real};
    rng::AbstractRNG,
    mode::String="dirichlet",
    min_abs_sum::Float64=1e-12,
    max_slice_tries::Int=10_000)
    n = length(r)
    if mode == "dirichlet"
        B = zeros(Float64, n, n, n)
        for i in 1:n
            pairs = _unique_pairs_excluding_i(n, i)
            m = length(pairs)
            m == 0 && continue
            w = _rand_dirichlet_ones(rng, m)
            @inbounds for t in 1:m
                j, k = pairs[t]
                B[j, k, i] = w[t]
            end
            @views B[:, :, i] .*= -r[i]
        end
        return B
    elseif mode == "signed_flip"
        B = zeros(Float64, n, n, n)
        for i in 1:n
            pairs = _unique_pairs_excluding_i(n, i)
            m = length(pairs)
            m == 0 && continue
            accepted = false
            for _ in 1:max_slice_tries
                vals = randn(rng, m)
                s = sum(vals)
                if s > 0.0
                    vals .*= -1.0
                    s = -s
                end
                abs(s) <= min_abs_sum && continue
                scale = (-r[i]) / s
                @inbounds for t in 1:m
                    j, k = pairs[t]
                    B[j, k, i] = vals[t] * scale
                end
                accepted = true
                break
            end
            accepted || error("signed_flip B sampling failed for slice i=$i")
        end
        return B
    else
        error("Unknown B sampling mode: $mode")
    end
end

function hatB_from_tensor(B::Array{<:Real,3})
    n = size(B, 1)
    @assert size(B, 2) == n && size(B, 3) == n
    Bhat = zeros(Float64, n, n)
    @inbounds for i in 1:n
        for j in 1:n
            s = 0.0
            for k in 1:n
                s += B[j, k, i] + B[k, j, i]
            end
            Bhat[i, j] = s
        end
    end
    return Bhat
end

function negdef_diag_shift(M::AbstractMatrix{<:Real}; safety::Float64=1.2, eps::Float64=1e-12)
    S = Symmetric((M + M') / 2)
    lam_max = eigmax(S)
    delta = safety * max(0.0, lam_max) + eps
    return -delta
end

function stabilize_and_rescale_A_section2(A0::AbstractMatrix{<:Real}, r::AbstractVector{<:Real};
    safety::Float64=1.2, eps::Float64=1e-12)
    n = size(A0, 1)
    @assert size(A0, 2) == n
    @assert length(r) == n

    dA = negdef_diag_shift(A0; safety=safety, eps=eps)
    A1 = Matrix{Float64}(A0)
    @inbounds for i in 1:n
        A1[i, i] += dA
    end
    cA = (-Float64.(r)) ./ (-Float64.(r) .+ dA)
    A2 = Diagonal(cA) * A1
    return (A=A2, dA=dA, row_scale=cA)
end

function stabilize_and_rescale_B_section2(B0::Array{<:Real,3}, r::AbstractVector{<:Real};
    safety::Float64=1.2, eps::Float64=1e-12, half_shift::Bool=true)
    n = size(B0, 1)
    @assert size(B0, 2) == n && size(B0, 3) == n
    @assert length(r) == n

    B = Array{Float64,3}(B0)
    Bhat0 = hatB_from_tensor(B)
    dB = negdef_diag_shift(Bhat0; safety=safety, eps=eps)
    delta = half_shift ? (dB / 2) : dB
    @inbounds for i in 1:n
        B[i, i, i] += delta
    end
    cB = (-Float64.(r)) ./ (-Float64.(r) .+ delta)
    @inbounds for i in 1:n
        @views B[:, :, i] .*= cB[i]
    end
    return (B=B, dB=dB, slice_shift=delta, slice_scale=cB)
end

function sample_parameter_set_balanced(n::Int;
    sigmaA::Float64,
    sigmaB::Float64,
    muA::Float64    = 0.0,
    muB::Float64    = 0.0,
    rng::AbstractRNG,
    safety::Float64 = 1.2,
    eps::Float64    = 1e-9)

    r = ones(Float64, n)

    # 1. Draw: fully iid Gaussian (no pre-constrained diagonals)
    A0 = muA .+ (sigmaA / sqrt(Float64(n))) .* randn(rng, n, n)
    B0 = muB .+ (sigmaB / Float64(n))       .* randn(rng, n, n, n)

    # 2. Shift A: δ_A = safety·max(½λ_max(A0+A0ᵀ), max_row_sum) + ε
    # Multiplicative safety bounds rescaling factors to ≤ 1/((safety-1)·max(λ_A,s_A)).
    SA          = A0 + A0'
    lambda_A    = eigmax(Symmetric(SA)) / 2
    row_sums_A0 = vec(sum(A0, dims=2))
    s_A         = maximum(row_sums_A0)
    delta_A     = safety * max(lambda_A, s_A) + eps
    As          = A0 - delta_A * I

    # 3. Row-wise rescale A: (d_A)_i = 1/(δ_A − row_sum_i) → row sums = −1 exactly
    dA = 1.0 ./ (delta_A .- row_sums_A0)   # all positive: δ_A > s_A ≥ each row_sum
    A  = Diagonal(dA) * As

    # 4. Shift B (only pure-diagonal entries B[i,i,i]):
    #    shifting B[i,i,i] by δ_B moves B̂[i,i] by −2δ_B → (B̂+B̂ᵀ) by −4δ_B,
    #    so need δ_B > ¼λ_max(B̂0+B̂0ᵀ). Same multiplicative safety as A.
    Bhat0         = hatB_from_tensor(B0)
    SB            = Bhat0 + Bhat0'
    lambda_B      = eigmax(Symmetric(SB)) / 4
    slice_sums_B0 = [sum(@view B0[:, :, i]) for i in 1:n]
    s_B           = maximum(slice_sums_B0)
    delta_B       = safety * max(lambda_B, s_B) + eps
    Bs            = copy(B0)
    for i in 1:n
        Bs[i, i, i] -= delta_B
    end

    # 5. Row-wise rescale B (3rd index): (d_B)_i = 1/(δ_B − slice_sum_i) → sums = −1 exactly
    dB = 1.0 ./ (delta_B .- slice_sums_B0)  # all positive: δ_B > s_B ≥ each slice_sum
    B  = copy(Bs)
    for i in 1:n
        @views B[:, :, i] .*= dB[i]
    end

    # 6. Verify stability after row-wise rescaling (non-uniform scaling may break ND)
    A_stable    = eigmax(Symmetric((A + A') / 2)) < 0.0
    Bhat        = hatB_from_tensor(B)
    Bhat_stable = eigmax(Symmetric((Bhat + Bhat') / 2)) < 0.0
    (A_stable && Bhat_stable) || return nothing

    return r, A, B, delta_A, delta_B
end

function sample_parameter_set_balanced_stable(n::Int;
    sigmaA::Float64,
    sigmaB::Float64,
    rng::AbstractRNG,
    safety::Float64 = 1.2,
    eps::Float64    = 1e-12)

    n >= 2 || error("balanced_stable requires n >= 2")

    # Step 1: r = ones(n)
    r = ones(Float64, n)

    # Step 2: A off-diagonal ~ N(0, sigmaA²/n); diagonal enforces row sum = -1
    A0 = (sigmaA / sqrt(Float64(n))) .* randn(rng, n, n)
    A0[diagind(A0)] .= 0.0
    for i in 1:n
        A0[i, i] = -1.0 - sum(A0[i, :])
    end

    # Step 3: Stabilise A only — no delta_B needed (B̂ diagonal is pinned to A diagonal)
    S_A     = A0 + A0'
    delta_A = -negdef_diag_shift(S_A; safety=safety, eps=eps) / 2   # positive
    c_A     = 1.0 / (1.0 + delta_A)
    A       = copy(A0)
    for i in 1:n
        A[i, i] -= delta_A
    end
    A .*= c_A   # row sums: c_A * (-1 - delta_A) = -1 ✓

    # Step 4: B construction — balanced_stable recipe
    # All raw entries ~ N(0, sigmaB²/n²)
    B_raw = (sigmaB / Float64(n)) .* randn(rng, n, n, n)
    B     = zeros(Float64, n, n, n)

    for i in 1:n
        # ── Pure diagonal ────────────────────────────────────────────────────
        # B[i,i,i] = A[i,i]/2 so that the k=i contribution to
        # B̂_{ii} = Σ_k(B[i,k,i]+B[k,i,i]) is exactly A[i,i].
        B[i, i, i] = A[i, i] / 2.0

        # ── Non-pure diagonal shift (δ_i) ────────────────────────────────────
        # Require Σ_{k≠i}(B[i,k,i]+B[k,i,i]) = 0  ⟹  B̂_{ii} = A[i,i].
        # Raw contribution is Q_i; adding δ_i to each non-pure pair zeros it:
        #   Q_i + 2(n-1)·δ_i = 0  ⟹  δ_i = -Q_i / (2(n-1)).
        Q_i = 0.0
        for k in 1:n
            k == i && continue
            Q_i += B_raw[i, k, i] + B_raw[k, i, i]
        end
        delta_i = -Q_i / (2.0 * (n - 1))

        for k in 1:n
            k == i && continue
            B[i, k, i] = B_raw[i, k, i] + delta_i   # pair1 = i
            B[k, i, i] = B_raw[k, i, i] + delta_i   # pair2 = i
        end

        # ── Off-diagonal correction (ε_i) ────────────────────────────────────
        # After the above, slice sum = A[i,i]/2 + R_i.
        # Distribute the deficit uniformly over the (n-1)² off-diagonal pairs
        # to restore Σ_{j,k} B[j,k,i] = -1.
        R_i = 0.0
        for j in 1:n, k in 1:n
            (j == i || k == i) && continue
            R_i += B_raw[j, k, i]
        end
        corr = (-1.0 - A[i, i] / 2.0 - R_i) / Float64((n - 1)^2)

        for j in 1:n, k in 1:n
            (j == i || k == i) && continue
            B[j, k, i] = B_raw[j, k, i] + corr
        end
    end

    # Step 5: Stabilise B̂.
    # Shift B[i,i,i] by -δ_B and rescale each slice by c_B = 1/(1+δ_B).
    # This preserves slice sums = -1 exactly (c_B·(-1-δ_B) = -1) and
    # ensures (B̂+B̂ᵀ) ≺ 0.  The Jacobian diagonal becomes
    #   B̂_{ii} = c_B·(A[i,i] - 2δ_B) ≤ A[i,i],
    # so J_{ii}(α) is non-increasing in α (self-regulation does not weaken).
    Bhat0   = hatB_from_tensor(B)
    S_B     = Bhat0 + Bhat0'
    delta_B = -negdef_diag_shift(S_B; safety=safety, eps=eps) / 2   # positive
    c_B     = 1.0 / (1.0 + delta_B)
    for i in 1:n
        B[i, i, i] -= delta_B
        B[:, :, i] .*= c_B
    end

    return r, A, B, delta_A, delta_B
end

function sample_parameter_set_constrained_r(n::Int;
    sigmaA::Float64,
    sigmaB::Float64,
    muA::Float64    = 0.0,
    muB::Float64    = 0.0,
    rng::AbstractRNG,
    safety::Float64 = 1.2,
    eps::Float64    = 1e-9)

    # 1. Draw iid Gaussian (same scaling as balanced)
    A0 = muA .+ (sigmaA / sqrt(Float64(n))) .* randn(rng, n, n)
    B0 = muB .+ (sigmaB / Float64(n))       .* randn(rng, n, n, n)

    # 2. Shift A: δ_A = safety·max(½λ_max(A0+A0ᵀ), max_row_sum) + ε
    SA          = A0 + A0'
    lambda_A    = eigmax(Symmetric(SA)) / 2
    row_sums_A0 = vec(sum(A0, dims=2))
    s_A         = maximum(row_sums_A0)
    delta_A     = safety * max(lambda_A, s_A) + eps
    A           = A0 - delta_A * I    # shift only — NO row rescaling

    # 3. Plant equilibrium: r_i = -row_sum(A)[i] = δ_A - Σ_j A0[i,j]
    #    Guaranteed positive: δ_A > s_A ≥ each row_sum_A0[i]
    r = delta_A .- row_sums_A0

    # 4. Shift B pure-diagonal entries by δ_B
    Bhat0         = hatB_from_tensor(B0)
    SB            = Bhat0 + Bhat0'
    lambda_B      = eigmax(Symmetric(SB)) / 4
    slice_sums_B0 = [sum(@view B0[:, :, i]) for i in 1:n]
    s_B           = maximum(slice_sums_B0)
    delta_B       = safety * max(lambda_B, s_B) + eps
    Bs            = copy(B0)
    for i in 1:n
        Bs[i, i, i] -= delta_B
    end

    # 5. Rescale B slices to enforce slice_sum(B)[i] = -r[i]
    #    dB[i] = r[i] / (δ_B - slice_sum_B0[i]) — positive since δ_B > s_B and r > 0
    dB = r ./ (delta_B .- slice_sums_B0)
    B  = copy(Bs)
    for i in 1:n
        @views B[:, :, i] .*= dB[i]
    end

    # 6. Verify stability (non-uniform B rescaling may break negative-definiteness)
    A_stable    = eigmax(Symmetric((A + A') / 2)) < 0.0
    Bhat        = hatB_from_tensor(B)
    Bhat_stable = eigmax(Symmetric((Bhat + Bhat') / 2)) < 0.0
    (A_stable && Bhat_stable) || return nothing

    return r, A, B, delta_A, delta_B
end

function sample_parameter_set_elegant(n::Int;
    muA::Float64 = 0.0,
    muB::Float64 = 0.0,
    rng::AbstractRNG)

    r = ones(Float64, n)

    # 1. Sample A0 ~ N(μ_A, 1/n); pin diagonal to enforce zero row sums
    A0 = muA .+ randn(rng, n, n) / sqrt(Float64(n))
    for i in 1:n
        A0[i, i] = 0.0
        A0[i, i] = -sum(@view A0[i, :])
    end

    # 2. Spectral threshold δ_A
    #    S_A = (A0 + A0ᵀ)/2
    #    μ_A > 0 (facilitative): mean-field stabilising → use centered S_A − E[S_A]
    #    μ_A ≤ 0: mean-field neutral/destabilising → use full S_A
    S_A = (A0 + A0') / 2
    if muA > 0.0
        E_SA    = -Float64(n) * muA * (I - ones(n, n) / n)
        delta_A = max(eigmax(Symmetric(S_A - E_SA)), 1e-12)
    else
        delta_A = eigmax(Symmetric(S_A))
        delta_A <= 0.0 && return nothing  # degenerate sample
    end

    # 3. A = A0/δ_A − I  →  row sums = −1
    A = A0 / delta_A - I

    # 4. Sample B0 ~ N(μ_B, 1/n²); pin pure diagonal to enforce zero slice sums
    B0 = muB .+ randn(rng, n, n, n) / Float64(n)
    for i in 1:n
        B0[i, i, i] = 0.0
        B0[i, i, i] = -sum(@view B0[:, :, i])
    end

    # 5. Spectral threshold δ_B
    #    S̃ = B̂₀ + B̂₀ᵀ  (full symmetrized aggregation)
    #    μ_B > 0 (facilitative): mean-field stabilising → use centered Ŝ = S̃ − E[S̃]
    #    μ_B ≤ 0: mean-field neutral/destabilising → use full S̃
    Bhat0   = hatB_from_tensor(B0)
    S_tilde = Bhat0 + Bhat0'
    if muB > 0.0
        E_S     = -4.0 * n^2 * muB * (I - ones(n, n) / n)
        S_hat   = S_tilde - E_S
        delta_B = max(eigmax(Symmetric(S_hat)) / 4, 1e-12)
    else
        delta_B = max(eigmax(Symmetric(S_tilde)) / 4, 1e-12)
    end

    # 6. B = B0/δ_B − δ_{ijk}  →  slice sums = −1
    B = B0 / delta_B
    for i in 1:n
        B[i, i, i] -= 1.0
    end

    # 7. Verify stability: discard if (A+Aᵀ)/2 or (B̂+B̂ᵀ)/2 not negative-definite
    A_stable    = eigmax(Symmetric((A + A') / 2)) < 0.0
    Bhat        = hatB_from_tensor(B)
    Bhat_stable = eigmax(Symmetric((Bhat + Bhat') / 2)) < 0.0
    (A_stable && Bhat_stable) || return nothing

    return r, A, B, delta_A, delta_B
end

"""
    recover_B_from_T_slices(T_slices)

Given n symmetric Jacobian-slice matrices T_k, recover B[i,j,k] via minimum-norm
solution of:
  (T_k)_im = B[m,k,i] + B[k,m,i] + B[i,k,m] + B[k,i,m]   for all i ≤ m, k.

This is the contraction that controls the symmetric part of the per-capita Jacobian:
  J_F^s = (A_eff+A_eff')/2 + (α/2) Σ_k x_k T_k
"""
function sample_parameter_set_all_negative(n::Int;
    sigmaA::Float64,
    sigmaB::Float64,
    rng::AbstractRNG)

    # Half-normal A, negated:  A_ij = -|N(0, sigmaA²/n)|
    A = -abs.((sigmaA / sqrt(Float64(n))) .* randn(rng, n, n))

    # Half-normal B, negated:  B_ijk = -|N(0, sigmaB²/n²)|
    B = -abs.((sigmaB / Float64(n)) .* randn(rng, n, n, n))

    return A, B
end

function recover_B_from_T_slices(T_slices::Vector{Matrix{Float64}})
    n = size(T_slices[1], 1)
    @assert length(T_slices) == n

    n_unknowns = n^3
    n_upper = n * (n + 1) ÷ 2
    n_eqs = n * n_upper

    M = zeros(Float64, n_eqs, n_unknowns)
    s = zeros(Float64, n_eqs)

    lin(a, b, c) = a + n * (b - 1) + n^2 * (c - 1)

    row = 0
    for k in 1:n
        for i in 1:n, m in i:n
            row += 1
            s[row] = T_slices[k][i, m]
            # (T_k)_im = B[m,k,i] + B[k,m,i] + B[i,k,m] + B[k,i,m]
            for idx in (lin(m, k, i), lin(k, m, i), lin(i, k, m), lin(k, i, m))
                M[row, idx] += 1.0
            end
        end
    end

    b = M' * ((M * M') \ s)
    return reshape(b, n, n, n)
end

function sample_parameter_set_unique_equilibrium(n::Int; safety::Float64=1.2, rng::AbstractRNG)
    eps = 1e-12

    # 1. Draw A: off-diagonal ~ N(0, 1/n), diagonal = 0 temporarily
    A = randn(rng, n, n) / sqrt(n)
    @inbounds A[diagind(A)] .= 0.0

    # 2. Set diagonal to ensure (A+A')/2 ≺ 0
    S_A = Symmetric((A + A') / 2)
    d_A = safety * max(0.0, eigmax(S_A)) + eps
    @inbounds A[diagind(A)] .= -d_A

    # 3. Draw Jacobian-slice matrices T_k: off-diagonal ~ N(0, 1/n²), symmetric
    #    These control the symmetric part of the per-capita Jacobian:
    #    J_F^s = (A_eff+A_eff')/2 + (α/2) Σ_k x_k T_k
    T_slices = Vector{Matrix{Float64}}(undef, n)
    d_k_values = Vector{Float64}(undef, n)
    for k in 1:n
        Tk = zeros(Float64, n, n)
        for i in 1:n, j in i+1:n
            v = randn(rng) / n
            Tk[i, j] = v
            Tk[j, i] = v
        end
        lam_max = eigmax(Symmetric(Tk))
        d_k = safety * max(0.0, lam_max) + eps
        @inbounds Tk[diagind(Tk)] .= -d_k
        d_k_values[k] = d_k
        T_slices[k] = Tk
    end

    # 4. Recover B from Jacobian-slice matrices
    B = recover_B_from_T_slices(T_slices)

    return A, B, d_A, d_k_values
end

function sample_parameter_set(n::Int;
    sigmaA::Float64,
    rng::AbstractRNG,
    enforce_r_positive::Bool,
    b_sampler_mode::String="dirichlet",
    safety::Float64=1.2,
    eps::Float64=1e-12,
    max_tries::Int=200_000)
    for _ in 1:max_tries
        A0 = sigmaA .* randn(rng, n, n)
        @inbounds A0[diagind(A0)] .= 0.0
        r = -A0 * ones(n)
        if enforce_r_positive && any(r .<= 0.0)
            continue
        end
        B0 = sample_B_dirichlet_sum_neg_r(r; rng=rng, mode=b_sampler_mode)
        Ainfo = stabilize_and_rescale_A_section2(A0, r; safety=safety, eps=eps)
        Binfo = stabilize_and_rescale_B_section2(B0, r; safety=safety, eps=eps, half_shift=true)
        return r, Ainfo.A, Binfo.B
    end
    error("Failed to sample parameter set after max_tries=$max_tries")
end

function random_unit_rays(n::Int, n_dirs::Int; rng::AbstractRNG)
    U = zeros(Float64, n, n_dirs)
    @inbounds for k in 1:n_dirs
        v = randn(rng, n)
        U[:, k] .= v ./ norm(v)
    end
    return U
end

matrix_to_nested(M::AbstractMatrix{<:Real}) = [collect(@view M[i, :]) for i in 1:size(M, 1)]
tensor3_to_nested(T::Array{Float64,3}) = [[[T[i, j, k] for k in 1:size(T, 3)] for j in 1:size(T, 2)] for i in 1:size(T, 1)]

function range_label(vals::Vector{Int})
    return length(vals) == 1 ? string(vals[1]) : "$(minimum(vals))-$(maximum(vals))"
end

derive_seed(base_seed::Int, n::Int, n_dirs::Int, model_idx::Int) = base_seed + 10_000_000 * n + 100_000 * n_dirs + model_idx

function main()
    n_values, n_models, n_dirs_values, alpha_grid = parse_args(ARGS)

    DYNAMICS_MODE in ("standard", "balanced", "balanced_stable", "constrained_r", "elegant", "unique_equilibrium", "all_negative") ||
        error("DYNAMICS_MODE must be 'standard', 'balanced', 'balanced_stable', 'constrained_r', 'elegant', 'unique_equilibrium', or 'all_negative' (set in pipeline_config.jl)")

    out_root = joinpath(@__DIR__, "model_runs")
    n_written = 0

    if DYNAMICS_MODE == "balanced"
        bank_name = "$(BANK_BASE_SEED)_bank_balanced_$(n_models)_models_n_$(range_label(n_values))_$(range_label(n_dirs_values))_dirs_sA_$(BANK_SIGMA_A)_sB_$(BANK_SIGMA_B)_muA_$(BANK_MU_A)_muB_$(BANK_MU_B)"
        out_dir   = joinpath(out_root, bank_name)
        mkpath(out_dir)

        for n in n_values
            for n_dirs in n_dirs_values
                for model_idx in 1:n_models
                    seed = derive_seed(BANK_BASE_SEED, n, n_dirs, model_idx)
                    rng  = MersenneTwister(seed)

                    result = nothing
                    for attempt in 1:100_000
                        result = sample_parameter_set_balanced(
                            n; sigmaA=BANK_SIGMA_A, sigmaB=BANK_SIGMA_B,
                            muA=BANK_MU_A, muB=BANK_MU_B,
                            safety=BANK_SAFETY, rng=rng,
                        )
                        result === nothing || break
                        attempt == 100_000 && error("sample_parameter_set_balanced: failed after 100_000 tries (n=$n, model_idx=$model_idx)")
                    end
                    r, A, B, delta_A, delta_B = result
                    U = random_unit_rays(n, n_dirs; rng=rng)

                    payload = Dict(
                        "n"             => n,
                        "n_dirs"        => n_dirs,
                        "model_idx"     => model_idx,
                        "seed"          => seed,
                        "r"             => collect(r),
                        "A"             => matrix_to_nested(A),
                        "B"             => tensor3_to_nested(B),
                        "U"             => matrix_to_nested(U),
                        "x_star"        => ones(n),
                        "alpha_grid"    => alpha_grid,
                        "sigmaA"        => BANK_SIGMA_A,
                        "sigmaB"        => BANK_SIGMA_B,
                        "muA"           => BANK_MU_A,
                        "muB"           => BANK_MU_B,
                        "dynamics_mode" => "balanced",
                        "metadata"      => Dict(
                            "parameterization" => "shift_row_rescale",
                            "r_base"           => 1.0,
                            "safety"           => BANK_SAFETY,
                            "delta_A"          => delta_A,
                            "delta_B"          => delta_B,
                        ),
                    )

                    file_name = "$(BANK_BASE_SEED)_model_n_$(n)_seed_$(seed)_n_dirs_$(n_dirs).json"
                    safe_write_json(joinpath(out_dir, file_name), payload)
                    n_written += 1
                end
            end
        end

    elseif DYNAMICS_MODE == "balanced_stable"
        bank_name = "$(BANK_BASE_SEED)_bank_balanced_stable_$(n_models)_models_n_$(range_label(n_values))_$(range_label(n_dirs_values))_dirs_sA_$(BANK_SIGMA_A)_sB_$(BANK_SIGMA_B)"
        out_dir   = joinpath(out_root, bank_name)
        mkpath(out_dir)

        for n in n_values
            for n_dirs in n_dirs_values
                for model_idx in 1:n_models
                    seed = derive_seed(BANK_BASE_SEED, n, n_dirs, model_idx)
                    rng  = MersenneTwister(seed)

                    r, A, B, delta_A, delta_B = sample_parameter_set_balanced_stable(
                        n; sigmaA=BANK_SIGMA_A, sigmaB=BANK_SIGMA_B, rng=rng,
                    )
                    U = random_unit_rays(n, n_dirs; rng=rng)

                    payload = Dict(
                        "n"             => n,
                        "n_dirs"        => n_dirs,
                        "model_idx"     => model_idx,
                        "seed"          => seed,
                        "r"             => collect(r),
                        "A"             => matrix_to_nested(A),
                        "B"             => tensor3_to_nested(B),
                        "U"             => matrix_to_nested(U),
                        "x_star"        => ones(n),
                        "alpha_grid"    => alpha_grid,
                        "sigmaA"        => BANK_SIGMA_A,
                        "sigmaB"        => BANK_SIGMA_B,
                        "dynamics_mode" => "balanced_stable",
                        "metadata"      => Dict(
                            "parameterization" => "jacobian_diagonal_stable",
                            "r_base"           => 1.0,
                            "delta_A"          => delta_A,
                            "delta_B"          => delta_B,
                        ),
                    )

                    file_name = "$(BANK_BASE_SEED)_model_n_$(n)_seed_$(seed)_n_dirs_$(n_dirs).json"
                    safe_write_json(joinpath(out_dir, file_name), payload)
                    n_written += 1
                end
            end
        end

    elseif DYNAMICS_MODE == "constrained_r"
        bank_name = "$(BANK_BASE_SEED)_bank_constrained_r_$(n_models)_models_n_$(range_label(n_values))_$(range_label(n_dirs_values))_dirs_sA_$(BANK_SIGMA_A)_sB_$(BANK_SIGMA_B)_muA_$(BANK_MU_A)_muB_$(BANK_MU_B)"
        out_dir   = joinpath(out_root, bank_name)
        mkpath(out_dir)

        for n in n_values
            for n_dirs in n_dirs_values
                for model_idx in 1:n_models
                    seed = derive_seed(BANK_BASE_SEED, n, n_dirs, model_idx)
                    rng  = MersenneTwister(seed)

                    result = nothing
                    for attempt in 1:100_000
                        result = sample_parameter_set_constrained_r(
                            n; sigmaA=BANK_SIGMA_A, sigmaB=BANK_SIGMA_B,
                            muA=BANK_MU_A, muB=BANK_MU_B,
                            safety=BANK_SAFETY, rng=rng,
                        )
                        result === nothing || break
                        attempt == 100_000 && error("sample_parameter_set_constrained_r: failed after 100_000 tries (n=$n, model_idx=$model_idx)")
                    end
                    r, A, B, delta_A, delta_B = result
                    U = random_unit_rays(n, n_dirs; rng=rng)

                    payload = Dict(
                        "n"             => n,
                        "n_dirs"        => n_dirs,
                        "model_idx"     => model_idx,
                        "seed"          => seed,
                        "r"             => collect(r),
                        "A"             => matrix_to_nested(A),
                        "B"             => tensor3_to_nested(B),
                        "U"             => matrix_to_nested(U),
                        "x_star"        => ones(n),
                        "alpha_grid"    => alpha_grid,
                        "sigmaA"        => BANK_SIGMA_A,
                        "sigmaB"        => BANK_SIGMA_B,
                        "muA"           => BANK_MU_A,
                        "muB"           => BANK_MU_B,
                        "dynamics_mode" => "constrained_r",
                        "metadata"      => Dict(
                            "parameterization" => "shift_only_r_from_rowsum",
                            "safety"           => BANK_SAFETY,
                            "delta_A"          => delta_A,
                            "delta_B"          => delta_B,
                        ),
                    )

                    file_name = "$(BANK_BASE_SEED)_model_n_$(n)_seed_$(seed)_n_dirs_$(n_dirs).json"
                    safe_write_json(joinpath(out_dir, file_name), payload)
                    n_written += 1
                end
            end
        end

    elseif DYNAMICS_MODE == "elegant"
        bank_name = "$(BANK_BASE_SEED)_bank_elegant_$(n_models)_models_n_$(range_label(n_values))_$(range_label(n_dirs_values))_dirs_muA_$(BANK_MU_A)_muB_$(BANK_MU_B)"
        out_dir   = joinpath(out_root, bank_name)
        mkpath(out_dir)

        max_tries = 20_000_000
        for n in n_values
            for n_dirs in n_dirs_values
                for model_idx in 1:n_models
                    seed = derive_seed(BANK_BASE_SEED, n, n_dirs, model_idx)

                    result = nothing
                    for attempt in 0:(max_tries - 1)
                        rng = MersenneTwister(seed + attempt)
                        result = sample_parameter_set_elegant(n; muA=BANK_MU_A, muB=BANK_MU_B, rng=rng)
                        result !== nothing && break
                    end
                    result === nothing && error("Failed to sample stable elegant system after $max_tries tries (n=$n, model_idx=$model_idx)")
                    r, A, B, delta_A, delta_B = result

                    U = random_unit_rays(n, n_dirs; rng=MersenneTwister(seed))

                    payload = Dict(
                        "n"             => n,
                        "n_dirs"        => n_dirs,
                        "model_idx"     => model_idx,
                        "seed"          => seed,
                        "r"             => collect(r),
                        "A"             => matrix_to_nested(A),
                        "B"             => tensor3_to_nested(B),
                        "U"             => matrix_to_nested(U),
                        "x_star"        => ones(n),
                        "alpha_grid"    => alpha_grid,
                        "muA"           => BANK_MU_A,
                        "muB"           => BANK_MU_B,
                        "dynamics_mode" => "elegant",
                        "metadata"      => Dict(
                            "parameterization" => "elegant_planted",
                            "delta_A"          => delta_A,
                            "delta_B"          => delta_B,
                        ),
                    )

                    file_name = "$(BANK_BASE_SEED)_model_n_$(n)_seed_$(seed)_n_dirs_$(n_dirs).json"
                    safe_write_json(joinpath(out_dir, file_name), payload)
                    n_written += 1
                end
            end
        end

    elseif DYNAMICS_MODE == "unique_equilibrium"
        bank_name = "$(BANK_BASE_SEED)_bank_unique_equilibrium_$(n_models)_models_n_$(range_label(n_values))_$(range_label(n_dirs_values))_dirs"
        out_dir   = joinpath(out_root, bank_name)
        mkpath(out_dir)

        for n in n_values
            for n_dirs in n_dirs_values
                for model_idx in 1:n_models
                    seed = derive_seed(BANK_BASE_SEED, n, n_dirs, model_idx)
                    rng  = MersenneTwister(seed)

                    A, B, d_A, d_k_values = sample_parameter_set_unique_equilibrium(
                        n; safety=BANK_SAFETY_UE, rng=rng,
                    )
                    U = random_unit_rays(n, n_dirs; rng=rng)

                    payload = Dict(
                        "n"             => n,
                        "n_dirs"        => n_dirs,
                        "model_idx"     => model_idx,
                        "seed"          => seed,
                        "A"             => matrix_to_nested(A),
                        "B"             => tensor3_to_nested(B),
                        "U"             => matrix_to_nested(U),
                        "x_star"        => ones(n),
                        "alpha_grid"    => alpha_grid,
                        "dynamics_mode" => "unique_equilibrium",
                        "metadata"      => Dict(
                            "parameterization" => "slicewise_negdef",
                            "safety"           => BANK_SAFETY_UE,
                            "d_A"              => d_A,
                            "d_k_values"       => d_k_values,
                        ),
                    )

                    file_name = "$(BANK_BASE_SEED)_model_n_$(n)_seed_$(seed)_n_dirs_$(n_dirs).json"
                    safe_write_json(joinpath(out_dir, file_name), payload)
                    n_written += 1
                end
            end
        end

    elseif DYNAMICS_MODE == "all_negative"
        bank_name = "$(BANK_BASE_SEED)_bank_all_negative_$(n_models)_models_n_$(range_label(n_values))_$(range_label(n_dirs_values))_dirs_sA_$(BANK_SIGMA_A)_sB_$(BANK_SIGMA_B)"
        out_dir   = joinpath(out_root, bank_name)
        mkpath(out_dir)

        for n in n_values
            for n_dirs in n_dirs_values
                for model_idx in 1:n_models
                    seed = derive_seed(BANK_BASE_SEED, n, n_dirs, model_idx)
                    rng  = MersenneTwister(seed)

                    A, B = sample_parameter_set_all_negative(
                        n; sigmaA=BANK_SIGMA_A, sigmaB=BANK_SIGMA_B, rng=rng,
                    )
                    U = random_unit_rays(n, n_dirs; rng=rng)

                    payload = Dict(
                        "n"             => n,
                        "n_dirs"        => n_dirs,
                        "model_idx"     => model_idx,
                        "seed"          => seed,
                        "A"             => matrix_to_nested(A),
                        "B"             => tensor3_to_nested(B),
                        "U"             => matrix_to_nested(U),
                        "x_star"        => ones(n),
                        "alpha_grid"    => alpha_grid,
                        "sigmaA"        => BANK_SIGMA_A,
                        "sigmaB"        => BANK_SIGMA_B,
                        "dynamics_mode" => "all_negative",
                        "metadata"      => Dict(
                            "parameterization" => "half_normal_negated",
                        ),
                    )

                    file_name = "$(BANK_BASE_SEED)_model_n_$(n)_seed_$(seed)_n_dirs_$(n_dirs).json"
                    safe_write_json(joinpath(out_dir, file_name), payload)
                    n_written += 1
                end
            end
        end

    else  # "standard"
        BANK_B_SAMPLER in ("dirichlet", "signed_flip") ||
            error("BANK_B_SAMPLER must be 'dirichlet' or 'signed_flip' (set in pipeline_config.jl)")

        bank_name = "$(BANK_BASE_SEED)_bank_$(n_models)_models_n_$(range_label(n_values))_$(range_label(n_dirs_values))_dirs_b_$(BANK_B_SAMPLER)"
        out_dir   = joinpath(out_root, bank_name)
        mkpath(out_dir)

        for n in n_values
            for n_dirs in n_dirs_values
                for model_idx in 1:n_models
                    seed = derive_seed(BANK_BASE_SEED, n, n_dirs, model_idx)
                    rng  = MersenneTwister(seed)

                    r, A, B = sample_parameter_set(
                        n; sigmaA=BANK_SIGMA_A, rng=rng,
                        enforce_r_positive=BANK_ENFORCE_R_POS,
                        b_sampler_mode=BANK_B_SAMPLER,
                        max_tries=20_000_000,
                    )
                    U = random_unit_rays(n, n_dirs; rng=rng)

                    payload = Dict(
                        "n"                  => n,
                        "n_dirs"             => n_dirs,
                        "model_idx"          => model_idx,
                        "seed"               => seed,
                        "sigmaA"             => BANK_SIGMA_A,
                        "enforce_r_positive" => BANK_ENFORCE_R_POS,
                        "b_sampler_mode"     => BANK_B_SAMPLER,
                        "r"                  => collect(r),
                        "A"                  => matrix_to_nested(A),
                        "B"                  => tensor3_to_nested(B),
                        "U"                  => matrix_to_nested(U),
                        "x_star"             => ones(n),
                        "alpha_grid"         => alpha_grid,
                    )

                    file_name = "$(BANK_BASE_SEED)_model_n_$(n)_seed_$(seed)_n_dirs_$(n_dirs).json"
                    safe_write_json(joinpath(out_dir, file_name), payload)
                    n_written += 1
                end
            end
        end
    end

    println("Wrote $n_written models to: $out_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
