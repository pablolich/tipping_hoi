#!/usr/bin/env julia
# Example terminal calls:
#   julia --startup-file=no pipeline/generate_bank.jl 3:5 10 8,16
#   julia --startup-file=no pipeline/generate_bank.jl 3 50 100
# All other settings (seed, sigmaA, etc.) are set in pipeline_config.jl.

using Random
using LinearAlgebra
using JSON3

include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))

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
      julia pipeline/generate_bank.jl <n_values> <n_models> <n_dirs> [--alpha-grid VALUES]

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

function sample_parameter_set_standard(n::Int;
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

    DYNAMICS_MODE in ("standard", "unique_equilibrium", "all_negative") ||
        error("DYNAMICS_MODE must be 'standard', 'unique_equilibrium', or 'all_negative' (set in pipeline_config.jl)")

    out_root = joinpath(@__DIR__, "..", "model_runs")
    n_written = 0

    if DYNAMICS_MODE == "standard"
        bank_name = "$(BANK_BASE_SEED)_bank_standard_$(n_models)_models_n_$(range_label(n_values))_$(range_label(n_dirs_values))_dirs_muA_$(BANK_MU_A)_muB_$(BANK_MU_B)"
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
                        result = sample_parameter_set_standard(n; muA=BANK_MU_A, muB=BANK_MU_B, rng=rng)
                        result !== nothing && break
                    end
                    result === nothing && error("Failed to sample stable standard system after $max_tries tries (n=$n, model_idx=$model_idx)")
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
                        "dynamics_mode" => "standard",
                        "metadata"      => Dict(
                            "parameterization" => "standard_planted",
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

    end

    println("Wrote $n_written models to: $out_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
