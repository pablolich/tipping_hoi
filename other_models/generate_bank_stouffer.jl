#!/usr/bin/env julia

using Random
using LinearAlgebra
using JSON3
using Distributions

include(joinpath(@__DIR__, "..", "model_store_utils.jl"))
include(joinpath(@__DIR__, "stouffer_model.jl"))

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no new_code/other_models/generate_bank_stouffer.jl \\
            <n_models> <n_dirs> [seed] [n]

    Args:
      n_models   Number of feasible stable models to generate
      n_dirs     Number of random ray directions per model
      seed       RNG seed (default 1)
      n          Species count (default 6)
    """
    error(msg)
end

function parse_args(args::Vector{String})
    (length(args) < 2 || length(args) > 4) && usage_error()
    n_models = parse(Int, args[1])
    n_dirs   = parse(Int, args[2])
    seed     = length(args) >= 3 ? parse(Int, args[3]) : 1
    n        = length(args) >= 4 ? parse(Int, args[4]) : 6
    n_models >= 1 || error("n_models must be >= 1")
    n_dirs   >= 1 || error("n_dirs must be >= 1")
    n        >= 2 || error("n must be >= 2")
    return n_models, n_dirs, seed, n
end

matrix_to_nested(M::AbstractMatrix{<:Real}) =
    [collect(@view M[i, :]) for i in 1:size(M, 1)]

function random_full_rays(n::Int, n_dirs::Int; rng::AbstractRNG)
    U = zeros(Float64, n, n_dirs)
    @inbounds for d in 1:n_dirs
        v = randn(rng, n)
        v ./= norm(v)
        U[:, d] .= v
    end
    return U
end

function main()
    n_models, n_dirs, seed, n = parse_args(ARGS)

    bank_name = "stouffer_random_n$(n)_$(n_models)models_$(n_dirs)dirs_seed$(seed)"
    out_dir = joinpath(@__DIR__, "..", "model_runs", bank_name)
    mkpath(out_dir)

    println("Stouffer bank generation")
    println("  topology=niche_model  n=$(n)")
    println("  connectance_set=(0.10, 0.12, 0.14, 0.16, 0.18, 0.20)")
    println("  n_models=$(n_models)  n_dirs=$(n_dirs)  seed=$(seed)")
    println("  out_dir=$(out_dir)")

    rng = MersenneTwister(seed)
    x0_dist = Uniform(0.05, 1.0)
    n_written = 0
    attempts = 0
    max_attempts = 2000 * n_models

    while n_written < n_models && attempts < max_attempts
        attempts += 1

        p  = sample_stouffer_params(n; rng=rng)
        x0 = rand(rng, x0_dist, n)
        res = integrate_stouffer_to_steady(p, x0; tmax=10_000.0, steady_tol=1e-8)
        res.success || continue

        x_eq = res.x_eq
        all(x_eq .> 1e-2) || continue

        alpha_eff = stouffer_alpha_eff_symbolic(p, x_eq)
        isfinite(alpha_eff) || continue

        U = random_full_rays(n, n_dirs; rng=rng)
        pp = stouffer_params_payload(p)
        model_idx = n_written + 1
        payload = Dict{String, Any}(
            "dynamics_mode"         => "stouffer",
            "n"                     => n,
            "n_dirs"                => n_dirs,
            "model_idx"             => model_idx,
            "r"                     => collect(stouffer_baseline_r(p)),
            "x_star"                => collect(x_eq),
            "alpha_eff"             => Float64(alpha_eff),
            "U"                     => matrix_to_nested(U),
            "stouffer_n"            => pp.stouffer_n,
            "stouffer_connectance"  => pp.stouffer_connectance,
            "stouffer_adj"          => pp.stouffer_adj,
            "stouffer_basal_mask"   => collect(pp.stouffer_basal_mask),
            "stouffer_w"            => pp.stouffer_w,
            "stouffer_M"            => collect(pp.stouffer_M),
            "stouffer_x"            => collect(pp.stouffer_x),
            "stouffer_y"            => collect(pp.stouffer_y),
            "stouffer_e"            => pp.stouffer_e,
            "stouffer_K"            => pp.stouffer_K,
            "stouffer_B0"           => pp.stouffer_B0,
            "stouffer_Mb"           => pp.stouffer_Mb,
            "stouffer_ar"           => pp.stouffer_ar,
            "stouffer_ax"           => pp.stouffer_ax,
            "stouffer_ay"           => pp.stouffer_ay,
        )

        file_name = "model_stouffer_n$(n)_idx$(lpad(model_idx, 4, '0'))_ndirs$(n_dirs).json"
        safe_write_json(joinpath(out_dir, file_name), payload)

        n_written += 1
        println("  [$(n_written)/$(n_models)] wrote $(file_name)" *
                "  connectance=$(round(p.connectance; digits=2))" *
                "  alpha_eff=$(round(alpha_eff; digits=3))" *
                "")
    end

    if n_written < n_models
        @warn "Only generated $(n_written)/$(n_models) models after $(attempts) attempts"
    else
        println("Done. Wrote $(n_written) models to: $(out_dir)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
