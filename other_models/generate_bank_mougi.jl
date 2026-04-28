#!/usr/bin/env julia
# Generate a bank of Mougi ecosystem-engineering food-web models and write
# new_code-compatible JSONs processable by boundary_scan.jl (dynamics_mode = "mougi").
#
# Usage:
#   julia --startup-file=no new_code/other_models/generate_bank_mougi.jl \
#         <n_models> <n_dirs> [seed] [n]
#
# Arguments:
#   n_models   Number of feasible, stable models to generate
#   n_dirs     Number of random perturbation directions per model
#   seed       RNG seed (default 1)
#   n          Species richness (default 6)
#
# Output:
#   new_code/model_runs/mougi_random_n<N>_<n_models>models_<n_dirs>dirs_seed<seed>/

using Random
using LinearAlgebra
using JSON3

include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))
include(joinpath(@__DIR__, "mougi_model.jl"))   # also loads lever_model.jl

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no new_code/other_models/generate_bank_mougi.jl \\
            <n_models> <n_dirs> [seed] [n]

    Args:
      n_models   Number of feasible stable models to generate
      n_dirs     Number of random ray directions per model
      seed       RNG seed (default 1)
      n          Species richness (default 6)
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

    connectance = 1.0

    bank_name = "mougi_random_n$(n)_$(n_models)models_$(n_dirs)dirs_seed$(seed)"
    out_dir   = joinpath(@__DIR__, "..", "model_runs", bank_name)
    mkpath(out_dir)

    println("Mougi bank generation")
    println("  topology=random  n=$(n)  connectance=$(connectance)")
    println("  n_models=$(n_models)  n_dirs=$(n_dirs)  seed=$(seed)")
    println("  out_dir=$(out_dir)")

    rng          = MersenneTwister(seed)
    n_written    = 0
    attempts     = 0
    max_attempts = 1000 * n_models
    x0           = fill(0.5, n)

    while n_written < n_models && attempts < max_attempts
        attempts += 1

        p = sample_mougi_params(n; connectance=connectance, rng=rng)
        res = integrate_mougi_to_steady(p, x0; tmax=4000.0, steady_tol=1e-8)

        if !res.success
            continue
        end

        x_eq = res.x_eq

        if !all(x_eq .> 1e-2)
            continue
        end

        alpha_eff = mougi_alpha_eff_symbolic(p, x_eq)
        if !isfinite(alpha_eff)
            continue
        end

        U = random_full_rays(n, n_dirs; rng=rng)
        pp = mougi_params_payload(p)
        model_idx = n_written + 1
        payload = Dict{String, Any}(
            "dynamics_mode"       => "mougi",
            "n"                   => n,
            "n_dirs"              => n_dirs,
            "model_idx"           => model_idx,
            "r"                   => collect(p.r0),
            "x_star"              => collect(x_eq),
            "alpha_eff"           => Float64(alpha_eff),
            "U"                   => matrix_to_nested(U),
            "mougi_n"             => pp.mougi_n,
            "mougi_connectance"   => pp.mougi_connectance,
            "mougi_topology_mode" => pp.mougi_topology_mode,
            "mougi_r0"            => collect(pp.mougi_r0),
            "mougi_s"             => collect(pp.mougi_s),
            "mougi_e"             => pp.mougi_e,
            "mougi_a"             => pp.mougi_a,
            "mougi_h"             => pp.mougi_h,
            "mougi_E0"            => collect(pp.mougi_E0),
            "mougi_engineer_mask" => collect(pp.mougi_engineer_mask),
            "mougi_receiver_mask" => collect(pp.mougi_receiver_mask),
            "mougi_betaE"         => pp.mougi_betaE,
            "mougi_gammaE"        => pp.mougi_gammaE,
            "mougi_q_r"           => pp.mougi_q_r,
            "mougi_q_a"           => pp.mougi_q_a,
        )

        file_name = "model_mougi_n$(n)_idx$(lpad(model_idx, 4, '0'))_ndirs$(n_dirs).json"
        safe_write_json(joinpath(out_dir, file_name), payload)

        n_written += 1
        println("  [$(n_written)/$(n_models)] wrote $(file_name)" *
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
