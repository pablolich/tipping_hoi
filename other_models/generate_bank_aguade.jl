#!/usr/bin/env julia
# Generate a bank of Aguadé-Gorgorió Allee-effect community models and write
# new_code-compatible JSONs processable by boundary_scan.jl (dynamics_mode = "aguade").
#
# Usage:
#   julia --startup-file=no new_code/other_models/generate_bank_aguade.jl \
#         <n_models> <n_dirs> [seed] [n]
#
# Arguments:
#   n_models   Number of feasible, stable models to generate
#   n_dirs     Number of random perturbation directions per model
#   seed       RNG seed (default 1)
#   n          Species richness (default 6)
#
# Output:
#   new_code/model_runs/aguade_n<N>_<n_models>models_<n_dirs>dirs_seed<seed>/
#   Files: model_aguade_n<N>_idx<XXXX>_ndirs<n_dirs>.json

using Random
using LinearAlgebra
using JSON3

include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))
include(joinpath(@__DIR__, "aguade_model.jl"))   # also loads lever_model.jl

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no new_code/other_models/generate_bank_aguade.jl \\
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

# Sample unit vectors in R^n with all rows active (all species perturbed).
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

    bank_name = "aguade_n$(n)_$(n_models)models_$(n_dirs)dirs_seed$(seed)"
    out_dir   = joinpath(@__DIR__, "..", "model_runs", bank_name)
    mkpath(out_dir)

    println("Aguadé bank generation")
    println("  n=$(n)  n_models=$(n_models)  n_dirs=$(n_dirs)  seed=$(seed)")
    println("  out_dir=$(out_dir)")

    rng          = MersenneTwister(seed)
    n_written    = 0
    attempts     = 0
    max_attempts = 1000 * n_models
    x0 = fill(0.5, n)   # initial condition: x0 = 0.5 * ones(n)

    while n_written < n_models && attempts < max_attempts
        attempts += 1

        p   = sample_aguade_params(n; rng=rng)
        res = integrate_aguade_to_steady(p, x0; tmax=4000.0, steady_tol=1e-8)

        if !res.success
            continue
        end

        x_eq = res.x_eq

        if !all(x_eq .> 1e-2)
            continue
        end

        # Compute α_eff from grouped symbolic contributions on the cleared HC system
        alpha_eff = aguade_alpha_eff_symbolic(p, x_eq)

        if isnan(alpha_eff)
            continue
        end

        # U: n × n_dirs, all rows active (full perturbation in R^n)
        U = random_full_rays(n, n_dirs; rng=rng)

        # r field: -d_i  (baseline per-capita constant term; d_i is perturbed by HC scan)
        r_field = -copy(p.d)

        # Build payload
        pp        = aguade_params_payload(p)
        model_idx = n_written + 1
        payload   = Dict{String, Any}(
            "dynamics_mode" => "aguade",
            "n"             => n,
            "n_dirs"        => n_dirs,
            "model_idx"     => model_idx,
            "r"             => collect(r_field),
            "x_star"        => collect(x_eq),
            "alpha_eff"     => Float64(alpha_eff),
            "U"             => matrix_to_nested(U),
            "aguade_n"      => n,
            "aguade_A"      => pp.aguade_A,
            "aguade_B"      => pp.aguade_B,
            "aguade_d"      => collect(pp.aguade_d),
            "aguade_gamma"  => collect(pp.aguade_gamma),
        )

        file_name = "model_aguade_n$(n)_idx$(lpad(model_idx, 4, '0'))_ndirs$(n_dirs).json"
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
