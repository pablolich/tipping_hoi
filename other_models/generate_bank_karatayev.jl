#!/usr/bin/env julia
# Generate a bank of Karatayev food-web models and write new_code-compatible JSONs
# that can be processed by boundary_scan.jl (dynamics_mode = "karatayev").
#
# Usage:
#   julia --startup-file=no new_code/other_models/generate_bank_karatayev.jl \
#         <n_models> <n_dirs> [seed] [feedback_mode] [n]
#
# Arguments:
#   n_models       Number of feasible, stable models to generate
#   n_dirs         Number of random perturbation directions per model
#   seed           RNG seed (default 1)
#   feedback_mode  "FMI" (default) or "RMI"
#   n              Total species richness (default 6); split as n_R = n÷2, n_C = n-n_R
#
# Output:
#   new_code/model_runs/karatayev_<FMI|RMI>_n<N>_<n_models>models_<n_dirs>dirs_seed<seed>/

using Random
using LinearAlgebra
using JSON3

include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))
include(joinpath(@__DIR__, "karatayev_model.jl"))   # also loads lever_model.jl

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no new_code/other_models/generate_bank_karatayev.jl \\
            <n_models> <n_dirs> [seed] [feedback_mode] [n]

    Args:
      n_models       Number of feasible stable models to generate
      n_dirs         Number of random ray directions per model
      seed           RNG seed (default 1)
      feedback_mode  "FMI" (default) or "RMI"
      n              Total species richness (default 6); split as n_R = n÷2, n_C = n-n_R
    """
    error(msg)
end

function parse_args(args::Vector{String})
    (length(args) < 2 || length(args) > 5) && usage_error()
    n_models      = parse(Int, args[1])
    n_dirs        = parse(Int, args[2])
    seed          = length(args) >= 3 ? parse(Int, args[3]) : 1
    feedback_mode = length(args) >= 4 ? args[4] : "FMI"
    n_total       = length(args) >= 5 ? parse(Int, args[5]) : 6
    n_models >= 1                   || error("n_models must be >= 1")
    n_dirs   >= 1                   || error("n_dirs must be >= 1")
    n_total  >= 4                   || error("n must be >= 4 (need at least 2 resources and 2 consumers)")
    feedback_mode in ("FMI", "RMI") || error("feedback_mode must be FMI or RMI")
    return n_models, n_dirs, seed, feedback_mode, n_total
end

matrix_to_nested(M::AbstractMatrix{<:Real}) =
    [collect(@view M[i, :]) for i in 1:size(M, 1)]

# Sample unit rays in R^n with zeros in resource rows (1:n_R),
# unit vectors in consumer subspace — mirrors random_pollinator_rays in Lever.
function random_consumer_rays(n::Int, n_R::Int, n_C::Int, n_dirs::Int;
                               rng::AbstractRNG)
    U = zeros(Float64, n, n_dirs)
    @inbounds for d in 1:n_dirs
        v = randn(rng, n_C)
        v ./= norm(v)
        U[n_R+1:end, d] .= v
    end
    return U
end

function main()
    n_models, n_dirs, seed, feedback_mode, n_total = parse_args(ARGS)

    n_R = n_total ÷ 2
    n_C = n_total - n_R
    n   = n_R + n_C

    bank_name = "karatayev_$(feedback_mode)_n$(n)_$(n_models)models_$(n_dirs)dirs_seed$(seed)"
    out_dir   = joinpath(@__DIR__, "..", "model_runs", bank_name)
    mkpath(out_dir)

    println("Karatayev bank generation")
    println("  feedback_mode=$(feedback_mode)  n_R=$(n_R)  n_C=$(n_C)  n=$(n)")
    println("  n_models=$(n_models)  n_dirs=$(n_dirs)  seed=$(seed)")
    println("  out_dir=$(out_dir)")

    rng          = MersenneTwister(seed)
    n_written    = 0
    attempts     = 0
    max_attempts = 200 * n_models

    while n_written < n_models && attempts < max_attempts
        attempts += 1

        p   = sample_karatayev_params(n_R, n_C; feedback_mode=feedback_mode, rng=rng)
        x0  = vcat(fill(0.5, n_R), fill(0.15, n_C))
        res = integrate_karatayev_to_steady(x0, p; tmax=2000.0, steady_tol=1e-8)

        if !res.success
            continue
        end

        x_eq = res.x_eq

        if !all(x_eq .> 1e-2)
            continue
        end

        alpha_eff = karatayev_alpha_eff_symbolic(p, x_eq)

        # U: n × n_dirs, zeros in resource rows, unit vecs in consumer subspace
        U = random_consumer_rays(n, n_R, n_C, n_dirs; rng=rng)

        # r field: resource growth rates + negative baseline mortality for consumers
        r_field = vcat(copy(p.r), fill(-p.m0, n_C))

        # Build payload
        pp        = karatayev_params_payload(p)
        model_idx = n_written + 1
        payload   = Dict{String, Any}(
            "n"                    => n,
            "n_dirs"               => n_dirs,
            "model_idx"            => model_idx,
            "dynamics_mode"        => "karatayev",
            "feedback_mode"        => feedback_mode,
            "x_star"               => collect(x_eq),
            "r"                    => collect(r_field),
            "alpha_eff"            => Float64(alpha_eff),
            "U"                    => matrix_to_nested(U),
            "karat_n_R"            => n_R,
            "karat_n_C"            => n_C,
            "karat_r"              => collect(pp.karat_r),
            "karat_K"              => collect(pp.karat_K),
            "karat_f"              => collect(pp.karat_f),
            "karat_delta"          => collect(pp.karat_delta),
            "karat_b"              => collect(pp.karat_b),
            "karat_beta"           => collect(pp.karat_beta),
            "karat_m0"             => pp.karat_m0,
            "karat_a"              => pp.karat_a,
            "karat_specials"       => matrix_to_nested(pp.karat_specials),
            "karat_comps"          => matrix_to_nested(pp.karat_comps),
            "karat_feedback_mode"  => pp.karat_feedback_mode,
        )

        file_name = "model_karatayev_$(feedback_mode)_n$(n)_idx$(lpad(model_idx, 4, '0'))_ndirs$(n_dirs).json"
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
