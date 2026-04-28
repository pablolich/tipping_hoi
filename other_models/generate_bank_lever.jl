#!/usr/bin/env julia
# Generate a bank of Lever plant-pollinator models and write new_code-compatible JSONs
# that can be processed by boundary_scan.jl (dynamics_mode = "lever").
#
# Usage:
#   julia --startup-file=no new_code/other_models/generate_bank_lever.jl <n_models> <n_dirs> [seed] [Sp] [Sa]
#
# Arguments:
#   n_models   Number of feasible, stable models to generate
#   n_dirs     Number of random perturbation directions per model
#   seed       RNG seed (default 1)
#   Sp         Number of plant species (default 3)
#   Sa         Number of pollinator species (default 3)
#
# Output: new_code/model_runs/lever_Sp{Sp}Sa{Sa}_{n_models}models_{n_dirs}dirs_seed{seed}/

using Random
using LinearAlgebra
using JSON3

include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))
include(joinpath(@__DIR__, "lever_model.jl"))

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no new_code/other_models/generate_bank_lever.jl <n_models> <n_dirs> [seed] [Sp] [Sa]

    Args:
      n_models   Number of feasible stable models to generate
      n_dirs     Number of random ray directions per model
      seed       RNG seed (default 1)
      Sp         Plant species count (default 3)
      Sa         Pollinator species count (default 3)
    """
    error(msg)
end

function parse_args(args::Vector{String})
    (length(args) < 2 || length(args) > 5) && usage_error()
    n_models = parse(Int, args[1])
    n_dirs   = parse(Int, args[2])
    seed     = length(args) >= 3 ? parse(Int, args[3]) : 1
    Sp       = length(args) >= 4 ? parse(Int, args[4]) : 3
    Sa       = length(args) >= 5 ? parse(Int, args[5]) : 3
    n_models >= 1 || error("n_models must be >= 1")
    n_dirs   >= 1 || error("n_dirs must be >= 1")
    Sp >= 1 || error("Sp must be >= 1")
    Sa >= 1 || error("Sa must be >= 1")
    return n_models, n_dirs, seed, Sp, Sa
end

matrix_to_nested(M::AbstractMatrix{<:Real}) =
    [collect(@view M[i, :]) for i in 1:size(M, 1)]

# Sample unit rays in R^n with zeros in plant rows (1:Sp), unit in pollinator rows.
function random_pollinator_rays(n::Int, Sp::Int, Sa::Int, n_dirs::Int; rng::AbstractRNG)
    U = zeros(Float64, n, n_dirs)
    @inbounds for k in 1:n_dirs
        v = randn(rng, Sa)
        v ./= norm(v)
        U[Sp+1:end, k] .= v
    end
    return U
end

function main()
    n_models, n_dirs, seed, Sp, Sa = parse_args(ARGS)
    n = Sp + Sa

    bank_name = "lever_Sp$(Sp)Sa$(Sa)_$(n_models)models_$(n_dirs)dirs_seed$(seed)"
    out_dir   = joinpath(@__DIR__, "..", "model_runs", bank_name)
    mkpath(out_dir)

    println("Lever bank generation")
    println("  Sp=$(Sp)  Sa=$(Sa)  n=$(n)")
    println("  n_models=$(n_models)  n_dirs=$(n_dirs)  seed=$(seed)")
    println("  out_dir=$(out_dir)")

    rng        = MersenneTwister(seed)
    n_written  = 0
    attempts   = 0
    max_attempts = 200 * n_models

    while n_written < n_models && attempts < max_attempts
        attempts += 1

        p   = sample_lever_original_params(Sp, Sa; rng=rng)
        x0  = ones(Float64, n)
        res = integrate_lever_to_steady(x0, p; tmax=4000.0, steady_tol=1e-8)

        if !res.success
            continue
        end

        x_eq = res.x_eq

        if !all(x_eq .> 1e-2)
            continue
        end

        alpha_eff = lever_alpha_eff_symbolic(p, x_eq)

        # U: n × n_dirs with zeros in plant rows, pollinator-subspace unit vectors
        U = random_pollinator_rays(n, Sp, Sa, n_dirs; rng=rng)

        # r = [rP..., rA...] baseline growth rates
        r = vcat(copy(p.rP), copy(p.rA))

        # Build payload including all lever parameters for HC dispatch
        pp = lever_params_payload(p)
        model_idx = n_written + 1
        payload = Dict{String,Any}(
            "n"            => n,
            "n_dirs"       => n_dirs,
            "model_idx"    => model_idx,
            "dynamics_mode" => "lever",
            "x_star"       => collect(x_eq),
            "r"            => collect(r),
            "alpha_eff"    => Float64(alpha_eff),
            "U"            => matrix_to_nested(U),
            "lever_Sp"     => Sp,
            "lever_Sa"     => Sa,
            "lever_t"      => pp.lever_t,
            "lever_muP"    => pp.lever_muP,
            "lever_muA"    => pp.lever_muA,
            "lever_dA"     => pp.lever_dA,
            "lever_rP"     => collect(pp.lever_rP),
            "lever_rA"     => collect(pp.lever_rA),
            "lever_hP"     => collect(pp.lever_hP),
            "lever_hA"     => collect(pp.lever_hA),
            "lever_CP"     => matrix_to_nested(pp.lever_CP),
            "lever_CA"     => matrix_to_nested(pp.lever_CA),
            "lever_GP"     => matrix_to_nested(pp.lever_GP),
            "lever_GA"     => matrix_to_nested(pp.lever_GA),
        )

        file_name = "model_lever_n$(n)_idx$(lpad(model_idx, 4, '0'))_ndirs$(n_dirs).json"
        safe_write_json(joinpath(out_dir, file_name), payload)

        n_written += 1
        println("  [$(n_written)/$(n_models)] wrote $(file_name)  alpha_eff=$(round(alpha_eff; digits=3))")
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
