#!/usr/bin/env julia
# Generate a bank of feasible–stable Marsland (2019) MiCRM parameter sets and
# write one JSON file per accepted sample into
#   new_code/model_runs/marsland_l<l>_n<n>_<n_models>models_<n_dirs>dirs_seed<seed>/
# ready to be consumed by `new_code/boundary_scan.jl` via the
# `dynamics_mode = "marsland"` dispatcher added to `utils/glvhoi_utils.jl`.
#
# The sampler and equilibrium solver live in the `pipeline_marsland_hc`
# environment (not the global Julia env), so the script activates that project
# before doing any Marsland work.  JSON3 and model_store_utils.jl's
# `safe_write_json` come along for the ride.
#
# Usage:
#   julia --startup-file=no new_code/generate_bank_marsland.jl \
#         <n_models> <n_dirs> [seed] [leakage] [n]
#
# Arguments:
#   n_models   Number of feasible–stable models to write                  (required)
#   n_dirs     Number of random perturbation rays per model               (required)
#   seed       RNG seed                                                   (default 1)
#   leakage    Leakage fraction l ∈ [0, 1)                                (default 0.3)
#   n          Shared community size n_C = n_R                            (default 4)
#
# The perturbation space is the consumer maintenance vector `m`, so every ray
# `u` is an `n_C`-dimensional unit vector and `dr` shifts `m_i → m_i + dr_i`.

import Pkg

# Activate the Marsland project so ForwardDiff / NLsolve / OrdinaryDiffEq /
# HomotopyContinuation / DynamicPolynomials / JSON3 are all visible.
const _MARSLAND_PROJ = joinpath(@__DIR__, "other_models", "papers",
                                "Marsland_et_al_2019", "pipeline_marsland_hc")
Pkg.activate(_MARSLAND_PROJ; io = devnull)

using Random
using LinearAlgebra
using JSON3

# `lever_model.jl` is loaded first because `marsland_alpha_eff_symbolic`
# (defined in pipeline_marsland_hc/src/hc_system.jl) calls the shared
# `symbolic_alpha_eff` helper that lives there — same trick that
# `karatayev_model.jl:10` uses.
include(joinpath(@__DIR__, "other_models", "lever_model.jl"))

# The full pipeline_marsland_hc umbrella (sampler, solver, stability, HC bridge).
include(joinpath(_MARSLAND_PROJ, "src", "Marsland.jl"))

# `safe_write_json` + `canonical_models_root` from the repo-wide helpers.
include(joinpath(@__DIR__, "model_store_utils.jl"))

# ----------------------------------------------------------------------------
# Helpers — mirror the shape of generate_bank_karatayev.jl
# ----------------------------------------------------------------------------

matrix_to_nested(M::AbstractMatrix{<:Real}) =
    [collect(@view M[i, :]) for i in 1:size(M, 1)]

"""
    random_consumer_rays(n_C, n_dirs; rng) -> Matrix{Float64}

Return an `n_C × n_dirs` matrix whose columns are independent uniform
samples from the unit sphere `S^{n_C - 1}`.  Each column `U[:, k]` is the
perturbation direction for the `k`-th ray in consumer-maintenance space.
"""
function random_consumer_rays(n_C::Int, n_dirs::Int;
                              rng::AbstractRNG = Random.default_rng())
    U = Matrix{Float64}(undef, n_C, n_dirs)
    @inbounds for d in 1:n_dirs
        v = randn(rng, n_C)
        v ./= norm(v)
        U[:, d] .= v
    end
    return U
end

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no new_code/generate_bank_marsland.jl \\
            <n_models> <n_dirs> [seed] [leakage] [n]

    Arguments:
      n_models  Number of feasible–stable models to write    (required)
      n_dirs    Number of perturbation rays per model        (required)
      seed      RNG seed                                     (default 1)
      leakage   Leakage fraction l ∈ [0, 1)                  (default 0.3)
      n         Shared community size n_C = n_R              (default 4)
    """
    error(msg)
end

function parse_args(args::Vector{String})
    (length(args) < 2 || length(args) > 5) && usage_error()
    n_models = parse(Int,     args[1])
    n_dirs   = parse(Int,     args[2])
    seed     = length(args) >= 3 ? parse(Int,     args[3]) : 1
    leakage  = length(args) >= 4 ? parse(Float64, args[4]) : 0.3
    n        = length(args) >= 5 ? parse(Int,     args[5]) : 4
    n_models >= 1            || error("n_models must be >= 1")
    n_dirs   >= 1            || error("n_dirs must be >= 1")
    0.0 ≤ leakage < 1.0      || error("leakage must satisfy 0 ≤ l < 1")
    n >= 2                   || error("n must be >= 2")
    return n_models, n_dirs, seed, leakage, n
end

# ----------------------------------------------------------------------------
# Main bank loop
# ----------------------------------------------------------------------------

function main()
    n_models, n_dirs, seed, leakage, n = parse_args(ARGS)
    n_C = n
    n_R = n

    leak_tag  = replace(string(leakage), '.' => 'p')
    bank_name = "marsland_l$(leak_tag)_n$(n)_$(n_models)models_" *
                "$(n_dirs)dirs_seed$(seed)"
    out_dir   = joinpath(@__DIR__, "model_runs", bank_name)
    mkpath(out_dir)

    println("Marsland MiCRM bank generation")
    println("  n_C = $n_C   n_R = $n_R   l = $leakage   seed = $seed")
    println("  n_models = $n_models   n_dirs = $n_dirs")
    println("  out_dir  = $out_dir")

    rng          = MersenneTwister(seed)
    n_written    = 0
    attempts     = 0
    max_attempts = 20_000 * n_models   # yield ≈ 0.02–1 % at n = 4 depending on l

    while n_written < n_models && attempts < max_attempts
        attempts += 1

        p  = sample_marsland_params(n_C, n_R, leakage; rng = rng)
        eq = try
            solve_equilibrium(p)
        catch err
            @warn "solve_equilibrium threw; skipping" exception = err
            continue
        end

        eq.converged                        || continue
        is_strictly_feasible(eq.N, eq.R)   || continue
        (stable_ok, max_re, _) = is_stable(p, eq.N, eq.R)
        stable_ok                           || continue

        # ----- assemble JSON payload ---------------------------------
        N_star = collect(Float64, eq.N)
        U      = random_consumer_rays(n_C, n_dirs; rng = rng)

        # Grouped symbolic alpha_eff on the polynomialized (N-only)
        # system — the same System that boundary_scan tracks.
        alpha_eff = marsland_alpha_eff_symbolic(p, N_star)

        pp         = marsland_params_payload(p)
        model_idx  = n_written + 1
        payload    = Dict{String, Any}(
            "n"             => n_C,                      # HC state dim
            "n_dirs"        => n_dirs,
            "model_idx"     => model_idx,
            "dynamics_mode" => "marsland",
            "x_star"        => N_star,
            "r"             => collect(pp.marsland_m),    # baseline m
            "alpha_eff"     => Float64(alpha_eff),
            "U"             => matrix_to_nested(U),
            # --- Marsland-specific payload (mirror karat_* / lever_*) ---
            "marsland_n_C"   => pp.marsland_n_C,
            "marsland_n_R"   => pp.marsland_n_R,
            "marsland_c"     => matrix_to_nested(pp.marsland_c),
            "marsland_D"     => matrix_to_nested(pp.marsland_D),
            "marsland_l"     => pp.marsland_l,
            "marsland_w"     => collect(pp.marsland_w),
            "marsland_g"     => collect(pp.marsland_g),
            "marsland_m"     => collect(pp.marsland_m),
            "marsland_kappa" => collect(pp.marsland_kappa),
            "marsland_tau_R" => pp.marsland_tau_R,
        )

        file_name = "model_marsland_l$(leak_tag)_n$(n_C)_" *
                    "idx$(lpad(model_idx, 4, '0'))_ndirs$(n_dirs).json"
        safe_write_json(joinpath(out_dir, file_name), payload)

        n_written += 1
        println("  [$(n_written)/$(n_models)] wrote $(file_name)   " *
                "alpha_eff = $(round(alpha_eff; sigdigits = 4))   " *
                "max Re λ = $(round(max_re; sigdigits = 3))   " *
                "‖residual‖ = $(round(norm(eq.residual); sigdigits = 3))")
    end

    if n_written < n_models
        @warn "Only generated $(n_written)/$(n_models) models after $(attempts) attempts."
    else
        println("Done. Wrote $(n_written) models to: $(out_dir)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
