#!/usr/bin/env julia
# Post-boundary ODE dynamics driven by canonical model JSON files.
# Example:
#   julia --startup-file=no pipeline/post_boundary_dynamics.jl 1_bank_50_models_n_2-7_100_dirs_b_dirichlet
#   julia --startup-file=no pipeline/post_boundary_dynamics.jl 1_bank_50_models_n_2-7_100_dirs_b_dirichlet --model-file model_n_3_seed_40000002_n_dirs_100.json
# All dynamics settings are in pipeline_config.jl.

using LinearAlgebra
using JSON3
using Random
using DifferentialEquations
using SciMLBase

include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "math_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "json_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "glvhoi_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "dynamics_cfg_utils.jl"))

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no pipeline/post_boundary_dynamics.jl <run_dir> [--model-file FILE]

    Required:
      <run_dir>          Folder inside model_runs/

    Options:
      --model-file FILE  Process one model JSON file from <run_dir>

    All dynamics settings are in pipeline_config.jl.
    """
    error(msg)
end

function parse_args(args::Vector{String})
    isempty(args) && usage_error()

    run_dir    = ""
    model_file = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--model-file" && i < length(args)
            model_file = args[i + 1]
            i += 2
        elseif startswith(arg, "--")
            error("Unknown flag: $arg. Configure dynamics settings in pipeline_config.jl.")
        elseif run_dir == ""
            run_dir = arg
            i += 1
        else
            error("Unexpected argument: $arg")
        end
    end

    run_dir == "" && usage_error()
    return (run_dir=run_dir, model_file=model_file)
end

include(joinpath(@__DIR__, "..", "utils", "ode_snap_utils.jl"))

function compute_abundance_ranks(x_minus::AbstractVector{<:Real})
    n = length(x_minus)
    order = sortperm(x_minus)
    ranks = zeros(Int, n)
    @inbounds for (rk, idx) in enumerate(order)
        ranks[idx] = rk
    end
    return ranks
end

function postboundary_fraction_for_model(model::Dict{String,Any})
    scan_cfg = get(model, "scan_config", Dict{String,Any}())
    pre_frac = Float64(get(scan_cfg, "preboundary_fraction", SCAN_PREBOUNDARY_FRAC))
    0 < pre_frac < 1 || error("preboundary_fraction must be in (0, 1), got $pre_frac")
    post_frac = POST_POSTBOUNDARY_FRAC === nothing ? (2.0 - pre_frac) : Float64(POST_POSTBOUNDARY_FRAC)
    post_frac > 1.0 || error("postboundary_fraction must be > 1, got $post_frac (set POST_POSTBOUNDARY_FRAC in pipeline_config.jl)")
    return pre_frac, post_frac
end

function scan_model_post_dynamics(model::Dict{String,Any},
                                  dyn::Dict{String,Any},
                                  seq::Dict{String,Any})
    n = Int(model["n"])
    dmode = get(model, "dynamics_mode", "standard")
    A = nested_to_matrix(model["A"])
    B = nested_to_tensor3(model["B"])
    r0_fixed = dmode == "unique_equilibrium" ? nothing : Float64.(model["r"])
    scan_results = get(model, "scan_results", Any[])
    isempty(scan_results) && error("Missing or empty scan_results in input model.")

    size(A, 1) == n && size(A, 2) == n || error("A has wrong shape: $(size(A)) for n=$n.")
    size(B, 1) == n && size(B, 2) == n && size(B, 3) == n || error("B has wrong shape: $(size(B)) for n=$n.")

    is_gibbs = dmode == "gibbs"
    pre_frac, post_frac = postboundary_fraction_for_model(model)
    halfwidth_frac = 1.0 - pre_frac

    alpha_results = Vector{Any}()
    for alpha_block in scan_results
        a_idx = Int(alpha_block["alpha_idx"])
        alpha = Float64(alpha_block["alpha"])
        dir_rows = get(alpha_block, "directions", Any[])

        r0 = dmode == "unique_equilibrium" ? compute_r_unique_equilibrium(A, B, alpha) : r0_fixed
        A_eff, B_eff = prescale(A, B, alpha, is_gibbs)

        post_rows = Vector{Any}()
        for row in dir_rows
            ray_id = Int(row["ray_id"])

            delta_c = Float64(row["delta_c"])
            drcrit = Float64.(row["drcrit"])
            x_pre = Float64.(row["x_preboundary"])
            length(drcrit) == n || error("drcrit length $(length(drcrit)) does not match n=$n for ray_id=$ray_id.")
            length(x_pre) == n || error("x_preboundary length $(length(x_pre)) does not match n=$n for ray_id=$ray_id.")

            u = drcrit ./ delta_c
            delta_pre = pre_frac * delta_c
            delta_post = post_frac * delta_c
            dr_post = delta_post .* u
            r_post = r0 .+ dr_post

            snap = integrate_and_snap(A_eff, B_eff, r0, u, delta_post, x_pre, dyn, seq)

            abundance_ranks = compute_abundance_ranks(x_pre)
            ode_ok = snap.reason == :ok
            x_snap = snap.x_snap === nothing ? nothing : Vector{Float64}(snap.x_snap)
            extinct_species = x_snap === nothing ? Int[] : findall(x_snap .<= dyn["eps_extinct"])
            n_lost = length(extinct_species)
            any_rank_gt1 = !isempty(extinct_species) && any(abundance_ranks[extinct_species] .> 1)

            push!(post_rows, Dict(
                "ray_id" => ray_id,
                "flag" => String(row["flag"]),
                "status" => String(row["status"]),
                "alpha_idx" => a_idx,
                "alpha" => alpha,
                "delta_c" => delta_c,
                "delta_pre" => delta_pre,
                "delta_post" => delta_post,
                "preboundary_fraction" => pre_frac,
                "postboundary_fraction" => post_frac,
                "postboundary_ball_halfwidth_fraction" => halfwidth_frac,
                "drcrit" => drcrit,
                "dr_post" => collect(dr_post),
                "r_post" => collect(r_post),
                "x_preboundary" => x_pre,
                "x_postboundary_snap" => x_snap,
                "snap_distance" => ismissing(snap.dist) ? nothing : Float64(snap.dist),
                "snap_reason" => String(snap.reason),
                "ode_status" => Bool(ode_ok),
                "n_skeleton_stable_equilibria" => snap.n_equilibria,
                "abundance_ranks_preboundary" => abundance_ranks,
                "extinct_species_post" => extinct_species,
                "n_lost_post" => Int(n_lost),
                "any_extinct_abundance_rank_gt1_post" => Bool(any_rank_gt1),
            ))
        end

        push!(alpha_results, Dict(
            "alpha_idx" => a_idx,
            "alpha" => alpha,
            "directions" => post_rows,
        ))
    end

    output = Dict{String,Any}()
    for (k, v) in model
        output[k] = v
    end
    output["post_dynamics_config"] = Dict(
        "preboundary_fraction_source"          => pre_frac,
        "postboundary_fraction"                => post_frac,
        "postboundary_fraction_mode"           =>
            (POST_POSTBOUNDARY_FRAC === nothing ? "symmetric_from_preboundary" : "override"),
        "postboundary_ball_halfwidth_fraction"  => halfwidth_frac,
        "dynamics" => Dict(
            "tspan"       => [dyn["tspan"][1], dyn["tspan"][2]],
            "reltol"      => dyn["reltol"],
            "abstol"      => dyn["abstol"],
            "eps_extinct" => dyn["eps_extinct"],
        ),
        "selection" => Dict("mode" => "full_model"),
    )
    output["post_dynamics_results"] = alpha_results
    haskey(output, "backtrack_results") && delete!(output, "backtrack_results")
    return output
end

function main()
    opts = parse_args(ARGS)
    run_root = canonical_models_root(@__DIR__, opts.run_dir)
    isdir(run_root) || error("Run directory not found: $run_root")
    model_paths = resolve_model_paths(run_root, opts.model_file)

    dyn = Dict{String,Any}(
        "tspan"       => (0.0, ODE_TSPAN_END),
        "reltol"      => ODE_RELTOL,
        "abstol"      => ODE_ABSTOL,
        "saveat"      => nothing,
        "eps_extinct" => ZERO_ABUNDANCE,
    )
    seq = build_seq_cfg()

    println("Post-boundary dynamics")
    println("  run: $run_root")
    println("  files: $(length(model_paths))")
    println("  output: canonical model files in-place")
    println("  tspan: $(dyn["tspan"])")
    println("  reltol: $(dyn["reltol"])")
    println("  abstol: $(dyn["abstol"])")
    println("  eps_extinct: $(dyn["eps_extinct"])")
    if POST_POSTBOUNDARY_FRAC === nothing
        println("  postboundary_fraction: symmetric from scan_config.preboundary_fraction")
    else
        println("  postboundary_fraction: $POST_POSTBOUNDARY_FRAC")
    end

    n_models = length(model_paths)
    for (idx, model_path) in enumerate(model_paths)
        model_name = basename(model_path)
        println("[$idx/$n_models] processing $model_name")

        if POST_SEED !== nothing
            Random.seed!(POST_SEED + idx - 1)
        end

        model = to_dict(JSON3.read(read(model_path, String)))
        payload = scan_model_post_dynamics(model, dyn, seq)
        safe_write_json(model_path, payload)
        println("      wrote $model_name")
    end

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
