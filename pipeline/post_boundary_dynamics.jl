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
      <run_dir>          Folder inside model_runs/ (searched recursively)

    Options:
      --model-file FILE  Process one model JSON file from <run_dir>
      --force            Recompute files that already have post_dynamics_results

    Set JULIA_NUM_THREADS to fan the rays of each alpha block across threads.
    All dynamics settings are in pipeline_config.jl.
    """
    error(msg)
end

function parse_args(args::Vector{String})
    isempty(args) && usage_error()

    run_dir    = ""
    model_file = nothing
    force      = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--model-file" && i < length(args)
            model_file = args[i + 1]
            i += 2
        elseif arg == "--force"
            force = true
            i += 1
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
    return (run_dir=run_dir, model_file=model_file, force=force)
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

"""
    ray_rng(tag, alpha_idx, ray_id)

Nudge RNG for one ray, seeded from the ray's identity rather than drawn from the
global stream.  Two consequences, both wanted: the result no longer depends on
how rays are scheduled across threads, and it no longer depends on the order the
driver walks the files in (the old `Random.seed!(POST_SEED + idx - 1)` keyed on
position in the directory listing, so resharding moved every seed).

`scratch/repro_gate/NOTE.md` bounds what this can move: between two unseeded
re-runs the landing state differs by 4.8e-15 and no discrete field moves at all,
against reltol 1e-8.
"""
ray_rng(tag::AbstractString, alpha_idx::Integer, ray_id::Integer) =
    MersenneTwister(hash((tag, alpha_idx, ray_id, POST_SEED)) % typemax(Int32))

function scan_model_post_dynamics(model::Dict{String,Any},
                                  dyn::Dict{String,Any},
                                  seq::Dict{String,Any},
                                  tag::AbstractString="")
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

        post_rows = Vector{Any}(undef, length(dir_rows))
        # Rays are independent; @threads is a plain serial loop at
        # JULIA_NUM_THREADS=1, so this is opt-in via the environment.
        Threads.@threads for ri in eachindex(dir_rows)
            row = dir_rows[ri]
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

            # A "success" ray is one the scan found no boundary for, so there is
            # nothing to step past — see POST_SKIP_NO_BOUNDARY.  The row is still
            # emitted, with the same null snap those rays almost always carry.
            snap = if POST_SKIP_NO_BOUNDARY && String(row["flag"]) == "success"
                (reason=:skipped_no_boundary, x_end=nothing, x_snap=nothing,
                 dist=missing, n_equilibria=0)
            else
                integrate_and_snap(A_eff, B_eff, r0, u, delta_post, x_pre, dyn, seq;
                                   rng=ray_rng(tag, a_idx, ray_id),
                                   record_retcode=POST_RECORD_RETCODE)
            end

            abundance_ranks = compute_abundance_ranks(x_pre)
            ode_ok = snap.reason == :ok
            x_snap = snap.x_snap === nothing ? nothing : Vector{Float64}(snap.x_snap)
            extinct_species = x_snap === nothing ? Int[] : findall(x_snap .<= dyn["eps_extinct"])
            n_lost = length(extinct_species)
            any_rank_gt1 = !isempty(extinct_species) && any(abundance_ranks[extinct_species] .> 1)

            post_rows[ri] = Dict(
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
            )
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
    println("  threads: $(Threads.nthreads())")
    println("  skip_no_boundary: $POST_SKIP_NO_BOUNDARY")
    println("  record_retcode: $POST_RECORD_RETCODE")
    println("  skip_done: $(POST_SKIP_DONE && !opts.force)$(opts.force ? " (--force)" : "")")
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
    n_written = 0
    n_skipped = 0
    failures  = String[]

    for (idx, model_path) in enumerate(model_paths)
        model_name = basename(model_path)

        # One bad file must not cost the whole run: this driver rewrites in
        # place, so an abort halfway leaves a half-enriched directory with no
        # record of where it stopped.
        try
            model = to_dict(JSON3.read(read(model_path, String)))

            # Not every JSON beside a model is a model — a sharded scan leaves a
            # scan_manifest.json in each shard dir, and the pre-scan gibbs
            # refgrid files have no scan_results at all.  Skip, do not die.
            if !haskey(model, "scan_results") || isempty(model["scan_results"])
                println("[$idx/$n_models] skip (no scan_results): $model_name")
                n_skipped += 1
                continue
            end

            # Re-running over a finished bank is destructive: the payload is
            # written in place and backtrack_results is dropped on the way
            # through.  Refuse by default.
            if POST_SKIP_DONE && !opts.force && haskey(model, "post_dynamics_results")
                println("[$idx/$n_models] skip (already has post_dynamics_results): $model_name")
                n_skipped += 1
                continue
            end

            println("[$idx/$n_models] processing $model_name")
            payload = scan_model_post_dynamics(model, dyn, seq, model_name)
            safe_write_json(model_path, payload)
            n_written += 1
            println("      wrote $model_name")
        catch err
            msg = sprint(showerror, err)
            println("[$idx/$n_models] FAILED $model_name: $msg")
            push!(failures, "$model_name: $msg")
        end
    end

    println("Done. written=$n_written skipped=$n_skipped failed=$(length(failures))")
    if !isempty(failures)
        println("Failures:")
        for f in failures
            println("  $f")
        end
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
