#!/usr/bin/env julia
# Rescan only the directions that previously flagged :success (i.e. reached
# SCAN_MAX_PERT without hitting a boundary) using a much larger delta_max.
#
# Reads <run_dir>/<model_file>, finds every (alpha, ray_id) with flag=="success"
# in scan_results, re-invokes scan_ray / scan_ray_linear_alpha0 with the
# user-supplied delta_max, and overwrites the same entries in place. The rest
# of the JSON (A, B, r, U, other flags, metadata) is preserved.
#
# Example:
#   julia --startup-file=no hpc/rescan_success_directions.jl \
#       2_bank_standard_50_models_n_4-20_128_dirs_muB_0.1 \
#       --model-file 2_model_n_10_seed_112800003_n_dirs_128.json \
#       --delta-max 1000.0

using LinearAlgebra
using JSON3
using HomotopyContinuation

include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "json_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "hc_param_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "glvhoi_utils.jl"))
include(joinpath(@__DIR__, "..", "pipeline", "boundary_scan.jl"))  # reuse scan_ray / scan_ray_linear_alpha0 / ScanWorkspace

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no hpc/rescan_success_directions.jl <run_dir>
                              [--model-file FILE] [--delta-max FLOAT]

    Required:
      <run_dir>          Folder inside model_runs/

    Options:
      --model-file FILE  Rescan only one model JSON (default: all in run_dir)
      --delta-max FLOAT  New max perturbation magnitude (default: 1000.0)
    """
    error(msg)
end

function parse_rescan_args(args::Vector{String})
    isempty(args) && usage_error()
    run_dir    = ""
    model_file = nothing
    delta_max  = 1000.0
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--model-file" && i < length(args)
            model_file = args[i+1]; i += 2
        elseif a == "--delta-max" && i < length(args)
            delta_max = parse(Float64, args[i+1]); i += 2
        elseif startswith(a, "--")
            error("Unknown flag: $a")
        elseif run_dir == ""
            run_dir = a; i += 1
        else
            error("Unexpected argument: $a")
        end
    end
    run_dir == "" && usage_error()
    return (run_dir=run_dir, model_file=model_file, delta_max=delta_max)
end

# Rescan the success-flagged rays of one model in place and write back to path.
function rescan_model_file!(model_path::AbstractString, delta_max::Float64)
    println("  $(basename(model_path))")
    raw    = JSON3.read(read(model_path, String))
    model  = to_dict(raw)

    haskey(model, "scan_results") ||
        (println("    no scan_results — skipping"); return 0)

    ctx  = build_hc_system(model)
    U    = ctx.U
    r0_f = ctx.baseline_r
    mode = get(model, "dynamics_mode", "standard")
    total_rescanned = 0

    for alpha_block in model["scan_results"]
        alpha = Float64(alpha_block["alpha"])

        # Indices (within alpha_block["directions"]) of rays flagged :success
        success_idx = findall(d -> String(d["flag"]) == "success",
                              alpha_block["directions"])
        isempty(success_idx) && continue

        r0_val = r0_f isa Function ? r0_f(alpha) : r0_f
        ws     = ctx.make_workspace(alpha)  # nothing if alpha ≈ 0 (linear fallback)
        x0_val = ws === nothing ? nothing :
                 (ctx.x0 isa Function ? ctx.x0(alpha) : ctx.x0)

        for idx in success_idx
            d       = alpha_block["directions"][idx]
            ray_id  = Int(d["ray_id"])
            u_raw   = @view U[:, ray_id]
            nrm     = norm(u_raw)
            nrm > 0 || error("Ray $ray_id has zero norm.")
            u = collect(u_raw ./ nrm)

            res = if ws === nothing
                lf = ctx.linear_fallback
                scan_ray_linear_alpha0(lf.A, lf.A_fac, lf.x_base_linear,
                                       r0_val, u; max_pert_mag=delta_max)
            else
                scan_ray(ws, x0_val, r0_val, u; max_pert_mag=delta_max)
            end

            # Overwrite just this direction entry (ray_id preserved).
            new_row = Dict{String,Any}(
                "ray_id"        => ray_id,
                "flag"          => String(res.flag),
                "status"        => string(res.status),
                "drcrit"        => res.drcrit,
                "rcrit"         => res.rcrit,
                "delta_c"       => Float64(res.delta_c),
                "x_preboundary" => res.x_preboundary,
                "x_boundary"    => res.x_crit,
            )
            alpha_block["directions"][idx] = new_row
            total_rescanned += 1
        end
        println("    alpha=$alpha  rescanned=$(length(success_idx))")
    end

    # Record the rescan in scan_config; leave original "max_pert" field so the
    # provenance of non-success rays is still readable.
    cfg = get(model, "scan_config", Dict{String,Any}())
    cfg["rescan_success_max_pert"] = delta_max
    cfg["rescan_success_count"]    = total_rescanned
    model["scan_config"] = cfg

    # Downstream ODE / backtrack results referring to rescanned rays are now
    # stale. Safest default: drop them so downstream stages must be re-run.
    if haskey(model, "post_dynamics_results")
        delete!(model, "post_dynamics_results")
    end
    if haskey(model, "backtrack_results")
        delete!(model, "backtrack_results")
    end

    open(model_path, "w") do io
        JSON3.write(io, model)
    end
    println("    total rescanned in file: $total_rescanned")
    return total_rescanned
end

function main()
    opts     = parse_rescan_args(ARGS)
    run_root = canonical_models_root(@__DIR__, opts.run_dir)
    isdir(run_root) || error("Run directory not found: $run_root")
    paths    = resolve_model_paths(run_root, opts.model_file)

    println("Rescan success directions")
    println("  run: $run_root")
    println("  models: $(length(paths))")
    println("  delta_max: $(opts.delta_max)")

    grand_total = 0
    for p in paths
        grand_total += rescan_model_file!(p, opts.delta_max)
    end
    println("Done. Total directions rescanned: $grand_total")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
