#!/usr/bin/env julia
# Add the Taylor-expansion-based effective nonlinearity `alpha_eff_taylor`
# (and P_taylor, Q_taylor) to existing bank JSONs.
#
# The existing `alpha_eff` field is NEVER modified — this is an additive
# augmentation for cross-model comparison.
#
# Usage:
#   julia --startup-file=no new_code/add_alpha_eff_taylor.jl <file-or-dir> [...] [--force] [--dry-run]
#
# Flags:
#   --force      Reprocess JSONs that already have alpha_eff_taylor[_grid].
#   --dry-run    Compute but do not write; log what would change.

using JSON3

include(joinpath(@__DIR__, "utils", "alpha_eff_taylor.jl"))

const SCALAR_MODES = ("gibbs", "lever", "karatayev", "aguade", "mougi", "stouffer")
const GRID_MODES = (
    "standard", "elegant", "balanced", "balanced_stable",
    "constrained_r", "unique_equilibrium", "all_negative",
)

function usage()
    println("""
Usage: julia --startup-file=no new_code/add_alpha_eff_taylor.jl <path> [<path>...] [--force] [--dry-run]

Augments each bank JSON with:
  alpha_eff_taylor, P_taylor, Q_taylor                       (scalar modes)
  alpha_eff_taylor_grid, P_taylor_grid, Q_taylor_grid        (GLV+HOI α-grid modes)

Skips marsland banks. Existing alpha_eff field is never touched.
""")
end

function parse_cli_args(args::Vector{String})
    force = false
    dry_run = false
    paths = String[]
    for a in args
        if a in ("-h", "--help")
            usage(); exit(0)
        elseif a == "--force"
            force = true
        elseif a == "--dry-run"
            dry_run = true
        elseif startswith(a, "--")
            error("Unknown flag: $a")
        else
            push!(paths, a)
        end
    end
    isempty(paths) && (usage(); error("At least one path is required."))
    return (paths, force, dry_run)
end

function collect_json_paths(args::Vector{String})
    paths = String[]
    for arg in args
        if isdir(arg)
            for (root, _, files) in walkdir(arg)
                for file in files
                    endswith(file, ".json") || continue
                    occursin("chunk", file) && continue
                    push!(paths, joinpath(root, file))
                end
            end
        elseif isfile(arg)
            push!(paths, arg)
        else
            error("Path does not exist: $(arg)")
        end
    end
    sort!(unique!(paths))
    return paths
end

function already_processed(bank::AbstractDict)
    return haskey(bank, "alpha_eff_taylor") || haskey(bank, "alpha_eff_taylor_grid")
end

# Returns (:updated|:skipped|:error, reason_or_nothing)
function process_bank!(bank::Dict{String,Any})
    mode = String(get(bank, "dynamics_mode", ""))
    isempty(mode) && return (:skipped, "no dynamics_mode")
    haskey(bank, "x_star") || return (:skipped, "no x_star")
    x_star = Float64.(bank["x_star"])

    if mode == "marsland"
        return (:skipped, "marsland (skipped by design)")
    elseif mode in SCALAR_MODES
        f, _, _ = build_per_capita_rates(bank)
        out = compute_alpha_eff_taylor(f, x_star)
        (isfinite(out.alpha_eff_taylor) && isfinite(out.P) && isfinite(out.Q)) ||
            return (:error, "non-finite taylor result")
        bank["alpha_eff_taylor"] = out.alpha_eff_taylor
        bank["P_taylor"]         = out.P
        bank["Q_taylor"]         = out.Q
        return (:updated, nothing)
    elseif mode in GRID_MODES
        haskey(bank, "alpha_grid") || return (:skipped, "no alpha_grid for $mode")
        αgrid = collect(Float64, bank["alpha_grid"])
        grid  = Vector{Float64}(undef, length(αgrid))
        Pg    = Vector{Float64}(undef, length(αgrid))
        Qg    = Vector{Float64}(undef, length(αgrid))
        for (i, α) in pairs(αgrid)
            f, _, _ = build_per_capita_rates_for_alpha(bank, α)
            out = compute_alpha_eff_taylor(f, x_star)
            (isfinite(out.alpha_eff_taylor) && isfinite(out.P) && isfinite(out.Q)) ||
                return (:error, "non-finite taylor result at α=$α")
            grid[i] = out.alpha_eff_taylor
            Pg[i]   = out.P
            Qg[i]   = out.Q
        end
        bank["alpha_eff_taylor_grid"] = grid
        bank["P_taylor_grid"]         = Pg
        bank["Q_taylor_grid"]         = Qg
        return (:updated, nothing)
    else
        return (:skipped, "unknown dynamics_mode=$mode")
    end
end

function main(args::Vector{String})
    paths_arg, force, dry_run = parse_cli_args(args)
    paths = collect_json_paths(paths_arg)

    updated = 0
    skipped = 0
    errors  = 0

    for path in paths
        local bank::Dict{String,Any}
        try
            bank = JSON3.read(read(path, String), Dict{String,Any})
        catch err
            @warn "failed to parse JSON" path exception = err
            errors += 1
            continue
        end

        if !force && already_processed(bank)
            skipped += 1
            continue
        end

        local status::Symbol; local reason
        try
            status, reason = process_bank!(bank)
        catch err
            @warn "failed to process bank" path exception = err
            errors += 1
            continue
        end

        if status == :updated
            if !dry_run
                try
                    write(path, JSON3.write(bank))
                catch err
                    @warn "failed to write JSON" path exception = err
                    errors += 1
                    continue
                end
            end
            updated += 1
            tag = dry_run ? "would-update" : "updated"
            println("$tag $path")
        elseif status == :skipped
            skipped += 1
            println("skipped $path ($(reason))")
        else  # :error
            errors += 1
            @warn "error processing bank" path reason
        end
    end

    println("summary updated=$updated skipped=$skipped errors=$errors dry_run=$dry_run")
end

main(copy(ARGS))
