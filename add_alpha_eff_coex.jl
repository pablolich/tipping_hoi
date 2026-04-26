#!/usr/bin/env julia
# Augment bank JSONs with `alpha_eff_coex` — the L² nonlinearity fraction
# computed over the coexistence region of state space.
#
# For each scalar-α bank (gibbs, lever, karatayev, aguade, mougi, stouffer),
# we sample equilibria along every perturbation ray up to δ_c, then fit
# the best-affine map x → f_nominal(x) and take the residual fraction.
# See sample_coex_equilibria.jl for the math.
#
# Existing fields are never modified.  Marsland is skipped (same as the
# Taylor augmentation).  Multi-α GLV+HOI banks (elegant/standard/…) are
# also skipped here — they would need a grid, which we defer.
#
# Usage:
#   julia --startup-file=no new_code/add_alpha_eff_coex.jl <path> [...] [--force] [--dry-run] [--K N]

using JSON3

include(joinpath(@__DIR__, "sample_coex_equilibria.jl"))

const SCALAR_ALPHA_MODES = ("gibbs", "lever", "karatayev", "aguade", "mougi", "stouffer")
const SKIPPED_MODES      = ("marsland",)

function usage()
    println("""
Usage: julia --startup-file=no new_code/add_alpha_eff_coex.jl <path> [<path>...] [--force] [--dry-run] [--K N]

Augments scalar-α bank JSONs with:
  alpha_eff_coex                (scalar)
  alpha_eff_coex_R2
  alpha_eff_coex_N_samples
  alpha_eff_coex_K              (samples per ray used)

Only processes dynamics_mode ∈ $(SCALAR_ALPHA_MODES).
Skips marsland and α-grid GLV+HOI banks.  Existing `alpha_eff` is never touched.
""")
end

function parse_cli_args(args::Vector{String})
    force = false
    dry_run = false
    K = 10
    paths = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help"); usage(); exit(0)
        elseif a == "--force"; force = true; i += 1
        elseif a == "--dry-run"; dry_run = true; i += 1
        elseif a == "--K"
            i + 1 <= length(args) || error("--K requires a value")
            K = parse(Int, args[i + 1])
            i += 2
        elseif startswith(a, "--"); error("Unknown flag: $a")
        else; push!(paths, a); i += 1
        end
    end
    isempty(paths) && (usage(); error("At least one path is required."))
    return (paths, force, dry_run, K)
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

already_processed(bank::AbstractDict) = haskey(bank, "alpha_eff_coex")

# Returns (:updated|:skipped|:error, reason_or_nothing)
function process_bank!(bank::Dict{String,Any}, K::Int)
    mode = String(get(bank, "dynamics_mode", ""))
    isempty(mode) && return (:skipped, "no dynamics_mode")
    mode in SKIPPED_MODES && return (:skipped, "mode=$mode (skipped by design)")
    mode in SCALAR_ALPHA_MODES || return (:skipped, "mode=$mode (not in scalar-α set)")

    haskey(bank, "x_star")      || return (:skipped, "no x_star")
    haskey(bank, "U")           || return (:skipped, "no U")
    haskey(bank, "scan_results") || return (:skipped, "no scan_results")

    r = compute_alpha_eff_coex(bank; K=K, α_index=1)
    (isfinite(r.alpha_eff_coex) && isfinite(r.R2)) ||
        return (:error, "non-finite alpha_eff_coex result")

    bank["alpha_eff_coex"]            = r.alpha_eff_coex
    bank["alpha_eff_coex_R2"]         = r.R2
    bank["alpha_eff_coex_N_samples"]  = r.N
    bank["alpha_eff_coex_K"]          = K
    return (:updated, nothing)
end

function main(args::Vector{String})
    paths_arg, force, dry_run, K = parse_cli_args(args)
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

        local status::Symbol
        local reason
        try
            status, reason = process_bank!(bank, K)
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
            α = bank["alpha_eff_coex"]
            N = bank["alpha_eff_coex_N_samples"]
            println("$tag α_eff_coex=$(round(α, digits=4))  N=$N  $path")
        elseif status == :skipped
            skipped += 1
            println("skipped $path ($(reason))")
        else  # :error
            errors += 1
            @warn "error processing bank" path reason
        end
    end

    println("summary updated=$updated skipped=$skipped errors=$errors dry_run=$dry_run K=$K")
end

main(copy(ARGS))
