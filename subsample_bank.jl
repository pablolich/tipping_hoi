"""
    Create a representative subsample of a Gibbs bank.

    Stratified by parameterization: each parameterization keeps
    ceil(fraction * n) systems, randomly selected. This preserves
    the relative representation across parameterizations/blocks/regimes.

    Usage:
      julia --project=~/Desktop/tipping_points_hoi/new_code subsample_bank.jl [fraction]

    fraction defaults to 0.1 (10%). The subsampled bank is written to a sibling
    directory with "_sub{pct}pct" appended to the name.
"""

using Pkg
Pkg.activate(joinpath(homedir(), "Desktop", "tipping_points_hoi", "new_code"))

using JSON
using Random
using Printf

const REPO_ROOT = joinpath(homedir(), "Desktop", "tipping_points_hoi", "new_code")
const SOURCE_DIR = joinpath(REPO_ROOT, "model_runs",
                            "gibbs_128_dirs_from_gibbs_figures_n20_seed12345")
const SEED = 42

function parse_param_id(fname::String)
    m = match(r"_param(\d+)_", fname)
    m === nothing && error("Cannot parse param_id from $fname")
    return parse(Int, m.captures[1])
end

function subsample_bank(fraction::Float64)
    0 < fraction < 1 || error("fraction must be in (0, 1), got $fraction")

    pct = round(Int, fraction * 100)
    out_dir = SOURCE_DIR * "_sub$(pct)pct"
    mkpath(out_dir)

    all_files = filter(f -> endswith(f, ".json"), readdir(SOURCE_DIR))
    println("Source bank: $(length(all_files)) files in\n  $SOURCE_DIR")

    # Group files by parameterization
    by_param = Dict{Int, Vector{String}}()
    for f in all_files
        pid = parse_param_id(f)
        push!(get!(by_param, pid, String[]), f)
    end

    rng = MersenneTwister(SEED)
    total_kept = 0

    for pid in sort(collect(keys(by_param)))
        files = by_param[pid]
        n_keep = max(1, ceil(Int, fraction * length(files)))
        selected = shuffle(rng, files)[1:n_keep]

        for (new_rep, fname) in enumerate(sort(selected))
            src = joinpath(SOURCE_DIR, fname)
            payload = open(src) do io; JSON.parse(io); end

            # Update replicate_id to be contiguous in the subsample
            payload["metadata"]["replicate_id"] = new_rep

            new_fname = replace(fname, r"_rep\d+" => @sprintf("_rep%03d", new_rep))
            dst = joinpath(out_dir, new_fname)
            tmp = dst * ".tmp"
            open(tmp, "w") do io; JSON.print(io, payload); end
            mv(tmp, dst; force=true)
        end

        @printf("  param %3d: %3d / %3d kept\n", pid, n_keep, length(files))
        total_kept += n_keep
    end

    println("\n" * "=" ^ 60)
    println("Subsample complete")
    println("  Fraction: $(fraction) ($(pct)%)")
    println("  Parameterizations: $(length(by_param))")
    println("  Files: $total_kept / $(length(all_files))")
    println("  Output: $out_dir")
end

fraction = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.1
subsample_bank(fraction)
