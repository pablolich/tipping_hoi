#!/usr/bin/env julia
# Merge boundary-scan chunk files into canonical model JSONs.
# Chunk files are produced by boundary_scan.jl when --dir-chunk-start/--dir-chunk-end are passed.
#
# Usage:
#   julia --startup-file=no hpc/merge_chunks.jl <run_dir> [options]
#
# Options:
#   --model-file FILE    Process only this one model (stem or full filename)
#   --keep-chunks        Keep chunk files after successful merge (default: delete)
#   --no-overwrite       Skip (warn) if canonical already has scan_results
#   --dry-run            Validate and report, write nothing

using JSON3

include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "json_utils.jl"))

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no hpc/merge_chunks.jl <run_dir> [options]

    Required:
      <run_dir>            Folder inside model_runs/

    Options:
      --model-file FILE    Process only this one model (stem or full filename)
      --keep-chunks        Keep chunk files after successful merge (default: delete)
      --no-overwrite       Skip (warn) if canonical already has scan_results
      --dry-run            Validate and report, write nothing
    """
    error(msg)
end

function strip_chunk_suffix(name::AbstractString)
    # Remove _chunk_START_END.json suffix if present, leaving the stem
    base = endswith(name, ".json") ? name[1:end-5] : name
    m = match(r"^(.+)_chunk_\d+_\d+$", base)
    m !== nothing ? m.captures[1] : base
end

function parse_args(args::Vector{String})
    isempty(args) && usage_error()

    run_dir      = ""
    model_file   = nothing
    keep_chunks  = false
    no_overwrite = false
    dry_run      = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--model-file" && i < length(args)
            raw = args[i + 1]
            # Strip _chunk_... suffix if user accidentally passed a chunk filename
            stem = strip_chunk_suffix(basename(raw))
            model_file = stem * ".json"
            i += 2
        elseif arg == "--keep-chunks"
            keep_chunks = true
            i += 1
        elseif arg == "--no-overwrite"
            no_overwrite = true
            i += 1
        elseif arg == "--dry-run"
            dry_run = true
            i += 1
        elseif startswith(arg, "--")
            error("Unknown flag: $arg")
        elseif run_dir == ""
            run_dir = arg
            i += 1
        else
            error("Unexpected argument: $arg")
        end
    end

    run_dir == "" && usage_error()
    return (run_dir=run_dir, model_file=model_file,
            keep_chunks=keep_chunks, no_overwrite=no_overwrite, dry_run=dry_run)
end

# ---------------------------------------------------------------------------
# Chunk discovery
# ---------------------------------------------------------------------------

# Returns Vector of (start_0based::Int, end_0based::Int, path::String), sorted by start.
function find_chunks(root::AbstractString, stem::AbstractString)
    escaped = replace(stem, r"([.+*?()\[\]{}|^$\\])" => s"\\\1")
    chunk_re = Regex("^$(escaped)_chunk_(\\d+)_(\\d+)\\.json\$")
    results = Tuple{Int,Int,String}[]
    for f in readdir(root)
        m = match(chunk_re, f)
        m === nothing && continue
        s = parse(Int, m.captures[1])
        e = parse(Int, m.captures[2])
        push!(results, (s, e, joinpath(root, f)))
    end
    sort!(results; by = x -> x[1])
    return results
end

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

function validate_chunks(chunks::Vector{Tuple{Int,Int,String}}, n_dirs::Int, stem::AbstractString)
    # Gap / overlap check
    for i in 2:length(chunks)
        prev_end   = chunks[i-1][2]
        curr_start = chunks[i][1]
        if prev_end + 1 != curr_start
            error("Gap or overlap in chunks for '$stem': " *
                  "chunk ending at $prev_end followed by chunk starting at $curr_start.")
        end
    end
    # Coverage check
    first_start = chunks[1][1]
    last_end    = chunks[end][2]
    if first_start != 0
        error("Chunks for '$stem' do not start at 0 (first chunk starts at $first_start).")
    end
    if last_end != n_dirs - 1
        error("Chunks for '$stem' cover only up to direction $last_end " *
              "but n_dirs=$n_dirs requires coverage up to $(n_dirs - 1).")
    end
end

function alpha_grids_consistent(ag1, ag2; tol=1e-12)
    length(ag1) != length(ag2) && return false
    for (a, b) in zip(ag1, ag2)
        abs(Float64(a) - Float64(b)) > tol && return false
    end
    return true
end

# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

function merge_model_chunks(canonical_path::AbstractString,
                             chunks::Vector{Tuple{Int,Int,String}},
                             stem::AbstractString;
                             no_overwrite::Bool,
                             dry_run::Bool,
                             keep_chunks::Bool)

    # Load all chunk dicts
    chunk_dicts = [to_dict(JSON3.read(read(path, String))) for (_, _, path) in chunks]

    first_chunk = chunk_dicts[1]
    n_dirs = Int(first_chunk["n_dirs"])

    validate_chunks(chunks, n_dirs, stem)

    # Alpha grid consistency
    ref_ag = first_chunk["scan_config"]["alpha_grid"]
    for (ci, cd) in enumerate(chunk_dicts[2:end])
        ag = cd["scan_config"]["alpha_grid"]
        alpha_grids_consistent(ref_ag, ag) ||
            error("alpha_grid mismatch in chunk $(ci+1) for '$stem'.")
    end

    # no-overwrite check on canonical
    if no_overwrite && isfile(canonical_path)
        canonical_data = to_dict(JSON3.read(read(canonical_path, String)))
        if haskey(canonical_data, "scan_results")
            @warn "Skipping '$stem': canonical already has scan_results (--no-overwrite)."
            return false
        end
    end

    # Determine number of alpha blocks from first chunk
    n_alpha = length(first_chunk["scan_results"])

    # Merge directions for each alpha block
    merged_results = Vector{Dict{String,Any}}(undef, n_alpha)
    for a_idx in 1:n_alpha
        alpha_val = first_chunk["scan_results"][a_idx]["alpha"]
        alpha_idx_val = first_chunk["scan_results"][a_idx]["alpha_idx"]

        all_dirs = Dict{String,Any}[]
        for cd in chunk_dicts
            block = cd["scan_results"][a_idx]
            append!(all_dirs, block["directions"])
        end

        # Sort by ray_id (1-based Int)
        sort!(all_dirs; by = d -> Int(d["ray_id"]))

        merged_results[a_idx] = Dict{String,Any}(
            "alpha"     => alpha_val,
            "alpha_idx" => alpha_idx_val,
            "directions" => all_dirs,
        )
    end

    # Report
    total_dirs = length(merged_results[1]["directions"])
    println("  $stem: merging $(length(chunks)) chunks → $total_dirs directions, $n_alpha alpha blocks")

    dry_run && return true

    # Build output: copy first chunk as base, patch scan_config, replace scan_results
    output = copy(first_chunk)

    # Build clean scan_config (drop dir_start/dir_end, update scanned range)
    old_cfg = first_chunk["scan_config"]
    new_cfg = Dict{String,Any}()
    for (k, v) in old_cfg
        k in ("dir_start", "dir_end") && continue
        new_cfg[k] = v
    end
    new_cfg["scanned_dir_start"] = 1
    new_cfg["scanned_dir_end"]   = n_dirs

    output["scan_config"]  = new_cfg
    output["scan_results"] = merged_results
    delete!(output, "chunk_info")

    safe_write_json(canonical_path, output)
    println("    wrote $(basename(canonical_path))")

    if !keep_chunks
        for (_, _, path) in chunks
            rm(path; force=true)
            println("    deleted $(basename(path))")
        end
    end

    return true
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    opts = parse_args(ARGS)
    run_root = canonical_models_root(@__DIR__, opts.run_dir)
    isdir(run_root) || error("Run directory not found: $run_root")

    canonical_paths = resolve_model_paths(run_root, opts.model_file)

    opts.dry_run && println("[dry-run] no files will be written")

    n_merged  = 0
    n_skipped = 0

    for canonical_path in canonical_paths
        stem = splitext(basename(canonical_path))[1]
        chunks = find_chunks(run_root, stem)

        if isempty(chunks)
            println("  $stem: no chunk files found, skipping")
            n_skipped += 1
            continue
        end

        ok = merge_model_chunks(canonical_path, chunks, stem;
                                no_overwrite=opts.no_overwrite,
                                dry_run=opts.dry_run,
                                keep_chunks=opts.keep_chunks)
        ok ? (n_merged += 1) : (n_skipped += 1)
    end

    println("\nMerged: $n_merged, Skipped: $n_skipped")
    n_merged == 0 && n_skipped == length(canonical_paths) && exit(1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
