#!/usr/bin/env julia
# Regression test: verify new Newton-polish logic in post_boundary_dynamics.jl
# produces x_postboundary_snap within 1e-6 L2 of the reference run.
#
# Usage:
#   julia --startup-file=no new_code/test_post_boundary_regression.jl

using JSON3

const REF_RUN_SUBDIR = "1_bank_10_models_n_3-4_4_dirs_b_dirichlet"
const SNAP_TOL = 1e-6

# ── inline to_dict (from json_utils.jl) ──────────────────────────────────────
function to_dict(x)
    if x isa JSON3.Object
        d = Dict{String,Any}()
        for (k, v) in pairs(x)
            d[String(k)] = to_dict(v)
        end
        return d
    elseif x isa JSON3.Array
        return [to_dict(v) for v in x]
    else
        return x
    end
end

# ── helpers ───────────────────────────────────────────────────────────────────
function load_json(path::String)
    to_dict(JSON3.read(read(path, String)))
end

function safe_write_json(path::AbstractString, payload)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON3.write(io, payload)
    end
    mv(tmp, path; force=true)
end

# Extract reference snap data: file → list of (alpha_idx, ray_id, x_snap_or_nothing)
function extract_snaps(post_dynamics_results)
    entries = Tuple{Int,Int,Union{Nothing,Vector{Float64}}}[]
    for ab in post_dynamics_results
        a_idx = Int(ab["alpha_idx"])
        for row in ab["directions"]
            ray_id = Int(row["ray_id"])
            raw = row["x_postboundary_snap"]
            x_snap = raw === nothing ? nothing : Float64.(raw)
            push!(entries, (a_idx, ray_id, x_snap))
        end
    end
    return entries
end

# ── main ──────────────────────────────────────────────────────────────────────
function main()
    script_dir = @__DIR__
    ref_run_dir = joinpath(script_dir, "model_runs", REF_RUN_SUBDIR)
    isdir(ref_run_dir) || error("Reference run directory not found: $ref_run_dir")

    json_files = sort([f for f in readdir(ref_run_dir)
                       if endswith(f, ".json") && !contains(f, "_chunk_")])
    isempty(json_files) && error("No JSON files in $ref_run_dir")

    # ── Step 1: load all reference results into memory ────────────────────────
    println("Loading $(length(json_files)) reference models from:\n  $ref_run_dir")
    ref_data = Dict{String,Any}()   # filename → full parsed dict
    for fname in json_files
        d = load_json(joinpath(ref_run_dir, fname))
        haskey(d, "post_dynamics_results") ||
            error("Reference file $fname has no post_dynamics_results — run Stage 3 first.")
        ref_data[fname] = d
    end

    # ── Step 2: write stripped copies to a temp dir ───────────────────────────
    tmpdir = mktempdir(; cleanup=false)
    println("Temp dir: $tmpdir")
    STRIP_KEYS = ("post_dynamics_results", "post_dynamics_config", "backtrack_results")
    for fname in json_files
        d = copy(ref_data[fname])
        for k in STRIP_KEYS
            delete!(d, k)
        end
        safe_write_json(joinpath(tmpdir, fname), d)
    end

    # ── Step 3: run post_boundary_dynamics.jl on the temp dir ─────────────────
    # Pass tmpdir as an absolute path — canonical_models_root(@__DIR__, tmpdir)
    # calls joinpath(script_dir, "model_runs", tmpdir) which, because tmpdir is
    # absolute, resolves to just tmpdir (Julia joinpath drops earlier components
    # when a later component is absolute).
    script_path = joinpath(script_dir, "post_boundary_dynamics.jl")
    cmd = `julia --startup-file=no $script_path $tmpdir`
    println("\nRunning:\n  $cmd\n")
    t0 = time()
    success = run(ignorestatus(cmd)).exitcode == 0
    elapsed = round(time() - t0; digits=1)
    println("\nScript exited $(success ? "OK" : "with error") in $(elapsed)s\n")
    if !success
        error("post_boundary_dynamics.jl exited with non-zero status. Inspect $tmpdir.")
    end

    # ── Step 4: compare x_postboundary_snap ──────────────────────────────────
    total_dirs  = 0
    n_pass      = 0
    n_fail      = 0
    fail_lines  = String[]

    println("Per-model comparison:")
    println(rpad("File", 50), "  Dirs  Pass  Fail")
    println("-"^75)

    for fname in json_files
        new_path = joinpath(tmpdir, fname)
        isfile(new_path) || error("Expected output not found: $new_path")
        new_model = load_json(new_path)
        haskey(new_model, "post_dynamics_results") ||
            error("post_dynamics_results missing from new output: $fname")

        ref_snaps = extract_snaps(ref_data[fname]["post_dynamics_results"])
        new_snaps = extract_snaps(new_model["post_dynamics_results"])

        # Build lookup: (alpha_idx, ray_id) → x_snap
        new_lookup = Dict((a, r) => x for (a, r, x) in new_snaps)

        file_pass = 0
        file_fail = 0
        for (a_idx, ray_id, ref_x) in ref_snaps
            new_x = get(new_lookup, (a_idx, ray_id), :missing)
            new_x === :missing && error("Missing (alpha_idx=$a_idx, ray_id=$ray_id) in new output for $fname")

            if ref_x === nothing && new_x === nothing
                file_pass += 1
            elseif ref_x === nothing || new_x === nothing
                file_fail += 1
                push!(fail_lines,
                    "  FAIL $fname  alpha_idx=$a_idx ray_id=$ray_id  " *
                    "ref=$(ref_x === nothing ? "null" : "vec")  " *
                    "new=$(new_x === nothing ? "null" : "vec")")
            else
                diff = norm(ref_x .- new_x)
                if diff < SNAP_TOL
                    file_pass += 1
                else
                    file_fail += 1
                    push!(fail_lines,
                        "  FAIL $fname  alpha_idx=$a_idx ray_id=$ray_id  " *
                        "L2_diff=$(round(diff; sigdigits=4))  (tol=$SNAP_TOL)")
                end
            end
        end

        total_dirs += file_pass + file_fail
        n_pass     += file_pass
        n_fail     += file_fail
        println(rpad(fname, 50), "  ", lpad(file_pass + file_fail, 4),
                "  ", lpad(file_pass, 4), "  ", lpad(file_fail, 4))
    end

    println("-"^75)
    println()
    if !isempty(fail_lines)
        println("FAILURES:")
        for ln in fail_lines
            println(ln)
        end
        println()
    end

    println("REGRESSION TEST SUMMARY")
    println("  Directions compared : $total_dirs")
    println("  PASS                : $n_pass")
    println("  FAIL                : $n_fail")
    if n_fail == 0
        println("  PASS All comparisons passed.")
        exit(0)
    else
        println("  FAIL $n_fail comparison(s) failed.")
        println("  Inspect temp dir: $tmpdir")
        exit(1)
    end
end

using LinearAlgebra
main()
