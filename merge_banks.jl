"""
merge_banks.jl

Merges two bank directories that differ only in n-range into a single new bank.
Originals are left untouched.

Usage:
    julia merge_banks.jl
"""

using Printf

const MODEL_RUNS_DIR = joinpath(@__DIR__, "model_runs")

const BANK_A = "1_bank_50_models_n_3-15_128_dirs_b_dirichlet"
const BANK_B = "1_bank_50_models_n_16-20_128_dirs_b_dirichlet"
const MERGED  = "1_bank_50_models_n_3-20_128_dirs_b_dirichlet"  # re-merge to pick up updated 16-20 results

function merge_banks(bank_a::String, bank_b::String, merged::String)
    src_a  = joinpath(MODEL_RUNS_DIR, bank_a)
    src_b  = joinpath(MODEL_RUNS_DIR, bank_b)
    dst    = joinpath(MODEL_RUNS_DIR, merged)

    isdir(src_a) || error("Bank A not found: $src_a")
    isdir(src_b) || error("Bank B not found: $src_b")

    if isdir(dst)
        println("Destination already exists: $dst")
        print("Overwrite? [y/N]: ")
        ans = strip(readline())
        ans in ("y", "Y") || (println("Aborted."); return)
        rm(dst; recursive=true)
    end

    mkpath(dst)

    files_a = filter(f -> endswith(f, ".json"), readdir(src_a))
    files_b = filter(f -> endswith(f, ".json"), readdir(src_b))

    # Check for filename collisions
    overlap = intersect(Set(files_a), Set(files_b))
    if !isempty(overlap)
        error("Filename collision between the two banks ($(length(overlap)) files). Aborting.")
    end

    for f in files_a
        cp(joinpath(src_a, f), joinpath(dst, f))
    end
    for f in files_b
        cp(joinpath(src_b, f), joinpath(dst, f))
    end

    total = length(files_a) + length(files_b)
    println("Merged $total files into: $dst")
    println("  $(length(files_a)) from $bank_a")
    println("  $(length(files_b)) from $bank_b")
end

merge_banks(BANK_A, BANK_B, MERGED)
