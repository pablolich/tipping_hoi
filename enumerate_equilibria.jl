using JSON3
using HomotopyContinuation
const HC = HomotopyContinuation
using LinearAlgebra
using DataFrames
using CSV
using Printf
using Base.Threads

include(joinpath(@__DIR__, "utils", "json_utils.jl"))
include(joinpath(@__DIR__, "utils", "math_utils.jl"))
include(joinpath(@__DIR__, "model_store_utils.jl"))

const MAX_N              = 10
const MAX_N_OUTPUT       = 10
const ZERO_ABUNDANCE_TOL = 1e-8
const IMAG_TOL           = 1e-8
const DEDUP_TOL          = 1e-6

const BANK_ROOT = joinpath(@__DIR__, "model_runs")

const BANKS = [
    ("2_bank_unique_equilibrium_50_models_n_4-10_128_dirs",
        joinpath(BANK_ROOT, "2_bank_unique_equilibrium_50_models_n_4-10_128_dirs")),
    ("2_bank_elegant_50_models_n_4-20_128_dirs_muB_-0.1",
        joinpath(BANK_ROOT, "2_bank_elegant_50_models_n_4-20_128_dirs_muB_-0.1")),
    ("2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.0",
        joinpath(BANK_ROOT, "2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.0")),
    ("2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.1",
        joinpath(BANK_ROOT, "2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.1")),
]

const LOG_LOCK = ReentrantLock()
log_safe(msg::AbstractString) = @lock LOG_LOCK println(msg)

function load_model(path::AbstractString)
    return to_dict(JSON3.read(read(path, String)))
end

prescale_local(A, B, alpha) = ((1 - alpha) .* A, alpha .* B)

function determine_r(mode::AbstractString,
                     A::Matrix{Float64},
                     B::Array{Float64,3},
                     r_static::Vector{Float64},
                     alpha::Float64)
    mode == "unique_equilibrium" && return compute_r_unique_equilibrium(A, B, alpha)
    return r_static
end

function build_factored_system(r_eff::AbstractVector{<:Real},
                                A_eff::AbstractMatrix{<:Real},
                                B_eff::Array{<:Real,3})
    n = length(r_eff)
    HC.@var x[1:n]
    eqs = Vector{HC.Expression}(undef, n)
    @inbounds for i in 1:n
        lin  = sum(A_eff[i, j] * x[j] for j in 1:n)
        quad = sum(B_eff[j, k, i] * x[j] * x[k] for j in 1:n for k in 1:n)
        eqs[i] = x[i] * (r_eff[i] + lin + quad)
    end
    return HC.System(eqs; variables = collect(x))
end

function dedup_solutions(sols::Vector{Vector{Float64}}, tol::Float64)
    uniques = Vector{Vector{Float64}}()
    for s in sols
        dup = false
        for u in uniques
            if norm(s .- u) < tol
                dup = true
                break
            end
        end
        dup || push!(uniques, copy(s))
    end
    return uniques
end

function dynamics_jacobian(A_eff::AbstractMatrix{<:Real},
                           B_eff::Array{<:Real,3},
                           r_eff::AbstractVector{<:Real},
                           x::AbstractVector{<:Real})
    n  = length(x)
    F  = per_capita_growth(A_eff, B_eff, r_eff, x)
    JF = jacobian_F(A_eff, B_eff, x)
    J  = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        J[i, j] = (i == j ? F[i] : 0.0) + x[i] * JF[i, j]
    end
    return J
end

function dynamics_lambda_max(A_eff, B_eff, r_eff, x)
    J = dynamics_jacobian(A_eff, B_eff, r_eff, x)
    return maximum(real, eigvals(J))
end

function enumerate_equilibria_at(A::Matrix{Float64},
                                 B::Array{Float64,3},
                                 r_eff::Vector{Float64},
                                 alpha::Float64)
    A_eff, B_eff = prescale_local(A, B, alpha)
    syst   = build_factored_system(r_eff, A_eff, B_eff)
    result = HC.solve(syst; show_progress = false, threading = false)
    cands  = HC.real_solutions(result; tol = IMAG_TOL)

    feasible = Vector{Vector{Float64}}()
    for c in cands
        cc = Float64.(c)
        clamp_small_negatives!(cc, ZERO_ABUNDANCE_TOL)
        all(cc .>= -ZERO_ABUNDANCE_TOL) || continue
        push!(feasible, cc)
    end

    uniques = dedup_solutions(feasible, DEDUP_TOL)

    out = Vector{NamedTuple{(:x, :stable, :lam), Tuple{Vector{Float64}, Bool, Float64}}}()
    for x in uniques
        lam = NaN
        stable = false
        try
            lam = dynamics_lambda_max(A_eff, B_eff, r_eff, x)
            stable = lam < 0
        catch err
            err isa InterruptException && rethrow(err)
            lam = NaN
            stable = false
        end
        push!(out, (x = x, stable = stable, lam = lam))
    end
    return out
end

function process_model(json_path::String, bank_name::String)
    rows = NamedTuple[]

    model = try
        load_model(json_path)
    catch err
        err isa InterruptException && rethrow(err)
        log_safe("[WARN] JSON parse failed: $(basename(json_path)) — $(err)")
        return rows
    end

    n = Int(model["n"])
    n > MAX_N && return rows

    A = nested_to_matrix(model["A"])
    B = nested_to_tensor3(model["B"])
    mode = String(model["dynamics_mode"])

    r_static = Float64[]
    if mode != "unique_equilibrium"
        if !haskey(model, "r") || model["r"] === nothing
            log_safe("[WARN] missing r field in $(basename(json_path)) (mode=$mode); skipping")
            return rows
        end
        r_static = Float64.(model["r"])
    end

    alpha_grid = collect(Float64, model["alpha_grid"])
    total_eqs  = 0

    for alpha in alpha_grid
        r_eff = determine_r(mode, A, B, r_static, alpha)
        eqs = try
            enumerate_equilibria_at(A, B, r_eff, alpha)
        catch err
            err isa InterruptException && rethrow(err)
            log_safe("[WARN] HC.solve failed: $(basename(json_path)) α=$alpha — $(err)")
            NamedTuple{(:x, :stable, :lam), Tuple{Vector{Float64}, Bool, Float64}}[]
        end
        for (idx, eq) in pairs(eqs)
            push!(rows, (
                bank                = bank_name,
                model_file          = basename(json_path),
                n                   = n,
                alpha               = alpha,
                equilibrium_id      = idx,
                x                   = eq.x,
                stable              = eq.stable,
                max_real_eigenvalue = eq.lam,
            ))
        end
        total_eqs += length(eqs)
    end

    log_safe(@sprintf("[%s] n=%d %s → %d feasible eqs across %d α",
                      bank_name, n, basename(json_path), total_eqs, length(alpha_grid)))
    return rows
end

function pad_row(row::NamedTuple, n_out::Int)
    xvec = row.x
    xs = ntuple(i -> i <= length(xvec) ? xvec[i] : NaN, n_out)
    x_pairs = NamedTuple{ntuple(i -> Symbol("x_", i), n_out)}(xs)
    return merge(
        (bank = row.bank,
         model_file = row.model_file,
         n = row.n,
         alpha = row.alpha,
         equilibrium_id = row.equilibrium_id),
        x_pairs,
        (stable = row.stable,
         max_real_eigenvalue = row.max_real_eigenvalue),
    )
end

function prefilter_paths(bank_dir::AbstractString, min_n::Int, max_n::Int)
    files = sort(readdir(bank_dir; join = true))
    keep  = String[]
    rx    = r"_n_(\d+)_seed_"
    for p in files
        endswith(p, ".json")               || continue
        contains(basename(p), "_chunk_")   && continue
        m = match(rx, basename(p))
        if m === nothing
            push!(keep, p)
            continue
        end
        n = parse(Int, m.captures[1])
        (min_n <= n <= max_n) && push!(keep, p)
    end
    return keep
end

function empty_results_dataframe(n_out::Int = MAX_N_OUTPUT)
    cols = Pair{Symbol, Vector}[
        :bank => String[],
        :model_file => String[],
        :n => Int[],
        :alpha => Float64[],
        :equilibrium_id => Int[],
    ]
    for i in 1:n_out
        push!(cols, Symbol("x_", i) => Float64[])
    end
    push!(cols, :stable => Bool[])
    push!(cols, :max_real_eigenvalue => Float64[])
    return DataFrame(cols...)
end

function rows_to_dataframe(rows::Vector{<:NamedTuple}, n_out::Int = MAX_N_OUTPUT)
    isempty(rows) && return empty_results_dataframe(n_out)
    padded = [pad_row(r, n_out) for r in rows]
    df = DataFrame(padded)
    col_order = vcat(
        [:bank, :model_file, :n, :alpha, :equilibrium_id],
        [Symbol("x_", i) for i in 1:n_out],
        [:stable, :max_real_eigenvalue],
    )
    return df[:, col_order]
end

function run_enumeration(banks::Vector{<:Tuple{AbstractString, AbstractString}};
                         output_path::AbstractString,
                         min_n::Int = 1,
                         max_n::Int = MAX_N)
    mkpath(dirname(output_path))
    all_rows = Vector{NamedTuple}()

    for (bank_name, bank_dir) in banks
        if !isdir(bank_dir)
            log_safe("[WARN] bank directory not found: $bank_dir — skipping")
            continue
        end
        paths = prefilter_paths(bank_dir, min_n, max_n)
        log_safe("═══ Bank: $bank_name — $(length(paths)) files ($min_n ≤ n ≤ $max_n)")

        bank_rows = Vector{NamedTuple}()
        rows_lock = ReentrantLock()

        @threads for i in eachindex(paths)
            local_rows = process_model(paths[i], bank_name)
            @lock rows_lock append!(bank_rows, local_rows)
        end

        log_safe("═══ Bank $bank_name complete — $(length(bank_rows)) rows")
        append!(all_rows, bank_rows)
    end

    df = rows_to_dataframe(all_rows, MAX_N_OUTPUT)
    CSV.write(output_path, df)
    log_safe("Wrote $(nrow(df)) rows to $output_path")
    return df
end

function run_single_model(; bank::AbstractString,
                            model_file::AbstractString,
                            output_path::AbstractString)
    bank_dir = joinpath(BANK_ROOT, bank)
    isdir(bank_dir) || error("bank directory not found: $bank_dir")

    json_path = joinpath(bank_dir, model_file)
    isfile(json_path) || error("model file not found: $json_path")

    mkpath(dirname(output_path))
    rows = process_model(json_path, String(bank))
    df = rows_to_dataframe(rows, MAX_N_OUTPUT)
    CSV.write(output_path, df)
    log_safe("Wrote $(nrow(df)) rows to $output_path")
    return df
end

function parse_cli_args(argv::Vector{String})
    opts = Dict{String, String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--bank" && i + 1 <= length(argv)
            opts["bank"] = argv[i + 1]; i += 2
        elseif a == "--model-file" && i + 1 <= length(argv)
            opts["model-file"] = argv[i + 1]; i += 2
        elseif a == "--output" && i + 1 <= length(argv)
            opts["output"] = argv[i + 1]; i += 2
        elseif a == "--only-n" && i + 1 <= length(argv)
            opts["only-n"] = argv[i + 1]; i += 2
        elseif a == "--min-n" && i + 1 <= length(argv)
            opts["min-n"] = argv[i + 1]; i += 2
        elseif a == "--max-n" && i + 1 <= length(argv)
            opts["max-n"] = argv[i + 1]; i += 2
        elseif a == "--help" || a == "-h"
            println("""
            Usage:
              # Full run (all 4 banks, one CSV):
              julia --threads auto enumerate_equilibria.jl

              # Single-model run (one chunk CSV, for cluster array tasks):
              julia --threads 1 enumerate_equilibria.jl \\
                    --bank <bank_dir_basename> \\
                    --model-file <model.json> \\
                    [--output <path.csv>]
            """)
            exit(0)
        else
            error("unknown or malformed argument: $a")
        end
        i
    end
    return opts
end

function default_chunk_output(bank::AbstractString, model_file::AbstractString)
    stem = replace(model_file, r"\.json$" => "")
    fname = string(bank, "__", stem, ".csv")
    return joinpath(@__DIR__, "cluster", "chunks", "enumerate_equilibria", fname)
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_cli_args(ARGS)
    if haskey(opts, "bank") && haskey(opts, "model-file")
        out = get(opts, "output",
                  default_chunk_output(opts["bank"], opts["model-file"]))
        run_single_model(bank = opts["bank"],
                         model_file = opts["model-file"],
                         output_path = out)
    else
        min_n = haskey(opts, "only-n") ? parse(Int, opts["only-n"]) :
                haskey(opts, "min-n")  ? parse(Int, opts["min-n"])  : 1
        max_n = haskey(opts, "only-n") ? parse(Int, opts["only-n"]) :
                haskey(opts, "max-n")  ? parse(Int, opts["max-n"])  : MAX_N
        default_out = haskey(opts, "only-n") ?
            joinpath(@__DIR__, "figures", "all_equilibria_n$(opts["only-n"]).csv") :
            joinpath(@__DIR__, "figures", "all_equilibria.csv")
        output_path = get(opts, "output", default_out)
        run_enumeration(BANKS; output_path = output_path, min_n = min_n, max_n = max_n)
    end
end
