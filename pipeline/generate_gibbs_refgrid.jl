#!/usr/bin/env julia
# Generate a systems bank from reference_grid_prob_gt_0p9.csv, iterating over
# every parameterization row explicitly and producing reps_per_param stable
# replicates per (row × n).  Mirrors systems_refgrid_bank/ structure so that
# generate_gibbs_grid.jl can consume it directly.
#
# Usage:
#   julia --startup-file=no new_code/generate_gibbs_refgrid.jl \
#         <ref_grid_csv> <n_values> [reps_per_param] [max_attempts_mult] [seed]
#
# Arguments:
#   ref_grid_csv        Path to reference_grid_prob_gt_0p9.csv
#   n_values            Comma-separated or range, e.g. "3,4,5,6,7" or "3:7"
#   reps_per_param      Stable replicates per (parameterization × n) (default 100)
#   max_attempts_mult   max_attempts = mult × reps_per_param (default 20)
#   seed                Base RNG seed (default 12345)
#
# Output:
#   new_code/model_runs/gibbs_refgrid_<tag>/{Q1,Q2,Q3}/param_XXXX/nN/system_repXXX.json

using Random
using LinearAlgebra
using JSON3

include(joinpath(@__DIR__, "model_store_utils.jl"))

# ─── argument parsing ────────────────────────────────────────────────────────

function parse_n_values(s::AbstractString)
    t = strip(s)
    if occursin(":", t)
        parts = split(t, ":")
        a = parse(Int, strip(parts[1]))
        b = parse(Int, strip(parts[end]))
        step = length(parts) == 3 ? parse(Int, strip(parts[2])) : 1
        return collect(a:step:b)
    end
    return [parse(Int, strip(x)) for x in split(t, ",") if !isempty(strip(x))]
end

function parse_args(args::Vector{String})
    length(args) >= 2 || error("""
    Usage:
      julia --startup-file=no new_code/generate_gibbs_refgrid.jl \\
            <ref_grid_csv> <n_values> [reps_per_param] [max_attempts_mult] [seed]
    """)
    ref_grid_csv     = args[1]
    n_values         = parse_n_values(args[2])
    reps_per_param   = length(args) >= 3 ? parse(Int, args[3]) : 100
    max_att_mult     = length(args) >= 4 ? parse(Int, args[4]) : 40
    seed             = length(args) >= 5 ? parse(Int, args[5]) : 12345
    isfile(ref_grid_csv)   || error("CSV not found: $ref_grid_csv")
    isempty(n_values)      && error("n_values cannot be empty")
    reps_per_param >= 1    || error("reps_per_param must be >= 1")
    max_att_mult   >= 1    || error("max_attempts_mult must be >= 1")
    return ref_grid_csv, n_values, reps_per_param, max_att_mult, seed
end

# ─── reference grid loading ──────────────────────────────────────────────────

struct RefGridRow
    param_id      :: Int
    regime        :: String
    mu_A          :: Float64
    sigma_A       :: Float64
    target_mean   :: Float64
    target_sd     :: Float64
    corr_strength :: Float64
    mode          :: String
    prob_stable   :: Float64
end

function load_reference_grid(filepath::AbstractString)
    lines = readlines(filepath)
    header = split(lines[1], ',')
    col(name) = findfirst(==(name), header)

    c_mu_A  = col("mu_A");        c_mu_A  !== nothing || error("Missing: mu_A")
    c_sig_A = col("sigma_A");     c_sig_A !== nothing || error("Missing: sigma_A")
    c_tmean = col("target_mean"); c_tmean !== nothing || error("Missing: target_mean")
    c_tsd   = col("target_sd");   c_tsd   !== nothing || error("Missing: target_sd")
    c_corr  = col("corr_strength"); c_corr !== nothing || error("Missing: corr_strength")
    c_mode  = col("mode");        c_mode  !== nothing || error("Missing: mode")
    c_qreg  = col("Q_regime");    c_qreg  !== nothing || error("Missing: Q_regime")
    c_prob  = col("prob_stable")

    rows = RefGridRow[]
    for (idx, line) in enumerate(lines[2:end])
        isempty(strip(line)) && continue
        f = split(line, ',')
        regime = strip(f[c_qreg])
        isempty(regime) && continue
        mu_A    = parse(Float64, strip(f[c_mu_A]))
        sigma_A = parse(Float64, strip(f[c_sig_A]))
        tmean   = parse(Float64, strip(f[c_tmean]))
        tsd     = parse(Float64, strip(f[c_tsd]))
        corr    = parse(Float64, strip(f[c_corr]))
        mode    = strip(f[c_mode])
        prob    = c_prob !== nothing ? parse(Float64, strip(f[c_prob])) : NaN
        push!(rows, RefGridRow(idx, regime, mu_A, sigma_A, tmean, tsd, corr, mode, prob))
    end
    return rows
end

# ─── system generation (matching generate_gibbs_scaled.jl) ──────────────────

function sample_A(n::Int, mu_A::Float64, sigma_A::Float64; rng::AbstractRNG)
    A = randn(rng, n, n) .* sigma_A .+ mu_A
    for i in 1:n; A[i, i] = 1.0; end
    return A
end

function dirichlet_ones(k::Int; rng::AbstractRNG)
    w = [-log(rand(rng)) for _ in 1:k]
    w ./= sum(w)
    return w
end

function distinct_pairs_excl_i(n::Int, i::Int)
    pairs = Tuple{Int,Int}[]
    for j in 1:n-1, k in j+1:n
        (j == i || k == i) && continue
        push!(pairs, (j, k))
    end
    return pairs
end

# Q1/Q2: positive constrained HOI tensor (B in B_ijk convention, receiver=i)
function build_B_q12(n::Int, A::AbstractMatrix{Float64},
                     R::AbstractVector{Float64}, x_star::AbstractVector{Float64};
                     rng::AbstractRNG)
    B_ijk = zeros(Float64, n, n, n)
    target = R - A * x_star
    for i in 1:n
        pairs = distinct_pairs_excl_i(n, i)
        isempty(pairs) && continue
        w = dirichlet_ones(length(pairs); rng=rng)
        for (idx, (j, k)) in enumerate(pairs)
            denom = 2.0 * x_star[j] * x_star[k]
            denom <= 0.0 && return nothing
            coeff = w[idx] * target[i] / denom
            B_ijk[i, j, k] = coeff
            B_ijk[i, k, j] = coeff
        end
    end
    return B_ijk
end

# Q3: correlated HOI tensor with equilibrium constraint, Y ~ Uniform(0, 1/S)
function build_B_q3(n::Int, A::AbstractMatrix{Float64}, corr::Float64,
                    x_star::AbstractVector{Float64}, R::AbstractVector{Float64};
                    rng::AbstractRNG)
    B_ijk = zeros(Float64, n, n, n)
    n_others = n - 2
    n_others <= 0 && return B_ijk

    Y = rand(rng, Float64, n, n) ./ n   # Uniform(0, 1/S)
    for i in 1:n; Y[i, i] = 0.0; end

    if corr < 1.0
        sum_x = sum(x_star)
        for i in 1:n
            off_Ax_i = sum(A[i, j] * x_star[j] for j in 1:n if j != i)
            target_i  = (R[i] - x_star[i]) / (1.0 - corr) - off_Ax_i
            cur_Y_i   = sum(Y[i, j] * x_star[j] for j in 1:n if j != i)
            correction = (target_i - cur_Y_i) / (sum_x - x_star[i])
            for j in 1:n; j == i && continue; Y[i, j] += correction; end
        end
    end

    B_tilde = -corr .* A .+ (1.0 - corr) .* Y
    for i in 1:n; B_tilde[i, i] = 0.0; end

    w = 1.0 / n_others
    for i in 1:n, j in 1:n
        j == i && continue
        for k in 1:n
            (k == i || k == j) && continue
            coeff = B_tilde[i, j] * w / (2.0 * x_star[k])
            B_ijk[i, j, k] += coeff
            B_ijk[i, k, j] += coeff
        end
    end
    return B_ijk
end

function tensor_ijk_to_jki(B_ijk::Array{Float64,3})
    n = size(B_ijk, 1)
    B_jki = Array{Float64,3}(undef, n, n, n)
    for i in 1:n, j in 1:n, k in 1:n; B_jki[j, k, i] = B_ijk[i, j, k]; end
    return B_jki
end

function jacobian_F_gibbs(A, B_jki, x)
    n = length(x)
    JF = Matrix{Float64}(undef, n, n)
    for i in 1:n
        Bi = @view B_jki[:, :, i]
        vL = Bi' * x; vR = Bi * x
        for m in 1:n; JF[i, m] = -A[i, m] - vL[m] - vR[m]; end
    end
    return JF
end

function lambda_max_at_xstar(A, B_jki, x)
    return maximum(real.(eigvals(Diagonal(x) * jacobian_F_gibbs(A, B_jki, x))))
end

function equilibrium_residual(R, A, B_jki, x)
    n = length(x)
    Ax = A * x
    max_res = 0.0
    for i in 1:n
        s = sum(B_jki[j, k, i] * x[j] * x[k] for j in 1:n, k in 1:n)
        max_res = max(max_res, abs(R[i] - Ax[i] - s))
    end
    return max_res
end

# ─── JSON helpers ────────────────────────────────────────────────────────────

matrix_to_nested(M::AbstractMatrix{<:Real}) =
    [collect(@view M[i, :]) for i in 1:size(M, 1)]

tensor3_to_nested(T::Array{Float64,3}) =
    [[[T[i, j, k] for k in 1:size(T, 3)] for j in 1:size(T, 2)] for i in 1:size(T, 1)]

# ─── deterministic per-attempt seed (mirrors R's attempt_seed) ──────────────

derive_seed(base::Int, param_id::Int, n::Int, attempt::Int) =
    base + param_id * 1_000_000 + n * 10_000 + attempt

# ─── main ────────────────────────────────────────────────────────────────────

function main()
    ref_grid_csv, n_values, reps_per_param, max_att_mult, seed_base = parse_args(ARGS)
    max_attempts = max_att_mult * reps_per_param

    rows = load_reference_grid(ref_grid_csv)
    println("generate_gibbs_refgrid")
    println("  ref_grid_csv:   $ref_grid_csv")
    println("  parameterizations: $(length(rows))")
    println("  n_values:       $n_values")
    println("  reps_per_param: $reps_per_param")
    println("  max_attempts:   $max_attempts")
    println("  seed_base:      $seed_base")

    n_str   = length(n_values) == 1 ? "n$(n_values[1])" : "n$(minimum(n_values))to$(maximum(n_values))"
    tag     = "$(n_str)_$(reps_per_param)reps_seed$(seed_base)"
    out_root = joinpath(@__DIR__, "model_runs", "gibbs_refgrid_$(tag)")
    for regime in ("Q1", "Q2", "Q3")
        mkpath(joinpath(out_root, regime))
    end
    println("  out_root:       $out_root\n")

    total_written = 0
    total_cells   = length(rows) * length(n_values)
    cell_idx      = 0

    for row in rows
        param_id = row.param_id
        regime   = row.regime
        is_q3    = row.mode == "correlated_btilde" || row.corr_strength > 0.0

        for n in n_values
            cell_idx += 1
            x_star = ones(Float64, n)
            R      = ones(Float64, n)

            out_dir = joinpath(out_root, regime,
                               "param_$(lpad(param_id, 4, '0'))",
                               "n$(n)")
            mkpath(out_dir)

            stable_count = 0
            attempt      = 0

            print("  [$(cell_idx)/$(total_cells)] $(regime) param=$(lpad(param_id,4,'0')) n=$(n): ")

            while stable_count < reps_per_param && attempt < max_attempts
                attempt += 1
                rng = MersenneTwister(derive_seed(seed_base, param_id, n, attempt))

                A = sample_A(n, row.mu_A, row.sigma_A; rng=rng)

                B_ijk = if is_q3
                    build_B_q3(n, A, row.corr_strength, x_star, R; rng=rng)
                else
                    build_B_q12(n, A, R, x_star; rng=rng)
                end

                (B_ijk === nothing || any(!isfinite, B_ijk)) && continue

                B_jki = tensor_ijk_to_jki(B_ijk)

                equilibrium_residual(R, A, B_jki, x_star) >= 1e-8 && continue
                lambda_max_at_xstar(A, B_jki, x_star) >= 0.0       && continue

                stable_count += 1
                out_path = joinpath(out_dir, "system_rep$(lpad(stable_count, 3, '0')).json")

                payload = Dict{String,Any}(
                    "metadata" => Dict{String,Any}(
                        "regime"             => regime,
                        "n"                  => n,
                        "parameterization_id" => param_id,
                        "replicate_id"       => stable_count,
                        "attempt_id"         => attempt,
                        "mu_A"               => row.mu_A,
                        "sigma_A"            => row.sigma_A,
                        "corr_strength"      => row.corr_strength,
                        "mode"               => row.mode,
                        "prob_stable_ref"    => row.prob_stable,
                        "source"             => "generate_gibbs_refgrid.jl",
                    ),
                    "parameters" => Dict{String,Any}(
                        "R"      => collect(R),
                        "A"      => matrix_to_nested(A),
                        "B"      => tensor3_to_nested(B_ijk),   # B_ijk convention
                        "x_star" => collect(x_star),
                    ),
                    "stability_check" => Dict{String,Any}(
                        "max_real_eigenvalue" => lambda_max_at_xstar(A, B_jki, x_star),
                    ),
                )

                safe_write_json(out_path, payload)
                total_written += 1
            end

            println("$(stable_count)/$(reps_per_param) stable ($(attempt) attempts)")
            if stable_count < reps_per_param
                @warn "Only $stable_count/$reps_per_param systems generated for " *
                      "$(regime) param=$param_id n=$n (max_attempts=$max_attempts reached)"
            end
        end
    end

    println("\nDone. Wrote $total_written systems to: $out_root")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
