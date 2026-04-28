#!/usr/bin/env julia
# Augment bank JSONs with `alpha_eff_hull` — the L² nonlinearity fraction
# of the per-capita rate f_nominal over the convex hull of the state-space
# points the boundary scan actually encountered:
#
#     hull = convex_hull({x_star} ∪ {x_preboundary_k : k = 1..n_dirs})
#
# This decouples the measured nonlinearity from the equilibrium manifold's
# curvature (the issue that crippled `alpha_eff_taylor` and the earlier
# ray-only attempt).  We sample points from the hull via uniform Dirichlet
# weights on the vertices, evaluate f_nominal directly, and fit the best
# affine model L(x) = a + B·x.  α_eff_hull = 1 − R² is the fraction of f's
# L² variation that no affine model captures.
#
# Scalar-α banks (gibbs, lever, karatayev, aguade, mougi, stouffer) get a
# scalar `alpha_eff_hull`; GLV+HOI grid banks (standard/balanced/…)
# get an `alpha_eff_hull_grid` of length len(alpha_grid), with f rebuilt per
# α via build_per_capita_rates_for_alpha and hull vertices drawn from the
# per-α scan entry's x_preboundary points.
# Existing `alpha_eff` is never touched.
#
# Usage:
#   julia --startup-file=no postprocess/add_alpha_eff_hull.jl <path> [<path>...] \
#          [--force] [--dry-run] [--M SAMPLES] [--seed N]

using JSON3
using LinearAlgebra
using Random
using Statistics

include(joinpath(@__DIR__, "..", "utils", "alpha_eff_taylor.jl"))

const SCALAR_ALPHA_MODES = ("gibbs", "lever", "karatayev", "aguade", "mougi", "stouffer")
const GLVHOI_GRID_MODES  = ("standard", "unique_equilibrium", "all_negative")

function usage()
    println("""
Usage: julia --startup-file=no postprocess/add_alpha_eff_hull.jl <path> [<path>...] [--force] [--dry-run] [--M SAMPLES] [--seed N]

Writes per-bank fields:
  Scalar-α modes ($(SCALAR_ALPHA_MODES)):
    alpha_eff_hull            (scalar, = 1 − R² of best-affine fit)
    alpha_eff_hull_R2
    alpha_eff_hull_M          (samples used)
    alpha_eff_hull_n_vertices (hull vertex count = 1 + n_dirs_nonzero)
    alpha_eff_hull_seed       (RNG seed used for sampling)
  GLV+HOI grid modes ($(GLVHOI_GRID_MODES)):
    alpha_eff_hull_grid              (vector, one entry per alpha_grid value)
    alpha_eff_hull_R2_grid
    alpha_eff_hull_n_vertices_grid
    alpha_eff_hull_M, _seed, _alpha_dir
""")
end

function parse_cli_args(args::Vector{String})
    force = false
    dry_run = false
    M = 2000
    seed = 20260418
    α_dir = 0.1  # Dirichlet concentration; < 1 spreads samples across hull
    paths = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help"); usage(); exit(0)
        elseif a == "--force"; force = true; i += 1
        elseif a == "--dry-run"; dry_run = true; i += 1
        elseif a == "--M"
            i + 1 <= length(args) || error("--M requires a value")
            M = parse(Int, args[i + 1]); i += 2
        elseif a == "--seed"
            i + 1 <= length(args) || error("--seed requires a value")
            seed = parse(Int, args[i + 1]); i += 2
        elseif a == "--alpha-dir"
            i + 1 <= length(args) || error("--alpha-dir requires a value")
            α_dir = parse(Float64, args[i + 1]); i += 2
        elseif startswith(a, "--"); error("Unknown flag: $a")
        else; push!(paths, a); i += 1
        end
    end
    isempty(paths) && (usage(); error("At least one path is required."))
    return (paths, force, dry_run, M, seed, α_dir)
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

already_processed(bank::AbstractDict) =
    haskey(bank, "alpha_eff_hull") || haskey(bank, "alpha_eff_hull_grid")

# ─── Hull vertex extraction ─────────────────────────────────────────────────

"""
    extract_hull_vertices(bank) -> Matrix{Float64}  (n × n_vertices)

Collect {x_star} ∪ {x_preboundary_k : k}, clamp tiny negatives to zero
(the boundary tracker sometimes slips <0 by numerical residuals), and
drop any degenerate rows (all-NaN etc.).
"""
function extract_hull_vertices(bank::AbstractDict)
    x_star = Float64.(bank["x_star"])
    n = length(x_star)

    vertices = [copy(x_star)]
    scan_results = bank["scan_results"]
    isempty(scan_results) && return reshape(x_star, n, 1)

    directions = scan_results[1]["directions"]
    @inbounds for d in directions
        haskey(d, "x_preboundary") || continue
        v = Float64.(d["x_preboundary"])
        length(v) == n || continue
        all(isfinite, v) || continue
        @. v = max(v, 0.0)  # clamp tracker undershoot into nonneg orthant
        push!(vertices, v)
    end

    V = Matrix{Float64}(undef, n, length(vertices))
    for j in eachindex(vertices)
        V[:, j] .= vertices[j]
    end
    return V
end

"""
    extract_hull_vertices_from_scan(bank, scan_entry) -> Matrix{Float64}

Like `extract_hull_vertices`, but uses the `directions` array of the given
scan_results entry (so each α in alpha_grid gets its own hull built from
the x_preboundary points of that α's scan).
"""
function extract_hull_vertices_from_scan(bank::AbstractDict,
                                         scan_entry::AbstractDict)
    x_star = Float64.(bank["x_star"])
    n = length(x_star)

    vertices = [copy(x_star)]
    directions = get(scan_entry, "directions", nothing)
    directions === nothing && return reshape(x_star, n, 1)

    @inbounds for d in directions
        haskey(d, "x_preboundary") || continue
        v = Float64.(d["x_preboundary"])
        length(v) == n || continue
        all(isfinite, v) || continue
        @. v = max(v, 0.0)
        push!(vertices, v)
    end

    V = Matrix{Float64}(undef, n, length(vertices))
    for j in eachindex(vertices)
        V[:, j] .= vertices[j]
    end
    return V
end

# ─── Hull sampling via uniform Dirichlet weights ────────────────────────────

"""
    sample_hull(V, M, rng) -> Matrix{Float64}  (n × M)

Uniform Dirichlet(1,...,1) weights over the vertex simplex. Each sample is
x = V·w with Σ w = 1, w ≥ 0. This stays inside the convex hull of V
(full-support), though not uniform wrt hull volume — which is fine for an
L² integral with a vertex-weighted measure.
"""
function sample_hull(V::AbstractMatrix{<:Real}, M::Int, rng::AbstractRNG;
                     α_dir::Real = 0.1)
    n, K = size(V)
    samples = Matrix{Float64}(undef, n, M)
    shape = Float64(α_dir)
    @inbounds for j in 1:M
        # Dirichlet(α,...,α) via Gamma(α, 1) shape / normalization.
        # Small α concentrates weight on few vertices → better hull coverage
        # than Dirichlet(1,...,1) which pulls samples to the centroid.
        w = [rand(rng, Gamma_dist(shape)) for _ in 1:K]
        s = sum(w)
        s == 0 && (w .= 1; s = K)
        w ./= s
        for i in 1:n
            acc = 0.0
            for k in 1:K
                acc += V[i, k] * w[k]
            end
            samples[i, j] = acc
        end
    end
    return samples
end

# Marsaglia-Tsang for α ≥ 1; Johnk/Ahrens-Dieter mix for 0 < α < 1.
# Avoids a full Distributions.jl dependency for one call.
struct Gamma_dist
    α::Float64
end

function Base.rand(rng::AbstractRNG, g::Gamma_dist)
    α = g.α
    if α >= 1.0
        # Marsaglia-Tsang
        d = α - 1.0 / 3.0
        c = 1.0 / sqrt(9.0 * d)
        while true
            x = randn(rng)
            v = (1 + c * x)^3
            v <= 0 && continue
            u = rand(rng)
            if u < 1 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1 - v + log(v))
                return d * v
            end
        end
    else
        # Boost: Gamma(α) = Gamma(α+1) · U^(1/α)
        g1 = Gamma_dist(α + 1.0)
        return rand(rng, g1) * rand(rng)^(1.0 / α)
    end
end

# ─── Best affine fit + R² ───────────────────────────────────────────────────

"""
    affine_fit_nonlinearity(X, Y) -> (alpha_nonlinear, R2)

X is (n × M) of sample points; Y is (n_out × M) of f evaluations. Fit
Y ≈ a + B·X by ordinary least squares, return α = 1 − R² where R² is
the centred coefficient of determination over the pooled outputs.
"""
function affine_fit_nonlinearity(X::AbstractMatrix{<:Real},
                                 Y::AbstractMatrix{<:Real})
    n_in,  M1 = size(X)
    n_out, M2 = size(Y)
    M1 == M2 || error("X and Y must have the same number of columns (samples).")
    M = M1

    # Build design matrix (M × (n_in + 1)) with leading intercept column.
    D = Matrix{Float64}(undef, M, n_in + 1)
    @inbounds for j in 1:M
        D[j, 1] = 1.0
        for i in 1:n_in
            D[j, 1 + i] = X[i, j]
        end
    end
    Ymat = permutedims(Y)                                  # M × n_out

    θ = D \ Ymat                                           # (n_in + 1) × n_out
    Yhat = D * θ

    Y_mean = mean(Ymat, dims = 1)
    SS_total = sum(abs2, Ymat .- Y_mean)
    SS_res   = sum(abs2, Ymat .- Yhat)
    R2 = SS_total > 0 ? 1 - SS_res / SS_total : 1.0
    α  = SS_total > 0 ? SS_res / SS_total : 0.0
    return (α, R2, SS_res, SS_total)
end

# ─── Per-bank processing ────────────────────────────────────────────────────

function process_bank!(bank::Dict{String,Any}, M::Int, seed::Int, α_dir::Real)
    mode = String(get(bank, "dynamics_mode", ""))
    isempty(mode) && return (:skipped, "no dynamics_mode")
    if mode in GLVHOI_GRID_MODES
        return process_bank_grid!(bank, M, seed, α_dir)
    end
    mode in SCALAR_ALPHA_MODES || return (:skipped, "mode=$mode (not scalar-α)")

    haskey(bank, "x_star")       || return (:skipped, "no x_star")
    haskey(bank, "scan_results") || return (:skipped, "no scan_results")

    V = extract_hull_vertices(bank)
    n_vertices = size(V, 2)
    n_vertices >= 2 || return (:skipped, "hull has fewer than 2 vertices")

    # Reject degenerate hulls (all vertices coincide / span near-zero volume).
    # Happens for Q3-style banks where the boundary scan could not step
    # (delta_c = 0 on every ray because the baseline is already at a
    # bifurcation).  Nothing meaningful to measure over a single point.
    vertex_spread = maximum(maximum(V, dims = 2) .- minimum(V, dims = 2))
    vertex_spread > 1e-8 || return (:skipped,
        "degenerate hull (vertex spread=$(vertex_spread))")

    f, _, _ = build_per_capita_rates(bank)

    # Sanity: x_star must be an equilibrium of f_nominal.  If it isn't, the
    # "coexistence region" framing doesn't apply and the LS residual becomes
    # numerically unstable (small variance / noise-dominated).
    fx_star_norm = norm(f(Float64.(bank["x_star"])))
    fx_star_norm < 1e-6 || return (:skipped,
        "x_star is not an equilibrium of f_nominal (||f(x_star)||=$(fx_star_norm))")

    rng = MersenneTwister(seed)
    X_int = sample_hull(V, M, rng; α_dir = α_dir)
    # Include the vertices themselves so the regression sees the extreme
    # boundary states, not just interior Dirichlet blends.
    X = hcat(X_int, V)
    M_total = size(X, 2)

    n_out = length(f(X[:, 1]))
    Y = Matrix{Float64}(undef, n_out, M_total)
    @inbounds for j in 1:M_total
        Y[:, j] .= f(@view X[:, j])
    end

    any(isnan, Y) && return (:error, "NaN in f(X) — possibly outside domain of f")

    α, R2, SS_res, SS_total = affine_fit_nonlinearity(X, Y)
    (isfinite(α) && isfinite(R2)) || return (:error, "non-finite fit result")
    # Defensive: clip out-of-range α (should already be in [0,1] by R²
    # construction, but near-singular LS solves can return NaN-like values).
    (0 <= α <= 1) || return (:error, "α_hull outside [0,1] (α=$(α), R²=$(R2)) — likely singular design")

    bank["alpha_eff_hull"]             = α
    bank["alpha_eff_hull_R2"]          = R2
    bank["alpha_eff_hull_M"]           = M
    bank["alpha_eff_hull_M_total"]     = M_total
    bank["alpha_eff_hull_n_vertices"]  = n_vertices
    bank["alpha_eff_hull_seed"]        = seed
    bank["alpha_eff_hull_alpha_dir"]   = α_dir
    return (:updated, nothing)
end

function process_bank_grid!(bank::Dict{String,Any}, M::Int, seed::Int, α_dir::Real)
    mode = String(get(bank, "dynamics_mode", ""))
    haskey(bank, "x_star")       || return (:skipped, "no x_star")
    haskey(bank, "alpha_grid")   || return (:skipped, "no alpha_grid for mode=$mode")
    haskey(bank, "scan_results") || return (:skipped, "no scan_results")

    α_grid = collect(Float64, bank["alpha_grid"])
    scan_results = bank["scan_results"]
    length(scan_results) == length(α_grid) || return (:error,
        "length(scan_results)=$(length(scan_results)) != length(alpha_grid)=$(length(α_grid))")

    G = length(α_grid)
    hull_grid = fill(NaN, G)
    R2_grid   = fill(NaN, G)
    nV_grid   = zeros(Int, G)

    master_rng = MersenneTwister(seed)
    x_star = Float64.(bank["x_star"])

    @inbounds for i in 1:G
        α = α_grid[i]
        scan_entry = scan_results[i]

        # Sanity: scan α must match grid α (within tolerance).
        scan_α = Float64(get(scan_entry, "alpha", α))
        abs(scan_α - α) < 1e-9 || return (:error,
            "scan_results[$i].alpha=$scan_α != alpha_grid[$i]=$α")

        V = extract_hull_vertices_from_scan(bank, scan_entry)
        n_vertices = size(V, 2)
        nV_grid[i] = n_vertices

        # Degenerate hulls (all vertices coincide / < 2 distinct points): leave
        # NaN and move on — meaningful for grid rows where the scan failed at
        # that α, not a hard error for the whole bank.
        n_vertices >= 2 || continue
        vertex_spread = maximum(maximum(V, dims = 2) .- minimum(V, dims = 2))
        vertex_spread > 1e-8 || continue

        f, _, _ = build_per_capita_rates_for_alpha(bank, α)

        # Sanity: x_star must be an equilibrium of f at this α.
        fx_star_norm = norm(f(x_star))
        fx_star_norm < 1e-6 || return (:error,
            "x_star not an equilibrium at α=$α (||f(x_star)||=$fx_star_norm)")

        rng_i = MersenneTwister(rand(master_rng, UInt32))
        X_int = sample_hull(V, M, rng_i; α_dir = α_dir)
        X = hcat(X_int, V)
        M_total = size(X, 2)

        n_out = length(f(@view X[:, 1]))
        Y = Matrix{Float64}(undef, n_out, M_total)
        for j in 1:M_total
            Y[:, j] .= f(@view X[:, j])
        end
        any(isnan, Y) && return (:error, "NaN in f(X) at α=$α (idx=$i)")

        α_hull, R2, _, _ = affine_fit_nonlinearity(X, Y)
        (isfinite(α_hull) && isfinite(R2)) ||
            return (:error, "non-finite fit result at α=$α (idx=$i)")
        (0 <= α_hull <= 1) || return (:error,
            "α_hull outside [0,1] at α=$α (got $α_hull)")

        hull_grid[i] = α_hull
        R2_grid[i]   = R2
    end

    bank["alpha_eff_hull_grid"]            = hull_grid
    bank["alpha_eff_hull_R2_grid"]         = R2_grid
    bank["alpha_eff_hull_n_vertices_grid"] = nV_grid
    bank["alpha_eff_hull_M"]               = M
    bank["alpha_eff_hull_seed"]            = seed
    bank["alpha_eff_hull_alpha_dir"]       = α_dir
    return (:updated, nothing)
end

function main(args::Vector{String})
    paths_arg, force, dry_run, M, seed, α_dir = parse_cli_args(args)
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
            status, reason = process_bank!(bank, M, seed, α_dir)
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
            if haskey(bank, "alpha_eff_hull_grid")
                hg   = bank["alpha_eff_hull_grid"]
                vg   = bank["alpha_eff_hull_n_vertices_grid"]
                finite_vals = Float64[x for x in hg if x isa Real && isfinite(x)]
                n_ok = length(finite_vals)
                if n_ok > 0
                    g_lo = round(minimum(finite_vals), digits = 4)
                    g_hi = round(maximum(finite_vals), digits = 4)
                    rng_str = "[$g_lo, $g_hi]"
                else
                    rng_str = "[—]"
                end
                v_med = length(vg) > 0 ? sort(vg)[div(length(vg) + 1, 2)] : 0
                println("$tag α_hull_grid[$n_ok/$(length(hg))] in $rng_str  V_med=$v_med  $path")
            else
                α = bank["alpha_eff_hull"]
                V = bank["alpha_eff_hull_n_vertices"]
                println("$tag α_hull=$(round(α, digits=4))  V=$V  $path")
            end
        elseif status == :skipped
            skipped += 1
            println("skipped $path ($(reason))")
        else
            errors += 1
            @warn "error processing bank" path reason
        end
    end

    println("summary updated=$updated skipped=$skipped errors=$errors dry_run=$dry_run M=$M alpha_dir=$α_dir")
end

main(copy(ARGS))
