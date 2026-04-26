#!/usr/bin/env julia
# generate_boundary_data.jl — compute all data needed for the boundary figure
# and save to boundary_data/ as .npy arrays + metadata.json.
#
# Usage:  julia --startup-file=no generate_boundary_data.jl [outdir]
#
# Produces the same data that draw_boundaries_minimal.jl computes internally,
# but saves it for plotting in Python (plot_boundaries.py).

using HomotopyContinuation
using Random
using DataFrames
using LinearAlgebra
using JSON3

include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "utils", "hc_tracker_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "boundary_event_utils.jl"))

# ═══════════════════════════════ NPY writer ═════════════════════════════════
# Writes Julia arrays in NumPy .npy v1.0 format (Fortran order).
function write_npy(path::String, arr::AbstractArray{Float64})
    open(path, "w") do io
        # Magic + version 1.0
        write(io, UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00])

        # Shape string  — keep Julia ordering; set fortran_order=True
        s = size(arr)
        shape_str = "(" * join(string.(s), ", ") * (ndims(arr) == 1 ? ",)" : ")")
        header = "{'descr': '<f8', 'fortran_order': True, 'shape': $shape_str, }"

        # Pad so (10 + HEADER_LEN) is a multiple of 64
        needed = length(header) + 1          # +1 for trailing \n
        total  = 10 + needed
        pad    = mod(64 - mod(total, 64), 64)
        header_padded = header * " "^pad * "\n"

        # Header length as little-endian uint16
        hl = length(header_padded)
        write(io, UInt8(hl & 0xFF))
        write(io, UInt8((hl >> 8) & 0xFF))
        write(io, header_padded)

        # Raw Float64 data in column-major (Fortran) order
        write(io, arr)
    end
end

# ═══════════════════════════════ JSON writer ═════════════════════════════════
_json(x::AbstractDict) = "{" * join(["\"$k\": " * _json(v) for (k,v) in x], ", ") * "}"
_json(x::AbstractVector) = "[" * join([_json(v) for v in x], ", ") * "]"
_json(x::AbstractFloat) = isnan(x) || !isfinite(x) ? "null" : string(x)
_json(x::Integer) = string(x)
_json(x::Bool) = x ? "true" : "false"
_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_json(x::Symbol) = _json(string(x))
_json(::Nothing) = "null"

function write_json(path::String, data)
    open(path, "w") do io
        println(io, _json(data))
    end
end

# ═══════════════════════ polynomial loading & evaluation ════════════════════
function load_poly_expr(path::AbstractString)::Expr
    s = read(path, String)
    s = replace(s, "\r\n" => "\n", "\r" => "\n")
    s = replace(s, r"^\ufeff" => "")
    s = replace(s, r"#.*"m => "")
    s = replace(s, '−' => '-', '–' => '-', '—' => '-')
    s = strip(s)
    endswith(s, ';') && (s = s[1:end-1])
    if (idx = findlast(==('='), s)) !== nothing
        s = strip(s[idx+1:end])
    end
    s = replace(s, r"\s+" => "")
    return Meta.parse(s)
end

function build_fun(expr::Expr)
    fexpr = :((r1,r2,a,A11,A12,A21,A22,B111,B112,B122,B211,B212,B222) -> $expr)
    return eval(fexpr)
end

polycall(f, args...) = Base.invokelatest(f, args...)

function value_on_slice_effective(f::Function, Δr1::Real, Δr2::Real;
                                  α::Real, r0::AbstractVector,
                                  A::AbstractMatrix, B::Array{<:Real,3})
    r1  = r0[1] + Δr1
    r2  = r0[2] + Δr2
    A11 = (1-α) * A[1,1];  A12 = (1-α) * A[1,2]
    A21 = (1-α) * A[2,1];  A22 = (1-α) * A[2,2]
    B111 = α * B[1,1,1]
    B112 = α * (B[1,2,1] + B[2,1,1])
    B122 = α * B[2,2,1]
    B211 = α * B[1,1,2]
    B212 = α * (B[1,2,2] + B[2,1,2])
    B222 = α * B[2,2,2]
    return polycall(f, r1, r2, α, A11, A12, A21, A22, B111, B112, B122, B211, B212, B222)
end

# ═══════════════════════ parameter sampling (r0, A, B) ══════════════════════
function build_C(ℓ::Int, m::Int, n::Int)
    D = n + m*n + ℓ*m*n
    C = zeros(Float64, 2n, D)
    for i in 1:n
        C[i, i] = 1.0
        for j in 1:n
            col = n + (j-1)*m + i
            C[i, col] = 1.0
        end
    end
    offsetB = n + m*n
    for i in 1:n
        row = n + i
        C[row, i] = 1.0
        for k in 1:m, j in 1:ℓ
            lin = j + (k-1)*ℓ + (i-1)*ℓ*m
            C[row, offsetB + lin] = 1.0
        end
    end
    return C
end

function hit_and_run_sample(C, L, U, x0, ℓ::Int, m::Int, n::Int; max_tries::Int=100)
    F     = svd(C; full=true)
    tol   = maximum(size(C)) * eps(eltype(C)) * maximum(F.S)
    rankC = count(>(tol), F.S)
    Q     = F.V[:, rankC+1:end]
    D     = length(x0)
    for _ in 1:max_tries
        u = Q * randn(size(Q,2))
        nu = norm(u); nu < 1e-12 && continue
        u ./= nu
        λmin, λmax = -Inf, Inf
        @inbounds for j in 1:D
            abs(u[j]) ≤ 1e-12 && continue
            a = (L[j]-x0[j]) / u[j]; b = (U[j]-x0[j]) / u[j]
            lo, hi = min(a,b), max(a,b)
            λmin = max(λmin, lo); λmax = min(λmax, hi)
        end
        (λmax < λmin || !isfinite(λmin) || !isfinite(λmax)) && continue
        λ = rand() * (λmax - λmin) + λmin
        x = x0 .+ λ .* u
        r =     x[1               : n]
        A = reshape(x[n+1         : n + m*n], m, n)
        B = reshape(x[n + m*n + 1 : end],       ℓ, m, n)
        return r, A, B
    end
    error("Failed to find a valid hit-and-run step.")
end

function generate_parameter_set(ℓ::Int, m::Int, n::Int;
                                lower::Float64=-10.0, upper::Float64=10.0,
                                seed::Union{Nothing,Int}=nothing)
    if seed !== nothing; Random.seed!(seed); end
    D = n + m*n + ℓ*m*n
    C = build_C(ℓ, m, n)
    L = fill(lower, D); U = fill(upper, D)
    x0 = zeros(D)
    return hit_and_run_sample(C, L, U, x0, ℓ, m, n)
end

# ═══════════════════════ HC system building & solvers ═══════════════════════
function build_num_eqs(r::AbstractVector, A::AbstractMatrix,
                       B::AbstractArray{<:Real,3}, x::Vector{Variable})
    n = length(x)
    eqs = Vector{Expression}(undef, n)
    for i in 1:n
        lin  = sum(A[i, j] * x[j] for j in 1:n)
        quad = sum(B[j, k, i] * x[j] * x[k] for j in 1:n, k in 1:n)
        eqs[i] = r[i] + lin + quad
    end
    return eqs
end

function get_parametrized_system(num_eqs::Vector{Expression},
                                 x::Vector{Variable},
                                 Δr::Vector{Variable},
                                 α::Variable)
    @assert length(num_eqs) == length(Δr)
    out = Vector{Expression}(undef, length(num_eqs))
    for i in eachindex(num_eqs)
        f = num_eqs[i]
        exps, coeffs = exponents_coefficients(f, x)
        orders = vec(sum(exps; dims = 1))
        terms = Vector{Expression}(undef, length(coeffs))
        @inbounds for k in eachindex(coeffs)
            mon = prod(x[j]^exps[j,k] for j in 1:length(x))
            c = coeffs[k]
            if orders[k] == 0
                c = c + Δr[i]
            elseif orders[k] == 1
                c = (1 - α) * c
            else
                c = α * c
            end
            terms[k] = c * mon
        end
        out[i] = sum(terms)
    end
    return System(out; variables = x, parameters = vcat(Δr, α))
end

function build_system_from_params(r0::AbstractVector, A::AbstractMatrix, B::Array{<:Real,3})
    n = length(r0)
    @var x[1:n] Δr[1:n] α
    num_f = build_num_eqs(r0, A, B, x)
    syst  = get_parametrized_system(num_f, x, Δr, α)
    return syst, x
end

function solve_at_params(syst::System, Δr::AbstractVector{<:Real}, α::Real)
    p = vcat(Float64.(Δr), Float64(α))
    Ffixed = fix_parameters(syst, p)
    return solve(Ffixed)
end

function pick_positive_equilibrium(syst::System, α::Real; tol::Float64 = 1e-9)
    n = length(variables(syst))
    res = solve_at_params(syst, zeros(n), α)
    sols = HomotopyContinuation.real_solutions(res; tol = tol)
    pos  = filter(s -> all(s .> 0), sols)
    isempty(pos) && error("No positive equilibrium at Δr=0, α=$(α)")
    target = ones(n)
    dists = map(s -> norm(s .- target), pos)
    _, idx = findmin(dists)
    return pos[idx]
end

# ═══════════════════════ ray scanning + bracketing ═════════════════════════
function unit_rays(n::Int)
    θ = range(0, 2π, length = n + 1)[1:end-1]
    U = hcat([ [cos(t); sin(t)] for t in θ ]...)
    return collect(θ), U
end

function scan_boundaries_from_params(syst::System,
                                     initialsol::AbstractVector;
                                     alpha::Float64 = 0.5,
                                     n_perts::Int   = 64,
                                     max_pert_mag::Float64 = 10.0)
    n = length(initialsol)
    θ, U = unit_rays(n_perts)
    ws = ScanWorkspace(syst, n + 1)
    rows = Vector{NamedTuple}()
    for k in 1:n_perts
        u = U[:, k]
        p_start  = vcat(zeros(n), alpha)
        p_target = vcat(max_pert_mag .* u, alpha)
        event, t_end, _ = find_event(p_start, p_target,
                                     collect(Float64.(initialsol)), ws,
                                     ZERO_ABUNDANCE; λ_tol=SCAN_LAMBDA_TOL)
        t_real = real(t_end)
        Δrcrit = (1 - t_real) .* p_target[1:n]
        push!(rows, (ray_id=k, theta=θ[k], u1=u[1], u2=u[2],
                     Δr1_crit=Δrcrit[1], Δr2_crit=Δrcrit[2],
                     scrit=norm(Δrcrit), flag=event))
    end
    return DataFrame(rows), (alpha=alpha, n_perts=n_perts, max_pert_mag=max_pert_mag)
end

# ═══════════════════════ shared ray selection ═══════════════════════════════
function select_shared_ray_id(dfs::Vector{DataFrame})::Int
    ray_ids = sort(unique(vcat([collect(df.ray_id) for df in dfs]...)))
    isempty(ray_ids) && error("No rays available to select a shared direction.")
    length(dfs) >= 5 || error("Need at least five panels.")

    function flags_for(ray_id::Int)
        flags = Symbol[]
        for df in dfs
            idx = findfirst(==(ray_id), df.ray_id)
            idx === nothing && return nothing
            push!(flags, df.flag[idx])
        end
        return flags
    end

    anchor_ok(flags::Vector{Symbol}) = (flags[1] == :negative && flags[end] ∈ (:fold, :unstable))
    target_ok(flags::Vector{Symbol}) =
        (all(flags[i] == :negative for i in 1:3) &&
         all(flags[i] ∈ (:fold, :unstable) for i in (length(flags)-1):length(flags)))

    old_best = nothing
    old_transition = typemax(Int)
    for ray_id in ray_ids
        flags = flags_for(ray_id)
        flags === nothing && continue
        anchor_ok(flags) || continue
        t_complex = findfirst(f -> f ∈ (:fold, :unstable), flags)
        t_complex === nothing && continue
        if t_complex < old_transition
            old_transition = t_complex
            old_best = ray_id
        end
    end

    if old_best !== nothing
        i0 = findfirst(==(old_best), ray_ids)
        i0 !== nothing || error("Internal error: old ray id not found.")
        n_rays = length(ray_ids)
        for k in 1:n_rays
            i = mod1(i0 - k, n_rays)
            rid = ray_ids[i]
            flags = flags_for(rid)
            flags === nothing && continue
            target_ok(flags) && return rid
        end
        for k in 1:n_rays
            i = mod1(i0 - k, n_rays)
            rid = ray_ids[i]
            flags = flags_for(rid)
            flags === nothing && continue
            (all(flags[i] == :negative for i in 1:3) && flags[end] ∈ (:fold, :unstable)) && return rid
        end
        for k in 1:n_rays
            i = mod1(i0 - k, n_rays)
            rid = ray_ids[i]
            flags = flags_for(rid)
            flags === nothing && continue
            anchor_ok(flags) && return rid
        end
    end

    for ray_id in ray_ids
        flags = flags_for(ray_id)
        flags === nothing && continue
        target_ok(flags) && return ray_id
    end
    for ray_id in ray_ids
        flags = flags_for(ray_id)
        flags === nothing && continue
        anchor_ok(flags) && return ray_id
    end
    for ray_id in ray_ids
        flags = flags_for(ray_id)
        flags === nothing && continue
        flags[1] != :success && return ray_id
    end
    return first(ray_ids)
end

# ═══════════════════════ solution tracking (HC) ════════════════════════════
function track_to_point(syst::System,
                        x0::AbstractVector,
                        p0::AbstractVector{<:Real},
                        pt::AbstractVector{<:Real};
                        max_step_ratio::Float64 = 0.3,
                        max_steps::Int = 1_000_000,
                        extended_precision::Bool = false)
    par_dist = norm(pt .- p0)
    tracker = Tracker(
        CoefficientHomotopy(syst; start_coefficients = p0, target_coefficients = pt);
        options = TrackerOptions(max_steps = max_steps)
    )
    init!(tracker, x0, 1.0, 0.0;
          max_initial_step_size = par_dist * max_step_ratio,
          extended_precision = extended_precision)
    while is_tracking(tracker.state.code)
        step!(tracker)
    end
    res = TrackerResult(tracker.homotopy, tracker.state)
    return (res.return_code == :success), res.solution
end

function heatmap_branch_min_via_tracking(syst::System,
                                         base_sol::AbstractVector;
                                         α::Real,
                                         Δr1_range::Tuple{<:Real,<:Real},
                                         Δr2_range::Tuple{<:Real,<:Real},
                                         nx::Int = 121, ny::Int = 121)
    xs = range(Δr1_range[1], Δr1_range[2], length=nx)
    ys = range(Δr2_range[1], Δr2_range[2], length=ny)
    H  = fill(NaN, ny, nx)
    p0 = [0.0, 0.0, float(α)]
    for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        pt = [float(x), float(y), float(α)]
        ok, xsol = track_to_point(syst, base_sol, p0, pt)
        if ok
            xr = real.(xsol)
            H[iy, ix] = min(xr[1], xr[2])
        end
    end
    return collect(xs), collect(ys), H
end

# ════════════════════════════════ main ══════════════════════════════════════
function main(; neg_path::AbstractString = joinpath(@__DIR__, "J1.jl"),
               cx_path::AbstractString  = joinpath(@__DIR__, "J2.jl"),
               alpha_vec::AbstractVector{<:Real} = collect(range(0.001, 0.374, length=5)),
               seed::Int = 52, n::Int = 2, m::Int = 2, ℓ::Int = 2,
               Δr1_range = (-7.1, 7.1), Δr2_range = (-7.1, 7.1),
               res::Int = 601, n_perts::Int = 64, max_pert_mag::Float64 = 50.0,
               nx_hm::Int = 100, ny_hm::Int = 100,
               outdir::AbstractString = joinpath(@__DIR__, "boundary_data"),
               rays_only::Bool = false)

    # In rays_only mode: reuse existing heatmap/contour data; only redo the ray scan.
    existing_meta = nothing
    if rays_only
        meta_path = joinpath(outdir, "metadata.json")
        isfile(meta_path) || error("rays_only=true requires existing metadata.json at $meta_path")
        existing_meta = JSON3.read(read(meta_path, String))
        alpha_vec = collect(Float64.(existing_meta["alpha_vec"]))
        println("rays_only=true — preserving heatmap/contour data in $outdir")
    end

    if !rays_only
        println("Loading boundary polynomials…")
    end
    expr_neg = rays_only ? nothing : load_poly_expr(neg_path)
    expr_cx  = rays_only ? nothing : load_poly_expr(cx_path)
    f_neg = rays_only ? nothing : build_fun(expr_neg)
    f_cx  = rays_only ? nothing : build_fun(expr_cx)

    println("Sampling parameters (seed=$seed)…")
    r0, A, B = generate_parameter_set(ℓ, m, n; seed=seed)
    syst, _  = build_system_from_params(r0, A, B)

    # Contour grid coordinates
    xs = collect(range(Δr1_range[1], Δr1_range[2], length=res))
    ys = collect(range(Δr2_range[1], Δr2_range[2], length=res))

    ncols = length(alpha_vec)
    dfs   = Vector{DataFrame}(undef, ncols)
    base_solutions = Vector{Vector{Float64}}(undef, ncols)
    heatmaps = Vector{NamedTuple}(undef, ncols)
    contour_neg = Vector{Matrix{Float64}}(undef, ncols)
    contour_cx  = Vector{Matrix{Float64}}(undef, ncols)
    global_min = Inf
    global_max = -Inf

    for (j, α) in enumerate(alpha_vec)
        println("  Panel $j/$ncols (α=$α)…")

        # Positive equilibrium
        xpos = pick_positive_equilibrium(syst, α; tol=1e-10)
        base_solutions[j] = Float64.(real.(xpos))

        # Boundary ray scan
        println("    Ray scan ($n_perts rays, max_pert_mag=$max_pert_mag)…")
        df, _ = scan_boundaries_from_params(syst, base_solutions[j];
                                            alpha=Float64(α), n_perts=n_perts,
                                            max_pert_mag=max_pert_mag)
        dfs[j] = df

        if !rays_only
            # Heatmap via tracking
            println("    Heatmap ($(nx_hm)×$(ny_hm))…")
            hxs, hys, H = heatmap_branch_min_via_tracking(syst, base_solutions[j];
                                                           α=α,
                                                           Δr1_range=Δr1_range,
                                                           Δr2_range=Δr2_range,
                                                           nx=nx_hm, ny=ny_hm)
            heatmaps[j] = (hxs=hxs, hys=hys, H=H)

            vals = H[.!isnan.(H)]
            if !isempty(vals)
                global_min = min(global_min, minimum(vals))
                global_max = max(global_max, maximum(vals))
            end

            # Contour grids for J1, J2
            println("    Contour grids ($(res)×$(res))…")
            Zneg = [value_on_slice_effective(f_neg, x, y; α=α, r0=r0, A=A, B=B) for y in ys, x in xs]
            Zcx  = [value_on_slice_effective(f_cx,  x, y; α=α, r0=r0, A=A, B=B) for y in ys, x in xs]
            contour_neg[j] = Zneg
            contour_cx[j]  = Zcx
        end
    end

    if rays_only
        global_min = Float64(existing_meta["clims"][1])
        global_max = Float64(existing_meta["clims"][2])
    end

    # Shared ray selection (same logic as draw_boundaries_minimal.jl).
    # Net offset of −1 shifts the final pick 4 directions clockwise relative to
    # the legacy −5 offset (rays are indexed in increasing θ, which is clockwise
    # on screen because the plot swaps axes to (Δr2, Δr1)).
    shared_ray_id = select_shared_ray_id(dfs)
    ray_ids_all = sort(unique(dfs[1].ray_id))
    i_sel = findfirst(==(shared_ray_id), ray_ids_all)
    if i_sel !== nothing && !isempty(ray_ids_all)
        shared_ray_id = ray_ids_all[mod1(i_sel - 1, length(ray_ids_all))]
    end

    if !isfinite(global_min) || !isfinite(global_max)
        global_min, global_max = -1.0, 1.0
    elseif global_max <= global_min
        δ = max(abs(global_max), 1.0) * 1e-6
        global_min -= δ
        global_max += δ
    end

    # ─── Save to disk ───────────────────────────────────────────────────────
    println("Saving to $outdir …")
    mkpath(outdir)

    # Grid coordinates (only rewritten in full mode)
    if !rays_only
        write_npy(joinpath(outdir, "xs.npy"), xs)
        write_npy(joinpath(outdir, "ys.npy"), ys)
    end

    # Per-panel data
    panels_meta = []
    for j in 1:ncols
        pdir = joinpath(outdir, "panel_$(j-1)")
        mkpath(pdir)
        if !rays_only
            write_npy(joinpath(pdir, "Zneg.npy"), contour_neg[j])
            write_npy(joinpath(pdir, "Zcx.npy"),  contour_cx[j])
            write_npy(joinpath(pdir, "H.npy"),     heatmaps[j].H)
            write_npy(joinpath(pdir, "hxs.npy"),   heatmaps[j].hxs)
            write_npy(joinpath(pdir, "hys.npy"),   heatmaps[j].hys)
        end

        df = dfs[j]
        neg_pts = df[df.flag .== :negative, :]
        cx_pts  = df[df.flag .∈ Ref((:fold, :unstable)), :]

        shared_hit = df[df.ray_id .== shared_ray_id, :]
        sh_x = nrow(shared_hit) > 0 ? shared_hit.Δr2_crit[1] : NaN
        sh_y = nrow(shared_hit) > 0 ? shared_hit.Δr1_crit[1] : NaN
        sh_flag = nrow(shared_hit) > 0 ? string(shared_hit.flag[1]) : "none"

        push!(panels_meta, Dict(
            "alpha"        => alpha_vec[j],
            "neg_x"        => collect(neg_pts.Δr2_crit),
            "neg_y"        => collect(neg_pts.Δr1_crit),
            "cx_x"         => collect(cx_pts.Δr2_crit),
            "cx_y"         => collect(cx_pts.Δr1_crit),
            "shared_x_end" => sh_x,
            "shared_y_end" => sh_y,
            "shared_flag"  => sh_flag,
        ))
    end

    # Preserve original heatmap/contour-grid metadata when in rays_only mode.
    meta_res       = rays_only ? Int(existing_meta["res"])   : res
    meta_nx_hm     = rays_only ? Int(existing_meta["nx_hm"]) : nx_hm
    meta_ny_hm     = rays_only ? Int(existing_meta["ny_hm"]) : ny_hm
    meta_dr1_range = rays_only ? collect(Float64.(existing_meta["dr1_range"])) :
                                  [Δr1_range[1], Δr1_range[2]]
    meta_dr2_range = rays_only ? collect(Float64.(existing_meta["dr2_range"])) :
                                  [Δr2_range[1], Δr2_range[2]]

    metadata = Dict(
        "alpha_vec"   => collect(alpha_vec),
        "clims"       => [global_min, global_max],
        "res"         => meta_res,
        "nx_hm"       => meta_nx_hm,
        "ny_hm"       => meta_ny_hm,
        "dr1_range"   => meta_dr1_range,
        "dr2_range"   => meta_dr2_range,
        "shared_ray_id" => shared_ray_id,
        "panels"      => panels_meta,
    )
    write_json(joinpath(outdir, "metadata.json"), metadata)

    println("Done. Saved $(ncols) panels to $outdir")
    return outdir
end

# ─── CLI entry point ────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    rays_only = "--rays-only" in ARGS
    positional = filter(a -> a != "--rays-only", ARGS)
    outdir = length(positional) >= 1 ? positional[1] : joinpath(@__DIR__, "boundary_data")
    main(; outdir=outdir, rays_only=rays_only)
end
