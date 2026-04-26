# GLV+HOI system builders shared across boundary_scan, post_boundary_dynamics,
# and backtrack_perturbation.

include(joinpath(@__DIR__, "..", "other_models", "lever_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "karatayev_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "aguade_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "mougi_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "stouffer_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "marsland_model.jl"))

"""
Single point of truth for the mathematical equivalence between model types.
Standard: A_eff = (1-alpha)*A,  B_eff = alpha*B
Gibbs:    A_eff = -A,           B_eff = -B
"""
function prescale(A::AbstractMatrix{<:Real},
                  B::Array{<:Real,3},
                  alpha::Real,
                  is_gibbs::Bool)
    if is_gibbs
        return -A, -B
    else
        return (1 - alpha) .* A, alpha .* B
    end
end

# Build the HomotopyContinuation System for a GLV+HOI model with pre-scaled matrices.
# F_i(x) = (r0[i] + dr[i]) + (A_eff*x)[i] + (B_eff*x*x)[i]
# Parameters are dr[1:n] only.  A_eff and B_eff are baked in as constants.
# Returns (System, x) — x is the variable vector (useful for backtrack).
function build_system(r0::AbstractVector,
                      A_eff::AbstractMatrix{<:Real},
                      B_eff::Array{<:Real,3})
    n = length(r0)
    @var x[1:n] dr[1:n]
    eqs = Vector{Expression}(undef, n)
    @inbounds for i in 1:n
        lin  = sum(A_eff[i, j] * x[j] for j in 1:n)
        diag = sum(B_eff[j, j, i] * x[j]^2 for j in 1:n)
        offd = n >= 2 ? sum((B_eff[j, k, i] + B_eff[k, j, i]) * x[j] * x[k]
                            for j in 1:n for k in j+1:n) : 0
        eqs[i] = (r0[i] + dr[i]) + lin + diag + offd
    end
    return System(eqs; variables=x, parameters=dr), x
end

# Build the in-place ODE right-hand side for a GLV+HOI model with pre-scaled matrices.
# dx/dt = x[i] * (r_eff[i] + (A_eff*x)[i] + (B_eff*x*x)[i])
function make_unified_rhs(A_eff::AbstractMatrix{<:Real},
                          B_eff::Array{<:Real,3},
                          r_eff::AbstractVector{<:Real})
    n = length(r_eff)
    Bi_list = [Matrix{Float64}(B_eff[:, :, i]) for i in 1:n]
    tmp = Vector{Float64}(undef, n)
    function f!(dx, x, p, t)
        mul!(dx, A_eff, x)
        @inbounds for i in 1:n
            mul!(tmp, Bi_list[i], x)
            quad_i = dot(x, tmp)
            dx[i] = x[i] * (r_eff[i] + dx[i] + quad_i)
        end
        return nothing
    end
    return f!
end

# ------------------------------ HC system builders ----------------------------

"""
Dispatcher: build all model-specific HC context from a model dict.
Returns a NamedTuple with fields:
  n, n_dirs, alpha_grid, x0, baseline_r, U, make_workspace, linear_fallback
"""
function build_hc_system(model::Dict)
    mode = get(model, "dynamics_mode", "standard")
    if mode == "unique_equilibrium" || mode == "all_negative"
        return _build_hc_system_unique_equilibrium(model)
    elseif mode == "standard" || mode == "balanced" ||
       mode == "balanced_stable" || mode == "constrained_r"
        # These banks differ only in how A/B/r are sampled; boundary/post/
        # backtrack still use the same canonical GLV+HOI equations.
        return _build_hc_system_standard(model)
    elseif mode == "elegant"
        return _build_hc_system_elegant(model)
    elseif mode == "gibbs"
        return _build_hc_system_gibbs(model)
    elseif mode == "lever"
        return _build_hc_system_lever(model)
    elseif mode == "karatayev"
        return _build_hc_system_karatayev(model)
    elseif mode == "aguade"
        return _build_hc_system_aguade(model)
    elseif mode == "mougi"
        return _build_hc_system_mougi(model)
    elseif mode == "stouffer"
        return _build_hc_system_stouffer(model)
    elseif mode == "marsland"
        return _build_hc_system_marsland(model)
    else
        error("Unknown dynamics_mode: $mode")
    end
end

function _build_hc_system_standard(model)
    n          = Int(model["n"])
    A          = nested_to_matrix(model["A"])
    B          = nested_to_tensor3(model["B"])
    U          = nested_to_matrix(model["U"])
    baseline_r = Float64.(model["r"])
    x0         = haskey(model, "x_star") ? Float64.(model["x_star"]) : ones(n)
    A_fac      = lu(A)
    x_base_lin = -(A_fac \ baseline_r)

    make_workspace = function(alpha::Float64)
        abs(alpha) <= SCAN_LINEAR_ALPHA_TOL && return nothing
        A_eff, B_eff = prescale(A, B, alpha, false)
        syst, _ = build_system(baseline_r, A_eff, B_eff)
        return ScanWorkspace(syst, n)
    end

    alpha_grid = haskey(model, "alpha_grid") ?
        collect(Float64, model["alpha_grid"]) :
        collect(Float64, SCAN_ALPHA_GRID)

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=alpha_grid,
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=(A=A, A_fac=A_fac, x_base_linear=x_base_lin),
    )
end

function _build_hc_system_elegant(model)
    n          = Int(model["n"])
    A          = nested_to_matrix(model["A"])
    B          = nested_to_tensor3(model["B"])
    U          = nested_to_matrix(model["U"])
    baseline_r = Float64.(model["r"])
    x0         = ones(n)
    A_fac      = lu(A)
    x_base_lin = -(A_fac \ baseline_r)

    make_workspace = function(alpha::Float64)
        abs(alpha) <= SCAN_LINEAR_ALPHA_TOL && return nothing
        A_eff, B_eff = prescale(A, B, alpha, false)
        syst, _ = build_system(baseline_r, A_eff, B_eff)
        return ScanWorkspace(syst, n)
    end

    alpha_grid = haskey(model, "alpha_grid") ?
        collect(Float64, model["alpha_grid"]) :
        collect(Float64, SCAN_ALPHA_GRID)

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=alpha_grid,
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=(A=A, A_fac=A_fac, x_base_linear=x_base_lin),
    )
end

function _build_hc_system_gibbs(model)
    n          = Int(model["n"])
    A          = nested_to_matrix(model["A"])
    B          = nested_to_tensor3(model["B"])
    U          = nested_to_matrix(model["U"])
    baseline_r = Float64.(model["r"])
    x0         = haskey(model, "x_star") ? Float64.(model["x_star"]) : ones(n)
    alpha_eff  = Float64(model["alpha_eff"])

    make_workspace = function(alpha::Float64)
        A_eff, B_eff = prescale(A, B, alpha, true)
        syst, _ = build_system(baseline_r, A_eff, B_eff)
        return ScanWorkspace(syst, n)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
    )
end

function _build_hc_system_lever(model)
    p          = lever_params_from_payload(model)
    n          = p.Sp + p.Sa
    x0         = Float64.(model["x_star"])
    alpha_eff  = Float64(model["alpha_eff"])
    baseline_r = Float64.(model["r"])
    U          = nested_to_matrix(model["U"])

    make_workspace = function(_::Float64)
        syst, _ = build_lever_cleared_system(p)
        return ScanWorkspace(syst, n)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
    )
end

function _build_hc_system_karatayev(model)
    p          = karatayev_params_from_payload(model)
    n          = karatayev_n_species(p)
    x0         = Float64.(model["x_star"])
    alpha_eff  = Float64(model["alpha_eff"])
    baseline_r = Float64.(model["r"])
    U          = nested_to_matrix(model["U"])

    make_workspace = function(_::Float64)
        syst, _ = build_karatayev_cleared_system(p)
        return ScanWorkspace(syst, n)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
    )
end

function _build_hc_system_aguade(model)
    p          = aguade_params_from_payload(model)
    n          = p.n
    x0         = Float64.(model["x_star"])
    alpha_eff  = Float64(model["alpha_eff"])
    baseline_r = Float64.(model["r"])
    U          = nested_to_matrix(model["U"])

    make_workspace = function(_::Float64)
        syst, _ = build_aguade_cleared_system(p)
        return ScanWorkspace(syst, n)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
    )
end

function _build_hc_system_mougi(model)
    p          = mougi_params_from_payload(model)
    n          = p.n
    x0         = Float64.(model["x_star"])
    alpha_eff  = Float64(model["alpha_eff"])
    baseline_r = Float64.(model["r"])
    U          = nested_to_matrix(model["U"])

    make_workspace = function(_::Float64)
        syst, _ = build_mougi_cleared_system(p)
        return ScanWorkspace(syst, n)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
    )
end

function _build_hc_system_stouffer(model)
    p          = stouffer_params_from_payload(model)
    n          = stouffer_n_species(p)
    x0         = Float64.(model["x_star"])
    alpha_eff  = Float64(model["alpha_eff"])
    baseline_r = Float64.(model["r"])
    U          = nested_to_matrix(model["U"])

    make_workspace = function(_::Float64)
        syst, _ = build_stouffer_cleared_system(p)
        return ScanWorkspace(syst, n)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
    )
end

# Marsland (2019) MiCRM — polynomialised step-6 system.
# State = N (consumers only, length n_C), after R has been eliminated
# analytically via Cramer's rule.  Perturbation parameters are the
# maintenance costs m_i (also length n_C), so that the boundary scan
# shifts m -> m + dr along random unit rays in consumer-cost space.
# The polynomial equations live in
# `pipeline_marsland_hc/src/hc_system.jl::build_marsland_cleared_system`.
function _build_hc_system_marsland(model)
    p          = marsland_params_from_payload(model)
    n          = marsland_n_species(p)            # = p.n_C
    x0         = Float64.(model["x_star"])        # length n_C
    alpha_eff  = Float64(get(model, "alpha_eff", 0.0))
    baseline_r = Float64.(model["r"])             # length n_C, = p.m
    U          = nested_to_matrix(model["U"])     # n_C × n_dirs

    make_workspace = function(_::Float64)
        syst, _ = build_marsland_cleared_system(p)
        return ScanWorkspace(syst, n)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
    )
end

function _build_hc_system_unique_equilibrium(model)
    n = Int(model["n"])
    A = nested_to_matrix(model["A"])
    B = nested_to_tensor3(model["B"])
    U = nested_to_matrix(model["U"])
    x0 = ones(n)

    baseline_r_fn = (alpha::Float64) -> compute_r_unique_equilibrium(A, B, alpha)

    make_workspace = function(alpha::Float64)
        r_alpha = compute_r_unique_equilibrium(A, B, alpha)
        A_eff, B_eff = prescale(A, B, alpha, false)
        syst, _ = build_system(r_alpha, A_eff, B_eff)
        return ScanWorkspace(syst, n)
    end

    alpha_grid = haskey(model, "alpha_grid") ?
        collect(Float64, model["alpha_grid"]) :
        collect(Float64, SCAN_ALPHA_GRID)

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=alpha_grid,
        x0=x0, baseline_r=baseline_r_fn, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
    )
end
