# ode_snap_utils.jl — shared ODE integration + snapping utilities.
# Extracted from post_boundary_dynamics.jl (random_direction, apply_signed_nudge,
# integrate_and_snap) and backtrack_perturbation.jl (restrict_params).
#
# Must be included after: pipeline_config.jl, math_utils.jl, equilibrium_utils.jl,
# glvhoi_utils.jl, dynamics_cfg_utils.jl.
# Caller must provide: using DifferentialEquations, SciMLBase (and Random for randn).

function make_extinction_cb(eps_extinct::Real)
    condition = (u, t, integrator) -> any(ui < eps_extinct for ui in u)
    affect!   = function(integrator)
        @inbounds for i in eachindex(integrator.u)
            integrator.u[i] < eps_extinct && (integrator.u[i] = 0.0)
        end
    end
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

# Per-capita steady-state terminator for GLV/HOI systems.
# Uses F_i = du_i / u_i (which Tsit5 already has in fsallast via get_du) instead
# of |du_i| alone — for a species heading to extinction, |du| shrinks with u and
# a plain-derivative test would fire prematurely.  Species with u <= u_thresh
# are ignored (the extinction callback handles them); termination requires
# |F_i| < f_tol for every still-active species.
function make_percap_terminate_cb(f_tol::Real, u_thresh::Real)
    condition = function(u, t, integrator)
        du = SciMLBase.get_du(integrator)
        @inbounds for i in eachindex(u)
            u[i] <= u_thresh && continue
            abs(du[i] / u[i]) > f_tol && return false
        end
        return true
    end
    return DiscreteCallback(condition, terminate!; save_positions=(false, false))
end

function random_direction(extinct_mask::AbstractVector{Bool})
    v = randn(length(extinct_mask))
    v[extinct_mask] .= 0.0
    nrm = norm(v)
    if nrm == 0.0
        v[.!extinct_mask] .= 1.0
        nrm = norm(v)
    end
    nrm == 0.0 && return nothing
    return v ./ nrm
end

function apply_signed_nudge(x_start::AbstractVector{<:Real},
                            dir::Union{Nothing,AbstractVector{<:Real}},
                            nudge_abs::Real,
                            tol_neg::Real)
    dir === nothing && return (x0=copy(x_start), dir_used=nothing, sign=0)
    x_plus  = x_start .+ nudge_abs .* dir
    x_minus = x_start .- nudge_abs .* dir
    neg_plus  = count(x -> x < -tol_neg, x_plus)
    neg_minus = count(x -> x < -tol_neg, x_minus)
    if neg_minus < neg_plus || (neg_minus == neg_plus && minimum(x_minus) > minimum(x_plus))
        return (x0=x_minus, dir_used=-dir, sign=-1)
    end
    return (x0=x_plus, dir_used=dir, sign=1)
end

function integrate_and_snap(A_eff::AbstractMatrix{<:Real},
                            B_eff::Array{<:Real,3},
                            r0::AbstractVector{<:Real},
                            u::AbstractVector{<:Real},
                            delta_post::Real,
                            x_start::AbstractVector{<:Real},
                            dyn::Dict{String,Any},
                            seq::Dict{String,Any})
    r_eff = r0 .+ delta_post .* u
    f! = make_unified_rhs(A_eff, B_eff, r_eff)

    extinct_mask = x_start .<= dyn["eps_extinct"]
    active_norm  = norm(x_start[.!extinct_mask])
    nudge_mag    = max(seq["nudge_abs"], seq["nudge_rel"] * active_norm)
    dir          = random_direction(extinct_mask)
    nudge        = apply_signed_nudge(x_start, dir, nudge_mag, seq["tol_neg"])
    x0           = nudge.x0

    ext_cb = make_extinction_cb(dyn["eps_extinct"])
    ss_cb  = make_percap_terminate_cb(POST_SS_PERCAP_TOL, POST_SS_U_THRESH)
    cbs    = CallbackSet(ext_cb, ss_cb)
    prob   = ODEProblem(f!, x0, dyn["tspan"])
    sol    = DifferentialEquations.solve(prob, Tsit5(); reltol=dyn["reltol"], abstol=dyn["abstol"],
                                         callback=cbs,
                                         save_everystep=false, save_start=false, dense=false)
    !SciMLBase.successful_retcode(sol) &&
        return (reason=:ode_fail, x_end=nothing, x_snap=nothing, dist=missing, n_equilibria=0)

    x_end   = sol.u[end]
    n       = length(x_end)
    n_alive = count(>(seq["tol_pos"]), x_end)

    # If all species still alive after first pass, run extended integration
    if n_alive == n
        t0, t1  = dyn["tspan"]
        t2      = t1 + (t1 - t0)
        prob2   = ODEProblem(f!, x_end, (t1, t2))
        sol2    = DifferentialEquations.solve(prob2, Tsit5(); reltol=dyn["reltol"], abstol=dyn["abstol"],
                                              callback=cbs,
                                              save_everystep=false, save_start=false, dense=false)
        !SciMLBase.successful_retcode(sol2) &&
            return (reason=:ode_fail, x_end=nothing, x_snap=nothing, dist=missing, n_equilibria=0)
        x_end   = sol2.u[end]
        n_alive = count(>(seq["tol_pos"]), x_end)
    end

    # Still all alive after two passes — no convergence
    n_alive == n &&
        return (reason=:no_extinction, x_end=x_end, x_snap=nothing, dist=missing, n_equilibria=0)

    # At least one species went extinct — clamp and return
    x_snap = copy(x_end)
    for i in eachindex(x_snap)
        x_snap[i] <= seq["tol_pos"] && (x_snap[i] = 0.0)
    end
    dist = norm(x_end .- x_snap)
    return (reason=:ok, x_end=x_end, x_snap=x_snap, dist=dist, n_equilibria=1)
end

function restrict_params(A::AbstractMatrix{<:Real},
                         B::Array{<:Real,3},
                         r0::AbstractVector{<:Real},
                         u::AbstractVector{<:Real},
                         idx::Vector{Int},
                         x_cur::Union{Nothing,AbstractVector}=nothing;
                         normalize_u::Bool=true)
    A2 = A[idx, idx]
    B2 = B[idx, idx, idx]
    r2 = r0[idx]
    u2 = u[idx]
    if normalize_u
        nrm = norm(u2)
        nrm > 0 && (u2 ./= nrm)
    end
    if x_cur === nothing
        return A2, B2, r2, u2
    else
        x2 = x_cur[idx]
        return A2, B2, r2, u2, x2
    end
end
