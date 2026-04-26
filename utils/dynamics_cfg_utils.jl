# ODE dynamics configuration helpers shared by post_boundary_dynamics.jl
# and backtrack_perturbation.jl.
#
# Constants (ODE_TSPAN_END, ODE_RELTOL, ODE_ABSTOL, SNAP_*) are defined in
# pipeline_config.jl, which must be included before this file.
# Requires: per_capita_growth (from math_utils.jl)

# Build the sequential (equilibrium-snapping) config dict.
# Snapping thresholds are shared across stages.
function build_seq_cfg()
    return Dict{String,Any}(
        "tol_pos"        => SNAP_TOL_POS,
        "tol_neg"        => SNAP_TOL_NEG,
        "imag_tol"       => SNAP_IMAG_TOL,
        "lambda_tol"     => SNAP_LAMBDA_TOL,
        "nudge_abs"      => POST_NUDGE_ABS,
        "nudge_rel"      => POST_NUDGE_REL,
        "snap_tie_tol"   => SNAP_TIE_TOL,
        "snap_dist_abs"  => SNAP_DIST_ABS,
        "snap_dist_rel"  => SNAP_DIST_REL,
        "snap_sep_ratio" => SNAP_SEP_RATIO,
        "snap_F_abs"     => SNAP_F_ABS,
        "snap_F_rel"     => SNAP_F_REL,
    )
end

# Canonical snap_confident: assess whether x_end is close enough to a stable
# equilibrium to be confidently snapped.  x_end must already be clamped by
# the caller if needed (backtrack clamps before calling; post_dynamics uses
# the raw ODE endpoint).
# A_eff and B_eff are pre-scaled by the caller via prescale().
function snap_confident(x_end::AbstractVector{<:Real},
                        equilibria::Vector{NamedTuple},
                        A_eff::AbstractMatrix{<:Real},
                        B_eff::Array{<:Real,3},
                        r_eff::AbstractVector{<:Real},
                        seq::Dict{String,Any})
    isempty(equilibria) && return (ok=false, reason=:no_equilibria, d1=Inf, d2=Inf, relF=Inf)
    dists = [norm(x_end .- eq.x) for eq in equilibria]
    idx = argmin(dists)
    d1 = dists[idx]
    d2 = Inf
    if length(dists) > 1
        tmp = copy(dists)
        tmp[idx] = Inf
        d2 = minimum(tmp)
    end

    dist_thresh = seq["snap_dist_abs"] + seq["snap_dist_rel"] * norm(x_end)
    if d1 > dist_thresh
        return (ok=false, reason=:too_far, d1=d1, d2=d2, relF=Inf)
    end
    if isfinite(d2) && d1 > seq["snap_sep_ratio"] * d2
        return (ok=false, reason=:ambiguous, d1=d1, d2=d2, relF=Inf)
    end

    Fk = per_capita_growth(A_eff, B_eff, r_eff, x_end)
    active = abs.(x_end) .> seq["tol_pos"]
    relF = norm(max.(Fk[active], 0.0))
    F_thresh = seq["snap_F_abs"] + seq["snap_F_rel"] * norm(x_end[active])
    if relF > F_thresh
        return (ok=false, reason=:residual, d1=d1, d2=d2, relF=relF)
    end

    return (ok=true, reason=:ok, d1=d1, d2=d2, relF=relF)
end
