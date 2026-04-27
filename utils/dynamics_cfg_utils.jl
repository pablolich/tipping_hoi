# ODE dynamics configuration helpers shared by post_boundary_dynamics.jl
# and backtrack_perturbation.jl.
#
# Constants (ODE_TSPAN_END, ODE_RELTOL, ODE_ABSTOL, SNAP_TOL_POS, SNAP_TOL_NEG,
# POST_NUDGE_ABS, POST_NUDGE_REL) are defined in pipeline_config.jl, which must
# be included before this file.

# Build the ODE-snap config dict consumed by integrate_and_snap.
function build_seq_cfg()
    return Dict{String,Any}(
        "tol_pos"   => SNAP_TOL_POS,
        "tol_neg"   => SNAP_TOL_NEG,
        "nudge_abs" => POST_NUDGE_ABS,
        "nudge_rel" => POST_NUDGE_REL,
    )
end
