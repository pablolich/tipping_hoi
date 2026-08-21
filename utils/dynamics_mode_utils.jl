# Single point of truth for which `dynamics_mode` families store (A, B) already
# negated, with α baked in:  A_eff = −A,  B_eff = −B.
#
# This used to be seven independent `mode == "gibbs"` string comparisons spread
# over glvhoi_utils.jl, alpha_eff_taylor.jl, post_boundary_dynamics.jl and
# backtrack_perturbation.jl.  Two of them (`is_gibbs` in the post-boundary and
# backtrack drivers) select between (−A, −B) and ((1−α)A, αB) in `prescale`, so
# a family that misses one of those sites silently integrates a *different*
# system — with α read from a field that means nothing for it.  Adding a family
# here is now the whole change.
#
# Included with an `@isdefined` guard by both glvhoi_utils.jl and
# alpha_eff_taylor.jl, which are separate include roots (the latter deliberately
# avoids pulling in HomotopyContinuation).

const NEGATED_AB_MODES = ("gibbs", "terry", "mougi2025")

negated_ab(mode::AbstractString) = String(mode) in NEGATED_AB_MODES

"""
    alpha_eff_label(model) -> Float64

`alpha_eff` for a NEGATED_AB_MODES bank is a *label*: `prescale` ignores it, and
the only thing that reads it downstream is `alpha_grid`, which lands in
`scan_config`.  Gibbs banks carry a meaningful α there; Terry banks do not, and
store JSON `null` rather than invent a fourth meaning for the legacy field — the
repo already has three coexisting `alpha_eff` definitions (CLAUDE.md), and
inventing a fourth is how that happened.  A missing or null field therefore reads
back as NaN, not as an error; `json_number` writes it out as null again.
"""
function alpha_eff_label(model)
    v = get(model, "alpha_eff", nothing)
    return v === nothing ? NaN : Float64(v)
end
