# hc_lambda_utils.jl — THE one place where the community-matrix eigenvalues are
# formed.  Group-A fix A3 (review-1_responses/stouffer_regeneration_plan.md §3,
# atn_bank_plan.md §3).
#
# Requires: LinearAlgebra, HomotopyContinuation (for evaluate_and_jacobian!).
#
# ---------------------------------------------------------------------------
# WHY THIS FILE EXISTS
#
# `lambda_max_equilibrium_hc!` used to be copy-pasted into three files
# (pipeline/boundary_scan.jl, utils/hc_tracker_utils.jl,
# pipeline/backtrack_perturbation.jl), and all three formed
#
#     diag(x) * J_G
#
# where G is the DENOMINATOR-CLEARED polynomial system that HomotopyContinuation
# tracks, not the per-capita rate the ODE is written in.  For every rational-RHS
# family in this repo the cleared system is
#
#     G_i = P_i * F_i,      P_i > 0
#
# with F the per-capita rate and P_i the product of denominators row i was
# multiplied through by.  Then
#
#     dG_i/dx_j = P_i dF_i/dx_j + F_i dP_i/dx_j
#
# and on the tracked path F == 0 identically (G == 0 and P > 0), so the second
# term vanishes and J_G = diag(P) * J_F exactly.  The ODE Jacobian at an
# equilibrium is diag(x) * J_F, so
#
#     J_ODE = diag(x ./ P) * J_G.
#
# This is an EXACT correction — one extra O(n + links) evaluation per call — not
# a tolerance argument.  What the old code diagonalised, diag(P) * J_ODE, is a
# positive ROW rescale, and a row rescale does not preserve eigenvalue signs: on
# the shipped Stouffer bank max(P)/min(P) had median 2.3 and worst 24x, and 151
# of 220 rays flagged `unstable` had lambda_max < 0 under the true Jacobian.
#
# ---------------------------------------------------------------------------
# THE MECHANISM
#
# `row_scale` is supplied by the model builder (utils/glvhoi_utils.jl) and maps
# x -> Vector of P_i.  `nothing` means identity.
#
# For GLV+HOI — the "standard", "unique_equilibrium", "all_negative", "gibbs"
# and "terry" modes — G IS the per-capita rate, so P == 1 and `nothing` is
# provably a no-op: the code path taken is byte-for-byte the old
# `mul!(jac_comm, Diagonal(x), jac_f)`.  Nothing about those banks moves.
#
# lever, karatayev, aguade, mougi and stouffer ARE rational-RHS families and DO
# own a P.  They are deliberately left at `nothing` — see the comment in
# utils/glvhoi_utils.jl for why (A6's read-only pre-measurement is owed first).

"""
    RowScaleError

Thrown when a model-supplied row scale `P` is not strictly positive and finite
at a point on the tracked path.

DECISION, and it is deliberate: this is a hard error, never a silent fallback to
the identity.  For every family that supplies a P, `P_i` is a product of
denominators that are bounded below by a positive constant ON THE DOMAIN THE
MODEL IS DEFINED ON, so a non-positive or non-finite `P_i` does not mean "the
correction is numerically delicate here" — it means x itself has left that
domain, i.e. the tracked path is broken.  Falling back to `diag(x)` would
substitute a Jacobian that is known to be wrong for one that is merely
unavailable, and would do it invisibly.

WHERE THAT BOUND HOLDS, STATED EXACTLY — it is a parity argument, not a
universal one.  For the ATN,

    Q_i = B0^h + c B0^h x_i + sum_j w_ij x_j^h

so `Q_i >= B0^h > 0` at any finite real x only when h is EVEN (and c = 0, or
x_i >= 0).  The shipped arm is h = 2, which is why the bound is quoted at all.
For ODD h the bound holds only on the non-negative orthant: at h = 1 a consumer
with a single prey has w_ij = 1, so x_prey = -1 gives Q_i = 0.5 - 1 = -0.5 and
this throws.

CONSEQUENCE, LEFT IN PLACE ON PURPOSE.  `find_stability`
(utils/boundary_event_utils.jl) evaluates lambda at refinement iterates whose
components are not guaranteed positive, so an odd-h run can throw on a merely
negative iterate.  The throw is caught by A5's per-model try/catch in
pipeline/boundary_scan.jl — there is no per-RAY catch — so the cost is the whole
model's rays, recorded as status "failed" in scan_manifest.json.  That is
visible rather than silent, and it cannot touch the h = 2 arm that ships or any
of the five shipped eco-model banks (all `row_scale = nothing`, so
`check_row_scale` is never called on them at all).  If an odd-h ATN arm is ever
scanned, this is the thing to fix first: classify the offending RAY (say
`:row_scale_invalid`) rather than failing the model.

The error surfaces through A5's per-model try/catch in pipeline/boundary_scan.jl:
the model is recorded in scan_manifest.json with status "failed" and this
message, and no model JSON is written.  That is the visibility.
"""
struct RowScaleError <: Exception
    index::Int
    value::Float64
    msg::String
end

function Base.showerror(io::IO, e::RowScaleError)
    print(io, "RowScaleError: row scale P[", e.index, "] = ", e.value,
              " is not strictly positive and finite. ", e.msg)
end

# Validate a model-supplied row scale.  Cheap (n comparisons) and unconditional:
# a scale that has gone bad is the whole reason the corrected Jacobian would be
# wrong, so it must not be discovered by reading a NaN eigenvalue later.
@inline function check_row_scale(P::AbstractVector)
    @inbounds for i in eachindex(P)
        p = P[i]
        if !(isfinite(p) && p > 0)
            throw(RowScaleError(i, Float64(real(p)),
                "The denominator-cleared system G = P .* F is only a valid " *
                "stand-in for F where P > 0; a non-positive or non-finite P " *
                "means the tracked path has left the domain, not that the " *
                "correction is delicate. See A3 in utils/hc_lambda_utils.jl."))
        end
    end
    return P
end

"""
    lambda_max_equilibrium_core!(f_eval, jac_f, jac_comm, compiled_system, x, p, row_scale)

Dominant real part of the ODE community matrix at `(x, p)`.

`row_scale === nothing` forms `diag(x) * J_G` — the identity case, bit-identical
to the pre-A3 code.  Otherwise it forms `diag(x ./ P) * J_G` with
`P = row_scale(x)`.

This is the ONLY place in the repo where these eigenvalues are formed; the three
workspace types each keep a one-line method that forwards here.
"""
function lambda_max_equilibrium_core!(f_eval::AbstractVector{Float64},
                                      jac_f::AbstractMatrix{Float64},
                                      jac_comm::AbstractMatrix{Float64},
                                      compiled_system,
                                      x::AbstractVector{<:Real},
                                      p::AbstractVector{<:Real},
                                      row_scale::Union{Nothing,Function})
    evaluate_and_jacobian!(f_eval, jac_f, compiled_system, x, p)
    if row_scale === nothing
        mul!(jac_comm, Diagonal(x), jac_f)
    else
        P = row_scale(x)
        length(P) == length(x) || error(
            "row_scale returned $(length(P)) entries for a state of length $(length(x)).")
        check_row_scale(P)
        mul!(jac_comm, Diagonal(x ./ P), jac_f)
    end
    e = eigvals!(jac_comm)
    return maximum(real, e)
end
