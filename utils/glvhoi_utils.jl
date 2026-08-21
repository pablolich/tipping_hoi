# GLV+HOI system builders shared across boundary_scan, post_boundary_dynamics,
# and backtrack_perturbation.

include(joinpath(@__DIR__, "..", "other_models", "lever_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "karatayev_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "aguade_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "mougi_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "stouffer_model.jl"))
include(joinpath(@__DIR__, "..", "other_models", "atn_model.jl"))

if !@isdefined(NEGATED_AB_MODES)
    include(joinpath(@__DIR__, "dynamics_mode_utils.jl"))
end

"""
Single point of truth for the mathematical equivalence between model types.
Standard:              A_eff = (1-alpha)*A,  B_eff = alpha*B
NEGATED_AB_MODES:      A_eff = -A,           B_eff = -B   (alpha already baked in)

`negate_ab` must come from `negated_ab(dynamics_mode)` — see dynamics_mode_utils.jl.
"""
function prescale(A::AbstractMatrix{<:Real},
                  B::Array{<:Real,3},
                  alpha::Real,
                  negate_ab::Bool)
    if negate_ab
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
  n, n_dirs, alpha_grid, x0, baseline_r, U, make_workspace, linear_fallback,
  row_scale

`row_scale` (A3, see utils/hc_lambda_utils.jl) is the model builder's statement
of the factor `P_i` its HC system multiplies row `i` by, so that lambda_max can
be taken from `diag(x ./ P) * J_G` — the true ODE Jacobian — rather than from
`diag(x) * J_G`.  It is carried on the NamedTuple for uniformity across branches
and handed to the ScanWorkspace by `make_workspace`.

`nothing` means identity.  Every GLV+HOI branch ("standard", "gibbs"/"terry",
"unique_equilibrium"/"all_negative") is `nothing` BY ALGEBRA, not by omission: G
is the per-capita rate itself there, so P == 1 and the A3 change is provably a
no-op on every bank in `data/` that those branches read.

The five rational-RHS eco-model branches are ALSO left at `nothing` — read the
comment on `_build_hc_system_lever` before changing that.
"""
function build_hc_system(model::Dict)
    mode = get(model, "dynamics_mode", "standard")
    if mode == "unique_equilibrium" || mode == "all_negative"
        return _build_hc_system_unique_equilibrium(model)
    elseif mode == "standard"
        return _build_hc_system_standard(model)
    elseif negated_ab(mode)
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
    elseif mode == "atn"
        return _build_hc_system_atn(model)
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
        row_scale=nothing,          # A3: G is the per-capita rate; P == 1
    )
end

function _build_hc_system_gibbs(model)
    n          = Int(model["n"])
    A          = nested_to_matrix(model["A"])
    B          = nested_to_tensor3(model["B"])
    U          = nested_to_matrix(model["U"])
    baseline_r = Float64.(model["r"])
    x0         = haskey(model, "x_star") ? Float64.(model["x_star"]) : ones(n)
    alpha_eff  = alpha_eff_label(model)

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
        row_scale=nothing,          # A3: G is the per-capita rate; P == 1
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
        # A3/A6: this family IS rational-RHS and DOES own a
        # P_i = D_i^[consumer] * prod_{k in pred(i)} D_k, so `nothing` here is
        # WRONG ALGEBRA that is being kept deliberately.  A6 (the read-only
        # pre-measurement of how many scan flags move for lever, karatayev
        # FMI/RMI, mougi and aguade under A1 and A3 — atn_bank_plan.md §13 says
        # it "is owed separately") has not been run.  Leaving row_scale at
        # `nothing` preserves today's measurement of these banks EXACTLY, so the
        # five points already on Fig. 4b do not move underneath a revision that
        # never claimed to re-measure them.  Supplying P here without running A6
        # first would silently change published numbers.  Do not "finish the
        # job" by filling these in: run A6, report the deltas, then change them.
        row_scale=nothing,
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
        # A3/A6: this family IS rational-RHS and DOES own a
        # P_i = D_i^[consumer] * prod_{k in pred(i)} D_k, so `nothing` here is
        # WRONG ALGEBRA that is being kept deliberately.  A6 (the read-only
        # pre-measurement of how many scan flags move for lever, karatayev
        # FMI/RMI, mougi and aguade under A1 and A3 — atn_bank_plan.md §13 says
        # it "is owed separately") has not been run.  Leaving row_scale at
        # `nothing` preserves today's measurement of these banks EXACTLY, so the
        # five points already on Fig. 4b do not move underneath a revision that
        # never claimed to re-measure them.  Supplying P here without running A6
        # first would silently change published numbers.  Do not "finish the
        # job" by filling these in: run A6, report the deltas, then change them.
        row_scale=nothing,
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
        # A3/A6: this family IS rational-RHS and DOES own a
        # P_i = D_i^[consumer] * prod_{k in pred(i)} D_k, so `nothing` here is
        # WRONG ALGEBRA that is being kept deliberately.  A6 (the read-only
        # pre-measurement of how many scan flags move for lever, karatayev
        # FMI/RMI, mougi and aguade under A1 and A3 — atn_bank_plan.md §13 says
        # it "is owed separately") has not been run.  Leaving row_scale at
        # `nothing` preserves today's measurement of these banks EXACTLY, so the
        # five points already on Fig. 4b do not move underneath a revision that
        # never claimed to re-measure them.  Supplying P here without running A6
        # first would silently change published numbers.  Do not "finish the
        # job" by filling these in: run A6, report the deltas, then change them.
        row_scale=nothing,
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
        # A3/A6: this family IS rational-RHS and DOES own a
        # P_i = D_i^[consumer] * prod_{k in pred(i)} D_k, so `nothing` here is
        # WRONG ALGEBRA that is being kept deliberately.  A6 (the read-only
        # pre-measurement of how many scan flags move for lever, karatayev
        # FMI/RMI, mougi and aguade under A1 and A3 — atn_bank_plan.md §13 says
        # it "is owed separately") has not been run.  Leaving row_scale at
        # `nothing` preserves today's measurement of these banks EXACTLY, so the
        # five points already on Fig. 4b do not move underneath a revision that
        # never claimed to re-measure them.  Supplying P here without running A6
        # first would silently change published numbers.  Do not "finish the
        # job" by filling these in: run A6, report the deltas, then change them.
        row_scale=nothing,
    )
end

function _build_hc_system_stouffer(model)
    p          = stouffer_params_from_payload(model)
    n          = stouffer_n_species(p)
    x0         = Float64.(model["x_star"])
    alpha_eff  = Float64(model["alpha_eff"])
    baseline_r = Float64.(model["r"])
    U          = nested_to_matrix(model["U"])

    # A3, OPT-IN PER MODEL.  This family IS rational-RHS and DOES own a
    #   P_i = D_i^[consumer] * prod_{k in pred(i)} D_k,
    # so `nothing` is wrong algebra — but it is the algebra every number
    # currently on Fig. 4b was measured under, and A6 (the read-only
    # pre-measurement of how many flags move for lever, karatayev FMI/RMI, mougi
    # and aguade under A1 and A3) has not been run.  So the correction is keyed
    # on a flag the MODEL carries:
    #
    #   frozen bank  — no `a3_row_scale` key  -> `nothing`, today's measurement,
    #                  bit-for-bit, so the submitted points cannot move
    #                  underneath a revision that never claimed to re-measure them;
    #   corrected bank — `a3_row_scale = true` -> the true ODE Jacobian.
    #
    # The corrected bank has never been measured, so there is no published number
    # to move, and A3 is not optional for it: 20 of 80 pilot models were rejected
    # as `unstable_at_start` under `diag(x) * J_G` and 0 of 80 under
    # `diag(x ./ P) * J_G`, which three independent implementations agree on
    # (`scratch/stouffer_p0/REPORT.md` §12).  At n = 6 the uncorrected test
    # reports a POSITIVE lambda_max on the median model.
    #
    # `stouffer_row_scale` and the cleared system both route through
    # `_stouffer_denominators`, so the two cannot drift apart.
    #
    # Do not "finish the job" by turning this on for the frozen bank: run A6,
    # report the deltas, then change it.
    row_scale = get(model, "a3_row_scale", false) === true ?
        (x -> stouffer_row_scale(p, x)) : nothing

    make_workspace = function(_::Float64)
        syst, _ = build_stouffer_cleared_system(p)
        return ScanWorkspace(syst, n; row_scale=row_scale)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
        row_scale=row_scale,
    )
end

function _build_hc_system_atn(model)
    p          = atn_params_from_payload(model)
    n          = atn_n_species(p)
    x0         = Float64.(model["x_star"])
    alpha_eff  = Float64(model["alpha_eff"])
    baseline_r = Float64.(model["r"])
    U          = nested_to_matrix(model["U"])

    # A3: build_atn_cleared_system multiplies row i through by
    #   P_i = Q_i^[i is a consumer] * prod_{k in pred(i)} Q_k,
    # so G = P .* F exactly and the ODE Jacobian is diag(x ./ P) * J_G.
    # atn_row_scale and the cleared system both route through _atn_denoms, so
    # the two cannot drift apart.  This is the ONE branch that supplies a P: the
    # ATN bank has never been measured, so there is no published number to move.
    row_scale = x -> atn_row_scale(p, x)

    make_workspace = function(_::Float64)
        syst, _ = build_atn_cleared_system(p)
        return ScanWorkspace(syst, n; row_scale=row_scale)
    end

    return (
        n=n, n_dirs=Int(model["n_dirs"]),
        alpha_grid=[alpha_eff],
        x0=x0, baseline_r=baseline_r, U=U,
        make_workspace=make_workspace,
        linear_fallback=nothing,
        row_scale=row_scale,
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
        row_scale=nothing,          # A3: G is the per-capita rate; P == 1
    )
end
