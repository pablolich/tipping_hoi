# boundary_event_utils.jl — step-constrained event detection for HC boundary scan.
# Included by boundary_scan.jl after ScanWorkspace and lambda_max_equilibrium_hc! are defined.

@inline function parameters_at_t!(p_eval, t, u0, u1)
    tt = real(t)
    omt = 1 - tt
    @inbounds @simd for i in eachindex(p_eval, u0, u1)
        p_eval[i] = tt * u0[i] + omt * u1[i]
    end
    return p_eval
end

# All three refiners below compute Δt = abs(t_previous - t_end), stop when it is
# zero, and then call this unconditionally.  Δt == 0.0 makes 1/Δt infinite and
# `Int(ceil(Inf))` throws InexactError, which propagates out of scan_model and
# costs the whole model: it is recorded as `failed` in the scan manifest and is
# simply absent from the scanned bank, i.e. a hole in the ensemble rather than an
# error anyone reading the output would see.
#
# Returning early is not a papering-over.  When Δt == 0 the caller has ALREADY
# assigned t_end and set keep_tracking = false, so the tracker is never stepped
# again and every option this function would have written is dead.  Skipping the
# write is therefore bit-identical for every model that does not reach the
# branch, and turns the ones that do into the stop the caller already asked for.
@inline function set_refinement_options!(tracker, Δt)
    Δt == 0.0 && return tracker
    tracker.options.max_step_size = Δt / 2
    tracker.options.max_steps = max(1, Int(ceil(1 / Δt)))
    tracker.options.min_step_size = Δt * 1e-48
    return tracker
end

@inline function reset_tracker_options!(tracker)
    tracker.options.max_step_size = Inf
    tracker.options.max_steps = 10000
    tracker.options.min_step_size = 1e-48
    return tracker
end

# Refine the homotopy parameter where x[i] crosses zero by retracking on
# progressively smaller t-intervals until x[i] is within tol of zero, or
# until the tracker can no longer reduce the step.
function find_zero(tracker, x_current, i, t_end, t_previous, tol)
    init!(tracker, x_current, t_end, t_previous)
    keep_tracking = true

    xᵢ = x_current[i]
    if real(xᵢ) < 0
        direction = -1
    else
        direction = 1
    end

    Δt = abs(t_previous - t_end)
    if Δt == 0.0
        t_end = tracker.state.t
        keep_tracking = false
    end
    set_refinement_options!(tracker, Δt)

    while keep_tracking
        t_previous = tracker.state.t
        HomotopyContinuation.step!(tracker)

        Δt = abs(t_previous - tracker.state.t)
        if Δt == 0.0
            t_end = tracker.state.t
            keep_tracking = false
        end

        x_current .= solution(tracker)
        xᵢ = x_current[i]
        if abs(real(xᵢ)) ≤ tol
            t_end = tracker.state.t
            keep_tracking = false
        elseif direction * real(xᵢ) < -tol
            t_end = find_zero(tracker, x_current, i, tracker.state.t, t_previous, tol)
            keep_tracking = false
        end
    end

    return t_end
end

function find_stability(tracker, ws, x_current, u0, u1, t_end, t_previous, λ_tol)
    init!(tracker, x_current, t_end, t_previous)
    keep_tracking = true

    parameters_at_t!(ws.p_eval, t_end, u0, u1)
    λ_shift = lambda_max_equilibrium_hc!(ws, x_current, ws.p_eval) - λ_tol
    direction = λ_shift < 0 ? -1 : 1

    Δt = abs(t_previous - t_end)
    if Δt == 0.0
        t_end = tracker.state.t
        keep_tracking = false
    end
    set_refinement_options!(tracker, Δt)

    while keep_tracking
        t_previous = tracker.state.t
        HomotopyContinuation.step!(tracker)

        Δt = abs(t_previous - tracker.state.t)
        if Δt == 0.0
            t_end = tracker.state.t
            keep_tracking = false
        end

        x_current .= solution(tracker)
        parameters_at_t!(ws.p_eval, tracker.state.t, u0, u1)
        λ_shift = lambda_max_equilibrium_hc!(ws, x_current, ws.p_eval) - λ_tol
        if abs(λ_shift) ≤ λ_tol
            t_end = tracker.state.t
            keep_tracking = false
        elseif direction * λ_shift < -λ_tol
            t_end = find_stability(tracker, ws, x_current, u0, u1, tracker.state.t, t_previous, λ_tol)
            keep_tracking = false
        end
    end

    return t_end
end

function find_invasion(tracker, x_current, invasion_fn, t_end, t_previous, inv_tol)
    init!(tracker, x_current, t_end, t_previous)
    keep_tracking = true

    inv_val = invasion_fn(x_current, t_end)
    direction = inv_val < inv_tol ? -1 : 1

    Δt = abs(t_previous - t_end)
    if Δt == 0.0
        t_end = tracker.state.t
        keep_tracking = false
    end
    set_refinement_options!(tracker, Δt)

    while keep_tracking
        t_previous = tracker.state.t
        HomotopyContinuation.step!(tracker)

        Δt = abs(t_previous - tracker.state.t)
        if Δt == 0.0
            t_end = tracker.state.t
            keep_tracking = false
        end

        x_current .= solution(tracker)
        inv_val = invasion_fn(x_current, tracker.state.t)
        if abs(inv_val - inv_tol) ≤ inv_tol
            t_end = tracker.state.t
            keep_tracking = false
        elseif direction * (inv_val - inv_tol) < -inv_tol
            t_end = find_invasion(tracker, x_current, invasion_fn, tracker.state.t, t_previous, inv_tol)
            keep_tracking = false
        end
    end

    return t_end
end

# The vanished species at a refined zero crossing: every entry at or below the
# zero floor `tol` is set to exactly 0.  A refiner that stopped inside the
# floor leaves |x_i| ≤ tol; one that stopped on a no-progress exit leaves the
# overshoot, x_i < -tol (2.4% of the `negative` rays in the shipped standard
# banks, signed minimum entry typically -1e-6 .. -1e-4).  Both are the species
# whose extinction defines the crossing, and an overshoot left in place is not
# harmless: row i of diag(x)·J then carries a factor x_i < 0, which turns the
# species' own stabilising J_ii into a spurious positive eigenvalue ~ -x_i·J_ii
# of 1e-7 .. 1e-5 — above λ_tol, so `abs(x_i) ≤ tol` alone would have
# reclassified ~9,200 GLV+HOI rays on nothing
# (review-1_responses/scratch/find_event_order/census/).  Entries above tol
# are left untouched.
@inline function zero_vanished!(x::AbstractVector{<:Real}, tol)
    @inbounds for i in eachindex(x)
        x[i] <= tol && (x[i] = 0.0)
    end
    return x
end

function find_event(p_start, p_target, x_start, ws, tol;
                    check_stability::Bool=true, λ_tol=LAMBDA_TOL,
                    check_invasibility::Bool=false,
                    invasion_fn=nothing, invasion_tol::Float64=1e-10)
    start_parameters!(ws.tracker, p_start)
    target_parameters!(ws.tracker, p_target)
    init!(ws.tracker, x_start, 1.0, 0.0)
    x_current = Vector{Float64}(undef, length(x_start))
    λ_previous = -Inf
    inv_previous = -Inf

    if check_stability
        parameters_at_t!(ws.p_eval, ws.tracker.state.t, p_start, p_target)
        λ_previous = lambda_max_equilibrium_hc!(ws, x_start, ws.p_eval)
        if λ_previous ≥ λ_tol
            x_crit = copy(x_start)
            reset_tracker_options!(ws.tracker)
            # A2 (stouffer_regeneration_plan.md §3, atn_bank_plan.md §3): the
            # homotopy has not moved — t is still 1, so delta_c = (1 − Re(t)) ·
            # max_pert = 0.  This says x_start itself is unstable; it says
            # nothing about where a boundary is.  Calling it `:unstable`
            # buried 384 such rays in the shipped Stouffer bank among genuine
            # instability boundaries found at delta_c > 0.
            #
            # STOP CONDITION, not a filterable artifact.  The ATN generator
            # gates stability at generation (atn_bank_plan.md §6 criterion 5:
            # λ_max(J_ODE(x*)) < -1e-9 by complex step), so this symbol must
            # come back EMPTY on the ATN bank.  One hit means the generator
            # accepted an unstable equilibrium — a generator bug — and the run
            # stops (plan §11 G2).  Do not remap it back to `:unstable`
            # downstream; being distinguishable is the entire point.
            return :unstable_at_start, ws.tracker.state.t, x_crit
        end
    end

    if check_invasibility && invasion_fn !== nothing
        inv_previous = invasion_fn(x_start, ws.tracker.state.t)
        if inv_previous > invasion_tol
            x_crit = copy(x_start)
            reset_tracker_options!(ws.tracker)
            return :invasion, ws.tracker.state.t, x_crit
        end
    end

    t_previous = Complex(0.0)
    t_end = Complex(0.0)
    event = :still_tracking
    x_crit = copy(x_start)
    # Set when a crossing's stability test moved the tracker but the onset was
    # not accepted: the crossing state as it was, so the ray is returned exactly
    # as it would have been without the test.
    x_crit_keep = nothing

    keep_tracking = true

    while keep_tracking
        t_previous = ws.tracker.state.t
        HomotopyContinuation.step!(ws.tracker)

        x_current .= real.(ws.tracker.state.x)
        if !is_tracking(ws.tracker.state.code) && !is_success(ws.tracker.state.code)
            t_end = ws.tracker.state.t
            # A1 (stouffer_regeneration_plan.md §3): a tracker that dies before
            # accepting a single step has not found a boundary — it never left
            # x_start.  The homotopy runs t: 1 → 0 and delta_c = (1 − Re(t_end))
            # · max_pert, so `accepted_steps == 0 ⟺ t_end == 1 ⟺ delta_c == 0`
            # exactly; the test is bit-precise and needs no tolerance.  Calling
            # that case `:fold` is what manufactured 3009 phantom Stouffer
            # tipping points at zero perturbation
            # (findings/stouffer_folds_are_unconverged_equilibria.md).
            event = real(t_end) == 1.0 ? :tracker_failure : :fold
            keep_tracking = false
        else
            if any(xᵢ -> abs(xᵢ) ≤ tol, x_current)
                event = :negative
                t_end = ws.tracker.state.t
                keep_tracking = false
            else
                neg_indices = findall(xᵢ -> xᵢ < -tol, x_current)
                if !isempty(neg_indices)
                    event = :negative
                    t_cur = ws.tracker.state.t
                    x_snap = copy(x_current)
                    best_i = neg_indices[1]
                    t_end = find_zero(ws.tracker, copy(x_snap), best_i, t_cur, t_previous, tol)
                    for i in neg_indices[2:end]
                        t_i = find_zero(ws.tracker, copy(x_snap), i, t_cur, t_previous, tol)
                        if real(t_i) > real(t_end)
                            t_end = t_i
                            best_i = i
                        end
                    end
                    # Re-align tracker state with the winner if it wasn't the last candidate evaluated
                    if best_i !== neg_indices[end]
                        find_zero(ws.tracker, copy(x_snap), best_i, t_cur, t_previous, tol)
                    end
                    keep_tracking = false
                end
            end

            # Negativity is tested before stability inside a step, so a loss of
            # stability that happens in the SAME step as a zero crossing was
            # never seen: the ray came back `negative` at the crossing although
            # the equilibrium had stopped being the attractor earlier.  Test the
            # crossing state with the vanished species set to exactly 0
            # (zero_vanished!).  Its community matrix then has a zero row, so
            # its spectrum is {0} plus that of the surviving block; the block
            # being unstable means the full equilibrium was already unstable
            # just before the crossing, and the boundary is that onset —
            # refined between the previous step and the crossing, exactly as a
            # mid-step instability is.
            #
            # The test itself touches only the workspace buffers.  The tracker
            # moves only when the block is unstable, and the ray is relabelled
            # only when find_stability then locates the onset STRICTLY before
            # the crossing, at a state with every species present; otherwise
            # the crossing is returned exactly as it was, so a ray that is not
            # reclassified comes out bit-identical.  The two ways the onset is
            # not located: the crossing state is not on the branch — a refiner
            # that ended on a no-progress exit far past zero (x_i of −1.5, −29
            # were seen on the aguade and karatayev banks), from which init!
            # cannot start — or λ_max sits inside find_stability's stopping band
            # at the crossing itself.  Either way `unstable` at an unchanged
            # δ_c would be a failed refinement relabelled as a boundary type.
            if !keep_tracking && event === :negative && check_stability
                x_zero = zero_vanished!(copy(real.(ws.tracker.state.x)), tol)
                parameters_at_t!(ws.p_eval, t_end, p_start, p_target)
                if lambda_max_equilibrium_hc!(ws, x_zero, ws.p_eval) >= λ_tol
                    x_cross = copy(real.(ws.tracker.state.x))
                    t_onset = find_stability(ws.tracker, ws, x_zero, p_start, p_target,
                                             t_end, t_previous, λ_tol)
                    if real(t_onset) > real(t_end) && all(xᵢ -> xᵢ > tol, real.(ws.tracker.state.x))
                        event = :unstable
                        t_end = t_onset
                    else
                        x_crit_keep = x_cross
                    end
                end
            end

            if !keep_tracking
                continue
            end

            if check_stability
                parameters_at_t!(ws.p_eval, ws.tracker.state.t, p_start, p_target)
                λ_current = lambda_max_equilibrium_hc!(ws, x_current, ws.p_eval)
                if (λ_previous ≤ λ_tol) && (λ_current ≥ λ_tol)
                    event = :unstable
                    if abs(λ_current - λ_tol) ≤ λ_tol
                        t_end = ws.tracker.state.t
                    else
                        t_end = find_stability(ws.tracker, ws, copy(x_current), p_start, p_target, ws.tracker.state.t, t_previous, λ_tol)
                    end
                    keep_tracking = false
                else
                    λ_previous = λ_current
                end
            end

            if !keep_tracking
                continue
            end

            if check_invasibility && invasion_fn !== nothing
                inv_current = invasion_fn(x_current, ws.tracker.state.t)
                if (inv_previous ≤ invasion_tol) && (inv_current > invasion_tol)
                    event = :invasion
                    if abs(inv_current - invasion_tol) ≤ invasion_tol
                        t_end = ws.tracker.state.t
                    else
                        t_end = find_invasion(ws.tracker, copy(x_current), invasion_fn, ws.tracker.state.t, t_previous, invasion_tol)
                    end
                    keep_tracking = false
                else
                    inv_previous = inv_current
                end
            end

            if !keep_tracking
                continue
            end

            if is_success(ws.tracker.state.code)
                event = :success
                t_end = ws.tracker.state.t
                keep_tracking = false
            end
        end
    end

    x_crit = x_crit_keep === nothing ? copy(real.(ws.tracker.state.x)) : x_crit_keep
    init!(ws.tracker, x_start, 1.0, 0.0)
    reset_tracker_options!(ws.tracker)
    return event, t_end, x_crit
end
