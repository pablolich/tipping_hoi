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

# ─── An instability window inside one tracker step ───────────────────────────
#
# find_event sees λ_max only at the steps its tracker accepts, and the tracker
# sets the step length from the path's curvature and the Newton radius: it
# knows nothing about λ_max.  A loss of stability that begins AND ends between
# two accepted steps — λ_max below λ_tol at both — was therefore invisible, and
# the ray went on to be reported at a later event, typically `negative` at an
# extinction, although the equilibrium had stopped being the attractor earlier
# (first seen on ray 93 of community 9 of the Patil–Altieri h = 0 control:
# accepted steps at δ = 0.792 and 1.192, unstable on 1.011–1.148, reported
# `negative` at 1.306).  An instability that begins in the step that ends at a
# zero crossing and lasts up to the crossing is the other half of the same
# blind spot, closed by the test of the crossing state (zero_vanished! above).
#
# The test, on every accepted step whose two ends are stable:
#   1. the slope of λ_max along the path at each end (lambda_slope): one more
#      λ_max per step, a one-sided difference along the tangent the tracker's
#      predictor already holds;
#   2. rising at the start and falling at the end means λ_max has a maximum
#      inside the step.  Estimate it with the cubic through the two values and
#      the two slopes (hermite_max) and compare with the trigger level
#      (window_trigger);
#   3. on a trigger, re-walk the step in WINDOW_SUBSTEPS pieces on an AUXILIARY
#      tracker over the same homotopy, with λ_max at every sub-step, and refine
#      the first unstable one with find_stability (find_window_onset).
#
# Steps 1–3 touch the workspace buffers and the auxiliary tracker only.  The
# main tracker is never stepped, re-initialised or re-optioned by any of it,
# so a ray on which no onset is accepted comes out bit-identical.
#
# Not seen, by construction: a bump the two end slopes do not notice (λ_max
# turning around twice inside one step); a species dipping below the zero
# floor and back inside a step — the re-walk stops there and counts it; and a
# window inside a step that ends the ray (at a crossing, a fold, or an onset
# further on), since only a step with two stable ends is tested.

const WINDOW_SLOPE_EPS = 1e-6
const WINDOW_SUBSTEPS  = 8

# Run totals of the window test, so the rates are on record without a per-ray
# field.  The scan is single-threaded; a caller that wants per-run numbers
# resets before and reads after.
mutable struct WindowStats
    turnarounds::Int       # steps, both ends stable, λ_max rising at the start and falling at the end
    triggers::Int          # turnarounds whose cubic maximum reaches the trigger level
    triggers_strict::Int   # turnarounds whose cubic maximum reaches λ_tol itself
    rewalks::Int           # steps re-walked on the auxiliary tracker
    substeps::Int          # accepted sub-steps over all re-walks
    onsets::Int            # re-walks that located an onset with every species present
    onsets_strict::Int     # onsets on a step the strict rule also triggers
    onsets_rejected::Int   # unstable sub-step found, a species missing at the refined onset
    stopped_negative::Int  # re-walks stopped on a sub-step with an abundance at or below the floor
    linear_windows::Int    # α = 0 analytic rays unstable on the grid and stable at its far end
end
WindowStats() = WindowStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

const FIND_EVENT_WINDOW_STATS = WindowStats()

function reset_window_stats!(stats::WindowStats=FIND_EVENT_WINDOW_STATS)
    for f in fieldnames(WindowStats)
        setfield!(stats, f, 0)
    end
    return stats
end

window_stats_dict(stats::WindowStats=FIND_EVENT_WINDOW_STATS) =
    Dict{String,Int}(String(f) => getfield(stats, f) for f in fieldnames(WindowStats))

# Slope of λ_max along the tracked path per unit s = 1 − t (t runs 1 → 0, so
# s grows with the perturbation: positive means rising in δ), at an accepted
# point (x, t) where λ_max = λ.  `tx¹[i, 2]` is dx_i/dt there — the predictor's
# tangent, filled by init! and after every accepted step — so the point a
# distance ε further along the path is x − ε·x¹ at t − ε, and
#
#     slope = [ λ_max(x − ε·x¹, t − ε) − λ ] / ε.
#
# ε is WINDOW_SLOPE_EPS in t, shortened where the tangent is long so the probe
# never sits more than 1e-6·max(1, ‖x‖) from x: on a bank scanned with
# max_pert = 1000 a fixed 1e-6 in t is 1e-3 in δ, and next to a fold, where
# ‖x¹‖ diverges, it would step off the branch.  `λ_at(x, t)` evaluates λ_max;
# `x_probe` is a buffer.  NaN (never a turnaround) if the tangent is not finite.
function lambda_slope(λ_at, x, tx¹, t, λ, x_probe; ε_max=WINDOW_SLOPE_EPS)
    norm_x = 0.0
    norm_v = 0.0
    @inbounds for i in eachindex(x)
        norm_x = max(norm_x, abs(x[i]))
        norm_v = max(norm_v, abs(real(tx¹[i, 2])))
    end
    isfinite(norm_v) || return NaN
    ε = ε_max * min(1.0, max(1.0, norm_x) / norm_v)
    ε > 0 || return NaN
    @inbounds for i in eachindex(x)
        x_probe[i] = x[i] - ε * real(tx¹[i, 2])
    end
    return (λ_at(x_probe, t - ε) - λ) / ε
end

# Maximum over τ ∈ [0, 1] of the cubic through the values λ_a, λ_b and the
# slopes m_a, m_b at the two ends of a step of length h (h in the slopes' own
# variable):
#
#     H(τ) = (2τ³ − 3τ² + 1) λ_a + (τ³ − 2τ² + τ) h m_a
#          + (−2τ³ + 3τ²) λ_b + (τ³ − τ²) h m_b.
#
# Candidates are the two ends and the roots of H′, a quadratic.  Returns
# (H_max, τ_max).
function hermite_max(λ_a, λ_b, m_a, m_b, h)
    d_a = h * m_a
    d_b = h * m_b
    c1 = d_a
    c2 = 3 * (λ_b - λ_a) - 2 * d_a - d_b
    c3 = 2 * (λ_a - λ_b) + d_a + d_b
    H(τ) = ((c3 * τ + c2) * τ + c1) * τ + λ_a

    # H′(τ) = 3 c3 τ² + 2 c2 τ + c1; a root that does not exist stays NaN
    a, b, c = 3 * c3, 2 * c2, c1
    τ₁ = τ₂ = NaN
    if a == 0
        b == 0 || (τ₁ = -c / b)
    else
        disc = b^2 - 4 * a * c
        if disc ≥ 0
            q = -(b + copysign(sqrt(disc), b)) / 2
            τ₁ = q / a
            q == 0 || (τ₂ = c / q)
        end
    end

    H_max, τ_max = λ_a ≥ λ_b ? (float(λ_a), 0.0) : (float(λ_b), 1.0)
    for τ in (τ₁, τ₂)
        if 0 < τ < 1 && H(τ) > H_max
            H_max, τ_max = H(τ), τ
        end
    end
    return H_max, τ_max
end

# Whether a turnaround is worth a re-walk.  The cubic is an estimate of the
# maximum, not a bound — on the ray that found the bug it gives +0.0047 against
# a true peak of +0.0018, the safe direction, but it can err the other way — so
# the default asks for less than λ_tol by the smaller of the two end margins.
# With two stable ends that is H_max ≥ max(λ_a, λ_b) + λ_tol: any interior
# maximum that clears the higher end, which at a turnaround is all of them.
# `strict` is the bare comparison with λ_tol.
window_trigger(H_max, λ_a, λ_b, λ_tol; strict::Bool=false) =
    H_max ≥ (strict ? λ_tol : λ_tol - min(abs(λ_a), abs(λ_b)))

# Re-walk one accepted step, from (x_from, t_from) to t_to, on the auxiliary
# tracker `aux` with the step capped at 1/WINDOW_SUBSTEPS of its length, and
# evaluate λ_max at every accepted sub-step.  The first sub-step with
# λ_max ≥ λ_tol brackets the onset against the sub-step before it, and
# find_stability refines it there exactly as find_event does for a sign change
# between two of its own steps.  Returns (t_onset, x_onset) when the onset is
# located at a state with every species present, `nothing` otherwise:
#   * no sub-step is unstable (the cubic overestimated, or the window is
#     narrower than a sub-step);
#   * a sub-step has an abundance at or below the floor — a species dipping
#     under zero and back inside a step whose ends are both positive.  That is
#     a different gap from this one; the re-walk stops and counts it;
#   * a species is missing at the refined onset.
# `aux` shares the main tracker's homotopy object, so it sees the ray's
# parameters without anything being set, and the homotopy's only state is a
# cache keyed on t.  Its options are rewritten here on every call.
function find_window_onset(aux, ws, x_from, t_from, t_to, p_start, p_target, tol, λ_tol;
                           stats::WindowStats=FIND_EVENT_WINDOW_STATS)
    stats.rewalks += 1
    reset_tracker_options!(aux)
    aux.options.max_step_size = abs(t_from - t_to) / WINDOW_SUBSTEPS
    init!(aux, x_from, t_from, t_to) || return nothing

    x_sub = Vector{Float64}(undef, length(x_from))
    while is_tracking(aux.state.code)
        t_sub_previous = aux.state.t
        HomotopyContinuation.step!(aux)
        aux.state.t == t_sub_previous && continue      # rejected: nothing new to test
        stats.substeps += 1

        x_sub .= real.(aux.state.x)
        if any(xᵢ -> xᵢ ≤ tol, x_sub)
            stats.stopped_negative += 1
            return nothing
        end

        parameters_at_t!(ws.p_eval, aux.state.t, p_start, p_target)
        λ_sub = lambda_max_equilibrium_hc!(ws, x_sub, ws.p_eval)
        if λ_sub ≥ λ_tol
            t_onset = aux.state.t
            if abs(λ_sub - λ_tol) > λ_tol
                t_onset = find_stability(aux, ws, copy(x_sub), p_start, p_target,
                                         aux.state.t, t_sub_previous, λ_tol)
            end
            x_onset = copy(real.(aux.state.x))
            if all(xᵢ -> xᵢ > tol, x_onset)
                return t_onset, x_onset
            end
            stats.onsets_rejected += 1
            return nothing
        end
    end
    return nothing
end

# The auxiliary tracker of the re-walks: a second Tracker over the main
# tracker's own homotopy object.  The assertion is for the compiler.  The
# constructor's return type is inferred as a union of two tracker types, and
# without it the whole tracker stack is compiled a second time for the one
# that is never built: 2 s more on every system scanned, a third of the time
# of a scan that is mostly compilation.
auxiliary_tracker(tracker) =
    Tracker(tracker.homotopy; options=tracker.options)::typeof(tracker)

function find_event(p_start, p_target, x_start, ws, tol;
                    check_stability::Bool=true, λ_tol=LAMBDA_TOL,
                    check_invasibility::Bool=false,
                    invasion_fn=nothing, invasion_tol::Float64=1e-10,
                    window_test::Bool=true)
    start_parameters!(ws.tracker, p_start)
    target_parameters!(ws.tracker, p_target)
    init!(ws.tracker, x_start, 1.0, 0.0)
    x_current = Vector{Float64}(undef, length(x_start))
    λ_previous = -Inf
    inv_previous = -Inf

    # The window test (see above), gated on check_stability like every other
    # stability test here: the slope and state at the previous accepted step,
    # and the auxiliary tracker, built on the first trigger of the ray.
    test_window = window_test && check_stability
    m_previous = NaN
    x_previous = collect(Float64, x_start)
    x_probe = similar(x_current)
    λ_at = (x, t) -> begin
        parameters_at_t!(ws.p_eval, t, p_start, p_target)
        lambda_max_equilibrium_hc!(ws, x, ws.p_eval)
    end
    aux = nothing

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
        if test_window
            m_previous = lambda_slope(λ_at, x_start, ws.tracker.predictor.tx¹,
                                      real(ws.tracker.state.t), λ_previous, x_probe)
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
    # The state to return when it is not the one the main tracker ends on.  Set
    # when a crossing's stability test moved the tracker but the onset was not
    # accepted — the crossing state as it was, so the ray is returned exactly as
    # it would have been without the test — and when the window test located an
    # onset, which lives on the auxiliary tracker.
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
                    # Both ends of the step are stable.  If λ_max was rising
                    # when the step began and is falling now, it had a maximum
                    # in between; when the cubic through the two ends puts that
                    # maximum high enough, re-walk the step on the auxiliary
                    # tracker.  A rejected step (t did not move) has nothing
                    # new to test.
                    if test_window && ws.tracker.state.t != t_previous
                        m_current = lambda_slope(λ_at, x_current, ws.tracker.predictor.tx¹,
                                                 real(ws.tracker.state.t), λ_current, x_probe)
                        if m_previous > 0 && m_current < 0
                            stats = FIND_EVENT_WINDOW_STATS
                            stats.turnarounds += 1
                            H_max, _ = hermite_max(λ_previous, λ_current, m_previous, m_current,
                                                   real(t_previous) - real(ws.tracker.state.t))
                            strict = window_trigger(H_max, λ_previous, λ_current, λ_tol; strict=true)
                            stats.triggers_strict += strict
                            if window_trigger(H_max, λ_previous, λ_current, λ_tol)
                                stats.triggers += 1
                                if aux === nothing
                                    aux = auxiliary_tracker(ws.tracker)
                                end
                                onset = find_window_onset(aux, ws, x_previous, t_previous,
                                                          ws.tracker.state.t, p_start, p_target,
                                                          tol, λ_tol)
                                if onset !== nothing
                                    event = :unstable
                                    t_end, x_crit_keep = onset
                                    keep_tracking = false
                                    stats.onsets += 1
                                    stats.onsets_strict += strict
                                end
                            end
                        end
                        m_previous = m_current
                        x_previous .= x_current
                    end
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
