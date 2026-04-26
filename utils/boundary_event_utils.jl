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

@inline function set_refinement_options!(tracker, Δt)
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

function find_event(p_start, p_target, x_start, ws, tol;
                    check_stability::Bool=true, λ_tol=SCAN_LAMBDA_TOL,
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
            return :unstable, ws.tracker.state.t, x_crit
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

    keep_tracking = true

    while keep_tracking
        t_previous = ws.tracker.state.t
        HomotopyContinuation.step!(ws.tracker)

        x_current .= real.(ws.tracker.state.x)
        if !is_tracking(ws.tracker.state.code) && !is_success(ws.tracker.state.code)
            event = :fold
            t_end = ws.tracker.state.t
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

    x_crit = copy(real.(ws.tracker.state.x))
    init!(ws.tracker, x_start, 1.0, 0.0)
    reset_tracker_options!(ws.tracker)
    return event, t_end, x_crit
end
