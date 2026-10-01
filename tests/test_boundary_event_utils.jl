#!/usr/bin/env julia
# Tests for utils/boundary_event_utils.jl's tracker-option helpers.
#
# Run with:
#   julia --project=. --startup-file=no tests/test_boundary_event_utils.jl
#
# boundary_event_utils.jl is included by boundary_scan.jl AFTER ScanWorkspace and
# lambda_max_equilibrium_hc! are defined, so it cannot be included on its own.
# Including the driver is the same pattern test_backtrack.jl uses; its main() is
# guarded on PROGRAM_FILE and does not run.

using Test

include(joinpath(@__DIR__, "..", "pipeline", "boundary_scan.jl"))

# The helpers touch only `tracker.options`, so a stub with the three fields is
# enough and keeps the test free of HomotopyContinuation setup.
mutable struct StubOptions
    max_step_size::Float64
    max_steps::Int
    min_step_size::Float64
end
struct StubTracker
    options::StubOptions
end
fresh_tracker() = StubTracker(StubOptions(Inf, 10_000, 1e-48))

@testset "set_refinement_options!" begin
    @testset "writes the step constraints for a positive Δt" begin
        t = fresh_tracker()
        set_refinement_options!(t, 0.25)
        @test t.options.max_step_size == 0.125
        @test t.options.max_steps == 4
        @test t.options.min_step_size == 0.25e-48
    end

    @testset "rounds the step budget up" begin
        t = fresh_tracker()
        set_refinement_options!(t, 0.3)              # 1/0.3 = 3.33...
        @test t.options.max_steps == 4
    end

    @testset "never asks for fewer than one step" begin
        t = fresh_tracker()
        set_refinement_options!(t, 2.0)              # 1/2 = 0.5, ceil -> 1
        @test t.options.max_steps == 1
    end

    # THE REGRESSION.  find_zero / find_stability / find_invasion each compute
    # Δt = abs(t_previous - t_end), stop when it is zero, and then call this
    # unconditionally.  Before the guard, Δt == 0 made 1/Δt infinite and
    # Int(ceil(Inf)) threw InexactError, which propagated out of scan_model and
    # cost the whole model -- observed on
    # review-1_responses/scratch/parameterization_v4a_bank at a = -1, b = -6,
    # n = 4, where the b = -6 arm's larger beta drives the tracker onto a step
    # the refiner cannot subdivide.
    @testset "Δt == 0 is a no-op, not an InexactError" begin
        t = fresh_tracker()
        @test set_refinement_options!(t, 0.0) === t
        # The caller has already set keep_tracking = false and assigned t_end,
        # so the options are dead -- and must be left exactly as they were.
        @test t.options.max_step_size == Inf
        @test t.options.max_steps == 10_000
        @test t.options.min_step_size == 1e-48
    end
end

@testset "reset_tracker_options!" begin
    t = StubTracker(StubOptions(0.5, 3, 1e-30))
    reset_tracker_options!(t)
    @test t.options.max_step_size == Inf
    @test t.options.max_steps == 10_000
    @test t.options.min_step_size == 1e-48
end

# ─── The order of events in find_event ───────────────────────────────────────
#
# Inside one accepted step find_event tests negativity before stability, so a
# loss of stability in the step that ends at a zero crossing was never seen and
# the ray came back `negative` at the crossing.  The fix tests the crossing
# state with the vanished species set to exactly 0 and, if the surviving block
# is unstable, refines the onset with find_stability — and relabels only when
# the onset is located strictly before the crossing with every species present.
# Fixtures (tests/fixtures/find_event_order*.json, provenance inside): a
# shipped gibbs model with one ray that is reclassified and one whose crossing
# is stable, and a shipped karatayev model with a ray whose crossing state is
# off the branch (a refiner that ended far past zero), which must stay
# `negative` untouched.  The expected numbers come from
# review-1_responses/scratch/find_event_order/ (step 3 of the plan: grid of
# track_to_params! + ODE from either side of delta_u).

@testset "zero_vanished!" begin
    tol = 1e-9
    # (c) every species present: left untouched, bit for bit
    x = [1.0, 0.5, 2.0e-9, 1.1e-9]
    @test zero_vanished!(copy(x), tol) == x
    # inside the floor, either sign
    @test zero_vanished!([1.0, 5e-10, -2e-10, 0.0], tol) == [1.0, 0.0, 0.0, 0.0]
    # the overshoot case: a refiner that stopped past zero leaves x_i < -tol,
    # and that entry is the vanished species too
    @test zero_vanished!([1.0, -3.6e-6], tol) == [1.0, 0.0]
    # exactly at the floor counts as vanished
    @test zero_vanished!([tol], tol) == [0.0]
    @test zero_vanished!([nextfloat(tol)], tol) == [nextfloat(tol)]
end

# ─── An instability window inside one tracker step: the pieces ───────────────
#
# find_event sees λ_max only at the steps its tracker accepts, so a loss of
# stability that begins and ends between two of them was invisible.  The test
# has three pieces, checked here on numbers that can be worked by hand: the
# slope of λ_max along the path, the maximum of the cubic through the two ends
# of a step, and the rule that decides whether that maximum earns a re-walk.
# The fixture tests further down run the whole thing on shipped models.

@testset "lambda_slope" begin
    # A stub path x(t) = x0 + (1 − t)·v: the state moves by v per unit s = 1 − t,
    # so dx/dt = −v, which is what the predictor's tangent holds in column 2.
    x0 = [1.0, 2.0, 0.5]
    v  = [0.3, -0.2, 0.1]
    tangent(v) = hcat(zeros(length(v)), -v)
    probe = similar(x0)

    # λ known along the line: λ = 2·x₁, so dλ/ds = 2·v₁ > 0 — rising in δ.
    λ_at = (x, t) -> 2 * x[1]
    m = lambda_slope(λ_at, x0, tangent(v), 1.0, λ_at(x0, 1.0), probe)
    @test m ≈ 2 * v[1] rtol = 1e-6
    @test m > 0
    # the same line walked the other way falls in δ
    @test lambda_slope(λ_at, x0, tangent(-v), 1.0, λ_at(x0, 1.0), probe) ≈ -2 * v[1] rtol = 1e-6
    # λ that depends on the parameters only: the probe sits at t − ε
    λ_p = (x, t) -> 5 * (1 - t)
    @test lambda_slope(λ_p, x0, tangent(v), 0.7, λ_p(x0, 0.7), probe) ≈ 5 rtol = 1e-6

    # A long tangent shortens ε: the probe never sits more than
    # 1e-6·max(1, ‖x‖) from x, and the slope is still the slope.
    seen = similar(x0)
    λ_rec = (x, t) -> (seen .= x; 2 * x[1])
    big = 1e9 .* v
    m_big = lambda_slope(λ_rec, x0, tangent(big), 1.0, 2 * x0[1], probe)
    @test maximum(abs, seen .- x0) <= WINDOW_SLOPE_EPS * maximum(abs, x0) * (1 + 1e-6)
    @test m_big ≈ 2 * big[1] rtol = 1e-6
    # a tangent that is not finite is never a turnaround
    @test isnan(lambda_slope(λ_at, x0, tangent([Inf, 0.0, 0.0]), 1.0, 2.0, probe))
    @test isnan(lambda_slope(λ_at, x0, tangent([NaN, 0.0, 0.0]), 1.0, 2.0, probe))
end

@testset "hermite_max" begin
    # monotone: the maximum is at an end
    @test hermite_max(-2.0, -1.0, 1.0, 1.0, 1.0) == (-1.0, 1.0)
    @test hermite_max(-1.0, -2.0, -1.0, -1.0, 1.0) == (-1.0, 0.0)
    # symmetric bump: the parabola −1 + 4·3·τ(1 − τ) on a step of length 2,
    # which the cubic reproduces exactly — maximum +2 in the middle
    H, τ = hermite_max(-1.0, -1.0, 6.0, -6.0, 2.0)
    @test H ≈ 2.0
    @test τ ≈ 0.5
    # a cubic proper: τ³ − 2τ² + τ − 0.1, stationary at 1/3 and 1
    H, τ = hermite_max(-0.1, -0.1, 1.0, 0.0, 1.0)
    @test H ≈ 4 / 27 - 0.1
    @test τ ≈ 1 / 3
    # the ray that found the bug (Patil–Altieri h = 0 control, community 9,
    # ray 93): accepted steps at δ = 0.792 and 1.192, both stable, unstable on
    # 1.011–1.148 with a true peak of +0.0018
    H, τ = hermite_max(-0.01433, -0.00491, 0.0760, -0.1649, 0.4004)
    @test H ≈ 0.0047 atol = 5e-5
    @test 0.792 + τ * 0.4004 ≈ 1.06 atol = 0.01
end

@testset "window_trigger" begin
    λ_tol = 1e-9
    # ray 93: the cubic's maximum is above λ_tol, both rules fire
    H, _ = hermite_max(-0.01433, -0.00491, 0.0760, -0.1649, 0.4004)
    @test window_trigger(H, -0.01433, -0.00491, λ_tol)
    @test window_trigger(H, -0.01433, -0.00491, λ_tol; strict=true)
    # a bump that the cubic keeps below zero (maximum −0.25 from ends at −0.5):
    # within the smaller end margin of λ_tol, so the default re-walks it; the
    # strict rule does not
    H, _ = hermite_max(-0.5, -0.5, 1.0, -1.0, 1.0)
    @test H ≈ -0.25
    @test window_trigger(H, -0.5, -0.5, λ_tol)
    @test !window_trigger(H, -0.5, -0.5, λ_tol; strict=true)
    # no bump at all (maximum at an end): neither rule fires
    H, _ = hermite_max(-2.0, -1.0, 1.0, 1.0, 1.0)
    @test !window_trigger(H, -2.0, -1.0, λ_tol)
    @test !window_trigger(H, -2.0, -1.0, λ_tol; strict=true)
end

@testset "window counters" begin
    stats = WindowStats()
    stats.turnarounds = 3; stats.onsets = 1
    @test window_stats_dict(stats)["turnarounds"] == 3
    @test window_stats_dict(stats)["onsets"] == 1
    reset_window_stats!(stats)
    @test all(==(0), values(window_stats_dict(stats)))
end

# The α = 0 analytic path has the same blind spot in its own form: its 128-point
# stability grid used to be consulted only when λ_max at the far end of the ray
# was unstable.  A three-species linear system built to have a window: x* = 1,
# stable there, and along u the community matrix diag(x(s))·A is unstable on
# s ∈ (2.2744, 4.6745) — a complex pair, peak +0.066 — and stable again before
# species 3 reaches zero at s = 5.8186.  (None of the 663,441 α = 0 rays in the
# shipped banks has a window, so this is the only place the new branch runs.)
@testset "scan_ray_linear_alpha0: a window on the α = 0 path" begin
    A = [-0.45 0.43 1.11; -0.23 -0.84 -1.18; 2.34 3.67 -1.17]
    x_base = ones(3)
    r0 = -A * x_base
    A_fac = lu(A)
    scan(u) = scan_ray_linear_alpha0(A, A_fac, x_base, r0, normalize(u); max_pert_mag=30.0)
    λ_at(s, u) = maximum(real, eigvals(Diagonal(x_base .- s .* (A_fac \ normalize(u))) * A))

    u = [0.405, -0.168, -0.899]
    @test λ_at(0.0, u) < 0 && λ_at(3.5, u) > 0.05 && λ_at(5.5, u) < 0      # the window is there
    reset_window_stats!()
    res = scan(u)
    @test res.flag === :unstable
    @test res.delta_c ≈ 2.2744273536 rtol = 1e-3       # one grid cell, bisected MAX_ITERS times
    @test res.delta_c < 5.8186
    @test all(>(0), res.x_crit)
    @test λ_at(res.delta_c, u) > LAMBDA_TOL
    @test FIND_EVENT_WINDOW_STATS.linear_windows == 1

    # the opposite ray has no instability anywhere: `negative` at its extinction
    reset_window_stats!()
    res = scan(-u)
    @test res.flag === :negative
    @test res.delta_c ≈ 2.447420933 rtol = 1e-8
    # a ray that is still unstable at the far end is found as it always was,
    # and is not counted as a window
    res = scan([1.0, 0.0, 0.0])
    @test res.flag === :unstable
    @test res.delta_c < 0.8082348
    @test FIND_EVENT_WINDOW_STATS.linear_windows == 0
end

const FIXTURES = [joinpath(@__DIR__, "fixtures", "find_event_order.json"),
                  joinpath(@__DIR__, "fixtures", "find_event_order_karatayev.json")]

@testset "find_event: instability behind a zero crossing ($(basename(FIXTURE)))" for FIXTURE in FIXTURES
    fx    = to_dict(JSON3.read(read(FIXTURE, String)))
    model = fx["model"]
    ctx   = build_hc_system(model)
    alpha = first(ctx.alpha_grid)
    ws    = ctx.make_workspace(alpha)
    x0    = collect(Float64, ctx.x0)
    r0    = collect(Float64, ctx.baseline_r)
    max_pert = Float64(fx["max_pert"])

    function run_ray(ray_id; check_stability=true)
        u = collect(ctx.U[:, ray_id]); u ./= norm(u)
        reset_ray_parameters!(ws)
        set_ray_target!(ws, u, max_pert)
        event, t_end, x_crit = find_event(ws.p_start, ws.p_target, copy(x0), ws, ZERO_ABUNDANCE;
                                          check_stability=check_stability, λ_tol=LAMBDA_TOL)
        delta = norm((1 - real(t_end)) .* ws.p_target)
        return (event=event, t_end=t_end, x_crit=x_crit, delta=delta)
    end

    for ray in fx["rays"]
        ray_id = Int(ray["ray_id"])
        kind   = String(ray["kind"])
        res    = run_ray(ray_id)
        expected_delta = Float64(ray["expected_delta"])
        rtol   = Float64(ray["delta_rtol"])
        @testset "ray $ray_id ($kind)" begin
            @test res.event === Symbol(ray["expected_flag"])
            @test abs(res.delta - expected_delta) <= rtol * expected_delta
            if kind == "reclassified"
                # (a) was `negative` at the shipped delta_c; now `unstable` at
                # delta_u < delta_c, with every species still present there.
                @test res.delta < Float64(ray["shipped_delta_c"])
                @test all(>(0), res.x_crit)
            elseif kind == "stable_crossing" || kind == "onset_not_located"
                # (b) a crossing whose surviving block is stable is still
                # `negative`, and the extra evaluation must not have moved the
                # tracker: with check_stability=false neither the new test nor
                # the per-step stability block runs, so the two calls must agree
                # bit for bit.  The same holds when the block IS unstable but
                # the onset cannot be located (`onset_not_located`: the crossing
                # state is off the branch, init! cannot start from it) — the
                # crossing is handed back exactly as it was.
                res_nostab = run_ray(ray_id; check_stability=false)
                @test res_nostab.event === :negative
                @test res_nostab.t_end == res.t_end
                @test res_nostab.x_crit == res.x_crit
                @test minimum(res.x_crit) <= ZERO_ABUNDANCE
            end
        end
    end
end

# ─── An instability window inside one tracker step: shipped models ───────────
#
# Fixtures (tests/fixtures/find_event_window*.json, provenance inside), rays of
# shipped models of three kinds:
#   window            λ_max crosses zero and comes back between two accepted
#                     steps, both stable, so the shipped bank and find_event
#                     without the window test report the ray at a later event:
#                     `negative` at the extinction (find_event_window.json), or
#                     `unstable` at a second, later onset
#                     (find_event_window_unstable.json).  With the test the ray
#                     is `unstable` at the first onset, every species present,
#                     inside the bracket an independent grid of track_to_params!
#                     puts around it (review-1_responses/scratch/find_event_window/).
#   crossing_window   the same inside the step that ends at the zero crossing
#                     (find_event_window_crossing.json): the last accepted state
#                     is stable, the surviving block at the crossing is stable,
#                     and the equilibrium is unstable on a stretch in between.
#   trigger_no_onset  a step turns around and is re-walked, and the re-walk
#                     finds nothing: the ray must come out bit for bit as it
#                     does with the test off — the auxiliary-tracker guarantee.
#   no_turnaround     no step turns around: the same bit-for-bit agreement, and
#                     agreement with the shipped delta_c.
# Every `negative` ray has its crossing step re-walked exactly once, whatever
# its kind, and comes out bit for bit when that re-walk finds nothing.

const WINDOW_FIXTURES = [joinpath(@__DIR__, "fixtures", "find_event_window.json"),
                         joinpath(@__DIR__, "fixtures", "find_event_window_unstable.json"),
                         joinpath(@__DIR__, "fixtures", "find_event_window_crossing.json")]

@testset "find_event: instability window ($(basename(FIXTURE)))" for FIXTURE in WINDOW_FIXTURES
    fx    = to_dict(JSON3.read(read(FIXTURE, String)))
    model = fx["model"]
    ctx   = build_hc_system(model)
    max_pert = Float64(fx["max_pert"])
    workspaces = Dict{Int,Any}()

    function run_ray(alpha_idx, ray_id; window_test)
        alpha = ctx.alpha_grid[alpha_idx]
        ws = get!(() -> ctx.make_workspace(alpha), workspaces, alpha_idx)
        x0 = collect(Float64, ctx.x0 isa Function ? ctx.x0(alpha) : ctx.x0)
        u  = collect(ctx.U[:, ray_id]); u ./= norm(u)
        reset_ray_parameters!(ws)
        set_ray_target!(ws, u, max_pert)
        reset_window_stats!()
        event, t_end, x_crit = find_event(ws.p_start, ws.p_target, x0, ws, ZERO_ABUNDANCE;
                                          check_stability=true, λ_tol=LAMBDA_TOL,
                                          window_test=window_test)
        delta = norm((1 - real(t_end)) .* ws.p_target)
        return (event=event, t_end=t_end, x_crit=x_crit, delta=delta, stats=window_stats_dict())
    end

    for ray in fx["rays"]
        alpha_idx = Int(ray["alpha_idx"])
        ray_id    = Int(ray["ray_id"])
        kind      = String(ray["kind"])
        rtol      = Float64(ray["delta_rtol"])
        on  = run_ray(alpha_idx, ray_id; window_test=true)
        off = run_ray(alpha_idx, ray_id; window_test=false)
        @testset "alpha $alpha_idx ray $ray_id ($kind)" begin
            @test on.event === Symbol(ray["expected_flag"])
            @test abs(on.delta - Float64(ray["expected_delta"])) <= rtol * Float64(ray["expected_delta"])
            if kind == "window" || kind == "crossing_window"
                # without the test: the shipped result
                @test off.event === Symbol(ray["baseline_flag"])
                @test abs(off.delta - Float64(ray["baseline_delta_c"])) <= rtol * Float64(ray["baseline_delta_c"])
                # with it: the onset, before the old boundary, every species present
                @test on.stats["onsets"] == (kind == "window" ? 1 : 0)
                @test on.stats["crossing_onsets"] == (kind == "crossing_window" ? 1 : 0)
                @test on.delta < Float64(ray["shipped_delta_c"])
                @test all(>(0), on.x_crit)
                lo, hi = Float64.(ray["verified_bracket"])
                @test lo * (1 - rtol) <= on.delta <= hi * (1 + rtol)
            else
                @test on.event === off.event
                @test on.t_end == off.t_end
                @test on.x_crit == off.x_crit
                @test on.stats["onsets"] == 0
                @test on.stats["crossing_onsets"] == 0
                @test on.stats["crossing_rewalks"] == (on.event === :negative ? 1 : 0)
                if kind == "trigger_no_onset"
                    @test on.stats["triggers"] >= 1
                    @test on.stats["rewalks"] >= 1
                elseif kind == "no_turnaround"
                    @test on.stats["turnarounds"] == 0
                    @test abs(on.delta - Float64(ray["shipped_delta_c"])) <= 2e-9 * Float64(ray["shipped_delta_c"])
                end
            end
        end
    end
end
