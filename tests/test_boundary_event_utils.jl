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
