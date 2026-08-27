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
