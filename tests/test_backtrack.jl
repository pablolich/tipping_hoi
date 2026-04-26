#!/usr/bin/env julia
# Exhaustive tests for backtrack_perturbation.jl.
#
# Run with:
#   julia --startup-file=no new_code/tests/test_backtrack.jl
#
# Covers: all pure-utility functions, all early-exit paths in
# process_direction_row, lambda_max_equilibrium_hc!, track_to_preboundary,
# HC event paths (:success, :unstable, :invasion), and backtrack_model.

using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "backtrack_perturbation.jl"))

# ─── Shared helper: build dyn config (mirrors main()) ────────────────────────

function make_dyn_cfg()
    post_delta_frac = 1.0 - Float64(SCAN_PREBOUNDARY_FRAC)   # 0.01
    return Dict{String,Any}(
        "tspan"                  => (0.0, ODE_TSPAN_END),
        "reltol"                 => ODE_RELTOL,
        "abstol"                 => ODE_ABSTOL,
        "saveat"                 => nothing,
        "eps_extinct"            => ODE_EPS_EXTINCT,
        "post_delta_frac"        => post_delta_frac,
        "post_delta_abs"         => BACK_POST_DELTA_ABS,      # nothing
        "post_delta_abs_default" => 1e-6,
    )
end

# ─── Section 1: Pure utility functions ───────────────────────────────────────

@testset "delta_from_dr" begin
    @test delta_from_dr([1.0, 0.0], [1.0, 0.0]) ≈ 1.0
    @test delta_from_dr([2.0, 2.0], [1.0, 1.0]) ≈ 2.0
    @test delta_from_dr([1.0, 0.0], [0.0, 0.0]) == 0.0   # denom=0 guard
end

@testset "reversal_frac" begin
    @test reversal_frac(2.0, 1.0) ≈ 0.5
    @test reversal_frac(1.0, 0.0) ≈ 1.0
    @test reversal_frac(1.0, 2.0) ≈ 0.0          # clamped from -1
    @test isnan(reversal_frac(0.0,  0.5))         # delta_post not > 0
    @test isnan(reversal_frac(NaN,  0.5))         # delta_post not finite
end

@testset "post_boundary_delta_s" begin
    # No abs override: uses frac * delta
    dyn1 = Dict{String,Any}(
        "post_delta_abs" => nothing, "post_delta_frac" => 0.01,
        "post_delta_abs_default" => 1e-6,
    )
    @test post_boundary_delta_s(dyn1, 2.0) ≈ 0.02

    # Abs override: ignores frac
    dyn2 = Dict{String,Any}(
        "post_delta_abs" => 0.5, "post_delta_frac" => 0.01,
        "post_delta_abs_default" => 1e-6,
    )
    @test post_boundary_delta_s(dyn2, 2.0) ≈ 0.5

    # Frac * 0 = 0, not > 0 → fallback to default
    @test post_boundary_delta_s(dyn1, 0.0) ≈ 1e-6
end

@testset "build_ode_seed" begin
    # Inactive species gets seed: seed_amp = max(0.01*min_active, floor)
    x1 = [2.0, 0.0]
    s1 = build_ode_seed(x1, [2]; seed_floor=1e-3)
    @test s1[1] ≈ 2.0
    @test s1[2] ≈ 0.02    # max(0.01*2, 1e-3) = 0.02

    # No inactive: no change
    x2 = [1.0, 1.0]
    s2 = build_ode_seed(x2, Int[])
    @test s2 == [1.0, 1.0]

    # All inactive, all zero: seed_amp = max(0.01*0, 1e-3) = 1e-3
    x3 = [0.0, 0.0]
    s3 = build_ode_seed(x3, [1, 2]; seed_floor=1e-3)
    @test s3 ≈ [1e-3, 1e-3]
end

@testset "invasion_max_at_state" begin
    n = 3
    r0  = [0.0, 0.0, 1.0]
    A   = zeros(Float64, 3, 3)
    B   = zeros(Float64, 3, 3, 3)
    u   = [1/√2, 1/√2, 0.0]

    # Species 3 inactive; its per-capita growth = r0[3] = 1.0
    val = invasion_max_at_state(r0, A, B, u, [1.0, 1.0], [1, 2], [3], 0.0)
    @test val ≈ 1.0

    # No inactive species → -Inf guard
    @test invasion_max_at_state(r0, A, B, u, [1.0, 1.0, 1.0], [1, 2, 3], Int[], 0.0) == -Inf
end

@testset "to_float_or_nan" begin
    @test isnan(to_float_or_nan(nothing))
    @test isnan(to_float_or_nan(Inf))
    @test to_float_or_nan(1.5) ≈ 1.5
end

@testset "to_float_vector_or_nothing" begin
    @test to_float_vector_or_nothing(nothing, 2)          === nothing
    @test to_float_vector_or_nothing([1.0, 2.0], 2)       == [1.0, 2.0]
    @test to_float_vector_or_nothing([1.0, 2.0, 3.0], 2)  === nothing   # wrong length
    @test to_float_vector_or_nothing([1.0, NaN], 2)        === nothing   # non-finite
end

@testset "compute_u_from_U" begin
    U = Matrix{Float64}([1.0 0.0; 0.0 1.0])
    @test compute_u_from_U(U, 1) ≈ [1.0, 0.0]
    @test compute_u_from_U(U, 3) === nothing           # out of bounds

    U_scaled = Matrix{Float64}([2.0 0.0; 0.0 3.0])
    @test compute_u_from_U(U_scaled, 1) ≈ [1.0, 0.0]  # normalized

    U_zero = reshape([0.0, 0.0], 2, 1)
    @test compute_u_from_U(U_zero, 1) === nothing      # zero norm
end

@testset "default_result_row field completeness" begin
    row = default_result_row(1, 0.5, 1, 2, "negative", "step_refined", 1.0, 0.5)
    required_keys = [
        "row_idx", "alpha", "alpha_idx", "ray_id", "boundary_flag",
        "boundary_status", "delta_boundary", "delta_post",
        "n_active_start", "n_inactive_start",
        "hc_event", "hc_status", "delta_event", "delta_probe", "delta_return",
        "delta_return_kind", "reversal_frac", "returned_n", "class_label",
        "ode_retcode", "ode_ran", "snap_reason", "n_alive_ode",
    ]
    for k in required_keys
        @test haskey(row, k)
    end
end

# ─── Section 2: process_direction_row early-exit paths ───────────────────────

@testset "process_direction_row early exits" begin
    r0    = [1.5, 1.5]
    A_eff = Matrix{Float64}([-2.0  0.5;  0.5 -2.0])
    B_eff = zeros(Float64, 2, 2, 2)
    U2    = Matrix{Float64}([1.0 0.0; 0.0 1.0])
    dyn   = make_dyn_cfg()
    seq   = build_seq_cfg()
    back  = build_backtrack_cfg()

    call = (row) -> process_direction_row(
        row, 1, 0.0, 1, r0, A_eff, B_eff, U2, dyn, seq, back
    )

    # 2a: invalid ray_id (out of bounds)
    r2a = Dict{String,Any}("ray_id"=>99, "flag"=>"negative",
                           "status"=>"step_refined", "delta_c"=>0.5,
                           "delta_post"=>0.5, "x_postboundary_snap"=>[1.0, 1.0])
    res2a = call(r2a)
    @test res2a["hc_status"] == "invalid_direction"
    @test res2a["ode_ran"]   == false

    # 2b: invalid delta_post (negative)
    r2b = Dict{String,Any}("ray_id"=>1, "flag"=>"negative",
                           "status"=>"step_refined", "delta_c"=>0.5,
                           "delta_post"=>-0.5, "x_postboundary_snap"=>[1.0, 1.0])
    res2b = call(r2b)
    @test res2b["hc_status"] == "invalid_delta_post"
    @test res2b["ode_ran"]   == false

    # 2c: missing x_post key entirely
    r2c = Dict{String,Any}("ray_id"=>1, "flag"=>"negative",
                           "status"=>"step_refined", "delta_c"=>0.5,
                           "delta_post"=>0.5)
    res2c = call(r2c)
    @test res2c["hc_status"] == "missing_x_post"
    @test res2c["ode_ran"]   == false

    # 2d: wrong-length x_post
    r2d = Dict{String,Any}("ray_id"=>1, "flag"=>"negative",
                           "status"=>"step_refined", "delta_c"=>0.5,
                           "delta_post"=>0.5, "x_postboundary_snap"=>[1.0])
    res2d = call(r2d)
    @test res2d["hc_status"] == "invalid_x_post"
    @test res2d["ode_ran"]   == false

    # 2e: NaN in x_post
    r2e = Dict{String,Any}("ray_id"=>1, "flag"=>"negative",
                           "status"=>"step_refined", "delta_c"=>0.5,
                           "delta_post"=>0.5, "x_postboundary_snap"=>[NaN, 1.0])
    res2e = call(r2e)
    @test res2e["hc_status"] == "invalid_x_post"
    @test res2e["ode_ran"]   == false

    # 2f: empty support (all zeros)
    r2f = Dict{String,Any}("ray_id"=>1, "flag"=>"negative",
                           "status"=>"step_refined", "delta_c"=>0.5,
                           "delta_post"=>0.5, "x_postboundary_snap"=>[0.0, 0.0])
    res2f = call(r2f)
    @test res2f["hc_status"] == "empty_support"
    @test res2f["ode_ran"]   == false

    # 2g: delta_post = 0 (already at zero)
    r2g = Dict{String,Any}("ray_id"=>1, "flag"=>"negative",
                           "status"=>"step_refined", "delta_c"=>0.5,
                           "delta_post"=>0.0, "x_postboundary_snap"=>[1.0, 1.0])
    res2g = call(r2g)
    @test res2g["hc_event"]    == "success"
    @test res2g["hc_status"]   == "already_at_zero"
    @test res2g["delta_event"] ≈ 0.0
    @test res2g["ode_ran"]     == false
end

# ─── Section 3: lambda_max_equilibrium_hc! ───────────────────────────────────

@testset "lambda_max_equilibrium_hc!" begin
    A_s  = Matrix{Float64}([-2.0  0.5;  0.5 -2.0])
    B_s  = zeros(Float64, 2, 2, 2)
    r0_s = [1.5, 1.5]
    syst, _ = build_system(r0_s, A_s, B_s)
    ws   = BacktrackWorkspace(syst, 2, 2)

    x_test = [1.0, 1.0]
    p_test = [0.0, 0.0]
    λ_hc  = lambda_max_equilibrium_hc!(ws, x_test, p_test)
    λ_ref = lambda_max_equilibrium(A_s, B_s, x_test)
    @test isapprox(λ_hc, λ_ref, atol=1e-12)
    @test λ_hc < 0.0    # stable system: λ_max = -1.5
end

# ─── Section 4: track_to_preboundary ─────────────────────────────────────────

@testset "track_to_preboundary" begin
    A_mat = Matrix{Float64}([-2.0  0.5;  0.5 -2.0])
    B_mat = zeros(Float64, 2, 2, 2)
    r0    = [1.5, 1.5]

    syst, _ = build_system(r0, A_mat, B_mat)
    ws = BacktrackWorkspace(syst, 2, 2)

    # x_act = equilibrium at p_start = [0.5, 0]:  -A⁻¹*(r0 + p_start)
    # det(A) = 4 - 0.25 = 3.75
    # A⁻¹ = (1/3.75) * [[-2, -0.5], [-0.5, -2]]
    # x_act = -A⁻¹ * [2.0, 1.5] = [4.75/3.75, 4.0/3.75] = [19/15, 16/15]
    x_act   = [19.0/15, 16.0/15]
    p_start = [0.5, 0.0]
    p_crit  = [0.4, 0.0]

    x_pre = track_to_preboundary(ws, x_act, p_start, p_crit, 0.99)

    # Expected: -A⁻¹*(r0 + 0.99*p_crit) = -A⁻¹*[1.896, 1.5]
    p_pre_val  = 0.99 * 0.4
    x_expected = -inv(A_mat) * (r0 .+ [p_pre_val, 0.0])
    @test norm(x_pre .- x_expected) < 1e-6

    # Zero-distance case: p_start=p_crit=zeros → p_pre==p_start → early return
    x_zero = copy(x_act)
    x_ret  = track_to_preboundary(ws, x_zero, zeros(2), zeros(2), 0.99)
    @test norm(x_ret .- x_zero) < 1e-14
end

# ─── Section 5: HC event paths via process_direction_row ─────────────────────

@testset "HC event: success (smooth backtrack)" begin
    r0    = [1.5, 1.5]
    A_eff = Matrix{Float64}([-2.0  0.5;  0.5 -2.0])
    B_eff = zeros(Float64, 2, 2, 2)
    U2    = Matrix{Float64}([1.0 0.0; 0.0 1.0])
    dyn   = make_dyn_cfg()
    seq   = build_seq_cfg()
    back  = build_backtrack_cfg()

    # Equilibrium at p_start=[0.5,0]: x_act = [19/15, 16/15]
    row = Dict{String,Any}(
        "ray_id"              => 1,
        "flag"                => "negative",
        "status"              => "step_refined",
        "delta_c"             => 0.5,
        "delta_post"          => 0.5,
        "x_postboundary_snap" => [19.0/15, 16.0/15],
    )
    result = process_direction_row(row, 1, 0.0, 1, r0, A_eff, B_eff, U2, dyn, seq, back)

    @test result["hc_event"]    == "success"
    @test result["ode_ran"]     == false
    @test result["delta_event"] < 1e-6    # tracked all the way to delta=0
end

@testset "HC event: unstable (λ_max ≥ 0 at start)" begin
    # Community Jacobian = Diagonal(x)*A = I at x=[1,1] → λ_max = 1 > 0
    r0_u    = [-1.0, -1.0]
    A_eff_u = Matrix{Float64}([1.0 0.0; 0.0 1.0])
    B_eff_u = zeros(Float64, 2, 2, 2)
    U_u     = reshape([1.0, 0.0], 2, 1)   # 2×1 matrix, single direction
    dyn     = make_dyn_cfg()
    seq     = build_seq_cfg()
    back    = build_backtrack_cfg()

    row_u = Dict{String,Any}(
        "ray_id"              => 1,
        "flag"                => "fold",
        "status"              => "step_refined",
        "delta_c"             => 0.1,
        "delta_post"          => 0.1,
        "x_postboundary_snap" => [1.0, 1.0],
    )
    result_u = process_direction_row(row_u, 1, 0.0, 1, r0_u, A_eff_u, B_eff_u, U_u, dyn, seq, back)

    @test result_u["hc_event"] == "unstable"
    @test result_u["ode_ran"]  == true
end

@testset "HC event: invasion (inactive species invades)" begin
    # 3-species system; species 3 extinct in x_post.
    # A[3,1]=A[3,2]=0, r0[3]=1.0 → F_3 = 1.0 >> invasion_tol → immediate :invasion.
    r0_i    = [1.5, 1.5, 1.0]
    A_eff_i = Matrix{Float64}([-2.0  0.5  0.0;
                                0.5 -2.0  0.0;
                                0.0  0.0 -1.0])
    B_eff_i = zeros(Float64, 3, 3, 3)
    U_i     = reshape([1/√2, 1/√2, 0.0], 3, 1)   # 3×1, direction [1/√2,1/√2,0]
    dyn     = make_dyn_cfg()
    seq     = build_seq_cfg()
    back    = build_backtrack_cfg()

    x_post_i = [19.0/15, 16.0/15, 0.0]   # species 3 extinct
    row_i = Dict{String,Any}(
        "ray_id"              => 1,
        "flag"                => "negative",
        "status"              => "step_refined",
        "delta_c"             => 0.5,
        "delta_post"          => 0.5,
        "x_postboundary_snap" => x_post_i,
    )
    result_i = process_direction_row(row_i, 1, 0.0, 1, r0_i, A_eff_i, B_eff_i, U_i, dyn, seq, back)

    @test result_i["hc_event"]         == "invasion"
    @test result_i["ode_ran"]          == true
    @test result_i["n_inactive_start"] == 1
end

# ─── Section 6: backtrack_model integration test ─────────────────────────────

@testset "backtrack_model integration" begin
    # Build a complete synthetic model dict with post_dynamics_results.
    # Single direction, alpha=0, same stable n=2 system.
    model = Dict{String,Any}(
        "n"          => 2,
        "n_dirs"     => 1,
        "model_idx"  => 1,
        "r"          => [1.5, 1.5],
        # A: outer=rows of A
        "A"          => [[-2.0, 0.5], [0.5, -2.0]],
        # B: nested B[j][k][i] — all zeros
        "B"          => [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        # U: 2 rows × 1 col  (direction [1,0])
        "U"          => [[1.0], [0.0]],
        "x_star"     => [1.0, 1.0],
        "dynamics_mode" => "standard",
        "post_dynamics_results" => [
            Dict{String,Any}(
                "alpha_idx" => 1,
                "alpha"     => 0.0,
                "directions" => [
                    Dict{String,Any}(
                        "ray_id"              => 1,
                        "flag"                => "negative",
                        "status"              => "step_refined",
                        "delta_c"             => 0.5,
                        "delta_post"          => 0.5,
                        "x_postboundary_snap" => [19.0/15, 16.0/15],
                    )
                ]
            )
        ]
    )

    dyn  = make_dyn_cfg()
    seq  = build_seq_cfg()
    back = build_backtrack_cfg()

    output = backtrack_model(model, dyn, seq, back)

    # Top-level keys present
    @test haskey(output, "backtrack_results")
    @test haskey(output, "backtrack_config")

    # Exactly one alpha block
    @test length(output["backtrack_results"]) == 1
    @test output["backtrack_results"][1]["alpha"] ≈ 0.0

    # Exactly one direction result
    dirs = output["backtrack_results"][1]["directions"]
    @test length(dirs) == 1

    # The direction completed with :success (smooth backtrack to zero)
    @test dirs[1]["hc_event"] == "success"
    @test dirs[1]["ode_ran"]  == false

    # Original model fields preserved
    @test output["n"] == 2
    @test output["r"] == model["r"]

    # Config fields present and correct
    @test haskey(output["backtrack_config"], "backtrack")
    @test output["backtrack_config"]["backtrack"]["check_stability"] == true
end
