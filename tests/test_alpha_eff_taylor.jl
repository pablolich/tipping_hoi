#!/usr/bin/env julia
# Tests for alpha_eff_taylor utility.
#
# Run with:
#   julia --startup-file=no new_code/tests/test_alpha_eff_taylor.jl

using Test
using LinearAlgebra
using Random
using ForwardDiff
using JSON3

include(joinpath(@__DIR__, "..", "utils", "alpha_eff_taylor.jl"))

# ─── analytic GLV+HOI match vs AD ────────────────────────────────────────────

@testset "analytic GLV+HOI match vs AD" begin
    # Small n=3 GLV+HOI with α baked in via A_eff, B_eff (standard convention).
    rng = MersenneTwister(20260417)
    n = 3
    α = 0.4
    A = randn(rng, n, n)
    B = randn(rng, n, n, n)
    r = randn(rng, n)
    A_eff, B_eff = ((1 - α) .* A, α .* B)
    x_star = 1.0 .+ 0.5 .* rand(rng, n)

    # f via the internal eltype-generic helper
    f = x -> _glvhoi_per_capita_rates(A_eff, B_eff, r, x)

    # Analytic J and H for F_i(x) = r[i] + Σ_j A_eff[i,j] x[j] + Σ_{jk} B_eff[j,k,i] x[j] x[k]
    # ∂F_i/∂x_m = A_eff[i,m] + Σ_j (B_eff[j,m,i] + B_eff[m,j,i]) x[j]
    # ∂²F_i/(∂x_j ∂x_k) = B_eff[j,k,i] + B_eff[k,j,i]
    J_analytic = copy(A_eff)
    for i in 1:n, m in 1:n
        s = 0.0
        for j in 1:n
            s += (B_eff[j, m, i] + B_eff[m, j, i]) * x_star[j]
        end
        J_analytic[i, m] += s
    end
    H_analytic = Array{Float64,3}(undef, n, n, n)
    for i in 1:n, j in 1:n, k in 1:n
        H_analytic[i, j, k] = B_eff[j, k, i] + B_eff[k, j, i]
    end

    T1_analytic = J_analytic * x_star
    T2_analytic = [0.5 * dot(x_star, H_analytic[i, :, :] * x_star) for i in 1:n]

    # Compare via compute_alpha_eff_taylor
    out = compute_alpha_eff_taylor(f, x_star)
    @test isapprox(out.T1, T1_analytic; atol = 1e-10)
    @test isapprox(out.T2, T2_analytic; atol = 1e-10)

    P_ref = sum(abs, T1_analytic) / n
    Q_ref = sum(abs, T2_analytic) / n
    α_ref = Q_ref / (P_ref + Q_ref)
    @test isapprox(out.P, P_ref; atol = 1e-12)
    @test isapprox(out.Q, Q_ref; atol = 1e-12)
    @test isapprox(out.alpha_eff_taylor, α_ref; atol = 1e-12)
end

# ─── α=0 and α=1 edge cases (GLV+HOI) ────────────────────────────────────────

@testset "α=0 ⇒ alpha_eff_taylor = 0" begin
    rng = MersenneTwister(2)
    n = 4
    A = randn(rng, n, n)
    B = randn(rng, n, n, n)
    r = randn(rng, n)
    α = 0.0
    A_eff, B_eff = ((1 - α) .* A, α .* B)
    x_star = ones(n)
    f = x -> _glvhoi_per_capita_rates(A_eff, B_eff, r, x)
    out = compute_alpha_eff_taylor(f, x_star)
    @test out.Q ≈ 0.0 atol = 1e-14
    @test out.alpha_eff_taylor == 0.0
end

@testset "α=1 (pure HOI) ⇒ alpha_eff_taylor = 1/3" begin
    # For a homogeneous degree-2 f(x) = Σ B[j,k,i] x[j] x[k], Euler's theorem
    # gives T1[i] = Σ_m (∂f_i/∂x_m) x_m = 2·f_i(x_star) and T2[i] = f_i(x_star).
    # The ratio then equals 1/3 identically, independent of B and x_star.
    # This is a structural fact of the Taylor metric, not a bug — the Taylor
    # definition differs from the existing Gibbs-style alpha_eff at α=1.
    rng = MersenneTwister(3)
    n = 4
    A = randn(rng, n, n)
    B = randn(rng, n, n, n)
    r = zeros(n)  # ensures f is purely degree-2 (homogeneous)
    α = 1.0
    A_eff, B_eff = ((1 - α) .* A, α .* B)
    x_star = ones(n) .+ 0.3 .* rand(rng, n)
    f = x -> _glvhoi_per_capita_rates(A_eff, B_eff, r, x)
    out = compute_alpha_eff_taylor(f, x_star)
    @test out.alpha_eff_taylor ≈ 1 / 3 atol = 1e-12
    # Also verify T1 ≈ 2*T2 componentwise (Euler's identity).
    fx_star = f(x_star)
    @test isapprox(out.T1, 2 .* fx_star; atol = 1e-10)
    @test isapprox(out.T2, fx_star; atol = 1e-10)
end

# ─── Holling-II: cleared vs original Hessian differ ──────────────────────────

@testset "Holling-II cleared-vs-original Hessian" begin
    # Original 1-species per-capita rate:
    #   f(x) = r + a * x / (1 + h * x) - b * x
    # Cleared polynomial (multiply by 1 + h*x):
    #   g(x) = (r - b*x) * (1 + h*x) + a * x
    # At x* = 1.0, r = 0.2, a = 1.0, b = 0.3, h = 0.5:
    r, a, b, h = 0.2, 1.0, 0.3, 0.5
    x_star = [1.0]

    f(x) = [r + a * x[1] / (1 + h * x[1]) - b * x[1]]
    g(x) = [(r - b * x[1]) * (1 + h * x[1]) + a * x[1]]

    # Hand-derived: f''(x) = -2 a h / (1 + h x)^3
    # g''(x) = -2 b h  (constant, no x-dependence)
    fpp_analytic = -2 * a * h / (1 + h * x_star[1])^3
    gpp_analytic = -2 * b * h

    H_f = ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(f, y)), x_star)[1, 1]
    H_g = ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(g, y)), x_star)[1, 1]

    @test isapprox(H_f, fpp_analytic; atol = 1e-10)
    @test isapprox(H_g, gpp_analytic; atol = 1e-10)
    @test !isapprox(H_f, H_g; atol = 1e-4)  # must differ meaningfully

    # Sanity: compute_alpha_eff_taylor runs on the original without error.
    out_f = compute_alpha_eff_taylor(f, x_star)
    @test 0.0 <= out_f.alpha_eff_taylor <= 1.0
end

# ─── Dispatch smoke tests: one JSON per model ────────────────────────────────

const MODEL_RUNS = joinpath(@__DIR__, "..", "model_runs")

function _first_json_in(dirname::AbstractString)
    dir = joinpath(MODEL_RUNS, dirname)
    isdir(dir) || return nothing
    for f in sort(readdir(dir))
        endswith(f, ".json") || continue
        occursin("chunk", f) && continue
        return joinpath(dir, f)
    end
    return nothing
end

function _load_bank(path::AbstractString)
    return JSON3.read(read(path, String), Dict{String,Any})
end

@testset "dispatch: published-model smoke tests" begin
    cases = [
        ("lever_n4.6.8.10_50models_128dirs_seed1",       :scalar),
        ("karatayev_FMI_n4.6.8.10_50models_128dirs_seed1", :scalar),
        ("karatayev_RMI_n4.6.8.10_50models_128dirs_seed1", :scalar),
        ("aguade_n4.6.8.10_50models_128dirs_seed1",      :scalar),
        ("mougi_random_n4.6.8.9_50models_128dirs_seed1", :scalar),
        ("stouffer_random_n4.6.8.10_50models_128dirs_seed1", :scalar),
    ]
    for (dirname, _) in cases
        path = _first_json_in(dirname)
        path === nothing && (@info "skipping (no data)" dirname; continue)
        bank = _load_bank(path)
        f, x_star, n = build_per_capita_rates(bank)
        out = compute_alpha_eff_taylor(f, x_star)
        @test isfinite(out.alpha_eff_taylor)
        @test 0.0 <= out.alpha_eff_taylor <= 1.0
    end
end

@testset "dispatch: Gibbs smoke test" begin
    path = _first_json_in("gibbs_128_dirs_from_gibbs_refgrid_n4to10_50reps_seed12345")
    if path !== nothing
        bank = _load_bank(path)
        f, x_star, _ = build_per_capita_rates(bank)
        out = compute_alpha_eff_taylor(f, x_star)
        @test isfinite(out.alpha_eff_taylor)
        @test 0.0 <= out.alpha_eff_taylor <= 1.0
    end
end

@testset "dispatch: standard (α-grid) smoke test" begin
    path = _first_json_in("2_bank_standard_50_models_n_4-20_128_dirs_muB_0.0")
    if path !== nothing
        bank = _load_bank(path)
        αgrid = collect(Float64, bank["alpha_grid"])
        grid_result = Float64[]
        for α in αgrid
            f, x_star, _ = build_per_capita_rates_for_alpha(bank, α)
            out = compute_alpha_eff_taylor(f, x_star)
            push!(grid_result, out.alpha_eff_taylor)
        end
        @test length(grid_result) == length(αgrid)
        @test all(isfinite, grid_result)
        @test all(x -> 0.0 <= x <= 1.0, grid_result)
        # α=0 entry must give 0 (pure linear: H=0, Q=0).
        # α=1 entry must give 1/3 (pure HOI: T1=2f, T2=f via Euler's identity).
        i0 = findfirst(==(0.0), αgrid)
        i1 = findfirst(==(1.0), αgrid)
        i0 === nothing || @test grid_result[i0] ≈ 0.0 atol = 1e-12
        i1 === nothing || @test grid_result[i1] ≈ 1 / 3 atol = 1e-10
    end
end

@testset "dispatch: r-less GLV+HOI (all_negative, unique_equilibrium)" begin
    # These banks don't store r — it's derived per-α from A, B to pin
    # x* = ones(n) as equilibrium.  Verify the builder handles this and
    # that the α=0 and α=1 edge cases still hold.
    for dirname in (
        "2_bank_all_negative_50_models_n_4-10_128_dirs_sA_1.0_sB_1.0",
        "2_bank_unique_equilibrium_50_models_n_4-10_128_dirs",
    )
        path = _first_json_in(dirname)
        path === nothing && (@info "skipping (no data)" dirname; continue)
        bank = _load_bank(path)
        @test !haskey(bank, "r")  # confirms the branch is actually exercised
        αgrid = collect(Float64, bank["alpha_grid"])
        results = Float64[]
        for α in αgrid
            f, x_star, _ = build_per_capita_rates_for_alpha(bank, α)
            out = compute_alpha_eff_taylor(f, x_star)
            @test isfinite(out.alpha_eff_taylor)
            push!(results, out.alpha_eff_taylor)
        end
        i0 = findfirst(==(0.0), αgrid)
        i1 = findfirst(==(1.0), αgrid)
        i0 === nothing || @test results[i0] ≈ 0.0 atol = 1e-12
        i1 === nothing || @test results[i1] ≈ 1 / 3 atol = 1e-10
    end
end

@testset "dispatch errors" begin
    glvhoi_bank = Dict{String,Any}(
        "dynamics_mode" => "standard",
        "x_star" => [1.0],
    )
    @test_throws ErrorException build_per_capita_rates(glvhoi_bank)

    unknown = Dict{String,Any}("dynamics_mode" => "nope", "x_star" => [1.0])
    @test_throws ErrorException build_per_capita_rates(unknown)
end
