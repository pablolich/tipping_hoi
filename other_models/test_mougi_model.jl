#!/usr/bin/env julia

using Test
using Random
using HomotopyContinuation

include(joinpath(@__DIR__, "mougi_model.jl"))

function sample_stable_mougi(seed::Int)
    rng = MersenneTwister(seed)
    x0  = fill(0.5, 6)

    for _ in 1:1000
        p = sample_mougi_params(6; rng=rng)
        res = integrate_mougi_to_steady(p, x0; tmax=4000.0, steady_tol=1e-8)
        res.success || continue
        x_eq = res.x_eq
        all(x_eq .> 1e-6) || continue
        mougi_lambda_max_equilibrium(p, x_eq) < 0.0 || continue
        return p, x_eq
    end

    error("Failed to sample a feasible stable Mougi system within 1000 attempts")
end

@testset "Mougi payload round-trip" begin
    p  = sample_mougi_params(6; rng=MersenneTwister(1))
    p2 = mougi_params_from_payload(mougi_params_payload(p))

    @test p2.n == p.n
    @test p2.connectance == p.connectance
    @test p2.topology_mode == p.topology_mode
    @test p2.r0 ≈ p.r0
    @test p2.s ≈ p.s
    @test p2.e ≈ p.e
    @test p2.a ≈ p.a
    @test p2.h ≈ p.h
    @test p2.E0 ≈ p.E0
    @test p2.engineer_mask == p.engineer_mask
    @test p2.receiver_mask == p.receiver_mask
    @test p2.betaE ≈ p.betaE
    @test p2.gammaE ≈ p.gammaE
    @test p2.q_r == p.q_r
    @test p2.q_a == p.q_a
end

@testset "No-engineer limit reduces to baseline self-regulation" begin
    n = 4
    p = MougiParams(
        n,
        1.0,
        "random",
        fill(0.8, n),
        ones(Float64, n),
        zeros(Float64, n, n),
        zeros(Float64, n, n),
        zeros(Float64, n, n),
        fill(0.1, n),
        collect(falses(n)),
        collect(falses(n)),
        ones(Float64, n, n),
        ones(Float64, n, n),
        0.2,
        0.8,
    )

    x = fill(0.5, n)
    B, G = mougi_engineering_factors(p, x)
    @test B == ones(n)
    @test G == ones(n)

    f! = make_mougi_rhs(p, zeros(n), 0.0)
    dx = zeros(Float64, n)
    f!(dx, x, nothing, 0.0)
    @test dx ≈ x .* (p.r0 .- p.s .* x) atol=1e-12 rtol=1e-12
end

@testset "Stable sample has negative lambda and consistent cleared system" begin
    p, x_eq = sample_stable_mougi(2)

    lam = mougi_lambda_max_equilibrium(p, x_eq)
    @test isfinite(lam)
    @test lam < 0.0

    alpha_eff = mougi_alpha_eff_symbolic(p, x_eq)
    @test isfinite(alpha_eff)
    @test 0.0 <= alpha_eff <= 1.0

    syst, _ = build_mougi_cleared_system(p)
    compiled = CompiledSystem(syst)
    vals = zeros(ComplexF64, p.n)
    evaluate!(vals, compiled, ComplexF64.(x_eq), ComplexF64.(zeros(p.n)))
    @test maximum(abs.(vals)) < 1e-6
end
