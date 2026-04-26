#!/usr/bin/env julia

using Test
using Random
using Distributions
using HomotopyContinuation

include(joinpath(@__DIR__, "..", "boundary_scan.jl"))

matrix_to_nested(M::AbstractMatrix{<:Real}) =
    [collect(@view M[i, :]) for i in 1:size(M, 1)]

function handbuilt_stouffer_chain()
    adj = falses(2, 2)
    adj[2, 1] = true
    basal_mask = Bool[true, false]

    w = zeros(Float64, 2, 2)
    w[2, 1] = 0.1

    M = [1.0, exp(2.0)]
    x = [0.0, (STOUFFER_DEFAULT_AX / STOUFFER_DEFAULT_AR) * (M[2] / STOUFFER_DEFAULT_Mb)^(-0.25)]
    y = [0.0, STOUFFER_DEFAULT_AY / STOUFFER_DEFAULT_AX]
    e = zeros(Float64, 2, 2)
    e[2, 1] = STOUFFER_ASSIMILATION

    p = StoufferParams(0.10, adj, basal_mask, w, M, x, y, e)

    B1 = p.B0 / (p.w[2, 1] * (p.y[2] - 1.0))
    B2 = p.e[2, 1] * (1.0 - B1 / p.K) / p.x[2]
    x_star = [B1, B2]
    return p, x_star
end

function first_accepted_stouffer(seed::Int; n::Int = 6, max_tries::Int = 500)
    rng = MersenneTwister(seed)
    x0_dist = Uniform(0.05, 1.0)

    for _ in 1:max_tries
        p = sample_stouffer_params(n;
            rng=rng,
            mcmc_burnin=250,
            mcmc_steps=1000,
            proposal_sd=1.0,
        )
        x0 = rand(rng, x0_dist, n)
        res = integrate_stouffer_to_steady(p, x0; tmax=10_000.0, steady_tol=1e-8)
        res.success || continue

        x_eq = res.x_eq
        refined = integrate_stouffer_to_steady(p, x_eq; tmax=20_000.0, steady_tol=1e-10)
        if refined.success
            x_eq = refined.x_eq
        end
        all(x_eq .> 1e-6) || continue
        stouffer_lambda_max_equilibrium(p, x_eq) < 0.0 || continue
        return p, x_eq
    end

    error("Failed to sample a feasible stable Stouffer system within $(max_tries) attempts")
end

function stouffer_payload(p::StoufferParams,
                          x_star::AbstractVector{<:Real})
    pp = stouffer_params_payload(p)
    U = zeros(Float64, p.n, 1)
    U[1, 1] = 1.0
    return Dict{String, Any}(
        "dynamics_mode"         => "stouffer",
        "n"                     => p.n,
        "n_dirs"                => 1,
        "model_idx"             => 1,
        "r"                     => collect(stouffer_baseline_r(p)),
        "x_star"                => collect(x_star),
        "alpha_eff"             => Float64(stouffer_alpha_eff_symbolic(p, x_star)),
        "U"                     => matrix_to_nested(U),
        "stouffer_n"            => pp.stouffer_n,
        "stouffer_connectance"  => pp.stouffer_connectance,
        "stouffer_adj"          => pp.stouffer_adj,
        "stouffer_basal_mask"   => collect(pp.stouffer_basal_mask),
        "stouffer_w"            => pp.stouffer_w,
        "stouffer_M"            => collect(pp.stouffer_M),
        "stouffer_x"            => collect(pp.stouffer_x),
        "stouffer_y"            => collect(pp.stouffer_y),
        "stouffer_e"            => pp.stouffer_e,
        "stouffer_K"            => pp.stouffer_K,
        "stouffer_B0"           => pp.stouffer_B0,
        "stouffer_Mb"           => pp.stouffer_Mb,
        "stouffer_ar"           => pp.stouffer_ar,
        "stouffer_ax"           => pp.stouffer_ax,
        "stouffer_ay"           => pp.stouffer_ay,
    )
end

@testset "Stouffer payload round-trip and niche-web validity" begin
    p = sample_stouffer_params(6;
        rng=MersenneTwister(1),
        mcmc_burnin=200,
        mcmc_steps=800,
    )
    p2 = stouffer_params_from_payload(stouffer_params_payload(p))

    @test p2.n == p.n
    @test p2.connectance == p.connectance
    @test p2.adj == p.adj
    @test p2.basal_mask == p.basal_mask
    @test p2.prey_lists == p.prey_lists
    @test p2.pred_lists == p.pred_lists
    @test p2.w ≈ p.w
    @test p2.M ≈ p.M
    @test p2.x ≈ p.x
    @test p2.y ≈ p.y
    @test p2.e ≈ p.e
    @test p2.K == p.K
    @test p2.B0 == p.B0
    @test p2.Mb == p.Mb
    @test p2.ar == p.ar
    @test p2.ax == p.ax
    @test p2.ay == p.ay

    @test any(p.basal_mask)
    @test any(.!p.basal_mask)
    for i in 1:p.n
        if p.basal_mask[i]
            @test isempty(p.prey_lists[i])
        else
            @test !isempty(p.prey_lists[i])
            @test _stouffer_consumer_has_basal_path(p.adj, p.basal_mask, i)
        end
    end
end

@testset "Stouffer body-mass sampler tracks target ratios broadly" begin
    rng = MersenneTwister(2)
    ratios = Float64[]

    for _ in 1:12
        p = sample_stouffer_params(6;
            rng=rng,
            mcmc_burnin=200,
            mcmc_steps=800,
        )
        @test all(p.M[p.basal_mask] .== 1.0)

        for predator in 1:p.n, prey in p.prey_lists[predator]
            push!(ratios, log(p.M[predator] / p.M[prey]))
        end
    end

    @test !isempty(ratios)
    @test 3.0 <= mean(ratios) <= 9.5
    @test 8.0 <= var(ratios) <= 60.0
end

@testset "Stouffer fixed chain equations and alpha" begin
    p, x_star = handbuilt_stouffer_chain()

    f! = make_stouffer_rhs(p, zeros(p.n), 0.0)
    dx = zeros(Float64, p.n)
    f!(dx, x_star, nothing, 0.0)
    @test maximum(abs.(dx)) < 1e-10

    res = integrate_stouffer_to_steady(p, [0.5, 1.0]; tmax=10_000.0, steady_tol=1e-8)
    @test res.success
    @test maximum(abs.(res.x_eq .- x_star)) < 1e-6
    @test res.du_max < 1e-8

    lam = stouffer_lambda_max_equilibrium(p, x_star)
    @test isfinite(lam)
    @test lam < 0.0

    syst, _ = build_stouffer_cleared_system(p)
    compiled = CompiledSystem(syst)
    vals = zeros(ComplexF64, p.n)
    evaluate!(vals, compiled, ComplexF64.(x_star), ComplexF64.(zeros(p.n)))
    @test maximum(abs.(vals)) < 1e-8

    alpha_eff = stouffer_alpha_eff_symbolic(p, x_star)
    alpha_mono = stouffer_alpha_eff_monomial(p, x_star)
    @test isfinite(alpha_eff)
    @test 0.0 <= alpha_eff <= 1.0
    @test isapprox(alpha_mono, symbolic_alpha_eff_monomial_abs(syst, x_star).alpha_eff;
        atol=1e-12, rtol=1e-12)
end

@testset "Stouffer generator-facing payload and HC dispatch smoke" begin
    p, x_star = first_accepted_stouffer(11)
    payload = stouffer_payload(p, x_star)

    expected = Set([
        "dynamics_mode", "n", "n_dirs", "model_idx", "r", "x_star", "alpha_eff", "U",
        "stouffer_n", "stouffer_connectance", "stouffer_adj", "stouffer_basal_mask",
        "stouffer_w", "stouffer_M", "stouffer_x", "stouffer_y", "stouffer_e",
        "stouffer_K", "stouffer_B0", "stouffer_Mb", "stouffer_ar",
        "stouffer_ax", "stouffer_ay",
    ])
    @test Set(keys(payload)) == expected

    p2 = stouffer_params_from_payload(payload)
    @test stouffer_baseline_r(p2) ≈ Float64.(payload["r"])

    syst, _ = build_stouffer_cleared_system(p2)
    compiled = CompiledSystem(syst)
    vals = zeros(ComplexF64, p2.n)
    evaluate!(vals, compiled, ComplexF64.(x_star), ComplexF64.(zeros(p2.n)))
    @test maximum(abs.(vals)) < 1e-6

    ctx = build_hc_system(payload)
    @test ctx.n == p.n
    @test ctx.n_dirs == 1
    @test ctx.alpha_grid == [payload["alpha_eff"]]
    ws = ctx.make_workspace(payload["alpha_eff"])
    @test ws.compiled_system isa CompiledSystem
end
