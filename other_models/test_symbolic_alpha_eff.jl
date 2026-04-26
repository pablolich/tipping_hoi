#!/usr/bin/env julia

using Test
using Random
using HomotopyContinuation

include(joinpath(@__DIR__, "lever_model.jl"))
include(joinpath(@__DIR__, "karatayev_model.jl"))
include(joinpath(@__DIR__, "aguade_model.jl"))

matrix_to_nested(M::AbstractMatrix{<:Real}) = [collect(@view M[i, :]) for i in 1:size(M, 1)]

function aguade_alpha_eff_reference(p::AguadeParams,
                                    x_star::AbstractVector{<:Real};
                                    monomial_abs::Bool)
    n    = p.n
    base = n + 2

    P_total = 0.0
    H_total = 0.0

    for i in 1:n
        mono = Dict{Int, Float64}()

        function add_monomial!(exp_vec, coef)
            abs(coef) < 1e-300 && return
            key = 0
            pw  = 1
            for k in 1:n
                key += exp_vec[k] * pw
                pw  *= base
            end
            mono[key] = get(mono, key, 0.0) + coef
        end

        for S_bits in 0:(1 << n - 1)
            coef    = -p.d[i]
            exp_vec = zeros(Int, n)
            for k in 1:n
                if (S_bits >> (k - 1)) & 1 == 1
                    exp_vec[k] = 1
                else
                    coef *= p.gamma[k]
                end
            end
            add_monomial!(exp_vec, coef)
        end

        for j in 1:n
            others = [k for k in 1:n if k != j]
            for S_bits in 0:(1 << (n - 1) - 1)
                coef    = p.A[i, j]
                exp_vec = zeros(Int, n)
                exp_vec[j] = 1
                for b in 1:(n - 1)
                    k = others[b]
                    if (S_bits >> (b - 1)) & 1 == 1
                        exp_vec[k] += 1
                    else
                        coef *= p.gamma[k]
                    end
                end
                add_monomial!(exp_vec, coef)
            end
        end

        for j in 1:n
            for S_bits in 0:(1 << n - 1)
                coef    = -p.B[i, j]
                exp_vec = zeros(Int, n)
                exp_vec[j] += 1
                for k in 1:n
                    if (S_bits >> (k - 1)) & 1 == 1
                        exp_vec[k] += 1
                    else
                        coef *= p.gamma[k]
                    end
                end
                add_monomial!(exp_vec, coef)
            end
        end

        P_i = 0.0
        H_i = 0.0
        for (key, coef) in mono
            val = coef
            kk  = key
            deg = 0
            for k in 1:n
                e    = kk % base
                val *= x_star[k]^e
                deg += e
                kk  ÷= base
            end
            if deg == 1
                P_i += monomial_abs ? abs(val) : val
            elseif deg >= 2
                H_i += monomial_abs ? abs(val) : val
            end
        end

        if monomial_abs
            P_total += P_i
            H_total += H_i
        else
            P_total += abs(P_i)
            H_total += abs(H_i)
        end
    end

    P_total /= n
    H_total /= n
    denom = P_total + H_total
    return denom > 0.0 ? H_total / denom : NaN
end

function find_stable_lever_sample(seed::Int)
    rng = MersenneTwister(seed)
    for _ in 1:100
        p = sample_lever_original_params(3, 3; rng=rng)
        x0 = ones(Float64, 6)
        res = integrate_lever_to_steady(x0, p; tmax=4000.0, steady_tol=1e-8)
        res.success || continue
        x_eq = res.x_eq
        all(x_eq .> 1e-6) || continue
        lever_lambda_max_equilibrium(p, x_eq) < 0.0 || continue
        return p, x_eq
    end
    error("No stable Lever sample found")
end

function find_stable_karatayev_sample(seed::Int)
    rng = MersenneTwister(seed)
    for _ in 1:100
        p = sample_karatayev_params(3, 3; feedback_mode="RMI", rng=rng)
        x0 = vcat(fill(0.5, 3), fill(0.15, 3))
        res = integrate_karatayev_to_steady(x0, p; tmax=2000.0, steady_tol=1e-8)
        res.success || continue
        x_eq = res.x_eq
        all(x_eq .> 1e-6) || continue
        karatayev_lambda_max_equilibrium(p, x_eq) < 0.0 || continue
        return p, x_eq
    end
    error("No stable Karatayev sample found")
end

function find_stable_aguade_sample(seed::Int)
    rng = MersenneTwister(seed)
    x0 = fill(0.5, 6)
    for _ in 1:250
        p = sample_aguade_params(6; rng=rng)
        res = integrate_aguade_to_steady(p, x0; tmax=4000.0, steady_tol=1e-8)
        res.success || continue
        x_eq = res.x_eq
        all(x_eq .> 1e-1) || continue
        aguade_lambda_max_equilibrium(p, x_eq) < 0.0 || continue
        return p, x_eq
    end
    error("No stable Aguade sample found")
end

@testset "symbolic_alpha_eff grouped and monomial metrics" begin
    @var x[1:2] dr[1:2]
    syst = System([
        7 + 2 * x[1] - 4 * x[2] + x[1]^2 - 5 * x[1] * x[2] + dr[1] * x[2]
    ]; variables=x, parameters=dr)

    grouped = symbolic_alpha_eff(syst, [2.0, 3.0]; parameter_values=[0.0, 0.0])
    mono    = symbolic_alpha_eff_monomial_abs(syst, [2.0, 3.0]; parameter_values=[0.0, 0.0])

    @test grouped.P_eff ≈ 8.0
    @test grouped.H_eff ≈ 26.0
    @test grouped.alpha_eff ≈ 26.0 / 34.0

    @test mono.P_eff ≈ 16.0
    @test mono.H_eff ≈ 34.0
    @test mono.alpha_eff ≈ 34.0 / 50.0
end

@testset "symbolic_alpha_eff rejects unresolved symbolic coefficients" begin
    @var x[1:1] c
    expr = expand(c * x[1])
    _, coeffs = exponents_coefficients(expr, x)
    @test_throws ErrorException _symbolic_alpha_eff_coefficient(coeffs[1], expr, 1)
end

@testset "Aguade grouped symbolic wrapper matches grouped reference" begin
    for seed in 1:3
        p      = sample_aguade_params(4; rng=MersenneTwister(seed))
        x_star = rand(MersenneTwister(100 + seed), 4) .+ 0.2
        @test aguade_alpha_eff_symbolic(p, x_star) ≈
              aguade_alpha_eff_reference(p, x_star; monomial_abs=false) atol=1e-12 rtol=1e-12
    end
end

@testset "Aguade monomial wrapper matches monomial reference" begin
    for seed in 1:3
        p      = sample_aguade_params(4; rng=MersenneTwister(seed))
        x_star = rand(MersenneTwister(100 + seed), 4) .+ 0.2
        @test aguade_alpha_eff_monomial(p, x_star) ≈
              aguade_alpha_eff_reference(p, x_star; monomial_abs=true) atol=1e-12 rtol=1e-12
    end
end

@testset "Lever and Karatayev grouped symbolic match legacy grouped metric" begin
    p_lever   = sample_lever_original_params(3, 3; rng=MersenneTwister(1))
    x_lever   = rand(MersenneTwister(2), 6) .+ 0.2
    _, A_l, B_l = lever_effective_coefficients(p_lever)
    old_lever = effective_interaction_metrics(A_l, B_l, x_lever).alpha_eff
    new_lever = lever_alpha_eff_symbolic(p_lever, x_lever)
    @test new_lever ≈ old_lever atol=1e-12 rtol=1e-12

    p_karat   = sample_karatayev_params(3, 3; feedback_mode="RMI", rng=MersenneTwister(3))
    x_karat   = rand(MersenneTwister(4), 6) .+ 0.2
    _, A_k, B_k = karatayev_effective_coefficients(p_karat)
    old_karat = effective_interaction_metrics(A_k, B_k, x_karat).alpha_eff
    new_karat = karatayev_alpha_eff_symbolic(p_karat, x_karat)
    @test new_karat ≈ old_karat atol=1e-12 rtol=1e-12
end

@testset "Explicit monomial helpers match system-level monomial metric" begin
    p_lever   = sample_lever_original_params(3, 3; rng=MersenneTwister(7))
    x_lever   = rand(MersenneTwister(8), 6) .+ 0.2
    syst_lever, _ = build_lever_cleared_system(p_lever)
    @test lever_alpha_eff_monomial(p_lever, x_lever) ≈
          symbolic_alpha_eff_monomial_abs(syst_lever, x_lever).alpha_eff atol=1e-12 rtol=1e-12

    p_karat   = sample_karatayev_params(3, 3; feedback_mode="FMI", rng=MersenneTwister(9))
    x_karat   = rand(MersenneTwister(10), 6) .+ 0.2
    syst_karat, _ = build_karatayev_alpha_eff_system(p_karat)
    @test karatayev_alpha_eff_monomial(p_karat, x_karat) ≈
          symbolic_alpha_eff_monomial_abs(syst_karat, x_karat).alpha_eff atol=1e-12 rtol=1e-12
end

@testset "Generator-facing smoke validation" begin
    expected_lever = Set([
        "n", "n_dirs", "model_idx", "dynamics_mode", "x_star", "r", "alpha_eff", "U",
        "lever_Sp", "lever_Sa", "lever_t", "lever_muP", "lever_muA", "lever_dA",
        "lever_rP", "lever_rA", "lever_hP", "lever_hA", "lever_CP", "lever_CA", "lever_GP", "lever_GA",
    ])
    expected_karat = Set([
        "n", "n_dirs", "model_idx", "dynamics_mode", "feedback_mode", "x_star", "r", "alpha_eff", "U",
        "karat_n_R", "karat_n_C", "karat_r", "karat_K", "karat_f", "karat_delta", "karat_b", "karat_beta",
        "karat_m0", "karat_a", "karat_specials", "karat_comps", "karat_feedback_mode",
    ])
    expected_aguade = Set([
        "dynamics_mode", "n", "n_dirs", "model_idx", "r", "x_star", "alpha_eff", "U",
        "aguade_n", "aguade_A", "aguade_B", "aguade_d", "aguade_gamma",
    ])

    let
        p, x_eq = find_stable_lever_sample(11)
        alpha_eff = lever_alpha_eff_symbolic(p, x_eq)
        U = zeros(Float64, 6, 1)
        U[6, 1] = 1.0
        pp = lever_params_payload(p)
        payload = Dict{String, Any}(
            "n"            => 6,
            "n_dirs"       => 1,
            "model_idx"    => 1,
            "dynamics_mode" => "lever",
            "x_star"       => collect(x_eq),
            "r"            => collect(vcat(copy(p.rP), copy(p.rA))),
            "alpha_eff"    => Float64(alpha_eff),
            "U"            => matrix_to_nested(U),
            "lever_Sp"     => 3,
            "lever_Sa"     => 3,
            "lever_t"      => pp.lever_t,
            "lever_muP"    => pp.lever_muP,
            "lever_muA"    => pp.lever_muA,
            "lever_dA"     => pp.lever_dA,
            "lever_rP"     => collect(pp.lever_rP),
            "lever_rA"     => collect(pp.lever_rA),
            "lever_hP"     => collect(pp.lever_hP),
            "lever_hA"     => collect(pp.lever_hA),
            "lever_CP"     => matrix_to_nested(pp.lever_CP),
            "lever_CA"     => matrix_to_nested(pp.lever_CA),
            "lever_GP"     => matrix_to_nested(pp.lever_GP),
            "lever_GA"     => matrix_to_nested(pp.lever_GA),
        )
        @test isfinite(alpha_eff)
        @test Set(keys(payload)) == expected_lever
    end

    let
        p, x_eq = find_stable_karatayev_sample(17)
        alpha_eff = karatayev_alpha_eff_symbolic(p, x_eq)
        U = zeros(Float64, 6, 1)
        U[6, 1] = 1.0
        pp = karatayev_params_payload(p)
        payload = Dict{String, Any}(
            "n"                    => 6,
            "n_dirs"               => 1,
            "model_idx"            => 1,
            "dynamics_mode"        => "karatayev",
            "feedback_mode"        => "RMI",
            "x_star"               => collect(x_eq),
            "r"                    => collect(vcat(copy(p.r), fill(-p.m0, 3))),
            "alpha_eff"            => Float64(alpha_eff),
            "U"                    => matrix_to_nested(U),
            "karat_n_R"            => 3,
            "karat_n_C"            => 3,
            "karat_r"              => collect(pp.karat_r),
            "karat_K"              => collect(pp.karat_K),
            "karat_f"              => collect(pp.karat_f),
            "karat_delta"          => collect(pp.karat_delta),
            "karat_b"              => collect(pp.karat_b),
            "karat_beta"           => collect(pp.karat_beta),
            "karat_m0"             => pp.karat_m0,
            "karat_a"              => pp.karat_a,
            "karat_specials"       => matrix_to_nested(pp.karat_specials),
            "karat_comps"          => matrix_to_nested(pp.karat_comps),
            "karat_feedback_mode"  => pp.karat_feedback_mode,
        )
        @test isfinite(alpha_eff)
        @test Set(keys(payload)) == expected_karat
    end

    let
        p, x_eq = find_stable_aguade_sample(23)
        alpha_eff = aguade_alpha_eff_symbolic(p, x_eq)
        U = zeros(Float64, 6, 1)
        U[1, 1] = 1.0
        pp = aguade_params_payload(p)
        payload = Dict{String, Any}(
            "dynamics_mode" => "aguade",
            "n"             => 6,
            "n_dirs"        => 1,
            "model_idx"     => 1,
            "r"             => collect(-copy(p.d)),
            "x_star"        => collect(x_eq),
            "alpha_eff"     => Float64(alpha_eff),
            "U"             => matrix_to_nested(U),
            "aguade_n"      => 6,
            "aguade_A"      => pp.aguade_A,
            "aguade_B"      => pp.aguade_B,
            "aguade_d"      => collect(pp.aguade_d),
            "aguade_gamma"  => collect(pp.aguade_gamma),
        )
        @test isfinite(alpha_eff)
        @test Set(keys(payload)) == expected_aguade
    end
end
