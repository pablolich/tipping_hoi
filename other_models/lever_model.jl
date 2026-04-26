# Utilities for the Lever-style plant-pollinator model.
# Used by generate_bank_lever.jl (bank generation) and glvhoi_utils.jl (HC dispatch).
#
# Species ordering: [plants..., pollinators...].
# Immigration fixed to zero (muP = muA = 0). dA fixed to 0.0 in bank generation.

using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using SciMLBase
using HomotopyContinuation

struct LeverOriginalParams
    Sp::Int
    Sa::Int
    rP::Vector{Float64}
    rA::Vector{Float64}
    hP::Vector{Float64}
    hA::Vector{Float64}
    CP::Matrix{Float64}
    CA::Matrix{Float64}
    GP::Matrix{Float64}
    GA::Matrix{Float64}
    muP::Float64
    muA::Float64
    dA::Float64
    t::Float64
end

@inline lever_n_species(p::LeverOriginalParams) = p.Sp + p.Sa

function _dict_or_prop_get(x, key::AbstractString)
    if x isa AbstractDict
        if haskey(x, key)
            return x[key]
        elseif haskey(x, Symbol(key))
            return x[Symbol(key)]
        else
            error("Missing key in bank payload: $(key)")
        end
    end
    sym = Symbol(key)
    if hasproperty(x, sym)
        return getproperty(x, sym)
    end
    error("Missing property in bank payload: $(key)")
end

function _lever_dict_get_with_default(d::AbstractDict, key::AbstractString, default)
    if haskey(d, key)
        return d[key]
    elseif haskey(d, Symbol(key))
        return d[Symbol(key)]
    else
        return default
    end
end

function _sample_scalar_or_uniform(spec, rng::AbstractRNG, name::AbstractString)
    if spec isa Number
        return Float64(spec)
    elseif spec isa AbstractDict
        kind = String(_lever_dict_get_with_default(spec, "kind", "uniform"))
        kind == "uniform" || error("$(name) range object only supports kind='uniform'")

        lo_raw = _lever_dict_get_with_default(spec, "low", _lever_dict_get_with_default(spec, "min", nothing))
        hi_raw = _lever_dict_get_with_default(spec, "high", _lever_dict_get_with_default(spec, "max", nothing))
        lo_raw === nothing && error("$(name) range object missing low/min")
        hi_raw === nothing && error("$(name) range object missing high/max")

        lo = Float64(lo_raw)
        hi = Float64(hi_raw)
        isfinite(lo) || error("$(name) low must be finite")
        isfinite(hi) || error("$(name) high must be finite")
        lo <= hi || error("$(name) requires low <= high, got low=$(lo), high=$(hi)")
        return lo == hi ? lo : rand(rng, Uniform(lo, hi))
    else
        error("$(name) must be a number or a range object")
    end
end

function lever_params_payload(p::LeverOriginalParams)
    return (
        lever_Sp  = p.Sp,
        lever_Sa  = p.Sa,
        lever_t   = p.t,
        lever_muP = p.muP,
        lever_muA = p.muA,
        lever_dA  = p.dA,
        lever_rP  = copy(p.rP),
        lever_rA  = copy(p.rA),
        lever_hP  = copy(p.hP),
        lever_hA  = copy(p.hA),
        lever_CP  = copy(p.CP),
        lever_CA  = copy(p.CA),
        lever_GP  = copy(p.GP),
        lever_GA  = copy(p.GA),
    )
end

# Convert a value to Matrix{Float64}: handles both Julia matrices and JSON-decoded
# nested arrays (Vector{<:AbstractVector}).
function _to_matrix_f64(x)
    if x isa AbstractMatrix
        return Matrix{Float64}(x)
    end
    # nested array from JSON
    rows = x
    m = length(rows)
    m == 0 && return zeros(Float64, 0, 0)
    n = length(rows[1])
    M = Matrix{Float64}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = Float64(rows[i][j])
    end
    return M
end

function lever_params_from_payload(payload)
    Sp = Int(_dict_or_prop_get(payload, "lever_Sp"))
    Sa = Int(_dict_or_prop_get(payload, "lever_Sa"))
    return LeverOriginalParams(
        Sp,
        Sa,
        Vector{Float64}(_dict_or_prop_get(payload, "lever_rP")),
        Vector{Float64}(_dict_or_prop_get(payload, "lever_rA")),
        Vector{Float64}(_dict_or_prop_get(payload, "lever_hP")),
        Vector{Float64}(_dict_or_prop_get(payload, "lever_hA")),
        _to_matrix_f64(_dict_or_prop_get(payload, "lever_CP")),
        _to_matrix_f64(_dict_or_prop_get(payload, "lever_CA")),
        _to_matrix_f64(_dict_or_prop_get(payload, "lever_GP")),
        _to_matrix_f64(_dict_or_prop_get(payload, "lever_GA")),
        Float64(_dict_or_prop_get(payload, "lever_muP")),
        Float64(_dict_or_prop_get(payload, "lever_muA")),
        Float64(_dict_or_prop_get(payload, "lever_dA")),
        Float64(_dict_or_prop_get(payload, "lever_t")),
    )
end

function sample_lever_original_params(Sp::Int, Sa::Int;
                                      t   = 0.5,
                                      dA0 = 0.0,
                                      h   = nothing,
                                      rng::AbstractRNG = Random.default_rng())
    adj = trues(Sp, Sa)
    t_sampled   = _sample_scalar_or_uniform(t,   rng, "t")
    dA0_sampled = _sample_scalar_or_uniform(dA0, rng, "dA0")
    h_spec = h === nothing ? Dict{String,Any}("kind" => "uniform", "low" => 0.15, "high" => 0.30) : h

    rP = rand(rng, Uniform(0.05, 0.35), Sp)
    rA = rand(rng, Uniform(0.05, 0.35), Sa)
    hP = [_sample_scalar_or_uniform(h_spec, rng, "h") for _ in 1:Sp]
    hA = [_sample_scalar_or_uniform(h_spec, rng, "h") for _ in 1:Sa]

    CP = rand(rng, Uniform(0.01, 0.05), Sp, Sp)
    CA = rand(rng, Uniform(0.01, 0.05), Sa, Sa)
    @inbounds for i in 1:Sp
        CP[i, i] = rand(rng, Uniform(0.8, 1.1))
    end
    @inbounds for k in 1:Sa
        CA[k, k] = rand(rng, Uniform(0.8, 1.1))
    end

    degP = vec(sum(adj; dims=2))
    degA = vec(sum(adj; dims=1))
    GP   = zeros(Float64, Sp, Sa)
    GA   = zeros(Float64, Sa, Sp)
    @inbounds for i in 1:Sp, k in 1:Sa
        if adj[i, k]
            γ0P = rand(rng, Uniform(0.8, 1.2))
            γ0A = rand(rng, Uniform(0.8, 1.2))
            GP[i, k] = γ0P / (degP[i]^Float64(t_sampled))
            GA[k, i] = γ0A / (degA[k]^Float64(t_sampled))
        end
    end

    return LeverOriginalParams(
        Sp, Sa,
        rP, rA,
        hP, hA,
        CP, CA,
        GP, GA,
        0.0, 0.0,
        Float64(dA0_sampled),
        Float64(t_sampled),
    )
end

function make_lever_original_rhs(p::LeverOriginalParams,
                                 u_full::AbstractVector{<:Real},
                                 delta::Real)
    n  = lever_n_species(p)
    Sp = p.Sp
    Sa = p.Sa
    dr = Float64(delta) .* Float64.(u_full)

    function f!(dx, x, _, _)
        @inbounds for i in 1:Sp
            m = 0.0; comp = 0.0
            for k in 1:Sa; m    += p.GP[i, k] * x[Sp + k]; end
            for j in 1:Sp; comp += p.CP[i, j] * x[j];      end
            benefit = m / (1 + p.hP[i] * m)
            f = (p.rP[i] + dr[i]) + benefit - comp
            dx[i] = x[i] * f + p.muP
        end
        @inbounds for k in 1:Sa
            idx = Sp + k
            m = 0.0; comp = 0.0
            for i in 1:Sp;  m    += p.GA[k, i] * x[i];         end
            for l in 1:Sa;  comp += p.CA[k, l] * x[Sp + l];    end
            benefit = m / (1 + p.hA[k] * m)
            f = (p.rA[k] + dr[idx]) - p.dA + benefit - comp
            dx[idx] = x[idx] * f + p.muA
        end
        return nothing
    end
    return f!
end

function integrate_lever_to_steady(x0::AbstractVector{<:Real},
                                   p::LeverOriginalParams;
                                   tmax::Real      = 4000.0,
                                   steady_tol::Real = 1e-8)
    n      = lever_n_species(p)
    u_zero = zeros(Float64, n)
    f!     = make_lever_original_rhs(p, u_zero, 0.0)
    u_init = Float64.(x0)
    prob   = ODEProblem(f!, u_init, (0.0, Float64(tmax)))

    condition(_, _, integrator) = begin
        du = integrator.du
        du === nothing && return false
        return maximum(abs, du) < steady_tol
    end
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!; save_positions=(false, false))

    sol = DifferentialEquations.solve(prob, Tsit5();
        callback       = cb,
        reltol         = 1e-9,
        abstol         = 1e-11,
        save_everystep = false,
        save_start     = false,
    )

    if !SciMLBase.successful_retcode(sol)
        return (success=false, retcode=string(sol.retcode), x_eq=fill(NaN, n), du_max=Inf)
    end

    x_eq = Vector{Float64}(sol.u[end])
    @. x_eq = max(x_eq, 0.0)
    du_end = similar(x_eq)
    f!(du_end, x_eq, nothing, 0.0)
    return (success=true, retcode=string(sol.retcode), x_eq=x_eq, du_max=maximum(abs.(du_end)))
end

function lever_jacobian_F(p::LeverOriginalParams,
                          x::AbstractVector{<:Real},
                          dr_full::AbstractVector{<:Real})
    n  = lever_n_species(p)
    Sp = p.Sp
    Sa = p.Sa
    JF = zeros(Float64, n, n)

    @inbounds for i in 1:Sp
        m = 0.0
        for k in 1:Sa; m += p.GP[i, k] * x[Sp + k]; end
        den2 = (1 + p.hP[i] * m)^2
        for j in 1:Sp;  JF[i, j]      = -p.CP[i, j];        end
        for k in 1:Sa;  JF[i, Sp + k] = p.GP[i, k] / den2;  end
    end

    @inbounds for k in 1:Sa
        idx = Sp + k
        m = 0.0
        for i in 1:Sp; m += p.GA[k, i] * x[i]; end
        den2 = (1 + p.hA[k] * m)^2
        for i in 1:Sp;  JF[idx, i]      = p.GA[k, i] / den2;  end
        for l in 1:Sa;  JF[idx, Sp + l] = -p.CA[k, l];         end
    end

    return JF
end

function lever_jacobian_F(p::LeverOriginalParams, x::AbstractVector{<:Real})
    return lever_jacobian_F(p, x, zeros(Float64, lever_n_species(p)))
end

function lever_lambda_max_equilibrium(p::LeverOriginalParams,
                                      x::AbstractVector{<:Real},
                                      dr_full::AbstractVector{<:Real})
    JF = lever_jacobian_F(p, x, dr_full)
    J  = Matrix{Float64}(JF)
    @inbounds for i in 1:length(x)
        @views J[i, :] .*= x[i]
    end
    return maximum(real.(eigvals(J)))
end

function lever_lambda_max_equilibrium(p::LeverOriginalParams, x::AbstractVector{<:Real})
    return lever_lambda_max_equilibrium(p, x, zeros(Float64, lever_n_species(p)))
end

function lever_effective_coefficients(p::LeverOriginalParams,
                                      dr_full::AbstractVector{<:Real})
    n  = lever_n_species(p)
    Sp = p.Sp
    Sa = p.Sa

    r_eff = zeros(Float64, n)
    Aeff  = zeros(Float64, n, n)
    Beff  = zeros(Float64, n, n, n)

    @inbounds for i in 1:Sp
        r_i     = p.rP[i] + dr_full[i]
        r_eff[i] = r_i
        for j in 1:Sp;  Aeff[i, j]      += -p.CP[i, j];                       end
        for k in 1:Sa
            idxA = Sp + k
            Aeff[i, idxA] += p.GP[i, k] * (1 + p.hP[i] * r_i)
        end
        for j in 1:Sp, k in 1:Sa
            idxA = Sp + k
            Beff[j, idxA, i] += -p.hP[i] * p.CP[i, j] * p.GP[i, k]
        end
    end

    @inbounds for k in 1:Sa
        idx  = Sp + k
        r_i  = (p.rA[k] + dr_full[idx]) - p.dA
        r_eff[idx] = r_i
        for l in 1:Sa;  Aeff[idx, Sp + l] += -p.CA[k, l];                     end
        for i in 1:Sp
            Aeff[idx, i] += p.GA[k, i] * (1 + p.hA[k] * r_i)
        end
        for l in 1:Sa, i in 1:Sp
            Beff[Sp + l, i, idx] += -p.hA[k] * p.CA[k, l] * p.GA[k, i]
        end
    end

    return r_eff, Aeff, Beff
end

function lever_effective_coefficients(p::LeverOriginalParams)
    return lever_effective_coefficients(p, zeros(Float64, lever_n_species(p)))
end

"""
    effective_interaction_metrics(Aeff, Beff, x_eq)

Legacy grouped interaction metric retained for comparison with the older
coefficient-tensor-based `alpha_eff`. Default bank generation now uses
`symbolic_alpha_eff`, which allows cancellations within each equation before
taking absolute values. The comparison-only monomial metric is available via
`symbolic_alpha_eff_monomial_abs`.
"""
function effective_interaction_metrics(Aeff::AbstractMatrix{<:Real},
                                       Beff::Array{<:Real,3},
                                       x_eq::AbstractVector{<:Real})
    n   = length(x_eq)
    lin = Matrix{Float64}(Aeff) * Float64.(x_eq)
    P_eff = sum(abs.(lin)) / n

    quad_abs_sum = 0.0
    @inbounds for i in 1:n
        q = 0.0
        for j in 1:n, k in 1:n
            q += Beff[j, k, i] * x_eq[j] * x_eq[k]
        end
        quad_abs_sum += abs(q)
    end
    H_eff  = quad_abs_sum / n
    denom  = P_eff + H_eff
    alpha_eff = denom > 0 ? (H_eff / denom) : NaN
    return (alpha_eff=alpha_eff, P_eff=P_eff, H_eff=H_eff)
end

function _symbolic_alpha_eff_coefficient(coef, expr, col_idx)
    if coef isa Real
        return Float64(coef)
    end
    try
        return Float64(coef)
    catch err
        error("symbolic_alpha_eff found a non-numeric coefficient after parameter substitution " *
              "(equation=$(expr), term_index=$(col_idx), coefficient=$(coef)): $(err)")
    end
end

"""
    _symbolic_alpha_eff_impl(syst, x_star; parameter_values, monomial_abs)
        → (alpha_eff, P_eff, H_eff)

Internal shared implementation for the grouped and monomial-absolute symbolic
`alpha_eff` metrics.
"""
function _symbolic_alpha_eff_impl(syst::System,
                                  x_star::AbstractVector{<:Real};
                                  parameter_values::AbstractVector{<:Real}=zeros(Float64, length(parameters(syst))),
                                  monomial_abs::Bool=false)
    vars = variables(syst)
    pars = parameters(syst)
    n    = length(vars)

    length(x_star) == n ||
        error("symbolic alpha_eff helper expected x_star of length $(n), got $(length(x_star))")
    length(parameter_values) == length(pars) ||
        error("symbolic alpha_eff helper expected $(length(pars)) parameter values, got $(length(parameter_values))")

    x_eval = Float64.(x_star)
    exprs  = if isempty(pars)
        expressions(syst)
    else
        subs(expressions(syst), [pars[i] => Float64(parameter_values[i]) for i in eachindex(pars)]...)
    end

    P_total = 0.0
    H_total = 0.0

    for expr in exprs
        exps, coeffs = exponents_coefficients(expand(expr), vars)
        P_i = 0.0
        H_i = 0.0

        for col in axes(exps, 2)
            deg   = 0
            x_val = 1.0
            for row in axes(exps, 1)
                e    = Int(exps[row, col])
                deg += e
                x_val *= x_eval[row]^e
            end

            deg == 0 && continue

            coef    = _symbolic_alpha_eff_coefficient(coeffs[col], expr, col)
            contrib = coef * x_val

            if deg == 1
                P_i += monomial_abs ? abs(contrib) : contrib
            else
                H_i += monomial_abs ? abs(contrib) : contrib
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

    P_eff = P_total / length(exprs)
    H_eff = H_total / length(exprs)
    denom = P_eff + H_eff
    alpha_eff = denom > 0.0 ? (H_eff / denom) : NaN
    return (alpha_eff=alpha_eff, P_eff=P_eff, H_eff=H_eff)
end

"""
    symbolic_alpha_eff(syst, x_star; parameter_values=zeros(length(parameters(syst))))
        → (alpha_eff, P_eff, H_eff)

Compute the default grouped `alpha_eff` directly from a polynomial HC system.
After parameter substitution, each equation is expanded and split into monomials
using `exponents_coefficients`. Degree-1 monomials contribute to `P`, and
degree-`>= 2` monomials contribute to `H`. Cancellations are allowed within each
equation and bucket before taking absolute values.
"""
function symbolic_alpha_eff(syst::System,
                            x_star::AbstractVector{<:Real};
                            parameter_values::AbstractVector{<:Real}=zeros(Float64, length(parameters(syst))))
    return _symbolic_alpha_eff_impl(syst, x_star;
        parameter_values=parameter_values,
        monomial_abs=false)
end

"""
    symbolic_alpha_eff_monomial_abs(syst, x_star; parameter_values=zeros(length(parameters(syst))))
        → (alpha_eff, P_eff, H_eff)

Comparison-only monomial metric on a polynomial HC system. This sums
`abs(coeff * x_star^alpha)` term-by-term before aggregating within each equation.
"""
function symbolic_alpha_eff_monomial_abs(syst::System,
                                         x_star::AbstractVector{<:Real};
                                         parameter_values::AbstractVector{<:Real}=zeros(Float64, length(parameters(syst))))
    return _symbolic_alpha_eff_impl(syst, x_star;
        parameter_values=parameter_values,
        monomial_abs=true)
end

# Denominator-cleared HC polynomial for the Lever model.
# Parameters: dr[1:n] (length n = Sp + Sa).
# Plant rows in dr are zeroed out in practice via zeros in U plant rows.
# Pollinators: equation k uses dr[Sp+k] for perturbation of rA[k].
# dA baked in at p.dA (stored as 0.0 in all bank JSONs).
function build_lever_cleared_system(p::LeverOriginalParams)
    n  = p.Sp + p.Sa
    Sp = p.Sp
    Sa = p.Sa

    @var x[1:n] dr[1:n]
    eqs = Vector{Expression}(undef, n)

    for i in 1:Sp
        m      = sum(p.GP[i, k] * x[Sp + k] for k in 1:Sa)
        comp   = sum(p.CP[i, j] * x[j]       for j in 1:Sp)
        r_eff  = p.rP[i] + dr[i]
        # cleared: (r_eff - comp) * (1 + hP*m) + m = 0
        eqs[i] = r_eff * (1 + p.hP[i] * m) + m - comp * (1 + p.hP[i] * m)
    end

    for k in 1:Sa
        idx    = Sp + k
        q      = sum(p.GA[k, i] * x[i]        for i in 1:Sp)
        comp   = sum(p.CA[k, l] * x[Sp + l]   for l in 1:Sa)
        r_eff  = (p.rA[k] - p.dA) + dr[idx]
        eqs[idx] = r_eff * (1 + p.hA[k] * q) + q - comp * (1 + p.hA[k] * q)
    end

    return System(eqs; variables=x, parameters=dr), x
end

"""
    lever_alpha_eff_symbolic(p, x_star, dr_full=zeros(n)) → Float64

Compute `alpha_eff` from the denominator-cleared HC system using the shared
symbolic grouped metric.
"""
function lever_alpha_eff_symbolic(p::LeverOriginalParams,
                                  x_star::AbstractVector{<:Real},
                                  dr_full::AbstractVector{<:Real})
    length(dr_full) == lever_n_species(p) ||
        error("lever_alpha_eff_symbolic expected dr_full of length $(lever_n_species(p)), got $(length(dr_full))")
    syst, _ = build_lever_cleared_system(p)
    return symbolic_alpha_eff(syst, x_star; parameter_values=Float64.(dr_full)).alpha_eff
end

function lever_alpha_eff_symbolic(p::LeverOriginalParams,
                                  x_star::AbstractVector{<:Real})
    return lever_alpha_eff_symbolic(p, x_star, zeros(Float64, lever_n_species(p)))
end

"""
    lever_alpha_eff_monomial(p, x_star, dr_full=zeros(n)) → Float64

Comparison-only Lever `alpha_eff` that sums absolute monomial contributions on
the denominator-cleared HC system before any cancellation.
"""
function lever_alpha_eff_monomial(p::LeverOriginalParams,
                                  x_star::AbstractVector{<:Real},
                                  dr_full::AbstractVector{<:Real})
    length(dr_full) == lever_n_species(p) ||
        error("lever_alpha_eff_monomial expected dr_full of length $(lever_n_species(p)), got $(length(dr_full))")
    syst, _ = build_lever_cleared_system(p)
    return symbolic_alpha_eff_monomial_abs(syst, x_star; parameter_values=Float64.(dr_full)).alpha_eff
end

function lever_alpha_eff_monomial(p::LeverOriginalParams,
                                  x_star::AbstractVector{<:Real})
    return lever_alpha_eff_monomial(p, x_star, zeros(Float64, lever_n_species(p)))
end
