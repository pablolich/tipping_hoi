using LinearAlgebra

# Equilibrium utilities for stages 3 & 4.
# Requires math_utils.jl to be included first (uses jacobian_F, per_capita_growth,
# clamp_small_negatives!, support_indices).

"""
Newton-polish x0 on the support-restricted subsystem F_support(x_support) = 0.
Returns a NamedTuple (x, support, lambda_max) if converged and stable, else nothing.
A_eff and B_eff are pre-scaled by the caller via prescale().
"""
function newton_polish_equilibrium(A_eff::AbstractMatrix{<:Real},
                                   B_eff::Array{<:Real,3},
                                   r_eff::AbstractVector{<:Real},
                                   x0::AbstractVector{<:Real};
                                   tol_neg::Real,
                                   tol_pos::Real,
                                   lambda_tol::Real,
                                   max_iter::Int=20,
                                   ftol::Real=1e-12,
                                   xtol::Real=1e-12)
    x = clamp.(Vector{Float64}(x0), 0.0, Inf)
    clamp_small_negatives!(x, tol_neg)
    support = support_indices(x, tol_pos)

    if isempty(support)
        return (x=x, support=Int[], lambda_max=-Inf)
    end

    x_s = x[support]
    n = length(x)

    for _ in 1:max_iter
        x_full = zeros(Float64, n)
        x_full[support] .= x_s
        F = per_capita_growth(A_eff, B_eff, r_eff, x_full)
        F_s = F[support]

        if norm(F_s) < ftol
            break
        end

        JF = jacobian_F(A_eff, B_eff, x_full)
        J_s = JF[support, support]

        dx = J_s \ (-F_s)
        x_s_new = x_s .+ dx

        if norm(dx) < xtol
            x_s = x_s_new
            break
        end
        x_s = x_s_new
    end

    x_out = zeros(Float64, n)
    x_out[support] .= x_s
    clamp_small_negatives!(x_out, tol_neg)

    if any(xi < -tol_neg for xi in x_out)
        return nothing
    end

    F_final = per_capita_growth(A_eff, B_eff, r_eff, x_out)
    if norm(F_final[support]) > sqrt(ftol)
        return nothing
    end

    JF_final = jacobian_F(A_eff, B_eff, x_out)
    JF_s = JF_final[support, support]
    J_comm = Diagonal(x_out[support]) * JF_s
    lambda_max = maximum(real.(eigvals(J_comm)))

    if lambda_max >= -lambda_tol
        return nothing
    end

    return (x=x_out, support=support, lambda_max=lambda_max)
end

"""
Wrap newton_polish_equilibrium into the same Vector{NamedTuple} interface
used by snap_confident and snap_to_equilibrium (returns 0 or 1 element).
"""
function stable_equilibria_newton(A_eff::AbstractMatrix{<:Real},
                                  B_eff::Array{<:Real,3},
                                  r_eff::AbstractVector{<:Real},
                                  x_seed::AbstractVector{<:Real};
                                  tol_neg::Real,
                                  tol_pos::Real,
                                  lambda_tol::Real,
                                  max_iter::Int=20,
                                  ftol::Real=1e-12,
                                  xtol::Real=1e-12)
    result = newton_polish_equilibrium(
        A_eff, B_eff, r_eff, x_seed;
        tol_neg=tol_neg, tol_pos=tol_pos, lambda_tol=lambda_tol,
        max_iter=max_iter, ftol=ftol, xtol=xtol,
    )
    result === nothing && return NamedTuple[]
    return NamedTuple[result]
end

"""
Choose the closest equilibrium to a trajectory endpoint, breaking ties
by smaller (more negative) face stability eigenvalue.
"""
function snap_to_equilibrium(x_end::AbstractVector{<:Real},
                             equilibria::Vector{NamedTuple},
                             tol_neg::Real,
                             snap_tie_tol::Real)
    if isempty(equilibria)
        return nothing
    end
    x_clamped = Vector{Float64}(x_end)
    clamp_small_negatives!(x_clamped, tol_neg)

    best = nothing
    best_dist = Inf
    best_lambda = Inf
    for eq in equilibria
        dist = norm(x_clamped .- eq.x)
        if dist < best_dist - snap_tie_tol
            best = eq
            best_dist = dist
            best_lambda = eq.lambda_max
        elseif abs(dist - best_dist) <= snap_tie_tol && eq.lambda_max < best_lambda
            best = eq
            best_dist = dist
            best_lambda = eq.lambda_max
        end
    end
    return (x_snap=best.x, dist=best_dist, lambda_max=best_lambda)
end
