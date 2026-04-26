using LinearAlgebra

# Mathematical primitives for GLV+HOI systems — included via include() in each pass

# Shared HC tracker constants (identical in all passes that use them)
const X_TOL = 1e-12
const PARAM_TOL = 1e-9
const MAX_STEPS_PT = 1_000_000

# Jacobian of the unified per-capita growth F_i(x) = r_eff[i] + (A_eff*x)[i] + (B_eff*x*x)[i]
# with respect to x[m].  A_eff and B_eff are pre-scaled by the caller via prescale().
function jacobian_F(A_eff::AbstractMatrix{<:Real},
                    B_eff::Array{<:Real,3},
                    x::AbstractVector{<:Real})
    n = length(x)
    JF = Matrix{Float64}(undef, n, n)
    @inbounds @views for i in 1:n
        Bi = B_eff[:, :, i]
        vL = Bi' * x
        vR = Bi * x
        @inbounds @simd for m in 1:n
            JF[i, m] = A_eff[i, m] + vL[m] + vR[m]
        end
    end
    return JF
end

# Unified per-capita growth: F_i(x) = r_eff[i] + (A_eff*x)[i] + (B_eff*x*x)[i]
# A_eff and B_eff are pre-scaled by the caller via prescale().
function per_capita_growth(A_eff::AbstractMatrix{<:Real},
                           B_eff::Array{<:Real,3},
                           r_eff::AbstractVector{<:Real},
                           x::AbstractVector{<:Real})
    n = length(x)
    fx = Vector{Float64}(undef, n)
    lin = similar(x, Float64)
    mul!(lin, A_eff, x)
    tmp = similar(x, Float64)
    @inbounds @views for i in 1:n
        Bi = B_eff[:, :, i]
        mul!(tmp, Bi, x)
        quad = dot(x, tmp)
        fx[i] = r_eff[i] + lin[i] + quad
    end
    return fx
end

"""
Clamp small negative entries to 0 to stabilize support detection.
Values with |x[i]| <= tol_neg are treated as numerical noise.
"""
function clamp_small_negatives!(x::Vector{Float64}, tol_neg::Real)
    @inbounds for i in eachindex(x)
        if x[i] < 0 && abs(x[i]) <= tol_neg
            x[i] = 0.0
        end
    end
    return x
end

"""Return indices of surviving species using a strict positivity threshold."""
support_indices(x::AbstractVector{<:Real}, tol_pos::Real) = findall(x .> tol_pos)

"""
Compute the leading eigenvalue of the community Jacobian x_i * dF_i/dx_j
at equilibrium x. A_eff and B_eff are pre-scaled by the caller via prescale().
"""
@inline function lambda_max_equilibrium(A_eff::AbstractMatrix{<:Real},
                                        B_eff::Array{<:Real,3},
                                        x::AbstractVector{<:Real})
    JF = jacobian_F(A_eff, B_eff, x)
    J = Matrix{Float64}(JF)
    @inbounds for i in 1:length(x)
        @views J[i, :] .*= x[i]
    end
    return maximum(real.(eigvals(J)))
end
