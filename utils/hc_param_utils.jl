# HomotopyContinuation parameter-space helpers.
# Shared by boundary_scan.jl and backtrack_perturbation.jl.
# Constants (ZERO_ABUNDANCE, PARAM_TOL, MAX_STEPS_PT, MAX_ITERS) are defined
# in pipeline_config.jl, which must be included before this file.

@inline function parameters_at_t!(out::AbstractVector{Float64},
                                  t::Real,
                                  p_start::AbstractVector{<:Real},
                                  p_target::AbstractVector{<:Real})
    omt = 1 - t
    @inbounds @simd for i in eachindex(out, p_start, p_target)
        out[i] = t * p_start[i] + omt * p_target[i]
    end
    return out
end

function get_parameters_at_t(t, p_start, p_target)
    out = Vector{Float64}(undef, length(p_start))
    return parameters_at_t!(out, t, p_start, p_target)
end

@inline function parameter_distance(p_start::AbstractVector{<:Real},
                                    p_target::AbstractVector{<:Real})
    s = 0.0
    @inbounds @simd for i in eachindex(p_start, p_target)
        d = p_target[i] - p_start[i]
        s += d * d
    end
    return sqrt(s)
end
