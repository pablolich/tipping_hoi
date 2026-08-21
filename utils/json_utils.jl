# JSON parsing helpers shared across boundary_scan, post_boundary_dynamics, backtrack_perturbation.

"""
    to_float_or_nan(x)

Read a JSON scalar as a Float64, mapping `null`, `missing`, an unconvertible
value, or a non-finite one to NaN.  The inverse of `json_number` below.
"""
function to_float_or_nan(x)
    if x === nothing || ismissing(x)
        return NaN
    end
    v = try
        Float64(x)
    catch
        return NaN
    end
    return isfinite(v) ? v : NaN
end

"""
    json_number(x)

JSON has no NaN/Inf literal and JSON3 refuses to write one (`error("NaN not
allowed to be written in JSON spec")`), so a non-finite scalar is written as
`null` — which is what it means here: *no number*.  `to_float_or_nan` reads it
back.  Used for the `alpha` / `alpha_grid` labels of families that have no α.
"""
json_number(x::Real) = isfinite(x) ? x : nothing
json_number(v::AbstractVector{<:Real}) = Any[json_number(xi) for xi in v]

function to_dict(x)
    if x isa JSON3.Object
        d = Dict{String,Any}()
        for (k, v) in pairs(x)
            d[String(k)] = to_dict(v)
        end
        return d
    elseif x isa JSON3.Array
        return [to_dict(v) for v in x]
    else
        return x
    end
end

function nested_to_matrix(rows)
    m = length(rows)
    m == 0 && return zeros(Float64, 0, 0)
    n = length(rows[1])
    M = Matrix{Float64}(undef, m, n)
    @inbounds for i in 1:m
        length(rows[i]) == n || error("Inconsistent matrix row length at row $i.")
        for j in 1:n
            M[i, j] = Float64(rows[i][j])
        end
    end
    return M
end

function nested_to_tensor3(slices)
    n1 = length(slices)
    n1 == 0 && return zeros(Float64, 0, 0, 0)
    n2 = length(slices[1])
    n2 == 0 && return zeros(Float64, n1, 0, 0)
    n3 = length(slices[1][1])
    T = Array{Float64,3}(undef, n1, n2, n3)
    @inbounds for i in 1:n1
        length(slices[i]) == n2 || error("Inconsistent tensor dim-2 length at i=$i.")
        for j in 1:n2
            length(slices[i][j]) == n3 || error("Inconsistent tensor dim-3 length at i=$i j=$j.")
            for k in 1:n3
                T[i, j, k] = Float64(slices[i][j][k])
            end
        end
    end
    return T
end
