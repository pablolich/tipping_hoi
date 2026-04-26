# JSON parsing helpers shared across boundary_scan, post_boundary_dynamics, backtrack_perturbation.

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
