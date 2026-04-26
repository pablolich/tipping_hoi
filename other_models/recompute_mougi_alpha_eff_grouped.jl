#!/usr/bin/env julia

using JSON3

include(joinpath(@__DIR__, "mougi_model.jl"))

function usage()
    println("Usage: julia --startup-file=no recompute_mougi_alpha_eff_grouped.jl <file-or-dir> [...]")
    println("Recompute grouped symbolic Mougi alpha_eff in-place for matching JSON files.")
end

function collect_json_paths(args::Vector{String})
    isempty(args) && error("Expected at least one file or directory path.")

    paths = String[]
    for arg in args
        if isdir(arg)
            for (root, _, files) in walkdir(arg)
                for file in files
                    endswith(file, ".json") || continue
                    push!(paths, joinpath(root, file))
                end
            end
        elseif isfile(arg)
            push!(paths, arg)
        else
            error("Path does not exist: $(arg)")
        end
    end

    sort!(unique!(paths))
    return paths
end

function recompute_mougi_alpha!(model::Dict{String, Any})
    p = mougi_params_from_payload(model)
    x_star = Float64.(model["x_star"])
    old_alpha = Float64(model["alpha_eff"])
    new_alpha = Float64(mougi_alpha_eff_symbolic(p, x_star))

    model["alpha_eff"] = new_alpha

    if haskey(model, "scan_results")
        for block in model["scan_results"]
            block isa Dict{String, Any} || continue
            block["alpha"] = new_alpha
        end
    end

    return old_alpha, new_alpha
end

function main(args::Vector{String})
    if any(arg -> arg in ("-h", "--help"), args)
        usage()
        return
    end

    paths = collect_json_paths(args)
    updated = 0
    skipped = 0

    for path in paths
        model = JSON3.read(read(path, String), Dict{String, Any})
        if get(model, "dynamics_mode", nothing) != "mougi"
            skipped += 1
            continue
        end

        old_alpha, new_alpha = recompute_mougi_alpha!(model)
        write(path, JSON3.write(model))
        updated += 1
        println("updated $(path): $(old_alpha) -> $(new_alpha)")
    end

    println("summary updated=$(updated) skipped=$(skipped)")
end

main(ARGS)
