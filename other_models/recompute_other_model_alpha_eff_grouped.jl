#!/usr/bin/env julia

using JSON3

include(joinpath(@__DIR__, "lever_model.jl"))
include(joinpath(@__DIR__, "karatayev_model.jl"))
include(joinpath(@__DIR__, "aguade_model.jl"))

function usage()
    println("Usage: julia --startup-file=no recompute_other_model_alpha_eff_grouped.jl <file-or-dir> [...]")
    println("Recompute grouped symbolic alpha_eff in-place for Lever, Karatayev, and Aguade JSON files.")
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

function grouped_alpha_eff(model::Dict{String, Any})
    mode = get(model, "dynamics_mode", nothing)
    x_star = Float64.(model["x_star"])

    if mode == "lever"
        p = lever_params_from_payload(model)
        return Float64(lever_alpha_eff_symbolic(p, x_star))
    elseif mode == "karatayev"
        p = karatayev_params_from_payload(model)
        return Float64(karatayev_alpha_eff_symbolic(p, x_star))
    elseif mode == "aguade"
        p = aguade_params_from_payload(model)
        return Float64(aguade_alpha_eff_symbolic(p, x_star))
    else
        return nothing
    end
end

function update_alpha_metadata!(model::Dict{String, Any}, new_alpha::Float64)
    model["alpha_eff"] = new_alpha

    if haskey(model, "scan_results")
        for block in model["scan_results"]
            block isa Dict{String, Any} || continue
            block["alpha"] = new_alpha
        end
    end

    if haskey(model, "scan_config")
        scan_config = model["scan_config"]
        if scan_config isa Dict{String, Any} && haskey(scan_config, "alpha_grid")
            alpha_grid = scan_config["alpha_grid"]
            if alpha_grid isa Vector
                if haskey(model, "scan_results") &&
                   length(alpha_grid) == length(model["scan_results"])
                    for i in eachindex(alpha_grid, model["scan_results"])
                        block = model["scan_results"][i]
                        block isa Dict{String, Any} || continue
                        alpha_grid[i] = get(block, "alpha", new_alpha)
                    end
                elseif length(alpha_grid) == 1
                    alpha_grid[1] = new_alpha
                end
            end
        end
    end
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
        new_alpha = grouped_alpha_eff(model)
        if isnothing(new_alpha)
            skipped += 1
            continue
        end

        old_alpha = Float64(model["alpha_eff"])
        update_alpha_metadata!(model, new_alpha)
        write(path, JSON3.write(model))
        updated += 1
        println("updated $(path): $(old_alpha) -> $(new_alpha)")
    end

    println("summary updated=$(updated) skipped=$(skipped)")
end

main(ARGS)
