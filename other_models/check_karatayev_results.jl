using JSON3

for mode in ["FMI", "RMI"]
    dir = joinpath(@__DIR__, "..", "model_runs", "karatayev_$(mode)_2models_5dirs_seed42")
    println("\n=== $mode ===")
    for f in sort(readdir(dir))
        endswith(f, ".json") || continue
        m = JSON3.read(read(joinpath(dir, f), String))
        dirs = m["directions"]
        n_found = sum(haskey(d, "delta_crit") && d["delta_crit"] isa Number for d in dirs)
        println("$f: $(length(dirs)) dirs, $n_found with delta_crit")
        for d in dirs
            if haskey(d, "delta_crit") && d["delta_crit"] isa Number
                println("    dir $(d["dir_idx"]): delta_crit=$(round(Float64(d["delta_crit"]), digits=4))  x_pre=$(round.(Float64.(d["x_pre"]), digits=3))")
            end
        end
    end
end
