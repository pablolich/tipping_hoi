# Connector that brings the minimum slice of `pipeline_marsland_hc` needed by
# `utils/glvhoi_utils.jl` into scope.  This file is intentionally lightweight:
# it only includes `MarslandParams.jl` (the struct) and `hc_system.jl` (the
# HomotopyContinuation bridge + JSON payload helpers), because the rest of the
# pipeline (equilibrium solver, stability check, etc.) pulls in packages like
# ForwardDiff, NLsolve and OrdinaryDiffEq that are not guaranteed to be in the
# global Julia environment used by `boundary_scan.jl`.
#
# Callers that need the full Marsland pipeline — most notably
# `generate_bank_marsland.jl` — should activate the project at
# `pipeline_marsland_hc/Project.toml` and include the umbrella
# `pipeline_marsland_hc/src/Marsland.jl` directly instead.
#
# Helpers `_dict_or_prop_get` and `_to_matrix_f64` come from `lever_model.jl`,
# following the same include-ordering trick used by `karatayev_model.jl:10`.

if !@isdefined(LeverOriginalParams)
    include(joinpath(@__DIR__, "lever_model.jl"))
end

if !@isdefined(MarslandParams)
    _marsland_src = joinpath(@__DIR__, "papers", "Marsland_et_al_2019",
                             "pipeline_marsland_hc", "src")
    include(joinpath(_marsland_src, "MarslandParams.jl"))
    include(joinpath(_marsland_src, "hc_system.jl"))
end
