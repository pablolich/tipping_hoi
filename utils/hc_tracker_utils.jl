# hc_tracker_utils.jl — ScanWorkspace and eigenvalue-based stability check.
# Included by boundary_scan.jl (and figures scripts) before boundary_event_utils.jl.
# Requires: HomotopyContinuation, LinearAlgebra, MAX_STEPS_PT (from pipeline_config.jl).

mutable struct ScanWorkspace{TTracker,TCompiled}
    tracker::TTracker
    compiled_system::TCompiled
    p_start::Vector{Float64}
    p_target::Vector{Float64}
    p_eval::Vector{Float64}
    f_eval::Vector{Float64}
    jac_f::Matrix{Float64}
    jac_comm::Matrix{Float64}
    x_eval::Vector{Float64}
end

# Build reusable HC buffers and tracker state for repeated scans of one system.
function ScanWorkspace(syst::System, n_params::Int)
    p_start  = zeros(Float64, n_params)
    p_target = zeros(Float64, n_params)
    p_eval   = zeros(Float64, n_params)
    H = ParameterHomotopy(syst, p_start, p_target)
    tracker = Tracker(H; options=TrackerOptions(max_steps=MAX_STEPS_PT))
    compiled_system = CompiledSystem(syst)
    n_state = length(syst.variables)
    f_eval   = Vector{Float64}(undef, n_state)
    jac_f    = Matrix{Float64}(undef, n_state, n_state)
    jac_comm = Matrix{Float64}(undef, n_state, n_state)
    x_eval   = Vector{Float64}(undef, n_state)
    return ScanWorkspace(tracker, compiled_system,
                         p_start, p_target, p_eval,
                         f_eval, jac_f, jac_comm, x_eval)
end

# Evaluate the dominant community-matrix eigenvalue for the current equilibrium.
function lambda_max_equilibrium_hc!(ws::ScanWorkspace,
                                    x::AbstractVector{<:Real},
                                    p::AbstractVector{<:Real})
    evaluate_and_jacobian!(ws.f_eval, ws.jac_f, ws.compiled_system, x, p)
    mul!(ws.jac_comm, Diagonal(x), ws.jac_f)
    e = eigvals!(ws.jac_comm)
    return maximum(real, e)
end
