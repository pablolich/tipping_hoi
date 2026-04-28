#!/usr/bin/env julia
# Backtrack perturbations from post-boundary states toward delta=0.
# This is the new_code JSON analogue of pass4_fold_backtrack.jl.
# Unlike pass4, this processes all boundary rows (not only folds).
#
# Example:
#   julia --startup-file=no pipeline/backtrack_perturbation.jl 1_bank_50_models_n_2-7_100_dirs_b_dirichlet
#   julia --startup-file=no pipeline/backtrack_perturbation.jl 1_bank_50_models_n_2-7_100_dirs_b_dirichlet \
#       --model-file model_n_3_seed_40000002_n_dirs_100.json
# All backtrack settings are in pipeline_config.jl.

using LinearAlgebra
using JSON3
using HomotopyContinuation
using DifferentialEquations
using SciMLBase

include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
include(joinpath(@__DIR__, "..", "utils", "model_store_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "math_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "json_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "hc_param_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "glvhoi_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "dynamics_cfg_utils.jl"))
include(joinpath(@__DIR__, "..", "utils", "boundary_event_utils.jl"))

function usage_error()
    msg = """
    Usage:
      julia --startup-file=no pipeline/backtrack_perturbation.jl <run_dir> [--model-file FILE]

    Required:
      <run_dir>          Folder inside model_runs/

    Options:
      --model-file FILE  Process one model JSON file from <run_dir>

    All backtrack settings are in pipeline_config.jl.
    """
    error(msg)
end

function parse_args(args::Vector{String})
    isempty(args) && usage_error()

    run_dir    = ""
    model_file = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--model-file" && i < length(args)
            model_file = args[i + 1]
            i += 2
        elseif startswith(arg, "--")
            error("Unknown flag: $arg. Configure backtrack settings in pipeline_config.jl.")
        elseif run_dir == ""
            run_dir = arg
            i += 1
        else
            error("Unexpected argument: $arg")
        end
    end

    run_dir == "" && usage_error()
    return (run_dir=run_dir, model_file=model_file)
end

function build_backtrack_cfg()
    return Dict{String,Any}(
        "check_stability"    => true,
        "check_invasibility" => true,
        "lambda_tol"         => BACK_LAMBDA_TOL,
        "invasion_tol"       => BACK_INVASION_TOL,
        "max_step_ratio"     => BACK_MAX_STEP_RATIO,
        "tol_pos"            => SNAP_TOL_POS,
        "tol_neg"            => SNAP_TOL_NEG,
        "eps_seed_extinct"   => BACK_EPS_SEED_EXTINCT === nothing ? 10.0 * ODE_EPS_EXTINCT : Float64(BACK_EPS_SEED_EXTINCT),
        "return_dist_abs"    => 1e-3,
        "return_dist_rel"    => 1e-3,
    )
end

mutable struct BacktrackWorkspace{TTracker,TCompiled}
    tracker::TTracker
    compiled_system::TCompiled
    p_eval::Vector{Float64}
    f_eval::Vector{Float64}
    jac_f::Matrix{Float64}
    jac_comm::Matrix{Float64}
    x_eval::Vector{Float64}
    x_prev::Vector{Float64}
    x_curr::Vector{Float64}
end

function BacktrackWorkspace(syst::System, n_state::Int, n_params::Int)
    p_start  = zeros(Float64, n_params)
    p_target = zeros(Float64, n_params)
    H = ParameterHomotopy(syst, p_start, p_target)
    tracker = Tracker(H; options=TrackerOptions(max_steps=MAX_STEPS_PT))
    compiled_system = CompiledSystem(syst)
    return BacktrackWorkspace(
        tracker,
        compiled_system,
        Vector{Float64}(undef, n_params),
        Vector{Float64}(undef, n_state),
        Matrix{Float64}(undef, n_state, n_state),
        Matrix{Float64}(undef, n_state, n_state),
        Vector{Float64}(undef, n_state),
        Vector{Float64}(undef, n_state),
        Vector{Float64}(undef, n_state),
    )
end

# Per-alpha-block cache keyed by the sorted active-support indices.  Active-support
# restriction fully determines (r0_act, A_act_eff, B_act_eff) within one alpha block,
# so one workspace per support can be reused across all direction rows that share it.
const BacktrackCache = Dict{Vector{Int}, BacktrackWorkspace}

function get_or_build_workspace!(cache::BacktrackCache,
                                  active_idx::Vector{Int},
                                  r0_act::AbstractVector{<:Real},
                                  A_act_eff::AbstractMatrix{<:Real},
                                  B_act_eff::Array{<:Real,3})
    key = sort(active_idx)
    ws = get(cache, key, nothing)
    if ws === nothing
        n_active = length(active_idx)
        syst, _ = build_system(r0_act, A_act_eff, B_act_eff)
        ws = BacktrackWorkspace(syst, n_active, n_active)
        cache[key] = ws
    end
    return ws
end

function lambda_max_equilibrium_hc!(ws::BacktrackWorkspace,
                                    x::AbstractVector{<:Real},
                                    p::AbstractVector{<:Real})
    evaluate_and_jacobian!(ws.f_eval, ws.jac_f, ws.compiled_system, x, p)
    mul!(ws.jac_comm, Diagonal(x), ws.jac_f)
    e = eigvals!(ws.jac_comm)
    return maximum(real, e)
end

function delta_from_dr(dr::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    denom = dot(u, u)
    return denom == 0.0 ? 0.0 : dot(dr, u) / denom
end

# Recover the equilibrium at frac * p_crit by tracking from x_start at p_start.
function track_to_preboundary(ws::BacktrackWorkspace,
                               x_start::AbstractVector{<:Real},
                               p_start::AbstractVector{<:Real},
                               p_crit::AbstractVector{<:Real},
                               frac::Real)
    p_pre = frac .* p_crit
    if parameter_distance(p_pre, p_start) == 0.0
        return collect(Float64.(x_start))
    end
    start_parameters!(ws.tracker, p_start)
    target_parameters!(ws.tracker, p_pre)
    init!(ws.tracker, x_start, 1.0, 0.0)
    while is_tracking(ws.tracker.state.code)
        HomotopyContinuation.step!(ws.tracker)
    end
    return copy(real.(ws.tracker.state.x))
end

include(joinpath(@__DIR__, "..", "utils", "ode_snap_utils.jl"))

function invasion_max_at_state(r0_full::AbstractVector{<:Real},
                               A_full_eff::AbstractMatrix{<:Real},
                               B_full_eff::Array{<:Real,3},
                               u_full::AbstractVector{<:Real},
                               x_active::AbstractVector{<:Real},
                               active_idx::Vector{Int},
                               inactive_idx::Vector{Int},
                               delta::Real)
    isempty(inactive_idx) && return -Inf
    n = length(r0_full)
    x_full = zeros(Float64, n)
    x_full[active_idx] .= x_active
    r_eff = r0_full .+ delta .* u_full
    growth = per_capita_growth(A_full_eff, B_full_eff, r_eff, x_full)
    return maximum(growth[inactive_idx])
end

function build_ode_seed(x_good_full::AbstractVector{<:Real},
                        inactive_idx::Vector{Int};
                        seed_floor::Float64=1e-3)
    x_seed = max.(Vector{Float64}(x_good_full), 0.0)
    n = length(x_seed)
    inactive_mask = falses(n)
    inactive_mask[inactive_idx] .= true
    active_idx = findall(x -> !x, inactive_mask)

    min_active = isempty(active_idx) ? 0.0 : minimum(x_seed[active_idx])
    seed_amp = max(0.01 * min_active, seed_floor)

    for idx in inactive_idx
        x_seed[idx] += seed_amp
    end
    return x_seed
end

function post_boundary_delta_s(dyn::Dict{String,Any}, delta_boundary::Real)
    delta_s = dyn["post_delta_abs"] === nothing ?
        dyn["post_delta_frac"] * delta_boundary :
        Float64(dyn["post_delta_abs"])
    if !(isfinite(delta_s) && delta_s > 0)
        delta_s = Float64(dyn["post_delta_abs_default"])
    end
    return delta_s
end

function reversal_frac(delta_post::Real, delta_return::Real)
    if !(isfinite(delta_post) && delta_post > 0)
        return NaN
    end
    v = (delta_post - delta_return) / delta_post
    return clamp(v, 0.0, 1.0)
end


function integrate_and_classify_return(A_eff::AbstractMatrix{<:Real},
                                       B_eff::Array{<:Real,3},
                                       r0::AbstractVector{<:Real},
                                       u_full::AbstractVector{<:Real},
                                       delta_probe::Real,
                                       x_seed::AbstractVector{<:Real},
                                       n_full::Int,
                                       dyn::Dict{String,Any},
                                       back::Dict{String,Any})
    r_eff = r0 .+ delta_probe .* u_full
    f! = make_unified_rhs(A_eff, B_eff, r_eff)
    ext_cb = make_extinction_cb(dyn["eps_extinct"])

    prob = ODEProblem(f!, x_seed, dyn["tspan"])
    sol = DifferentialEquations.solve(prob, Tsit5(); reltol=dyn["reltol"], abstol=dyn["abstol"],
                                      callback=ext_cb,
                                      save_everystep=false, save_start=false)

    ode_retcode = string(sol.retcode)
    x_end = isempty(sol.u) ? fill(NaN, n_full) : Vector{Float64}(sol.u[end])
    clamp_small_negatives!(x_end, back["tol_neg"])

    if !SciMLBase.successful_retcode(sol)
        return (class=:ode_fail, returned_n=false, ode_retcode=ode_retcode,
                x_end=x_end, converged=false, n_alive=0, snap_reason=:n_a)
    end

    n_alive = count(>(back["tol_pos"]), x_end)
    returned_n = (n_alive == n_full)

    class = if returned_n
        :returned_n
    elseif n_alive < n_full
        :boundary_persist
    else
        :other_attractor
    end

    return (class=class, returned_n=returned_n, ode_retcode=ode_retcode,
            x_end=x_end, converged=true, n_alive=n_alive, snap_reason=:n_a)
end

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

function to_float_vector_or_nothing(x, n_expected::Int)
    if x === nothing || ismissing(x)
        return nothing
    end
    v = try
        Vector{Float64}(Float64.(x))
    catch
        return nothing
    end
    length(v) == n_expected || return nothing
    all(isfinite.(v)) || return nothing
    return v
end

function default_result_row(row_idx::Int,
                            alpha::Float64,
                            alpha_idx::Int,
                            ray_id::Int,
                            boundary_flag::String,
                            boundary_status::String,
                            delta_boundary::Float64,
                            delta_post::Float64)
    δret = isfinite(delta_post) ? delta_post : (isfinite(delta_boundary) ? delta_boundary : 0.0)
    return Dict{String,Any}(
        "row_idx" => row_idx,
        "alpha" => alpha,
        "alpha_idx" => alpha_idx,
        "ray_id" => ray_id,
        "boundary_flag" => boundary_flag,
        "boundary_status" => boundary_status,
        "delta_boundary" => delta_boundary,
        "delta_post" => delta_post,
        "n_active_start" => 0,
        "n_inactive_start" => 0,
        "hc_event" => "none",
        "hc_status" => "not_run",
        "delta_event" => δret,
        "delta_probe" => δret,
        "delta_return" => δret,
        "delta_return_kind" => "event_no_ode",
        "reversal_frac" => reversal_frac(delta_post, δret),
        "returned_n" => false,
        "class_label" => "not_run",
        "ode_retcode" => "not_run",
        "ode_ran" => false,
        "snap_reason" => "not_run",
        "n_alive_ode" => 0,
    )
end

function compute_u_from_U(U::AbstractMatrix{<:Real}, ray_id::Int)
    if !(1 <= ray_id <= size(U, 2))
        return nothing
    end
    u = Vector{Float64}(U[:, ray_id])
    nrm = norm(u)
    if !(isfinite(nrm) && nrm > 0)
        return nothing
    end
    return u ./ nrm
end

function process_direction_row(row::Dict{String,Any},
                               row_idx::Int,
                               alpha::Float64,
                               alpha_idx::Int,
                               r0::Vector{Float64},
                               A_eff::Matrix{Float64},
                               B_eff::Array{Float64,3},
                               U::Matrix{Float64},
                               dyn::Dict{String,Any},
                               back::Dict{String,Any},
                               cache::BacktrackCache)
    n = length(r0)
    ray_id = try
        Int(row["ray_id"])
    catch
        -1
    end
    boundary_flag = string(get(row, "flag", "unknown"))
    boundary_status = string(get(row, "status", "unknown"))
    delta_boundary = to_float_or_nan(get(row, "delta_c", nothing))
    delta_post = to_float_or_nan(get(row, "delta_post", nothing))

    base = default_result_row(
        row_idx, alpha, alpha_idx, ray_id,
        boundary_flag, boundary_status, delta_boundary, delta_post,
    )

    u_full = compute_u_from_U(U, ray_id)
    if u_full === nothing
        return merge(base, Dict(
            "class_label" => "invalid_direction",
            "hc_status" => "invalid_direction",
            "ode_retcode" => "not_run",
        ))
    end

    if !(isfinite(delta_post) && delta_post >= 0)
        return merge(base, Dict(
            "class_label" => "invalid_delta_post",
            "hc_status" => "invalid_delta_post",
            "ode_retcode" => "not_run",
        ))
    end

    x_post_raw = get(row, "x_postboundary_snap", nothing)
    if x_post_raw === nothing
        return merge(base, Dict(
            "class_label" => "missing_x_post",
            "hc_status" => "missing_x_post",
            "ode_retcode" => "not_run",
        ))
    end

    x_post = to_float_vector_or_nothing(x_post_raw, n)
    if x_post === nothing
        return merge(base, Dict(
            "class_label" => "invalid_x_post",
            "hc_status" => "invalid_x_post",
            "ode_retcode" => "not_run",
        ))
    end

    clamp_small_negatives!(x_post, back["tol_neg"])
    active_idx = support_indices(x_post, back["tol_pos"])
    inactive_idx = setdiff(collect(1:n), active_idx)
    n_active = length(active_idx)
    n_inactive = length(inactive_idx)

    if n_active == 0
        return merge(base, Dict(
            "n_active_start" => n_active,
            "n_inactive_start" => n_inactive,
            "class_label" => "empty_support",
            "hc_status" => "empty_support",
            "ode_retcode" => "not_run",
        ))
    end

    if delta_post == 0.0
        return merge(base, Dict(
            "n_active_start" => n_active,
            "n_inactive_start" => n_inactive,
            "hc_event" => "success",
            "hc_status" => "already_at_zero",
            "delta_event" => 0.0,
            "delta_probe" => 0.0,
            "delta_return" => 0.0,
            "delta_return_kind" => "event_no_ode",
            "reversal_frac" => 1.0,
            "class_label" => "success_to_zero",
            "ode_retcode" => "not_run",
            "ode_ran" => false,
            "snap_reason" => "not_run",
            "n_alive_ode" => 0,
        ))
    end

    # Restrict pre-scaled matrices to the active subsystem
    A_act_eff, B_act_eff, r0_act, u_act, x_act = restrict_params(A_eff, B_eff, r0, u_full, active_idx, x_post)

    # Parameters are always n_active-dimensional (no alpha slot)
    p_start = delta_post .* collect(u_act)
    p_target = zeros(n_active)

    ws = get_or_build_workspace!(cache, active_idx, r0_act, A_act_eff, B_act_eff)

    invasion_fn = back["check_invasibility"] && !isempty(inactive_idx) ?
        (x, t) -> invasion_max_at_state(r0, A_eff, B_eff, u_full, x, active_idx, inactive_idx, real(t) * delta_post) :
        nothing

    event, t_end, x_crit = find_event(p_start, p_target, x_act, ws, ZERO_ABUNDANCE;
                                       check_stability=back["check_stability"],
                                       λ_tol=back["lambda_tol"],
                                       check_invasibility=back["check_invasibility"],
                                       invasion_fn=invasion_fn,
                                       invasion_tol=back["invasion_tol"])

    t_real = real(t_end)
    p_crit = t_real .* p_start   # p_target = zeros, so p(t) = t * p_start
    delta_event = delta_from_dr(p_crit, u_act)
    if !(isfinite(delta_event))
        delta_event = delta_post
    end
    delta_event = clamp(delta_event, 0.0, delta_post)
    status_str = "step_refined"

    if event == :success
        return merge(base, Dict(
            "n_active_start" => n_active,
            "n_inactive_start" => n_inactive,
            "hc_event" => string(event),
            "hc_status" => status_str,
            "delta_event" => delta_event,
            "delta_probe" => 0.0,
            "delta_return" => 0.0,
            "delta_return_kind" => "event_no_ode",
            "reversal_frac" => reversal_frac(delta_post, 0.0),
            "returned_n" => false,
            "class_label" => "success_to_zero",
            "ode_retcode" => "not_run",
            "ode_ran" => false,
            "snap_reason" => "not_run",
            "n_alive_ode" => 0,
        ))
    end

    delta_s = post_boundary_delta_s(dyn, delta_event)
    delta_probe = max(0.0, delta_event - delta_s)

    x_pre = track_to_preboundary(ws, x_act, p_start, p_crit, SCAN_PREBOUNDARY_FRAC)
    x_good_full = zeros(Float64, n)
    x_good_full[active_idx] .= x_pre
    seed_floor = max(back["eps_seed_extinct"], 10.0 * dyn["eps_extinct"])
    x_seed = build_ode_seed(x_good_full, inactive_idx; seed_floor=seed_floor)

    ode = integrate_and_classify_return(
        A_eff, B_eff, r0, u_full, delta_probe, x_seed, n, dyn, back
    )
    delta_return = ode.returned_n ? delta_post : delta_probe
    kind = ode.returned_n ? "returned_n" : "probe_nonreturn"

    return merge(base, Dict(
        "n_active_start" => n_active,
        "n_inactive_start" => n_inactive,
        "hc_event" => string(event),
        "hc_status" => status_str,
        "delta_event" => delta_event,
        "delta_probe" => delta_probe,
        "delta_return" => delta_return,
        "delta_return_kind" => kind,
        "reversal_frac" => reversal_frac(delta_post, delta_return),
        "returned_n" => Bool(ode.returned_n),
        "class_label" => string(ode.class),
        "ode_retcode" => String(ode.ode_retcode),
        "ode_ran" => true,
        "snap_reason" => String(ode.snap_reason),
        "n_alive_ode" => Int(ode.n_alive),
    ))
end

function backtrack_model(model::Dict{String,Any},
                         dyn::Dict{String,Any},
                         back::Dict{String,Any})
    n = Int(model["n"])
    dmode = get(model, "dynamics_mode", "standard")
    A = nested_to_matrix(model["A"])
    B = nested_to_tensor3(model["B"])
    U = nested_to_matrix(model["U"])
    r0_fixed = dmode == "unique_equilibrium" ? nothing : Float64.(model["r"])

    size(A, 1) == n && size(A, 2) == n || error("A has wrong shape: $(size(A)) for n=$n.")
    size(B, 1) == n && size(B, 2) == n && size(B, 3) == n || error("B has wrong shape: $(size(B)) for n=$n.")
    size(U, 1) == n || error("U has wrong row count: $(size(U, 1)) for n=$n.")

    is_gibbs = dmode == "gibbs"
    post_results = get(model, "post_dynamics_results", Any[])
    isempty(post_results) && error("Missing or empty post_dynamics_results in input model.")

    alpha_results = Vector{Any}()
    for alpha_block in post_results
        a_idx = Int(alpha_block["alpha_idx"])
        alpha = Float64(alpha_block["alpha"])
        dir_rows = get(alpha_block, "directions", Any[])

        r0 = dmode == "unique_equilibrium" ? compute_r_unique_equilibrium(A, B, alpha) : r0_fixed
        A_eff, B_eff = prescale(A, B, alpha, is_gibbs)

        cache = BacktrackCache()

        out_rows = Vector{Any}()
        for (row_idx, row_any) in enumerate(dir_rows)
            row = row_any isa Dict{String,Any} ? row_any : to_dict(row_any)
            ray_id = try
                Int(row["ray_id"])
            catch
                -1
            end

            row_out = try
                process_direction_row(row, row_idx, alpha, a_idx, r0, A_eff, B_eff, U, dyn, back, cache)
            catch err
                # A cached tracker may be mid-step after the exception — evict so the
                # next row with this support rebuilds from scratch.
                empty!(cache)
                delta_boundary = to_float_or_nan(get(row, "delta_c", nothing))
                delta_post = to_float_or_nan(get(row, "delta_post", nothing))
                fallback = default_result_row(row_idx, alpha, a_idx, ray_id,
                                              string(get(row, "flag", "unknown")),
                                              string(get(row, "status", "unknown")),
                                              delta_boundary, delta_post)
                merge(fallback, Dict(
                    "class_label" => "backtrack_error",
                    "hc_status" => sprint(showerror, err),
                    "ode_retcode" => "not_run",
                    "ode_ran" => false,
                ))
            end
            push!(out_rows, row_out)
        end

        push!(alpha_results, Dict(
            "alpha_idx" => a_idx,
            "alpha" => alpha,
            "directions" => out_rows,
        ))
    end

    output = Dict{String,Any}()
    for (k, v) in model
        output[k] = v
    end
    output["backtrack_config"] = Dict(
        "processed_boundaries" => "all",
        "dynamics" => Dict(
            "tspan" => [dyn["tspan"][1], dyn["tspan"][2]],
            "reltol" => dyn["reltol"],
            "abstol" => dyn["abstol"],
            "eps_extinct" => dyn["eps_extinct"],
            "post_delta_frac" => dyn["post_delta_frac"],
            "post_delta_abs" => dyn["post_delta_abs"],
        ),
        "backtrack" => Dict(
            "check_stability" => back["check_stability"],
            "check_invasibility" => back["check_invasibility"],
            "lambda_tol" => back["lambda_tol"],
            "invasion_tol" => back["invasion_tol"],
            "max_step_ratio" => back["max_step_ratio"],
            "eps_seed_extinct" => back["eps_seed_extinct"],
        ),
        "selection" => Dict(
            "mode" => "full_model",
        ),
    )
    output["backtrack_results"] = alpha_results
    return output
end

function main()
    opts = parse_args(ARGS)
    run_root = canonical_models_root(@__DIR__, opts.run_dir)
    isdir(run_root) || error("Run directory not found: $run_root")
    model_paths = resolve_model_paths(run_root, opts.model_file)
    post_delta_frac = 1.0 - Float64(SCAN_PREBOUNDARY_FRAC)
    0.0 < post_delta_frac < 1.0 || error(
        "Derived backtrack post_delta_frac must be in (0, 1), got $post_delta_frac " *
        "(from SCAN_PREBOUNDARY_FRAC=$SCAN_PREBOUNDARY_FRAC)."
    )

    dyn = Dict{String,Any}(
        "tspan"                  => (0.0, ODE_TSPAN_END),
        "reltol"                 => ODE_RELTOL,
        "abstol"                 => ODE_ABSTOL,
        "saveat"                 => nothing,
        "eps_extinct"            => ODE_EPS_EXTINCT,
        "post_delta_frac"        => post_delta_frac,
        "post_delta_abs"         => BACK_POST_DELTA_ABS,
        "post_delta_abs_default" => 1e-6,
    )
    back = build_backtrack_cfg()

    println("Backtrack perturbation")
    println("  run: $run_root")
    println("  files: $(length(model_paths))")
    println("  output: canonical model files in-place")
    println("  processed boundaries: all")
    println("  reltol: $(dyn["reltol"])")
    println("  abstol: $(dyn["abstol"])")
    println("  eps_extinct: $(dyn["eps_extinct"])")
    println("  check_stability: $(back["check_stability"])")
    println("  check_invasibility: $(back["check_invasibility"])")

    n_models = length(model_paths)
    for (idx, model_path) in enumerate(model_paths)
        model_name = basename(model_path)
        println("[$idx/$n_models] processing $model_name")

        model = to_dict(JSON3.read(read(model_path, String)))
        payload = backtrack_model(model, dyn, back)
        safe_write_json(model_path, payload)
        println("      wrote $model_name")
    end

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
