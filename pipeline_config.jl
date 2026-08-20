# pipeline_config.jl — central configuration for all four pipeline stages.
# Edit values here instead of passing command-line flags.

# ─── 1. Bank generation (generate_bank.jl) ──────────────────────────────────
const DYNAMICS_MODE      = "standard"  # "standard", "unique_equilibrium", or "all_negative"
const BANK_BASE_SEED     = 2
const BANK_SIGMA_A       = 1.0
const BANK_SIGMA_B       = 1.0          # σ_B for all_negative mode (B entries ~ N(0, σ_B²/n²))
const BANK_MU_A          = parse(Float64, get(ENV, "BANK_MU_A", "0.0"))   # mean of A0 entries; override via env var
const BANK_MU_B          = parse(Float64, get(ENV, "BANK_MU_B", "0.0"))   # mean of B0 entries; override via env var
const BANK_SAFETY_UE     = 1.2          # safety margin for unique_equilibrium negdef diagonal shift

# ─── 2. Shared numerical floors and HC solver settings ─────────────────────
const ZERO_ABUNDANCE = 1e-9       # extinction / zero-abundance floor; shared by HC solver, ODE callbacks, and final zero-clamp
const PARAM_TOL      = 1e-9       # bisection convergence in parameter space
const LAMBDA_TOL     = 1e-9       # eigenvalue/stability tolerance; shared by boundary scan and backtrack
const MAX_STEPS_PT   = 1_000_000
const MAX_ITERS      = 8          # max bisection depth

# ─── 3. Boundary scan (boundary_scan.jl) ────────────────────────────────────
# Fallback alpha grid — used only when a bank JSON has no "alpha_grid" field.
const SCAN_ALPHA_GRID       = collect(LinRange(0.0, 1.0, 11))  # 0.0, 0.1, 0.2, ..., 1.0
const SCAN_MAX_PERT         = 1000.0
const SCAN_PREBOUNDARY_FRAC = 0.99
const SCAN_LINEAR_ALPHA_TOL = 1e-14
const SCAN_UNSTABLE_STEPS   = 128
const SCAN_CHECK_STABILITY  = true   # set false to ignore instability events (only negativity/fold trigger boundaries)

# ─── 4. ODE integration — shared by post_boundary_dynamics.jl and backtrack_perturbation.jl ─
const ODE_TSPAN_END   = 10000.0
const ODE_RELTOL      = 1e-8
const ODE_ABSTOL      = 1e-10

# Per-capita steady-state terminator (post_boundary / backtrack ODE solves).
# Terminate once |F_i| = |du_i/u_i| < POST_SS_PERCAP_TOL for every species with
# u_i > ZERO_ABUNDANCE.  Species below the threshold are ignored — the
# extinction callback owns them.  For GLV/HOI, |F_i| is O(1) away from
# equilibrium, so 1e-6 is tight but comfortably within Tsit5 reltol=1e-8.
const POST_SS_PERCAP_TOL = 1e-6

# ─── 5. Post-boundary dynamics (post_boundary_dynamics.jl) ──────────────────
# Set POST_POSTBOUNDARY_FRAC to a Float64 > 1 to override the symmetric default
# (symmetric default: post = 2 - pre, e.g. pre=0.99 → post=1.01).
const POST_POSTBOUNDARY_FRAC = nothing
const POST_SEED              = nothing   # mixed into the per-ray nudge seed; see below
const POST_NUDGE_ABS         = 1e-8
const POST_NUDGE_REL         = 1e-6

# Skip the ODE for rays the scan never found a boundary for (flag == "success",
# i.e. delta_c == max_pert).  There is no boundary to step past, every consumer
# already discards the landing state — figures/explore_2D_julia_helper.jl:99
# skips them, backtrack_perturbation.jl maps the null snap to "missing_x_post" —
# and on the parameterization_v2 banks they are 4% of rays but ~64% of stage-3
# runtime (~0.7-1.7 s each against ~5 ms for a fold ray).
#
# Default `false` so a bank measured before this flag existed reproduces
# exactly.  Measured exposure of flipping it on the submitted banks: of 202,752
# rows sampled across the three banks that carry post_dynamics_results, 5,237
# are flag == "success" and 4 of those (0.002% of all rows) hold a real
# x_postboundary_snap that would become null.  Set `true` for new banks.
const POST_SKIP_NO_BOUNDARY = false

# Record the solver's own retcode in snap_reason ("ode_fail_Unstable",
# "ode_fail_MaxIters", ...) instead of a flat "ode_fail".  The two mean very
# different things: `Unstable` is a trajectory diverging in finite time — a real
# statement that the model has no bounded post-boundary state — while
# `MaxIters` is only a step-budget failure.  On parameterization_v2 the split is
# 2,069 vs 9 over 7,365 fold rays, which the flat label hides entirely.
#
# Nothing under figures/ or postprocess/ reads snap_reason, so this is
# diagnostic only; the "ode_fail" prefix is kept so a startswith test still
# works.  Default `false` to keep stored strings byte-identical.
const POST_RECORD_RETCODE = false

# Skip model files that already carry post_dynamics_results.  Default `true`:
# the driver rewrites each file IN PLACE and drops backtrack_results on the way
# through (see the delete! at the end of scan_model_post_dynamics), so a re-run
# over a finished bank silently destroys stage-4 output.  Pass --force to
# recompute anyway.
const POST_SKIP_DONE = true

# ─── 6. Backtrack perturbation (backtrack_perturbation.jl) ──────────────────
const BACK_POST_DELTA_ABS   = nothing   # set to Float64 to override (1 - SCAN_PREBOUNDARY_FRAC)
const BACK_INVASION_TOL     = 1e-10
const BACK_EPS_SEED_EXTINCT = nothing   # defaults to 10 * ZERO_ABUNDANCE
