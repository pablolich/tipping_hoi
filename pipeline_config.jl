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

# ─── 2. HC solver — shared by boundary_scan.jl and backtrack_perturbation.jl ─
const ZERO_ABUNDANCE = 1e-9       # extinction / zero-abundance floor
const PARAM_TOL      = 1e-9       # bisection convergence in parameter space
const MAX_STEPS_PT   = 1_000_000
const MAX_ITERS      = 8          # max bisection depth

# ─── 3. Boundary scan (boundary_scan.jl) ────────────────────────────────────
# Fallback alpha grid — used only when a bank JSON has no "alpha_grid" field.
const SCAN_ALPHA_GRID       = collect(LinRange(0.0, 1.0, 11))  # 0.0, 0.1, 0.2, ..., 1.0
const SCAN_MAX_PERT         = 1000.0
const SCAN_PREBOUNDARY_FRAC = 0.99
const SCAN_LINEAR_ALPHA_TOL = 1e-14
const SCAN_LAMBDA_TOL       = 1e-9
const SCAN_UNSTABLE_STEPS   = 128
const SCAN_CHECK_STABILITY  = true   # set false to ignore instability events (only negativity/fold trigger boundaries)

# ─── 4. ODE integration — shared by post_boundary_dynamics.jl and backtrack_perturbation.jl ─
const ODE_TSPAN_END   = 10000.0
const ODE_RELTOL      = 1e-8
const ODE_ABSTOL      = 1e-10
const ODE_EPS_EXTINCT = 1e-9

# Per-capita steady-state terminator (post_boundary / backtrack ODE solves).
# Terminate once |F_i| = |du_i/u_i| < POST_SS_PERCAP_TOL for every species with
# u_i > POST_SS_U_THRESH.  Species below the threshold are ignored — the
# extinction callback owns them.  For GLV/HOI, |F_i| is O(1) away from
# equilibrium, so 1e-6 is tight but comfortably within Tsit5 reltol=1e-8.
const POST_SS_PERCAP_TOL = 1e-6
const POST_SS_U_THRESH   = ODE_EPS_EXTINCT

# ─── 5. Post-boundary dynamics (post_boundary_dynamics.jl) ──────────────────
# Set POST_POSTBOUNDARY_FRAC to a Float64 > 1 to override the symmetric default
# (symmetric default: post = 2 - pre, e.g. pre=0.99 → post=1.01).
const POST_POSTBOUNDARY_FRAC = nothing
const POST_SEED              = nothing   # set to an Int for reproducible nudge directions
const POST_NUDGE_ABS         = 1e-8
const POST_NUDGE_REL         = 1e-6

# ─── 6. Backtrack perturbation (backtrack_perturbation.jl) ──────────────────
const BACK_POST_DELTA_ABS   = nothing   # set to Float64 to override (1 - SCAN_PREBOUNDARY_FRAC)
const BACK_MAX_STEP_RATIO   = 0.3
const BACK_LAMBDA_TOL       = 1e-9
const BACK_INVASION_TOL     = 1e-10
const BACK_EPS_SEED_EXTINCT = nothing   # defaults to 10 * ODE_EPS_EXTINCT

# ─── 7. Zero-clamp tolerances — used by integrate_and_snap (Stage 3) ────────
const SNAP_TOL_POS    = 1e-9
const SNAP_TOL_NEG    = 1e-9
