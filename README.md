# Tipping Points in GLV+HOI Communities

A Julia pipeline for detecting, characterising, and testing the reversibility of
tipping points in ecological communities described by Generalised Lotka–Volterra
models with Higher-Order Interactions (GLV+HOI).

---

## Scientific background

### The GLV+HOI model

Community dynamics are governed by the per-capita growth equation

```
dx_i/dt = x_i * f_i(x)
```

where the per-capita growth rate `f_i` interpolates between pairwise (linear) and
higher-order (quadratic) interaction terms via a mixing parameter α ∈ [0, 1]:

```
f_i(x) = r_i + (1 − α)(Ax)_i + α (Bxx)_i
```

- **r** ∈ ℝⁿ — intrinsic growth rates (the bifurcation parameter in this pipeline)
- **A** ∈ ℝⁿˣⁿ — pairwise interaction matrix; `A[i,j]` is the per-capita effect of species j on species i
- **B** ∈ ℝⁿˣⁿˣⁿ — higher-order interaction (HOI) tensor; stored in **[j,k,i]** convention so that `(Bxx)_i = Σ_{j,k} B[j,k,i] x_j x_k`
- **α** — interpolation weight: α = 0 recovers pure pairwise GLV; α = 1 gives a fully HOI-driven system

Equilibria satisfy `f_i(x*) = 0` for all surviving species i.

### Tipping points and critical transitions

A *tipping point* occurs when a small, continuous change in an external driver
(here the growth rate vector **r**) causes the community to undergo an abrupt,
discontinuous transition to a qualitatively different state — typically one with
fewer surviving species. These transitions are associated with three generic
bifurcation mechanisms:

| Mechanism | Flag | Description |
|-----------|------|-------------|
| **Fold** (saddle-node) | `fold` | The coexistence equilibrium collides with an unstable equilibrium and both disappear; the system jumps discontinuously. |
| **Transcritical / negative** | `negative` | A species abundance crosses zero and the equilibrium ceases to be feasible; one or more species go extinct. |
| **Hopf / instability** | `unstable` | The equilibrium loses Lyapunov stability (leading eigenvalue of the community Jacobian crosses zero) before abundances go negative. |

Whether a transition is a fold, a transcritical crossing, or an instability has
direct consequences for **reversibility**: fold bifurcations produce hysteresis
(the system does not return to the original state when the perturbation is
relaxed), while transcritical crossings may be reversible.

### Boundary detection via homotopy continuation

The critical perturbation magnitude along a given direction **u** is found by
numerically continuing the equilibrium `x*(δ)` as δ increases from 0, tracking
when the first catastrophic event (fold, species loss, or instability) occurs.
This uses **polynomial homotopy continuation** (HomotopyContinuation.jl), which
follows a branch of equilibria by deforming a polynomial system whose parameters
(the perturbation `δ·u`) are moved continuously from 0 to a target value. The
critical δ is pinpointed to tolerance by binary bisection in parameter space.

Scanning over many random directions **u** (sampled uniformly on the unit sphere)
maps out the tipping boundary as a distance function on the parameter sphere — a
high-dimensional analogue of a bifurcation diagram.

### Post-boundary dynamics and reversibility

After locating the boundary at δ_c, the ODE is integrated from a state just
*past* the boundary (at δ_post > δ_c). The community relaxes to a new attractor,
which is identified by comparison against all stable equilibria computed via HC
(the *skeleton system* `x_i · f_i(x) = 0`, which encodes every face equilibrium
simultaneously).

To test **reversibility / hysteresis**, the perturbation is then walked back
toward zero using a second round of HC tracking starting from the post-boundary
attractor. If the community can track the equilibrium branch all the way back to
δ = 0 (`returned_n` class), the transition is reversible. If the system instead
encounters another event (fold, invasion, instability) before returning, the
transition exhibits hysteresis — a hallmark of fold bifurcations.

### The role of α and the two parameterisations

The mixing parameter α controls how much of the community's self-regulation comes
from HOI versus pairwise terms. Two parameterisations are supported:

**Standard parameterisation** — The equilibrium is fixed at **x*** = **1** (all
species at equal abundance) by construction. The parameters `(r, A, B)` are
sampled and rescaled so that `f_i(1) = 0` exactly. Scanning is performed over a
grid of α values, allowing the role of HOI strength to be varied independently of
the interaction coefficients.

**Gibbs parameterisation** — Systems are loaded from a pre-existing bank
(`robustness_check/systems_refgrid_bank/`) generated under a different sampling
protocol, where equilibria need not be at **1**. The equilibrium condition is:

```
R_i = (Ax*)_i + (Bx*x*)_i
```

with no explicit α (the full HOI and pairwise terms both contribute). An effective
`alpha_eff = H / (P + H)` is computed post-hoc from the magnitudes of pairwise
(P) and HOI (H) contributions at **x***, but it is a derived quantity rather than
a free parameter. Three qualitative regimes are distinguished (Q1/Q2/Q3) based on
the sign structure of the interaction tensors.

---

## Pipeline overview

The pipeline consists of four sequential stages. All settings live in
`pipeline_config.jl`; no flags are passed at the command line (except the bank name).
The dynamics mode (standard vs. Gibbs) is read automatically from each model's JSON.

```
Stage 1                Stage 2                Stage 3                Stage 4
generate_bank.jl  →  boundary_scan.jl  →  post_boundary_dynamics.jl  →  backtrack_perturbation.jl
     ↓                     ↓                        ↓                           ↓
 model JSON           scan_results             post_dynamics_results       backtrack_results
 (r, A, B, U,        (flag, delta_c,          (x_postboundary_snap,       (reversal_frac,
  x_star)             x_preboundary)           n_lost)                     class_label)
```

Each stage reads and *overwrites* the same JSON file in `model_runs/<bank>/`,
progressively enriching it with new result keys.

### Stage 1 — Bank generation

**`generate_bank.jl`** (standard) or **`generate_bank_gibbs.jl`** (Gibbs)

Samples `n_models` random communities for each combination of `(n, n_dirs)` and
writes one JSON per model. For the standard mode, `(r, A, B)` are drawn so that
`x* = 1` is an equilibrium and the Jacobian is stable. For Gibbs mode the systems
are converted from the reference bank and stored with explicit `x_star`.

Random ray directions **U** (columns of an n × n_dirs matrix, each column a unit
vector) are sampled uniformly on the sphere and stored alongside the model
parameters. The same rays are reused by all subsequent stages.

### Stage 2 — Boundary scan

**`boundary_scan.jl`**

For each model and each ray direction **u**, the code tracks the equilibrium branch
`x*(δ)` from δ = 0 outward until a boundary event is detected. The result for
each (ray, alpha) combination is:

- `flag` — the type of boundary event: `negative`, `fold`, or `unstable`
- `delta_c` — the critical perturbation magnitude ‖δ_c · u‖
- `drcrit` — the critical perturbation vector δ_c · u
- `x_preboundary` — the equilibrium at a fraction `preboundary_frac · δ_c`, used as the ODE seed in Stage 3

For α = 0 (pure pairwise), equilibria are linear in δ and the boundary is found
analytically. For all other α, HomotopyContinuation's `CoefficientHomotopy` is
used, stepping through parameter space and halting on the first event.

### Stage 3 — Post-boundary dynamics

**`post_boundary_dynamics.jl`**

Integrates the GLV+HOI ODE (using DifferentialEquations.jl, Tsit5 solver) from
`x_preboundary` with a small random nudge, at a perturbation `delta_post >
delta_c` (by default, symmetric: `delta_post = (2 - preboundary_frac) · delta_c`).
The trajectory is snapped to the nearest stable equilibrium from the set computed
by solving the skeleton system via HC at `delta_post`.

Outputs per ray:
- `x_postboundary_snap` — the post-boundary attractor
- `n_lost_post` — number of species extinct in the new attractor
- `snap_reason` — whether snapping succeeded or failed

### Stage 4 — Backtrack perturbation

**`backtrack_perturbation.jl`**

Starting from `x_postboundary_snap`, HC tracks the equilibrium branch *back*
toward δ = 0, checking at each step for:
1. **Stability loss** (leading Jacobian eigenvalue > 0)
2. **Invasion** (a previously extinct species can reinvade the current attractor)
3. **Fold** (the branch terminates)

The first event encountered defines `delta_event`. A probe ODE integration at
`delta_probe ≈ delta_event - ε` tests whether the community actually returns to
the n-species coexistence state. The key output is:

- `returned_n` (bool) — did the community recover full coexistence?
- `reversal_frac` — `(delta_post − delta_return) / delta_post` — fraction of the perturbation that was reversed
- `class_label` — `returned_n`, `boundary_persist`, `success_to_zero`, `snap_too_far`, etc.

---

## Repository structure

```
new_code/
├── pipeline_config.jl          # All tunable constants (single source of truth)
├── generate_bank.jl            # Stage 1 — standard parameterisation
├── generate_bank_gibbs.jl      # Stage 1 — Gibbs parameterisation
├── boundary_scan.jl            # Stage 2 — HC boundary detection
├── post_boundary_dynamics.jl   # Stage 3 — ODE integration past boundary
├── backtrack_perturbation.jl   # Stage 4 — HC backtrack + reversibility test
├── model_store_utils.jl        # JSON I/O helpers (safe_write_json, resolve paths)
├── model_runs/                 # Output directory — one subdirectory per bank
│   └── <bank_name>/
│       └── model_*.json        # Enriched model files (all four stages in-place)
└── utils/
    ├── math_utils.jl           # Jacobians, per-capita growth, lambda_max (standard + Gibbs)
    ├── equilibrium_utils.jl    # HC skeleton solver, snap_to_equilibrium (standard + Gibbs)
    ├── glvhoi_utils.jl         # HC system builders and ODE rhs (standard + Gibbs)
    ├── hc_param_utils.jl       # Parameter-space helpers for HC tracking
    ├── json_utils.jl           # JSON ↔ Julia array/tensor converters
    └── dynamics_cfg_utils.jl   # Shared ODE config and snap_confident logic
```

---

## Quick start

### Standard mode

```bash
# Stage 1 — generate bank: n=3..5, 50 models each, 16 ray directions
julia --startup-file=no new_code/generate_bank.jl 3:5 50 16

# Stages 2–4 — use the bank name printed by Stage 1
BANK="1_bank_50_models_n_3-5_16_dirs_b_dirichlet"
julia --startup-file=no new_code/boundary_scan.jl           $BANK
julia --startup-file=no new_code/post_boundary_dynamics.jl  $BANK
julia --startup-file=no new_code/backtrack_perturbation.jl  $BANK
```

### Gibbs mode

```bash
# Stage 1 — convert reference systems bank (Q1/Q2/Q3) to new_code format
julia --startup-file=no new_code/generate_bank_gibbs.jl \
    robustness_check/systems_refgrid_bank/ 16

# Stages 2–4 — dynamics_mode is read per-model from the JSON, no env var needed
BANK="gibbs_16_dirs_from_systems_refgrid_bank"
julia --startup-file=no new_code/boundary_scan.jl           $BANK
julia --startup-file=no new_code/post_boundary_dynamics.jl  $BANK
julia --startup-file=no new_code/backtrack_perturbation.jl  $BANK
```

### Other-model Stage 1-2 banks

```bash
# Stage 1 — generate a Mougi random-web bank
julia --startup-file=no new_code/other_models/generate_bank_mougi.jl 10 50 42

# Stage 2 — scan the generated bank
BANK="mougi_random_n6_10models_50dirs_seed42"
julia --startup-file=no new_code/boundary_scan.jl $BANK

# Stage 1 — generate a Sanders carrying-capacity bank
julia --startup-file=no new_code/other_models/generate_bank_sanders.jl 10 50 42

# Stage 2 — scan the generated bank
BANK="sanders_carrycap_n3_10models_50dirs_seed42"
julia --startup-file=no new_code/boundary_scan.jl $BANK
```

`mougi`, `lever`, `karatayev`, `aguade`, and `sanders` banks carry model-specific
payloads instead of standard `A`/`B` tensors. In this pass, Mougi and Sanders
follow the same **Stage 1–2 only** pattern;
`post_boundary_dynamics.jl` and `backtrack_perturbation.jl` still assume
standard/Gibbs-style `A` and `B` fields.

To process a single file (useful for debugging or cluster parallelism):

```bash
julia --startup-file=no new_code/boundary_scan.jl $BANK \
    --model-file model_n_3_seed_31000002_n_dirs_10.json
```

---

## Configuration

All settings are in `pipeline_config.jl`. The most important ones:

| Constant | Default | Description |
|----------|---------|-------------|
| `SCAN_ALPHA_GRID` | `0.0:0.1:1.0` | α values to scan in standard mode |
| `SCAN_MAX_PERT` | `10.0` | Maximum perturbation magnitude ‖δ · u‖ |
| `SCAN_PREBOUNDARY_FRAC` | `0.99` | Fraction of δ_c used as ODE seed in Stage 3 |
| `MAX_ITERS` | `8` | Maximum bisection depth for boundary localisation |
| `PARAM_TOL` | `1e-9` | Convergence tolerance in parameter space |
| `ODE_TSPAN_END` | `10000.0` | Integration time for ODE stages |
| `BANK_SIGMA_A` | `1.0` | Standard deviation of off-diagonal entries in A |
| `BANK_B_SAMPLER` | `"dirichlet"` | HOI sampling mode: `"dirichlet"` or `"signed_flip"` |

---

## Model JSON schema

Each file in `model_runs/<bank>/` accumulates fields from all completed stages:

```
{
  // Bank fields (Stage 1)
  "n":         int,              // number of species
  "n_dirs":    int,              // number of ray directions
  "r":         [float × n],      // baseline growth rates
  "A":         [[float × n] × n],// pairwise interaction matrix
  "B":         [[[float]×n]×n]×n,// HOI tensor, B[j][k][i] convention
  "U":         [[float × n] × n_dirs], // ray directions (columns)
  "x_star":    [float × n],      // baseline equilibrium (ones for standard)
  "dynamics_mode": "standard" | "gibbs",

  // Stage 2 — boundary scan
  "scan_config":  { ... },
  "scan_results": [              // one entry per alpha (or one for Gibbs)
    {
      "alpha": float,
      "directions": [            // one entry per ray
        {
          "flag":          "negative" | "fold" | "unstable",
          "delta_c":       float,   // critical perturbation magnitude
          "drcrit":        [float × n],
          "x_preboundary": [float × n]
        }, ...
      ]
    }, ...
  ],

  // Stage 3 — post-boundary dynamics
  "post_dynamics_results": [
    { "directions": [{ "x_postboundary_snap": [...], "n_lost_post": int, ... }] }
  ],

  // Stage 4 — backtrack
  "backtrack_results": [
    { "directions": [{ "returned_n": bool, "reversal_frac": float,
                       "class_label": str, "delta_return": float }] }
  ]
}
```

---

## B tensor convention

Throughout this codebase, the HOI tensor **B** is stored with the **[j, k, i]**
index order, so the quadratic interaction term for species i is:

```
(Bxx)_i = Σ_{j,k} B[j, k, i] · x_j · x_k
```

When loading systems from the Gibbs reference bank (whose JSON stores **B** in
**[i, j, k]** order), `generate_bank_gibbs.jl` applies the permutation
`B_jki[j, k, i] = B_ijk[i, j, k]` before writing the output bank.

---

## Dependencies

Tested with **Julia 1.12**. Required packages:

| Package | Usage |
|---------|-------|
| `HomotopyContinuation` | Polynomial homotopy tracking (Stages 2 and 4) |
| `DifferentialEquations` | ODE integration — Tsit5 solver (Stages 3 and 4) |
| `SciMLBase` | Return-code inspection for ODE solves |
| `LinearAlgebra` | Eigenvalues, matrix operations |
| `JSON3` | Fast JSON serialisation / deserialisation |
| `Random` | Reproducible random number generation |
# tipping_hoi
