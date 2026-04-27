# Tipping Points in GLV+HOI Communities

Code for the paper **"Tipping points are typical in ecosystems with higher-order
interactions"** (`main.tex`, `supplementary_information.tex`).

This is a Julia/Python pipeline for detecting, characterising, and testing the
reversibility of tipping points in ecological communities described by
Generalised Lotka–Volterra models with Higher-Order Interactions (GLV+HOI), and
for replicating the same analysis in five published multispecies models
(Lever, Karatayev, Aguadé-Gorgorió, Mougi, Stouffer) and the Gibbs replication.

---

## Scientific background

### The GLV+HOI model

Community dynamics are governed by the per-capita growth equation

```
dx_i/dt = x_i * f_i(x)
```

where the per-capita growth rate `f_i` interpolates between pairwise (linear)
and higher-order (quadratic) interaction terms via a mixing parameter α ∈ [0, 1]:

```
f_i(x) = r_i + (1 − α)(Ax)_i + α (Bxx)_i
```

- **r** ∈ ℝⁿ — intrinsic growth rates (the bifurcation parameter in this pipeline)
- **A** ∈ ℝⁿˣⁿ — pairwise interaction matrix; `A[i,j]` is the per-capita effect of species j on species i
- **B** ∈ ℝⁿˣⁿˣⁿ — higher-order interaction tensor; stored in **[j,k,i]** convention so `(Bxx)_i = Σ_{j,k} B[j,k,i] x_j x_k`
- **α** — interpolation weight: α = 0 recovers pure pairwise GLV; α = 1 gives a fully HOI-driven system

Equilibria satisfy `f_i(x*) = 0` for all surviving species i.

### Tipping points and critical transitions

A tipping point occurs when a small change in an external driver (here the
growth rate vector **r**) causes the community to jump discontinuously to a
qualitatively different state. The pipeline classifies transitions into three
mechanisms:

| Mechanism | Flag | Description |
|-----------|------|-------------|
| **Fold** (saddle-node) | `fold` | Coexistence equilibrium collides with an unstable equilibrium and both disappear; the system jumps discontinuously. |
| **Transcritical / negative** | `negative` | A species abundance crosses zero and the equilibrium ceases to be feasible. |
| **Hopf / instability** | `unstable` | The equilibrium loses stability before abundances go negative. |

Whether the transition is a fold, transcritical crossing, or instability has
direct consequences for **reversibility**: fold bifurcations produce hysteresis,
while transcritical crossings may be reversible.

### Boundary detection via homotopy continuation

The critical perturbation magnitude along a direction **u** is found by
numerically continuing the equilibrium `x*(δ)` as δ increases from 0, using
**polynomial homotopy continuation** (HomotopyContinuation.jl). The critical δ
is pinpointed by binary bisection in parameter space.

Scanning over many random directions sampled uniformly on the unit sphere maps
the tipping boundary as a distance function on the parameter sphere — a
high-dimensional analogue of a bifurcation diagram.

### Post-boundary dynamics and reversibility

After locating the boundary at δ_c, the ODE is integrated past the boundary at
δ_post > δ_c until the per-capita rates fall below tolerance (with one doubled
retry if no species has gone extinct). Residual abundances below `tol_pos` are
zero-clamped, defining the post-boundary attractor. To test hysteresis, the
perturbation is then walked back toward zero using a second round of HC
tracking. If the community tracks the equilibrium branch all the way back to
δ = 0, the transition is reversible; otherwise it exhibits hysteresis.

---

## Pipeline overview

The pipeline has four sequential stages plus two post-processing steps. Each
stage reads and overwrites the same JSON file in `data/example_runs/<bank>/`,
progressively enriching it.

```
Stage 1                Stage 2                Stage 3                  Stage 4
generate            boundary_scan.jl       post_boundary_dynamics.jl    backtrack_perturbation.jl
     ↓                     ↓                        ↓                           ↓
 model JSON           scan_results            post_dynamics_results        backtrack_results
 (r, A, B, U,        (flag, delta_c,         (x_postboundary_snap,        (returned_n,
  x_star)             x_preboundary)          n_lost_post)                 reversal_frac, class_label)

Post-processing (additive, never overwrites):
   add_alpha_eff_taylor.jl  →  alpha_eff_taylor[_grid]
   add_alpha_eff_hull.jl    →  alpha_eff_hull[_grid]
```

### Stage 1 — Bank generation (3 paths)

| Generator | Modes produced | Used for |
|-----------|----------------|----------|
| `generate_bank.jl` | `elegant`, `unique_equilibrium`, `all_negative` | Random GLV+HOI ensembles (Figs 3, S3, S4, S10–S13) |
| `generate_gibbs_refgrid.jl` | `gibbs` | Gibbs replication, three regimes (Figs S6, S7) |
| `other_models/generate_bank_<X>.jl` | `lever`, `karatayev`, `aguade`, `mougi`, `stouffer` | Published multispecies models (Fig 4, Fig S8) |

Each generator writes one JSON per model with `(r, A, B, U, x_star,
dynamics_mode, alpha_grid, ...)`. Random ray directions **U** (columns of an
n × n_dirs matrix) are sampled uniformly on the sphere.

### Stage 2 — Boundary scan

`boundary_scan.jl`. For each model and ray direction **u**, tracks the
equilibrium branch `x*(δ)` from δ = 0 outward until a boundary event. For α = 0
(pure pairwise) the boundary is found analytically; for α > 0 HC is used.

Outputs per (ray, alpha): `flag` (`negative` / `fold` / `unstable` / `success`),
`delta_c`, `drcrit`, `x_preboundary`.

### Stage 3 — Post-boundary dynamics

`post_boundary_dynamics.jl`. Integrates the GLV+HOI ODE (Tsit5) from
`x_preboundary` at `delta_post > delta_c` with an extinction callback that
zero-clamps any species below `eps_extinct`, terminating once all surviving
species have per-capita rate below tolerance. If no species has gone extinct
when the time window ends, the integration is extended once for a doubled
window. Residual abundances below `tol_pos` are zero-clamped to define the
attractor; if no extinction has occurred after both passes, the result is
flagged `:no_extinction`. There is no algebraic equilibrium solve here.
Outputs `x_postboundary_snap`, `n_lost_post`, `snap_reason`.

### Stage 4 — Backtrack perturbation

`backtrack_perturbation.jl`. From `x_postboundary_snap`, HC tracks the
equilibrium branch back toward δ = 0, looking for stability loss, invasion, or
fold events. Outputs `returned_n`, `reversal_frac`, `class_label`,
`delta_return`. Hysteresis is `reversal_frac < 1` (path reversal incomplete).

### Post-processing — effective nonlinearity metrics

`add_alpha_eff_taylor.jl` and `add_alpha_eff_hull.jl` add per-system
nonlinearity metrics that allow cross-model comparison (used on Figs S5, S7,
S8). These are additive: they never overwrite scan / post / backtrack results.

---

## Reproducing the paper

The paper's figures are produced by Python scripts in `figures/`, which read
the enriched bank JSONs in `data/example_runs/`. The mapping:

| Figure | Producer | Banks required |
|--------|----------|----------------|
| Fig 1 | `figures/hysteresis_panels.py` (+ `build_hysteresis_table.jl` to assemble input) | `2_bank_elegant_*` (full 4-stage pipeline) |
| Fig 2 | `figures/plot_boundaries.py` (+ `figures/generate_boundary_data.jl`, plus `J1.jl` / `J2.jl` polynomial inputs) | n=2 analytical + numerical, no model bank needed |
| Fig 3 | `figures/figure_3.py` | `2_bank_elegant_50_models_n_4-20_128_dirs_muB_{-0.1, 0.0, 0.1}` |
| Fig 4 | `figures/figure_4.py` (panels A: `combined_pauls_prevalence_figure.py`, B: `figure_4_panel_b_metrics.py`) | All 5 other-models banks + Gibbs (`pauls_idea_branches.jl` produces panel A's CSV) |
| Fig S1 | `figures/boundary_types_fig.tex` | TikZ schematic only |
| Fig S2, S9, S10–S13 | `figures/si_figures.py` | All 4 banks (μ_B variants + unique_equilibrium) |
| Fig S3 | `figures/plot_mu_grid_panels.py` | 9 banks `2_bank_elegant_*_muA_*_muB_*` |
| Fig S4 | `figures/si_boundary_types_2x2_all_negative.py` | `2_bank_all_negative_*` |
| Fig S5 | `figures/alpha_hull_vs_taylor_by_muB.py` | `2_bank_elegant_*_muB_*` (with `alpha_eff_taylor` + `alpha_eff_hull`) |
| Fig S6 | `figures/combined_pauls_prevalence_figure.py` (Gibbs branch) | Gibbs bank |
| Fig S7 | `figures/plot_boundary_types_row_gibbs.py` | Gibbs bank (with `alpha_eff_taylor` + `alpha_eff_hull`) |
| Fig S8 | `figures/figure_4_panel_b_metrics.py` (log-log mode) | All 5 other-models banks + Gibbs |

Example bank JSONs for each required parameterisation are committed in
`data/example_runs/`. To reproduce a figure end-to-end from scratch, run the
matching Stage-1 generator with the paper's config (`pipeline_config.jl`),
followed by stages 2–4 (and α_eff post-processing where required), then run the
figure script.

---

## Repository structure

```
tipping_hoi/
├── pipeline_config.jl              # All tunable constants (single source of truth)
├── generate_bank.jl                # Stage 1 — random GLV+HOI (elegant / unique_equilibrium / all_negative)
├── generate_gibbs_refgrid.jl       # Stage 1 — Gibbs replication
├── boundary_scan.jl                # Stage 2 — HC boundary detection
├── post_boundary_dynamics.jl       # Stage 3 — ODE past boundary
├── backtrack_perturbation.jl       # Stage 4 — HC backtrack + reversibility test
├── add_alpha_eff_taylor.jl         # Post-processing — Taylor nonlinearity metric
├── add_alpha_eff_hull.jl           # Post-processing — convex-hull nonlinearity metric
├── model_store_utils.jl            # JSON I/O helpers
├── merge_chunks.jl                 # HPC: merge chunked Stage 2 outputs
├── rescan_success_directions.jl    # HPC: re-run "success" rays at a higher δ_max
├── utils/
│   ├── glvhoi_utils.jl             # HC system builders + ODE rhs (one per dynamics_mode)
│   ├── math_utils.jl               # Jacobians, eigenvalues
│   ├── boundary_event_utils.jl     # Fold / negative / unstable detection
│   ├── hc_param_utils.jl, hc_tracker_utils.jl   # HC setup + tracker init
│   ├── ode_snap_utils.jl, dynamics_cfg_utils.jl # ODE integration helpers
│   ├── json_utils.jl               # Nested-list ↔ array/tensor conversion
│   └── alpha_eff_taylor.jl         # Taylor metric implementation
├── other_models/                   # Five published multispecies models (Fig 4)
│   ├── lever_model.jl, generate_bank_lever.jl
│   ├── karatayev_model.jl, generate_bank_karatayev.jl
│   ├── aguade_model.jl, generate_bank_aguade.jl
│   ├── mougi_model.jl, generate_bank_mougi.jl
│   └── stouffer_model.jl, generate_bank_stouffer.jl
├── figures/                        # Python figure scripts + Julia data generators
├── tests/                          # Julia test suite
├── cluster/                        # SLURM submission scripts (HPC)
├── data/example_runs/              # Example enriched bank JSONs (one per required parameterisation)
├── main.tex                        # Paper main text
├── supplementary_information.tex   # Paper SI
└── Manifest.toml                   # Julia dependency lock
```

---

## Quick start

The dynamics mode is read automatically from each model's JSON, so stages 2–4
are oblivious to which generator produced the bank.

### Random GLV+HOI (e.g. main Fig 3 bank)

Edit `pipeline_config.jl`: set `DYNAMICS_MODE = "elegant"`, `BANK_BASE_SEED = 2`,
override `BANK_MU_B` via env var if needed.

```bash
# Stage 1 — generate bank (n=4..20, 50 models each, 128 directions)
BANK_MU_B=0.0 julia --startup-file=no generate_bank.jl 4:20 50 128

# Stages 2–4 — bank name is printed by Stage 1
BANK="2_bank_elegant_50_models_n_4-20_128_dirs_muA_0.0_muB_0.0"
julia --startup-file=no boundary_scan.jl           "data/example_runs/$BANK"
julia --startup-file=no post_boundary_dynamics.jl  "data/example_runs/$BANK"
julia --startup-file=no backtrack_perturbation.jl  "data/example_runs/$BANK"

# Optional post-processing
julia --startup-file=no add_alpha_eff_taylor.jl    "data/example_runs/$BANK"
julia --startup-file=no add_alpha_eff_hull.jl      "data/example_runs/$BANK"
```

### Gibbs replication (Figs S6, S7)

```bash
# Stage 1 — convert Gibbs reference grid CSV to per-system JSONs
julia --startup-file=no generate_gibbs_refgrid.jl \
    path/to/reference_grid_prob_gt_0p9.csv 4:10 50 40 12345

# Stages 2–4 same as above
```

### Published multispecies model (e.g. Lever, Fig 4A)

```bash
# Stage 1 — generate Lever plant–pollinator bank
julia --startup-file=no other_models/generate_bank_lever.jl 4 6 8 10 50 128 1

# Stages 2 (and α_eff post-processing) — Stages 3-4 not used for these models
BANK="lever_n4.6.8.10_50models_128dirs_seed1"
julia --startup-file=no boundary_scan.jl              "data/example_runs/$BANK"
julia --startup-file=no add_alpha_eff_taylor.jl       "data/example_runs/$BANK"
julia --startup-file=no add_alpha_eff_hull.jl         "data/example_runs/$BANK"
```

To process a single file (debugging or cluster parallelism):

```bash
julia --startup-file=no boundary_scan.jl "data/example_runs/$BANK" \
    --model-file model_n_4_seed_X_n_dirs_128.json
```

For HPC chunked runs see `cluster/` and `merge_chunks.jl`.

---

## Configuration

All settings are in `pipeline_config.jl`. The most important:

| Constant | Default | Description |
|----------|---------|-------------|
| `DYNAMICS_MODE` | `"elegant"` | Mode used by `generate_bank.jl`: `elegant`, `unique_equilibrium`, or `all_negative` |
| `BANK_SIGMA_A`, `BANK_SIGMA_B` | `1.0`, `1.0` | Standard deviation of A and B entries (for `all_negative`) |
| `BANK_MU_A`, `BANK_MU_B` | env-overridable | Means of A and B entries (for `elegant`) |
| `BANK_BASE_SEED` | `2` | Base seed; per-model seeds derived deterministically |
| `SCAN_ALPHA_GRID` | `LinRange(0, 1, 11)` | α values to scan in GLV+HOI modes |
| `SCAN_MAX_PERT` | `1000.0` | Maximum perturbation magnitude ‖δ · u‖ |
| `SCAN_PREBOUNDARY_FRAC` | `0.99` | Fraction of δ_c used as ODE seed in Stage 3 |
| `MAX_ITERS` | `8` | Maximum bisection depth for boundary localisation |
| `PARAM_TOL` | `1e-9` | Convergence tolerance in parameter space |
| `ODE_TSPAN_END` | `10000.0` | Integration time for ODE stages |

---

## Model JSON schema

Each file in `data/example_runs/<bank>/` accumulates fields from completed
stages:

```jsonc
{
  // Bank fields (Stage 1)
  "n":              int,
  "n_dirs":         int,
  "r":              [float × n],
  "A":              [[float × n] × n],
  "B":              [[[float] × n] × n] × n,   // [j][k][i] convention
  "U":              [[float × n_dirs] × n],
  "x_star":         [float × n],
  "alpha_grid":     [float],                    // present for GLV+HOI modes
  "dynamics_mode":  "elegant" | "unique_equilibrium" | "all_negative" |
                    "gibbs"   | "lever" | "karatayev" | "aguade" |
                    "mougi"   | "stouffer",

  // Stage 2
  "scan_config":  { ... },
  "scan_results": [ { "alpha": float, "directions": [ {flag, delta_c, drcrit, x_preboundary}, ... ] }, ... ],

  // Stage 3
  "post_dynamics_results": [ { "directions": [ {x_postboundary_snap, n_lost_post, snap_reason}, ... ] } ],

  // Stage 4
  "backtrack_results":     [ { "directions": [ {returned_n, reversal_frac, class_label, delta_return}, ... ] } ],

  // Post-processing (optional)
  "alpha_eff_taylor[_grid]": ...,
  "alpha_eff_hull[_grid]":   ...
}
```

---

## B tensor convention

**B** is stored with **[j, k, i]** index order:

```
(Bxx)_i = Σ_{j,k} B[j, k, i] · x_j · x_k
```

The Gibbs reference data uses **[i, j, k]** order; `generate_gibbs_refgrid.jl`
permutes on read.

---

## Dependencies

Tested with **Julia 1.12**. Key packages (full list in `Manifest.toml`):

| Package | Usage |
|---------|-------|
| `HomotopyContinuation` | Polynomial homotopy tracking (Stages 2 and 4) |
| `DifferentialEquations` | ODE integration — Tsit5 solver (Stages 3 and 4) |
| `ForwardDiff` | Jacobian/Hessian for `alpha_eff_taylor` |
| `JSON3` | JSON I/O |

Python figure scripts use **matplotlib**, **numpy**, **pandas**, **scipy**,
**seaborn**, and **pyarrow** (for the hysteresis Arrow tables).

---

## Tests

```bash
julia --startup-file=no --project=. tests/test_alpha_eff_taylor.jl
julia --startup-file=no --project=. tests/test_backtrack.jl
```
