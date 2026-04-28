# Midway3 Cluster Pipeline (`new_code`)

This folder provides Slurm wrappers for running the canonical JSON pipeline at scale on Midway3.

## Julia environment pinning

`env_midway3.sh` auto-reads `Manifest.toml` and enforces the same Julia
major.minor version (currently `1.10`). It also defaults to a project-local
depot to avoid cross-version cache conflicts:

- `MIDWAY3_JULIA_MODULE` default: `julia/<manifest julia_version>`
- `JULIA_DEPOT_PATH` default: `<repo>/.julia_depot/v<major.minor>`

One-time setup on Midway3:

```bash
cd ~/tipping_points_hoi
source /etc/profile.d/modules.sh
module load julia/1.10.2
JULIA_EXE="$(which julia)" JULIA_DEPOT_PATH="$PWD/.julia_depot/v1.10" \
  julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

## Files

- `env_midway3.sh`: shared environment defaults for submission and task scripts.
- `make_manifest.sh`: builds a deterministic model list (`.txt`) from `new_code/model_runs/<run_dir>`.
- `stage_task.sh`: array-task worker; runs one stage on one model file.
- `submit_stage.sh`: submits one stage as a Slurm array.
- `submit_pipeline.sh`: submits boundary -> post -> backtrack with verify dependencies.
- `verify_stage.jl`: verifies that all models contain stage output keys.

## 1) Prepare a run directory

From repo root:

```bash
julia --project=. --startup-file=no new_code/pipeline/generate_bank.jl 3:5 200 100
ls new_code/model_runs
```

Take the generated folder name as `<run_dir>`.

## 2) Create manifest

```bash
new_code/cluster/make_manifest.sh <run_dir>
```

This writes:

- `new_code/cluster/manifests/<run_dir>.models.txt`

## 3) Submit full pipeline

```bash
new_code/cluster/submit_pipeline.sh \
  --partition caslake \
  --run-dir <run_dir> \
  --array-throttle 100
```

Slurm account defaults to `pi-salesina` via `MIDWAY3_SLURM_ACCOUNT` in `env_midway3.sh`.

Default per-task resources:

- boundary: `1 CPU`, `8G`, `12:00:00`
- post: `1 CPU`, `8G`, `08:00:00`
- backtrack: `1 CPU`, `12G`, `24:00:00`

Override with:

- `--mem-boundary`, `--mem-post`, `--mem-backtrack`
- `--time-boundary`, `--time-post`, `--time-backtrack`
- `--cpus-boundary`, `--cpus-post`, `--cpus-backtrack`

## 4) Submit a single stage manually

```bash
new_code/cluster/submit_stage.sh \
  --stage boundary \
  --run-dir <run_dir> \
  --partition caslake
```

You can chain to a prior job with `--dependency <jobid>`.

## 5) Resume from later stage

If boundary is already complete and verified:

```bash
new_code/cluster/submit_pipeline.sh \
  --partition caslake \
  --run-dir <run_dir> \
  --start-stage post
```

## 6) Gibbs parameterization workflow

The Gibbs bank is generated from existing `robustness_check` systems rather than `generate_bank.jl`.

### Generate a Gibbs bank

```bash
julia --project=. --startup-file=no new_code/pipeline/generate_gibbs_refgrid.jl
ls new_code/model_runs
```

Take the generated folder name as `<run_dir>` (same flat-JSON format as a standard bank).

### Submit a Gibbs pipeline

Pass `--mode gibbs` to propagate `DYNAMICS_MODE=gibbs` to all stage tasks:

```bash
new_code/cluster/submit_pipeline.sh \
  --partition caslake \
  --run-dir <run_dir> \
  --mode gibbs \
  --array-throttle 100
```

Or submit a single stage:

```bash
new_code/cluster/submit_stage.sh \
  --stage boundary \
  --run-dir <run_dir> \
  --mode gibbs \
  --partition caslake
```

Standard and Gibbs pipelines can run concurrently from the same repo checkout because the mode is passed via environment variable rather than a shared config file.

### Gibbs resource hints

Gibbs boundary scan evaluates only one alpha value per model (vs. up to 11 for standard), so the 12-hour boundary walltime default is very conservative. You can safely shorten it:

```bash
new_code/cluster/submit_pipeline.sh \
  --run-dir <run_dir> \
  --mode gibbs \
  --time-boundary 02:00:00
```

## 7) Logs and monitoring

Logs are written to:

- `new_code/cluster/logs/<run_dir>/<stage>/%A_%a.out`
- `new_code/cluster/logs/<run_dir>/<stage>/%A_%a.err`
- `new_code/cluster/logs/<run_dir>/verify/%j.out`
- `new_code/cluster/logs/<run_dir>/verify/%j.err`

Useful commands:

```bash
squeue -u plechon
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS
```
