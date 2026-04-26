#!/usr/bin/env bash
# Submit one pipeline stage as a Slurm array job.
#
# Usage:
#   run_stage.sh <boundary|post|backtrack> <run_dir> [options]
#
# Options:
#   --partition <name>       Slurm partition          (default: caslake)
#   --throttle <N>           Max concurrent tasks     (default: 100)
#   --dependency <jobid>     afterok dependency jobid (default: none)
#   --dry-run                Print sbatch command without submitting
#
# Example — run all three stages, chaining dependencies manually:
#   jid1=$(./run_stage.sh boundary my_run)
#   jid2=$(./run_stage.sh post     my_run --dependency $jid1)
#   jid3=$(./run_stage.sh backtrack my_run --dependency $jid2)
#
# Check for missing outputs after each stage:
#   grep -rL scan_results      new_code/model_runs/my_run/*.json
#   grep -rL post_dynamics_results new_code/model_runs/my_run/*.json
#   grep -rL backtrack_results new_code/model_runs/my_run/*.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── args ──────────────────────────────────────────────────────────────────────
stage="${1:?Usage: run_stage.sh <boundary|post|backtrack> <run_dir>}"
run_dir="${2:?Usage: run_stage.sh <boundary|post|backtrack> <run_dir>}"
shift 2

partition="caslake"
throttle="100"
dependency=""
dry_run="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)  partition="$2";  shift 2 ;;
        --throttle)   throttle="$2";   shift 2 ;;
        --dependency) dependency="$2"; shift 2 ;;
        --dry-run)    dry_run="true";  shift   ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

# ── validate ──────────────────────────────────────────────────────────────────
account="${MIDWAY3_SLURM_ACCOUNT:-pi-salesina}"

case "${stage}" in
    boundary)  mem="8G";  tlim="12:00:00" ;;
    post)      mem="8G";  tlim="12:00:00" ;;
    backtrack) mem="12G"; tlim="1-12:00:00" ;;
    *) echo "ERROR: stage must be boundary|post|backtrack" >&2; exit 1 ;;
esac

run_root="${REPO_ROOT}/new_code/model_runs/${run_dir}"
[[ -d "${run_root}" ]] || { echo "ERROR: run directory not found: ${run_root}" >&2; exit 1; }

# ── find Julia ────────────────────────────────────────────────────────────────
if [[ -z "${JULIA_EXE:-}" ]]; then
    julia_module="${JULIA_MODULE:-julia/1.10.2}"
    if command -v module &>/dev/null; then
        module load "${julia_module}" 2>/dev/null || {
            echo "ERROR: failed to load module '${julia_module}'" >&2
            exit 1
        }
    fi
    JULIA_EXE="$(command -v julia 2>/dev/null)" ||
        { echo "ERROR: julia not found; set JULIA_EXE or load module '${julia_module}'" >&2; exit 1; }
fi
export JULIA_EXE

# Leave JULIA_DEPOT_PATH untouched unless provided by the user.
# This uses the default Julia depot (~/.julia).

# ── build manifest ────────────────────────────────────────────────────────────
manifest_dir="${SCRIPT_DIR}/manifests"
mkdir -p "${manifest_dir}"
manifest="${manifest_dir}/${run_dir}.txt"

json_files=("${run_root}"/*.json)
[[ -e "${json_files[0]}" ]] || { echo "ERROR: no .json files in ${run_root}" >&2; exit 1; }

# Write basenames (task.sh reconstructs the full path from RUN_DIR).
printf '%s\n' "${json_files[@]}" | xargs -I{} basename {} | sort > "${manifest}"
n="$(wc -l < "${manifest}" | tr -d ' ')"
echo "Manifest: ${manifest} (${n} models)" >&2

# ── build sbatch command ──────────────────────────────────────────────────────
log_dir="${SCRIPT_DIR}/logs/${run_dir}/${stage}"
mkdir -p "${log_dir}"

sbatch_cmd=(
    sbatch --parsable
    -A "${account}"
    -p "${partition}"
    --job-name "tp_${stage}_${run_dir}"
    --array "1-${n}%${throttle}"
    --cpus-per-task 1
    --mem "${mem}"
    --time "${tlim}"
    --output "${log_dir}/%A_%a.out"
    --error  "${log_dir}/%A_%a.err"
    --export "ALL,STAGE=${stage},RUN_DIR=${run_dir},MANIFEST=${manifest},REPO_ROOT=${REPO_ROOT},JULIA_EXE=${JULIA_EXE}"
)

[[ -n "${dependency}" ]] && sbatch_cmd+=(--dependency "afterok:${dependency}")
sbatch_cmd+=("${SCRIPT_DIR}/task.sh")

if [[ "${dry_run}" == "true" ]]; then
    echo "DRY RUN:" >&2
    printf '  %q' "${sbatch_cmd[@]}"
    printf '\n'
    exit 0
fi

# ── submit ────────────────────────────────────────────────────────────────────
job_id="$("${sbatch_cmd[@]}")"
job_id="${job_id%%;*}"  # strip trailing ';cluster' if present
[[ -n "${job_id}" ]] || { echo "ERROR: sbatch returned empty job id" >&2; exit 1; }

echo "Submitted ${stage} job ${job_id} (${n} tasks, throttle=${throttle})" >&2
printf '%s\n' "${job_id}"
