#!/usr/bin/env bash
# Submit a Slurm array job that rescans the success-flagged directions of every
# model file in <run_dir>, using a larger delta_max. Results overwrite the
# scan_results entries of those rays in the original bank JSONs.
#
# Usage:
#   rescan_success_submit.sh <run_dir> [options]
#
# Options:
#   --delta-max  <F>     New max perturbation magnitude   (default: 1000.0)
#   --partition  <name>  Slurm partition                  (default: caslake)
#   --throttle   <N>     Max concurrent tasks             (default: 100)
#   --mem        <MEM>   Memory per task                  (default: 8G)
#   --time       <TLIM>  Time limit                       (default: 12:00:00)
#   --dry-run            Print sbatch command without submitting
#
# Example — rescan the mu_B = 0.1 elegant bank with delta_max = 1000:
#   ./rescan_success_submit.sh 2_bank_elegant_50_models_n_4-20_128_dirs_muB_0.1 \
#       --delta-max 1000
#
# Afterwards verify with:
#   jq '.scan_config.rescan_success_max_pert' \
#       new_code/model_runs/<run_dir>/*.json | sort -u

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

run_dir="${1:?Usage: rescan_success_submit.sh <run_dir>}"
shift

delta_max="1000.0"
partition="caslake"
throttle="100"
mem="8G"
tlim="12:00:00"
dry_run="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delta-max) delta_max="$2"; shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --throttle)  throttle="$2";  shift 2 ;;
        --mem)       mem="$2";       shift 2 ;;
        --time)      tlim="$2";      shift 2 ;;
        --dry-run)   dry_run="true"; shift   ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

account="${MIDWAY3_SLURM_ACCOUNT:-pi-salesina}"
run_root="${REPO_ROOT}/new_code/model_runs/${run_dir}"
[[ -d "${run_root}" ]] || { echo "ERROR: run directory not found: ${run_root}" >&2; exit 1; }

# Find Julia (same logic as run_stage.sh).
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

manifest_dir="${SCRIPT_DIR}/manifests"
mkdir -p "${manifest_dir}"
manifest="${manifest_dir}/${run_dir}__rescan.txt"

json_files=("${run_root}"/*.json)
[[ -e "${json_files[0]}" ]] || { echo "ERROR: no .json files in ${run_root}" >&2; exit 1; }
printf '%s\n' "${json_files[@]}" | xargs -I{} basename {} | sort > "${manifest}"
n="$(wc -l < "${manifest}" | tr -d ' ')"
echo "Manifest: ${manifest} (${n} models)" >&2

log_dir="${SCRIPT_DIR}/logs/${run_dir}/rescan_success"
mkdir -p "${log_dir}"

sbatch_cmd=(
    sbatch --parsable
    -A "${account}"
    -p "${partition}"
    --job-name "tp_rescan_${run_dir}"
    --array "1-${n}%${throttle}"
    --cpus-per-task 1
    --mem "${mem}"
    --time "${tlim}"
    --output "${log_dir}/%A_%a.out"
    --error  "${log_dir}/%A_%a.err"
    --export "ALL,RUN_DIR=${run_dir},MANIFEST=${manifest},REPO_ROOT=${REPO_ROOT},JULIA_EXE=${JULIA_EXE},DELTA_MAX=${delta_max}"
    "${SCRIPT_DIR}/rescan_success_task.sh"
)

if [[ "${dry_run}" == "true" ]]; then
    echo "DRY RUN:" >&2
    printf '  %q' "${sbatch_cmd[@]}"
    printf '\n'
    exit 0
fi

job_id="$("${sbatch_cmd[@]}")"
job_id="${job_id%%;*}"
[[ -n "${job_id}" ]] || { echo "ERROR: sbatch returned empty job id" >&2; exit 1; }

echo "Submitted rescan job ${job_id} (${n} tasks, throttle=${throttle}, delta_max=${delta_max})" >&2
printf '%s\n' "${job_id}"
