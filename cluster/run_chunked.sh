#!/usr/bin/env bash
# Submit boundary_scan in direction chunks, then merge results.
#
# Usage:
#   run_chunked.sh <run_dir> [options]
#
# Options:
#   --chunk-size <N>     Directions per chunk           (default: 10)
#   --partition <name>   Slurm partition                (default: caslake)
#   --throttle <N>       Max concurrent tasks per chunk (default: 100)
#   --dependency <jobid> afterok dependency jobid       (default: none)
#   --dry-run            Print sbatch commands without submitting
#
# Returns the merge job ID on stdout.
#
# Example:
#   jid_merge=$(./cluster/run_chunked.sh my_run --chunk-size 10)
#   jid2=$(./run_stage.sh post     my_run --dependency $jid_merge)
#   jid3=$(./run_stage.sh backtrack my_run --dependency $jid2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── args ──────────────────────────────────────────────────────────────────────
run_dir="${1:?Usage: run_chunked.sh <run_dir> [options]}"
shift

chunk_size="10"
partition="caslake"
throttle="100"
dependency=""
dry_run="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunk-size)  chunk_size="$2";  shift 2 ;;
        --partition)   partition="$2";   shift 2 ;;
        --throttle)    throttle="$2";    shift 2 ;;
        --dependency)  dependency="$2";  shift 2 ;;
        --dry-run)     dry_run="true";   shift   ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

# ── validate ──────────────────────────────────────────────────────────────────
account="${MIDWAY3_SLURM_ACCOUNT:-pi-salesina}"

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

# ── build canonical manifest (exclude chunk shards) ───────────────────────────
manifest_dir="${SCRIPT_DIR}/manifests"
mkdir -p "${manifest_dir}"
manifest="${manifest_dir}/${run_dir}_canonical.txt"

canonical_files=()
while IFS= read -r -d '' f; do
    canonical_files+=("$f")
done < <(find "${run_root}" -maxdepth 1 -name 'model_*.json' ! -name '*_chunk_*' -print0 | sort -z)

[[ ${#canonical_files[@]} -gt 0 ]] || { echo "ERROR: no canonical model_*.json files in ${run_root}" >&2; exit 1; }

printf '%s\n' "${canonical_files[@]}" | xargs -I{} basename {} | sort > "${manifest}"
n="$(wc -l < "${manifest}" | tr -d ' ')"
echo "Canonical manifest: ${manifest} (${n} models)" >&2

# ── read n_dirs from first model file ────────────────────────────────────────
first_model="${canonical_files[0]}"
n_dirs="$(python3 -c "import json; print(json.load(open('${first_model}'))['n_dirs'])")" || {
    echo "ERROR: could not read n_dirs from ${first_model}" >&2
    exit 1
}
echo "n_dirs=${n_dirs}, chunk_size=${chunk_size}" >&2

# ── compute chunk ranges ──────────────────────────────────────────────────────
chunk_starts=()
chunk_ends=()
s=0
while [[ $s -lt $n_dirs ]]; do
    e=$(( s + chunk_size - 1 ))
    [[ $e -ge $n_dirs ]] && e=$(( n_dirs - 1 ))
    chunk_starts+=("$s")
    chunk_ends+=("$e")
    s=$(( e + 1 ))
done
n_chunks="${#chunk_starts[@]}"
echo "Submitting ${n_chunks} chunk(s) covering directions [0, $((n_dirs-1))]" >&2

# ── submit chunk array jobs ───────────────────────────────────────────────────
chunk_job_ids=()

for (( i=0; i<n_chunks; i++ )); do
    s="${chunk_starts[$i]}"
    e="${chunk_ends[$i]}"

    log_dir="${SCRIPT_DIR}/logs/${run_dir}/chunk_${s}_${e}"
    mkdir -p "${log_dir}"

    sbatch_cmd=(
        sbatch --parsable
        -A "${account}"
        -p "${partition}"
        --job-name "tp_chunk_${run_dir}_${s}_${e}"
        --array "1-${n}%${throttle}"
        --cpus-per-task 1
        --mem "8G"
        --time "12:00:00"
        --output "${log_dir}/%A_%a.out"
        --error  "${log_dir}/%A_%a.err"
        --export "ALL,STAGE=boundary,RUN_DIR=${run_dir},MANIFEST=${manifest},REPO_ROOT=${REPO_ROOT},JULIA_EXE=${JULIA_EXE},CHUNK_START=${s},CHUNK_END=${e}"
    )

    [[ -n "${dependency}" ]] && sbatch_cmd+=(--dependency "afterok:${dependency}")
    sbatch_cmd+=("${SCRIPT_DIR}/task.sh")

    if [[ "${dry_run}" == "true" ]]; then
        echo "DRY RUN chunk [${s},${e}]:" >&2
        printf '  %q' "${sbatch_cmd[@]}"
        printf '\n'
        chunk_job_ids+=("DRY_${s}_${e}")
        continue
    fi

    jid="$("${sbatch_cmd[@]}")"
    jid="${jid%%;*}"
    [[ -n "${jid}" ]] || { echo "ERROR: sbatch returned empty job id for chunk [${s},${e}]" >&2; exit 1; }
    echo "Submitted chunk [${s},${e}] job ${jid} (${n} tasks, throttle=${throttle})" >&2
    chunk_job_ids+=("${jid}")
done

# ── build afterok dependency string ──────────────────────────────────────────
dep_str="afterok"
for jid in "${chunk_job_ids[@]}"; do
    dep_str="${dep_str}:${jid}"
done

# ── submit merge job ──────────────────────────────────────────────────────────
merge_log_dir="${SCRIPT_DIR}/logs/${run_dir}/merge"
mkdir -p "${merge_log_dir}"

merge_cmd=(
    sbatch --parsable
    -A "${account}"
    -p "${partition}"
    --job-name "tp_merge_${run_dir}"
    --cpus-per-task 1
    --mem "4G"
    --time "01:00:00"
    --output "${merge_log_dir}/%j.out"
    --error  "${merge_log_dir}/%j.err"
    --dependency "${dep_str}"
    --wrap "cd '${REPO_ROOT}' && '${JULIA_EXE}' --startup-file=no new_code/hpc/merge_chunks.jl '${run_dir}'"
)

if [[ "${dry_run}" == "true" ]]; then
    echo "DRY RUN merge (dep=${dep_str}):" >&2
    printf '  %q' "${merge_cmd[@]}"
    printf '\n'
    echo "DRY_MERGE"
    exit 0
fi

merge_jid="$("${merge_cmd[@]}")"
merge_jid="${merge_jid%%;*}"
[[ -n "${merge_jid}" ]] || { echo "ERROR: sbatch returned empty job id for merge" >&2; exit 1; }
echo "Submitted merge job ${merge_jid} (dep=${dep_str})" >&2

printf '%s\n' "${merge_jid}"
