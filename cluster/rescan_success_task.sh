#!/usr/bin/env bash
# Slurm array task: rescan success-flagged directions for one model file.
# Called by rescan_success_submit.sh via sbatch; not meant to be run directly.
set -euo pipefail

: "${RUN_DIR:?Missing RUN_DIR}"
: "${MANIFEST:?Missing MANIFEST}"
: "${REPO_ROOT:?Missing REPO_ROOT}"
: "${JULIA_EXE:?Missing JULIA_EXE}"
: "${DELTA_MAX:?Missing DELTA_MAX}"
: "${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}"

[[ -f "${MANIFEST}" ]] || { echo "ERROR: manifest not found: ${MANIFEST}" >&2; exit 1; }

model_file="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}" | tr -d '\r')"
[[ -n "${model_file}" ]] || {
    echo "ERROR: index ${SLURM_ARRAY_TASK_ID} out of range in ${MANIFEST}" >&2
    exit 1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] rescan task=${SLURM_ARRAY_TASK_ID} model=${model_file}"

cd "${REPO_ROOT}"
"${JULIA_EXE}" --startup-file=no \
    "${REPO_ROOT}/new_code/rescan_success_directions.jl" "${RUN_DIR}" \
    --model-file "${model_file}" \
    --delta-max  "${DELTA_MAX}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done rescan task=${SLURM_ARRAY_TASK_ID}"
