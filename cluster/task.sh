#!/usr/bin/env bash
# Slurm array task: run one model through a pipeline stage.
# Called by run_stage.sh via sbatch; not meant to be run directly.
set -euo pipefail

: "${STAGE:?Missing STAGE}"
: "${RUN_DIR:?Missing RUN_DIR}"
: "${MANIFEST:?Missing MANIFEST}"
: "${REPO_ROOT:?Missing REPO_ROOT}"
: "${JULIA_EXE:?Missing JULIA_EXE}"
: "${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}"

[[ -f "${MANIFEST}" ]] || { echo "ERROR: manifest not found: ${MANIFEST}" >&2; exit 1; }

model_file="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}" | tr -d '\r')"
[[ -n "${model_file}" ]] || {
    echo "ERROR: index ${SLURM_ARRAY_TASK_ID} out of range in ${MANIFEST}" >&2
    exit 1
}

case "${STAGE}" in
    boundary)  script="pipeline/boundary_scan.jl" ;;
    post)      script="pipeline/post_boundary_dynamics.jl" ;;
    backtrack) script="pipeline/backtrack_perturbation.jl" ;;
    *) echo "ERROR: unknown STAGE='${STAGE}'" >&2; exit 1 ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] stage=${STAGE} task=${SLURM_ARRAY_TASK_ID} model=${model_file}"

chunk_args=()
if [[ -n "${CHUNK_START:-}" && -n "${CHUNK_END:-}" ]]; then
    chunk_args=(--dir-chunk-start "$CHUNK_START" --dir-chunk-end "$CHUNK_END")
fi

cd "${REPO_ROOT}"
"${JULIA_EXE}" --startup-file=no \
    "${REPO_ROOT}/new_code/${script}" "${RUN_DIR}" --model-file "${model_file}" \
    "${chunk_args[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done stage=${STAGE} task=${SLURM_ARRAY_TASK_ID}"
