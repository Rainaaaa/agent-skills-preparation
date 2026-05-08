#!/bin/bash
#
# Orchestrator: submits the full data preparation DAG to SLURM.
#
#   normalize  ─┬─>  build_phase1
#               ├─>  build_phase2
#               └─>  build_phase3
#
# Phase 1/2/3 read the same normalized dataset and write to different output
# directories, so they can run concurrently once normalization is done.
#
# Usage:
#   ./submit_all.sh [NORMALIZED_VERSION] [OUTPUT_VERSION_SUFFIX]
#
# Examples:
#   ./submit_all.sh v1 v1
#   ./submit_all.sh v2026_04_23 v2026_04_23

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/run_data_preparation.sh"

NORMALIZED_VERSION="${1:-v1}"
OUTPUT_SUFFIX="${2:-${NORMALIZED_VERSION}}"

mkdir -p "${SCRIPT_DIR}/output/logs" "${SCRIPT_DIR}/output/manifests"

submit() {
  local name="$1"
  local command="$2"
  local output_version="$3"
  local dep="$4"

  local args=(
    --job-name="agentprep_${name}"
    --export=ALL,PIPELINE_COMMAND="${command}",NORMALIZED_VERSION="${NORMALIZED_VERSION}",OUTPUT_VERSION="${output_version}"
  )
  if [[ -n "${dep}" ]]; then
    args+=(--dependency="afterok:${dep}")
  fi
  local jobid
  jobid="$(sbatch --parsable "${args[@]}" "${SBATCH_FILE}")"
  echo "${jobid}"
}

echo "Submitting normalize (version=${NORMALIZED_VERSION})..." >&2
NORM_ID="$(submit normalize normalize "" "")"
echo "  normalize job id: ${NORM_ID}" >&2

echo "Submitting phase1 / phase2 / phase3 (dependent on ${NORM_ID})..." >&2
P1_ID="$(submit phase1 build_phase1 "full_cpt_${OUTPUT_SUFFIX}"   "${NORM_ID}")"
P2_ID="$(submit phase2 build_phase2 "stage_cpt_${OUTPUT_SUFFIX}"  "${NORM_ID}")"
P3_ID="$(submit phase3 build_phase3 "sft_${OUTPUT_SUFFIX}"        "${NORM_ID}")"

cat >&2 <<EOF
  phase1 job id:   ${P1_ID}   (output_version=full_cpt_${OUTPUT_SUFFIX})
  phase2 job id:   ${P2_ID}   (output_version=stage_cpt_${OUTPUT_SUFFIX})
  phase3 job id:   ${P3_ID}   (output_version=sft_${OUTPUT_SUFFIX})

Watch:   squeue -u \$USER
Logs:    ${SCRIPT_DIR}/output/logs/
EOF
