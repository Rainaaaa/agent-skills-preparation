#!/bin/bash
#SBATCH -J agentskills_prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH -A r00954
#SBATCH -p general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/output/logs/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/output/logs/%x_%j.err

set -euo pipefail

source /N/slate/cz1/miniconda3/etc/profile.d/conda.sh
conda activate AgentSkillsOSS

CONFIG=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/config.yaml
CODE_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation

mkdir -p "${CODE_ROOT}/output/logs"
mkdir -p "${CODE_ROOT}/output/manifests"

COMMAND="${PIPELINE_COMMAND:-normalize}"
NORMALIZED_VERSION="${NORMALIZED_VERSION:-v1}"
OUTPUT_VERSION="${OUTPUT_VERSION:-}"
FULL_CPT_VERSION="${FULL_CPT_VERSION:-}"
WORKERS="${PIPELINE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
BATCH_SIZE="${PIPELINE_BATCH_SIZE:-2048}"

ARGS=(
  "${COMMAND}"
  --config "${CONFIG}"
  --workers "${WORKERS}"
  --batch-size "${BATCH_SIZE}"
)

# build_phase2_hcl uses --full-cpt-version instead of --normalized-version.
if [[ "${COMMAND}" == "build_phase2_hcl" ]]; then
  if [[ -z "${FULL_CPT_VERSION}" ]]; then
    echo "FULL_CPT_VERSION env var must be set for build_phase2_hcl" >&2
    exit 2
  fi
  ARGS+=(--full-cpt-version "${FULL_CPT_VERSION}")
else
  ARGS+=(--normalized-version "${NORMALIZED_VERSION}")
fi

if [[ -n "${OUTPUT_VERSION}" ]]; then
  ARGS+=(--output-version "${OUTPUT_VERSION}")
fi

if [[ "${SKIP_PARQUET:-0}" == "1" ]]; then
  ARGS+=(--skip-parquet)
fi

python3 -u "${CODE_ROOT}/pipeline/main.py" "${ARGS[@]}"
