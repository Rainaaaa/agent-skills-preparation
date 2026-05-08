#!/bin/bash
#SBATCH -J enrich_t3c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -A r00954
#SBATCH -p general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/output/logs/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/output/logs/%x_%j.err

# Phase 2 HCL — Step C (T3c, rule-based hallucinated-identifier injection).
#
# Pure CPU. Runs in minutes — no GPU, no LLM. Substitutes K real
# package/CLI/env identifiers per (anchor, layer) with curated
# plausible-but-non-existent variants. See pipeline/enrich_t3c.py for
# the registry. Skips `unseen` pool by default.
#
# Env vars:
#   OUTPUT_VERSION — pl_hcl version (default pl_hcl_v1)
#   K_SUBS         — substitutions per (anchor, layer) (default 2)
#   SPLITS         — comma list (default 'train,val,test')
#   FORCE          — '1' to ignore existing parquets and full-rebuild

set -euo pipefail

OUTPUT_VERSION="${OUTPUT_VERSION:-pl_hcl_v1}"
K_SUBS="${K_SUBS:-2}"
SPLITS="${SPLITS:-train,val,test}"
FORCE="${FORCE:-0}"

CODE_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation
PL_HCL_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment/pl_hcl
CONDA_BASE=/N/slate/cz1/miniconda3
CONDA_ENV=AgentSkillsOSS

mkdir -p "${CODE_ROOT}/output/logs"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

cd "${CODE_ROOT}"

echo "[launcher] T3c rule-based  k_subs=${K_SUBS}  splits=${SPLITS}  force=${FORCE}"

CMD=(python -u "${CODE_ROOT}/pipeline/enrich_t3c.py"
     --pl-hcl-root "${PL_HCL_ROOT}"
     --output-version "${OUTPUT_VERSION}"
     --k-subs "${K_SUBS}"
     --splits "${SPLITS}")
if [[ "${FORCE}" == "1" ]]; then CMD+=(--force); fi
"${CMD[@]}"
