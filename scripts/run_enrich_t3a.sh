#!/bin/bash
#SBATCH -J enrich_t3a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
# Default: single A100 — Llama-3.1-8B fits in 80GB, no P2P pain, much
# shorter queue wait on BR200. Override with `--gres=gpu:N TP_SIZE=N` for
# multi-GPU TP if the queue is empty AND the node has working P2P.
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -A r00954
#SBATCH -p gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/output/logs/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/output/logs/%x_%j.err

# Phase 2 HCL — Step B (T3a, span-based behavior corruption).
#
# Reads pl_hcl/<output_version>/stage{1,2}/{train,val,test}.parquet (the
# `unseen` pool is intentionally excluded — it serves only as a donor for
# T2/T3b elsewhere). For each (anchor, layer), samples K spans of ~8 lines
# from logic-heavy lines and asks the LLM to flip behavior on real
# identifiers. Splices rewrites back, recomposes, length-filters, writes
# {split}_t3a.parquet alongside the existing files.
#
# Env vars (override at submit time):
#   OUTPUT_VERSION   — pl_hcl version to enrich (default pl_hcl_v1)
#   MODEL_KEY        — registry key in model_path.json (default llama3.1-8b)
#   TP_SIZE          — vLLM tensor_parallel_size (default ${SLURM_GPUS_ON_NODE})
#   GPU_MEM_UTIL     — vLLM gpu_memory_utilization (default 0.88)
#   REQUEST_BATCH    — prompts per llm.generate() call (default 512)
#   TEMPERATURE      — sampling temperature (default 0.8)
#   K_SPANS          — spans corrupted per (anchor, layer) (default 2)
#   SPAN_LINES       — target lines per span (default 8)
#   SPLITS           — comma list (default 'train,val,test')
#   DRY_BATCH        — if set to '1', cap each split at ~50 anchors via
#                      DRY_LIMIT (sanity check before full run)
#   DRY_LIMIT        — max anchors per split when DRY_BATCH=1 (default 50)

set -euo pipefail

OUTPUT_VERSION="${OUTPUT_VERSION:-pl_hcl_v1}"
MODEL_KEY="${MODEL_KEY:-llama3.1-8b}"
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE:-1}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
REQUEST_BATCH="${REQUEST_BATCH:-512}"
TEMPERATURE="${TEMPERATURE:-0.8}"
K_SPANS="${K_SPANS:-2}"
SPAN_LINES="${SPAN_LINES:-8}"
SPLITS="${SPLITS:-train,val,test}"

CODE_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation
PL_HCL_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment/pl_hcl
REGISTRY=/N/slate/cz1/GitHub/AgentSkills-OSS/model_training/model_path.json
CONDA_BASE=/N/slate/cz1/miniconda3
CONDA_ENV=AgentSkillsOSS

mkdir -p "${CODE_ROOT}/output/logs"

# Resolve model path from the registry by key.
MODEL_PATH="$(/N/slate/cz1/conda/envs/AgentSkillsOSS/bin/python - "${REGISTRY}" "${MODEL_KEY}" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
key = sys.argv[2]
def find(name, d):
    for k, v in d.items():
        if isinstance(v, dict):
            r = find(name, v)
            if r: return r
        elif k == name:
            return v
    return None
p = find(key, reg)
if not p or p in ("downloading", "not_available"):
    sys.exit(f"model key {key!r} not resolvable in {sys.argv[1]}")
print(p)
PY
)"
echo "[launcher] Resolved ${MODEL_KEY} -> ${MODEL_PATH}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# CUDA toolchain.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in \
      /N/soft/sles15sp6/cuda/gnu/12.6 \
      /N/soft/sles15sp6/cuda/gnu/12.4 \
      /N/soft/sles15sp6/cuda/gnu/12.2 \
      /usr/local/cuda; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "[launcher] CUDA_HOME=${CUDA_HOME}"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Some BR200 GPU nodes can't do P2P between A100s (e.g. mixed PCIe roots).
# Disable vLLM's custom all-reduce (uses P2P) and NCCL P2P; both fall back
# to host-mediated paths that work everywhere. Tiny perf hit, big stability.
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

cd "${CODE_ROOT}"

echo "[launcher] T3a span-based  TP=${TP_SIZE}  K=${K_SPANS}  span_lines=${SPAN_LINES}  splits=${SPLITS}"

python -u "${CODE_ROOT}/pipeline/enrich_t3a.py" \
    --pl-hcl-root "${PL_HCL_ROOT}" \
    --output-version "${OUTPUT_VERSION}" \
    --model-path "${MODEL_PATH}" \
    --model-tag "${MODEL_KEY}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-mem-util "${GPU_MEM_UTIL}" \
    --temperature "${TEMPERATURE}" \
    --request-batch "${REQUEST_BATCH}" \
    --k-spans "${K_SPANS}" \
    --span-lines "${SPAN_LINES}" \
    --splits "${SPLITS}"
