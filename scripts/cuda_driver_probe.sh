#!/bin/bash
#SBATCH -J cuda_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -A r00954
#SBATCH -p gpu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/output/logs/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/output/logs/%x_%j.err

set -uo pipefail

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed"
echo ""
echo "=== nvidia-smi driver/cuda fields ==="
nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv 2>&1 || true
echo ""
echo "=== nvcc (if any) ==="
which nvcc && nvcc --version 2>&1 || echo "no nvcc on PATH"
echo ""

PY=/N/slate/cz1/conda/envs/AgentSkillsOSS/bin/python
echo "=== AgentSkillsOSS torch (cu130) cuda probe ==="
"$PY" - <<'PY'
import torch
print("torch.__version__ :", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda is_available :", torch.cuda.is_available())
try:
    print("device count     :", torch.cuda.device_count())
    print("device name      :", torch.cuda.get_device_name(0))
    print("device cap       :", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul OK, sum   :", float(y.sum().item()))
except Exception as e:
    print("CUDA RUNTIME FAILED:", repr(e)[:400])
PY
echo ""
echo "=== flash_attn import in AgentSkillsOSS ==="
"$PY" - <<'PY'
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__, "OK")
except Exception as e:
    print("flash_attn FAIL:", repr(e)[:400])
PY
echo ""
echo "=== BiasDetection torch (cu124) sanity ==="
/N/slate/cz1/conda/envs/BiasDetection/bin/python - <<'PY'
import torch
print("torch.__version__ :", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda is_available :", torch.cuda.is_available())
try:
    print("device name      :", torch.cuda.get_device_name(0))
    print("device cap       :", torch.cuda.get_device_capability(0))
except Exception as e:
    print("BD CUDA FAIL:", repr(e)[:300])
PY
echo "=== done ==="
