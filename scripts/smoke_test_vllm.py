"""Tiny vLLM smoke test for AgentSkillsOSS env — run on a GPU node.

Usage:
  python smoke_test_vllm.py <model_path> [tensor_parallel_size]

Loads the model, generates one chat completion, prints stats. Used to verify
that vLLM works end-to-end before kicking off the full T3a enrichment job.
"""
import sys
import time

if len(sys.argv) < 2:
    sys.exit(f"usage: {sys.argv[0]} <model_path> [tensor_parallel_size]")

model_path = sys.argv[1]
tp_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1

print(f"[smoke] vLLM import ...", flush=True)
from vllm import LLM, SamplingParams
import vllm
print(f"[smoke] vLLM {vllm.__version__}", flush=True)

print(f"[smoke] loading {model_path} (TP={tp_size}) ...", flush=True)
t0 = time.time()
llm = LLM(
    model=model_path,
    tensor_parallel_size=tp_size,
    gpu_memory_utilization=0.85,
    dtype="bfloat16",
    trust_remote_code=False,
    enforce_eager=False,
)
print(f"[smoke] load took {time.time()-t0:.1f}s", flush=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
print(f"[smoke] tokenizer vocab={tok.vocab_size}", flush=True)

messages = [
    {"role": "system", "content": "You are a careful editor."},
    {"role": "user", "content": "Rewrite this so its meaning is subtly changed: 'Open the file and append a line.'"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

t1 = time.time()
out = llm.generate([prompt] * 8, SamplingParams(max_tokens=128, temperature=0.8, top_p=0.9))
dt = time.time() - t1
total_out = sum(len(o.outputs[0].token_ids) for o in out)
print(f"[smoke] generated {total_out} tokens in {dt:.2f}s = {total_out/dt:.0f} tok/s aggregate", flush=True)
print(f"[smoke] first sample output:")
print(out[0].outputs[0].text)
print(f"[smoke] OK", flush=True)
