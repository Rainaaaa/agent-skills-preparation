"""Phase 2 HCL — Step B: T3a span-based behavior corruption via vLLM.

For each anchor (label='positive') in
``pl_hcl/<output_version>/stage{1,2}/{train,val,test}.parquet`` (the
`unseen` pool is intentionally skipped — its skills only ever appear as
donors elsewhere, never as anchors), select K spans of ~8 lines from each
target layer (INSTRUCTION for stage1; INSTRUCTION + RESOURCE for stage2),
ask `llama3.1-8b-Instruct` to rewrite each span so its **behavior** is
subtly but meaningfully changed *while keeping all identifiers real*,
splice the rewrites back into the original layer at their original
offsets, and emit:

    pl_hcl/<output_version>/stage{1,2}/{train,val,test}_t3a.parquet

Span-based corruption is ~5–8x cheaper than the previous whole-layer
rewrite path because the LLM only sees and produces ~K * span_lines worth
of text per call. It also enforces the T3a / T3c boundary at
**span-selection time**: T3a sees logic-heavy lines (conditionals,
comparisons, function calls), T3c sees identifier-heavy lines (imports,
deps, CLI flags). Different input → naturally disjoint outputs.

Implementation notes
--------------------
* vLLM batched generation, TP defaults to ``$SLURM_GPUS_ON_NODE``.
* One LLM call per (anchor, layer) returns all K rewrites delimited by
  ``===REWRITE===``. Stop tokens prevent the model from continuing past
  the last rewrite.
* Outputs are post-filtered: empty / refusal-shaped / preamble-only
  rewrites are dropped before splice.
* Length filter at recompose time discards rows that exceed the per-stage
  token budget (4096 / 10240).
* Resume: for each ``{split}_t3a.parquet`` already present, the
  ``(anchor_skill_id, target_layer)`` keys are skipped on re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline._corrupt_spans import (
    Span,
    build_spans_block,
    compose_layers,
    est_tokens,
    extract_layer,
    parse_rewrites,
    sample_spans,
    score_t3a,
    splice_spans,
    wrap,
)

LOGGER = logging.getLogger("enrich_phase2_t3a")


# ---------------------------------------------------------------------------
# Prompt design — span-based, behavior-shift only, REAL identifiers only.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meticulous skill-package editor. You receive K small spans "
    "extracted from one section of a software skill package. For each span, "
    "produce a subtly altered variant whose BEHAVIOR is meaningfully "
    "different from the original (a 'minor misalignment'). "
    "Strict rules:\n"
    "  - Use only REAL, existing identifiers — real package names, real "
    "library calls, real file paths, real CLI flags. Do NOT invent names.\n"
    "  - The rewrite must fail to match the description because of CHANGED "
    "BEHAVIOR (flipped logic, swapped real-but-different function, changed "
    "constant, reordered steps that change the result), not because of "
    "non-existent references.\n"
    "  - Keep formatting, indentation, language, vocabulary, and length close "
    "to the original.\n"
    "  - Output exactly K rewrites, one per input span, separated by the "
    "literal line `===REWRITE===`. No preamble, no commentary, no quoting, "
    "no markdown code fences. The first rewrite begins on the line after the "
    "first `===REWRITE===` separator."
)

USER_TEMPLATE_INSTRUCTION = (
    "The K input spans below are excerpts of an INSTRUCTION section "
    "(prose with steps, commands, or numbered actions). For each span, "
    "produce a corrupted variant whose meaning is subtly but meaningfully "
    "changed by editing logic, constraints, ordering, or the entity a step "
    "operates on — using only real, existing identifiers.\n\n"
    "Spans to rewrite (K=<<K>>):\n<<SPANS>>\n\n"
    "Output format: K rewrites, each preceded by a line containing exactly "
    "`===REWRITE===`. Do not output anything else."
)

USER_TEMPLATE_RESOURCE = (
    "The K input spans below are excerpts of a RESOURCE section "
    "(typically code or config). For each span, produce a corrupted variant "
    "whose runtime behavior is subtly but meaningfully changed: rename a "
    "function/method to a real-but-different one in the same library, flip a "
    "comparison (`==` ↔ `!=`, `<` ↔ `>`), change a constant, change an "
    "argument order, swap a return type. All identifiers must remain real "
    "and importable. Keep file format, language, indentation, and length the "
    "same.\n\n"
    "Spans to rewrite (K=<<K>>):\n<<SPANS>>\n\n"
    "Output format: K rewrites, each preceded by a line containing exactly "
    "`===REWRITE===`. Do not output anything else."
)


def _build_messages(layer: str, spans: List[Span]) -> List[Dict[str, str]]:
    template = USER_TEMPLATE_INSTRUCTION if layer == "I" else USER_TEMPLATE_RESOURCE
    user = template.replace("<<K>>", str(len(spans))).replace(
        "<<SPANS>>", build_spans_block(spans)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

STAGE_T3A_LAYERS: Dict[str, List[str]] = {
    "stage1": ["I"],
    "stage2": ["I", "R"],
}

STAGE_MAX_TOKENS: Dict[str, int] = {
    "stage1": 4096,
    "stage2": 10240,
}

# Skip 'unseen' — anchors only come from the cpt pool. Unseen rows exist as
# donor candidates for T2 / T3b only.
SPLITS = ("train", "val", "test")


def _read_positives(parquet_path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore
    cols = [
        "anchor_skill_id", "label", "pool", "split", "stage", "text",
        "normalized_version", "phase1_output_version", "output_version",
    ]
    sch_cols = pq.read_schema(parquet_path).names
    cols = [c for c in cols if c in sch_cols]
    t = pq.read_table(parquet_path, columns=cols)
    rows = []
    for r in t.to_pylist():
        if r.get("label") == "positive":
            rows.append(r)
    return rows


def _existing_t3a_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    try:
        import pyarrow.parquet as pq  # type: ignore
        sch_cols = pq.read_schema(out_path).names
        cols = [c for c in ("anchor_skill_id", "target_layer") if c in sch_cols]
        if len(cols) != 2:
            return set()
        t = pq.read_table(out_path, columns=cols)
        return set(zip(t.column("anchor_skill_id").to_pylist(),
                       t.column("target_layer").to_pylist()))
    except Exception as e:
        LOGGER.warning("could not read existing %s: %s — re-generating", out_path, e)
        return set()


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

def _make_llm(model_path: str, tensor_parallel_size: int, gpu_mem_util: float):
    from vllm import LLM  # type: ignore
    LOGGER.info("Loading vLLM (model=%s, TP=%d, mem=%.2f) ...",
                model_path, tensor_parallel_size, gpu_mem_util)
    # `disable_custom_all_reduce=True` is required when the cluster's GPUs
    # don't have working peer-to-peer (BR200 nodes hit
    # `cudaSetDevice ... CUDA-capable device(s) is/are busy or unavailable`
    # in vLLM 0.8.5's _can_p2p probe even with VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
    # set, because the env var only gates *use* of custom_all_reduce, not the
    # constructor's probe). Setting the flag explicitly skips the probe.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem_util,
        dtype="bfloat16",
        enforce_eager=False,
        trust_remote_code=False,
        disable_custom_all_reduce=True,
    )
    LOGGER.info("vLLM loaded.")
    return llm


def _make_sampling_params(max_new: int, temperature: float, top_p: float, repetition_penalty: float):
    from vllm import SamplingParams  # type: ignore
    return SamplingParams(
        max_tokens=max_new,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        n=1,
        stop=["</METADATA>", "</INSTRUCTION>", "</RESOURCE>", "===END==="],
    )


def _format_chat(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Task building
# ---------------------------------------------------------------------------

def _build_tasks(
    anchors: List[Dict[str, Any]],
    stage: str,
    skip_keys: set,
    k_spans: int,
    span_lines: int,
    rng_label: str,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    layers = STAGE_T3A_LAYERS[stage]
    for a in anchors:
        sid = a["anchor_skill_id"]
        full_text = a.get("text", "") or ""
        for layer in layers:
            if (sid, layer) in skip_keys:
                continue
            inner_text = extract_layer(full_text, layer)
            if not inner_text or not inner_text.strip():
                continue
            spans = sample_spans(
                inner_text,
                scorer=score_t3a,
                k=k_spans,
                span_lines=span_lines,
                seed_parts=(rng_label, sid, layer),
            )
            if not spans:
                continue
            tasks.append({
                "anchor": a,
                "layer": layer,
                "inner": inner_text,
                "spans": spans,
            })
    return tasks


def _make_row(
    *,
    anchor: Dict[str, Any],
    layer: str,
    spans: List[Span],
    rewritten_inner: str,
    text: str,
    output_version: str,
    model_tag: str,
) -> Dict[str, Any]:
    sub = f"layer={layer}|method=span|k={len(spans)}|llm={model_tag}"
    sample_id = hashlib.sha1(
        f"T3a|{sub}|{anchor['anchor_skill_id']}|{text[:128]}".encode("utf-8")
    ).hexdigest()
    return {
        "sample_id": sample_id,
        "anchor_skill_id": anchor["anchor_skill_id"],
        "label": "T3a",
        "sub_strategy": sub,
        "target_layer": layer,
        "source_skill_ids": [anchor["anchor_skill_id"]],
        "text": text,
        "text_chars": len(text),
        "text_tokens_est": est_tokens(text),
        "pool": anchor.get("pool", ""),
        "split": anchor.get("split", ""),
        "stage": anchor.get("stage", ""),
        "normalized_version": anchor.get("normalized_version", ""),
        "phase1_output_version": anchor.get("phase1_output_version", ""),
        "output_version": output_version,
    }


def _compose_with_swap(anchor: Dict[str, Any], layer: str, new_inner: str, stage: str) -> str:
    """Replace the target layer's inner text in the anchor's composed `text`."""
    full = anchor.get("text", "") or ""
    order = ("M", "I") if stage == "stage1" else ("M", "I", "R")
    parts = {ll: extract_layer(full, ll) for ll in order}
    parts[layer] = new_inner
    return compose_layers(parts, stage)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    pl_hcl_root: Path,
    output_version: str,
    model_path: str,
    model_tag: str,
    tensor_parallel_size: int,
    gpu_mem_util: float,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    request_batch: int,
    k_spans: int,
    span_lines: int,
    max_new_per_span: int,
    splits: List[str],
) -> None:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    target_root = pl_hcl_root / output_version
    if not target_root.exists():
        raise FileNotFoundError(f"pl_hcl version not found: {target_root}")

    LOGGER.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = _make_llm(model_path, tensor_parallel_size, gpu_mem_util)

    summary: Dict[str, Any] = {
        "model_path": model_path,
        "model_tag": model_tag,
        "tensor_parallel_size": tensor_parallel_size,
        "k_spans": k_spans,
        "span_lines": span_lines,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "request_batch": request_batch,
        "started_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "splits": splits,
        "per_stage_split": {},
    }

    rng_label = f"T3a|{output_version}|k={k_spans}|sl={span_lines}"

    for stage in ("stage1", "stage2"):
        stage_dir = target_root / stage
        if not stage_dir.exists():
            LOGGER.warning("Skipping %s: no dir %s", stage, stage_dir)
            continue
        max_total = STAGE_MAX_TOKENS[stage]

        for split in splits:
            in_path = stage_dir / f"{split}.parquet"
            out_path = stage_dir / f"{split}_t3a.parquet"
            if not in_path.exists():
                LOGGER.warning("[%s/%s] skip — no input %s", stage, split, in_path)
                continue

            anchors = _read_positives(in_path)
            LOGGER.info("[%s/%s] anchors: %d", stage, split, len(anchors))

            already = _existing_t3a_keys(out_path)
            if already:
                LOGGER.info("[%s/%s] resume: %d keys already present", stage, split, len(already))
            tasks = _build_tasks(anchors, stage, already, k_spans, span_lines, rng_label)
            LOGGER.info("[%s/%s] tasks: %d", stage, split, len(tasks))
            if not tasks:
                continue

            FLUSH_EVERY = 1024
            buffer: List[Dict[str, Any]] = []
            writer: Optional[Any] = None
            n_emitted = n_oversize = n_empty = n_parse_fail = 0

            def _flush(force: bool = False) -> None:
                nonlocal writer, buffer
                if not buffer or (not force and len(buffer) < FLUSH_EVERY):
                    return
                table = pa.Table.from_pylist(buffer)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
                writer.write_table(table)
                buffer = []

            for batch_start in range(0, len(tasks), request_batch):
                batch = tasks[batch_start:batch_start + request_batch]
                prompts = [_format_chat(tokenizer, _build_messages(t["layer"], t["spans"]))
                           for t in batch]
                # max_new = K spans worth of text + delimiter overhead
                max_new = min(
                    max(
                        sum(est_tokens(s.text) for s in t["spans"]) * 2
                        + 32 * len(t["spans"]) + max_new_per_span
                        for t in batch
                    ),
                    max_total,
                )
                sampling = _make_sampling_params(max_new, temperature, top_p, repetition_penalty)
                outputs = llm.generate(prompts, sampling)
                for t, out in zip(batch, outputs):
                    try:
                        txt_out = out.outputs[0].text or ""
                    except Exception:
                        txt_out = ""
                    txt_out = txt_out.strip()
                    if not txt_out:
                        n_empty += 1
                        continue
                    rewrites = parse_rewrites(txt_out, k=len(t["spans"]))
                    nonempty = [r for r in rewrites if r and r.strip()]
                    if not nonempty:
                        n_parse_fail += 1
                        continue
                    spans = t["spans"]
                    rewrites_padded = []
                    for r in rewrites:
                        rewrites_padded.append(r if r and r.strip() else "")
                    spliced_inner = splice_spans(t["inner"], spans, rewrites_padded)
                    full_text = _compose_with_swap(t["anchor"], t["layer"], spliced_inner, stage)
                    if est_tokens(full_text) > max_total:
                        n_oversize += 1
                        continue
                    row = _make_row(
                        anchor=t["anchor"], layer=t["layer"], spans=spans,
                        rewritten_inner=spliced_inner, text=full_text,
                        output_version=output_version, model_tag=model_tag,
                    )
                    buffer.append(row)
                    n_emitted += 1
                _flush()
                if (batch_start // request_batch) % 5 == 0:
                    LOGGER.info(
                        "[%s/%s] %d/%d tasks; emitted=%d oversize=%d empty=%d parse_fail=%d",
                        stage, split, batch_start + len(batch), len(tasks),
                        n_emitted, n_oversize, n_empty, n_parse_fail,
                    )

            _flush(force=True)
            if writer is not None:
                writer.close()
            LOGGER.info(
                "[%s/%s] wrote %s — emitted=%d oversize=%d empty=%d parse_fail=%d",
                stage, split, out_path, n_emitted, n_oversize, n_empty, n_parse_fail,
            )
            summary["per_stage_split"][f"{stage}/{split}"] = {
                "anchors": len(anchors),
                "tasks": len(tasks),
                "emitted": n_emitted,
                "oversize_dropped": n_oversize,
                "empty_dropped": n_empty,
                "parse_fail_dropped": n_parse_fail,
            }

    summary["finished_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary_path = target_root / "stats_t3a.json"
    with summary_path.open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote summary: %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="T3a span-based behavior corruption")
    parser.add_argument("--pl-hcl-root", required=True)
    parser.add_argument("--output-version", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-tag", default="llama3.1-8b",
                        help="Short label embedded into sub_strategy and stats")
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=int(os.environ.get("TP_SIZE", "1")))
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--request-batch", type=int, default=512)
    parser.add_argument("--k-spans", type=int, default=2,
                        help="Number of spans corrupted per (anchor, layer)")
    parser.add_argument("--span-lines", type=int, default=8,
                        help="Target lines per span")
    parser.add_argument("--max-new-per-span", type=int, default=64,
                        help="Headroom tokens per span on top of 2x input estimate")
    parser.add_argument("--splits", default="train,val,test",
                        help="Comma list. 'unseen' is intentionally excluded by default.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    run(
        pl_hcl_root=Path(args.pl_hcl_root),
        output_version=args.output_version,
        model_path=args.model_path,
        model_tag=args.model_tag,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_mem_util=args.gpu_mem_util,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        request_batch=args.request_batch,
        k_spans=args.k_spans,
        span_lines=args.span_lines,
        max_new_per_span=args.max_new_per_span,
        splits=splits,
    )


if __name__ == "__main__":
    main()
