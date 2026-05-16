"""T3a span-corruption with Claude Opus 4.7 — research comparison run.

Mirrors pipeline.enrich_t3a's task building + parsing exactly so the
output is directly comparable to the llama-8b T3a run. The only thing
that changes is the LLM backend: instead of vLLM batching, this issues
one Claude Code OAuth call per task, with quota gating.

Inputs (read-only):
  pl_hcl/<output_version>/stage{1,2}/test.parquet

Output (written alongside the llama version):
  pl_hcl/<output_version>/stage{1,2}/test_t3a_claude.parquet

Sample size: ~100 anchors total (50 per stage by default), chosen
deterministically by skill_id hash so a re-run gives the same set.
The K spans per (anchor, layer) come from pipeline.enrich_t3a._build_tasks
with the same rng_label as the llama run, so the spans are identical
across the two runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PREP_REPO = Path("/media/volume/skills/AgentSkills-OSS/agent-skills-preparation")
SCAN_REPO = Path("/media/volume/skills/AgentSkills-OSS/agent-skills-scanning")
# Prep imports must come from prep's `pipeline` package; scan-repo quota
# helpers come from a different module name we import below.
sys.path.insert(0, str(PREP_REPO))

from pipeline.enrich_t3a import (  # type: ignore
    STAGE_T3A_LAYERS,
    STAGE_MAX_TOKENS,
    SYSTEM_PROMPT,
    _build_messages,
    _build_tasks,
    _compose_with_swap,
    _existing_t3a_keys,
    _make_row,
    _read_positives,
)
from pipeline._corrupt_spans import parse_rewrites, splice_spans, est_tokens  # type: ignore

# Scan-repo's pipeline.quota lives at the same dotted name as prep's, so we
# load it from its file path directly to avoid the package shadow.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("scan_quota", SCAN_REPO / "pipeline" / "quota.py")
_scan_quota = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_scan_quota)  # type: ignore
RateLimitError = _scan_quota.RateLimitError
looks_rate_limited = _scan_quota.looks_rate_limited
record_call = _scan_quota.record_call
wait_for_quota = _scan_quota.wait_for_quota


LOGGER = logging.getLogger("enrich_t3a_claude")

CLAUDE_MODEL = "claude-opus-4-7"
CLAUDE_TIMEOUT_SEC = 240


def call_claude_opus(prompt: str, *, skill_id: str, scanner: str = "t3a_claude") -> Tuple[bool, str]:
    """Run `claude --model claude-opus-4-7 -p <prompt> --output-format json`.

    Gates via the same quota ledger the scanning pipeline uses. Raises
    RateLimitError on rate-limit signals so the caller waits the window out.
    """
    wait_for_quota()

    cmd = ["claude", "--model", CLAUDE_MODEL, "-p", prompt, "--output-format", "json"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        record_call(scanner=scanner, skill_id=skill_id, ok=False,
                    rate_limited=False, elapsed_sec=time.time() - t0)
        return False, "TIMEOUT"
    elapsed = time.time() - t0

    raw_out = p.stdout or ""
    in_toks = out_toks = 0
    is_error = (p.returncode != 0)
    result_text = ""
    rate_limited = False
    try:
        data = json.loads(raw_out)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        usage = data.get("usage") or {}
        try:
            in_toks = int(usage.get("input_tokens") or 0)
            out_toks = int(usage.get("output_tokens") or 0)
        except (TypeError, ValueError):
            pass
        is_error = bool(data.get("is_error", is_error))
        result_text = data.get("result") or ""
        stop_reason = data.get("stop_reason") or ""
        rate_limited = (
            looks_rate_limited(stop_reason)
            or (is_error and looks_rate_limited(result_text))
        )
    else:
        result_text = raw_out or (p.stderr or "")
        rate_limited = looks_rate_limited(result_text) or looks_rate_limited(p.stderr or "")

    record_call(
        scanner=scanner, skill_id=skill_id,
        ok=(not is_error) and (not rate_limited),
        rate_limited=rate_limited,
        input_tokens=in_toks, output_tokens=out_toks,
        elapsed_sec=elapsed,
    )
    if rate_limited:
        raise RateLimitError(f"[{scanner}/{skill_id}] claude rate-limit: {result_text[:240]}")
    if is_error or p.returncode != 0:
        return False, f"EXIT={p.returncode} {result_text[:200]}"
    return True, result_text


def _sample_anchors(anchors: List[Dict[str, Any]], n: int, seed_tag: str) -> List[Dict[str, Any]]:
    """Deterministic top-n by hash(seed_tag + skill_id)."""
    ranked = sorted(
        anchors,
        key=lambda a: hashlib.sha1(
            f"{seed_tag}|{a['anchor_skill_id']}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="T3a with Claude Opus 4.7 (comparison run)")
    parser.add_argument("--pl-hcl-root", type=Path, required=True)
    parser.add_argument("--output-version", required=True)
    parser.add_argument("--per-stage-sample", type=int, default=50,
                        help="anchors sampled per stage (default 50 → ~100 total)")
    parser.add_argument("--k-spans", type=int, default=2)
    parser.add_argument("--span-lines", type=int, default=8)
    parser.add_argument("--model-tag", default="claude-opus-4-7")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    target_root = args.pl_hcl_root / args.output_version
    if not target_root.exists():
        raise FileNotFoundError(target_root)

    # Same rng_label as enrich_t3a so spans match the llama-8b run exactly.
    rng_label = f"T3a|{args.output_version}|k={args.k_spans}|sl={args.span_lines}"
    seed_tag = f"t3a_claude_sample|{args.output_version}|{args.per_stage_sample}"

    summary: Dict[str, Any] = {
        "model_tag": args.model_tag,
        "claude_model": CLAUDE_MODEL,
        "per_stage_sample": args.per_stage_sample,
        "k_spans": args.k_spans,
        "span_lines": args.span_lines,
        "started_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "per_stage": {},
    }

    for stage in ("stage1", "stage2"):
        stage_dir = target_root / stage
        in_path = stage_dir / "test.parquet"
        out_path = stage_dir / "test_t3a_claude.parquet"
        if not in_path.exists():
            LOGGER.warning("[%s] skip — no input %s", stage, in_path)
            continue
        max_total = STAGE_MAX_TOKENS[stage]

        all_anchors = _read_positives(in_path)
        sampled = _sample_anchors(all_anchors, args.per_stage_sample, seed_tag)
        LOGGER.info("[%s/test] all=%d sampled=%d", stage, len(all_anchors), len(sampled))

        already = _existing_t3a_keys(out_path)
        if already:
            LOGGER.info("[%s/test] resume: %d keys already present", stage, len(already))
        tasks = _build_tasks(sampled, stage, already, args.k_spans, args.span_lines, rng_label)
        LOGGER.info("[%s/test] tasks: %d", stage, len(tasks))
        if not tasks:
            continue

        FLUSH_EVERY = 16
        buf: List[Dict[str, Any]] = []
        writer: Optional[Any] = None
        n_emitted = n_oversize = n_empty = n_parse_fail = n_err = 0

        def _flush(force: bool = False) -> None:
            nonlocal writer, buf
            if not buf or (not force and len(buf) < FLUSH_EVERY):
                return
            table = pa.Table.from_pylist(buf)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
            writer.write_table(table)
            buf = []

        for i, t in enumerate(tasks, 1):
            msgs = _build_messages(t["layer"], t["spans"])
            prompt = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
            ok, response = call_claude_opus(prompt, skill_id=t["anchor"]["anchor_skill_id"])
            if not ok:
                n_err += 1
                LOGGER.warning("[%s/test] %d/%d ERROR: %s", stage, i, len(tasks), response[:120])
                continue
            txt = response.strip()
            if not txt:
                n_empty += 1
                continue
            rewrites = parse_rewrites(txt, k=len(t["spans"]))
            nonempty = [r for r in rewrites if r and r.strip()]
            if not nonempty:
                n_parse_fail += 1
                continue
            spans = t["spans"]
            rewrites_padded = [r if r and r.strip() else "" for r in rewrites]
            spliced_inner = splice_spans(t["inner"], spans, rewrites_padded)
            full_text = _compose_with_swap(t["anchor"], t["layer"], spliced_inner, stage)
            if est_tokens(full_text) > max_total:
                n_oversize += 1
                continue
            row = _make_row(
                anchor=t["anchor"], layer=t["layer"], spans=spans,
                rewritten_inner=spliced_inner, text=full_text,
                output_version=args.output_version, model_tag=args.model_tag,
            )
            buf.append(row)
            n_emitted += 1
            _flush()

            if i % 10 == 0 or i == len(tasks):
                LOGGER.info("[%s/test] %d/%d  emitted=%d oversize=%d empty=%d parse_fail=%d err=%d",
                            stage, i, len(tasks), n_emitted, n_oversize, n_empty, n_parse_fail, n_err)

        _flush(force=True)
        if writer is not None:
            writer.close()
        LOGGER.info("[%s/test] wrote %s — emitted=%d oversize=%d empty=%d parse_fail=%d err=%d",
                    stage, out_path, n_emitted, n_oversize, n_empty, n_parse_fail, n_err)
        summary["per_stage"][stage] = {
            "anchors_sampled": len(sampled),
            "tasks": len(tasks),
            "emitted": n_emitted,
            "oversize": n_oversize,
            "empty": n_empty,
            "parse_fail": n_parse_fail,
            "error": n_err,
        }

    summary["finished_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    stats_path = target_root / "stats_t3a_claude.json"
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary: %s", stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
