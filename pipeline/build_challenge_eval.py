#!/usr/bin/env python3
"""Canonicalize + normalize the SFT challenge set into a gold-label eval dataset.

The challenge set lives at
    /N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-scanning/work/skills_sft_challenge/
with one subdir per skill_id and a sibling `scan_results_reviewed.csv`
holding the human-reviewed labels (`final_safe`, `final_align`).

Run `restore_challenge_payloads.py` first so the `source=collected`
sub-dirs contain the full payload from /N/project/AdversarialModeling/...

This script emits two artifacts under <out_dir>:

  1. challenge_canonical.jsonl
        One canonical record per skill (matches the schema used elsewhere
        in agent-skills-preparation): first_layer, skill_md sections,
        package_files, derived. Plus the gold-label CSV columns merged in.

  2. challenge_eval.parquet
        SFT-ready, one row per skill:
            skill_id        : str
            skill_text      : str   (rendered text the model will see)
            overall_class   : str   (gold malicious label: SAFE | SUSPICIOUS | MALICIOUS)
            alignment_class : str   (gold alignment label: ALIGNED | MISALIGNED)
            source          : str   (collected | masb | hsb)
            split           : str   (always "test" — challenge set is eval-only)

Schema rationale: `overall_class` + `alignment_class` align with the column
names the downstream SFT training in agent-skills-training/stages/downstream
already consumes (see sft_malicious_example.yaml / sft_misalignment_example.yaml).

Usage
-----
    python -m pipeline.build_challenge_eval \
        --reviewed-csv /N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-scanning/work/skills_sft_challenge/scan_results_reviewed.csv \
        --challenge-dir /N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-scanning/work/skills_sft_challenge \
        --out-dir /N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-preparation/outputs/challenge_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


MAX_FILE_BYTES = 200_000
MAX_SKILL_TEXT_CHARS = 16_000   # matches data-prep's stage-text cap
SKIP_TOPLEVEL = {"SKILL.md", "skill.md", "manifest.json"}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_DESC_HEADING_RE = re.compile(r"^(#+)\s+(description)\s*$", re.IGNORECASE | re.MULTILINE)
_ANY_HEADING_RE  = re.compile(r"^#+\s+\S", re.MULTILINE)


# ----------------------------------------------------------------------- canonicalize

def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
        if not isinstance(data, dict):
            data = {}
    except yaml.YAMLError:
        data = {}
    return data, m.group(2)


def find_description_section(skill_md: str) -> Tuple[Dict[str, Any], str]:
    """Locate the `## Description` block; return (section_dict, body-with-section-removed)."""
    m = _DESC_HEADING_RE.search(skill_md)
    if not m:
        return ({"found": False, "heading": None, "content": None}, skill_md)
    heading_start = m.start()
    heading_end = m.end()
    heading_text = m.group(2)
    nxt = _ANY_HEADING_RE.search(skill_md, heading_end)
    section_end = nxt.start() if nxt else len(skill_md)
    content = skill_md[heading_end:section_end].strip("\n").strip()
    non_description = skill_md[:heading_start] + skill_md[section_end:]
    non_description = re.sub(r"\n{3,}", "\n\n", non_description)
    return (
        {"found": True, "heading": heading_text, "content": content or None},
        non_description,
    )


def collect_package_files(skill_dir: Path) -> List[Dict[str, Any]]:
    """Walk skill_dir, capturing payload files (skipping top-level SKILL.md/manifest.json)."""
    out: List[Dict[str, Any]] = []
    if not skill_dir.exists():
        return out
    for p in sorted(skill_dir.rglob("*")):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(skill_dir).as_posix()
        except ValueError:
            continue
        if p.name in SKIP_TOPLEVEL and "/" not in rel:
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        size = len(raw)
        truncated = size > MAX_FILE_BYTES
        chunk = raw[:MAX_FILE_BYTES] if truncated else raw
        try:
            content = chunk.decode("utf-8")
        except UnicodeDecodeError:
            content = chunk.decode("utf-8", errors="replace")
        out.append({"path": rel, "content": content, "size_bytes": size, "truncated": truncated})
    return out


def canonicalize(sid: str, skill_dir: Path) -> Dict[str, Any]:
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.exists():
        skill_md_path = skill_dir / "skill.md"
    skill_md = skill_md_path.read_text(encoding="utf-8", errors="replace") if skill_md_path.exists() else ""

    front, _ = parse_frontmatter(skill_md)
    name = str(front.get("name") or "")
    description = str(front.get("description") or "")

    desc_section, non_description = find_description_section(skill_md)
    non_description = non_description.rstrip()
    used_fallback = not bool(desc_section["content"])

    return {
        "skill_id": sid,
        "first_layer": {"name": name, "description": description},
        "skill_md": {
            "description_section": desc_section,
            "non_description": non_description,
        },
        "package_files": collect_package_files(skill_dir),
        "derived": {
            "description_from_metadata": description,
            "has_explicit_description_heading": bool(desc_section["found"]),
            "used_metadata_description_fallback": used_fallback,
        },
    }


# ----------------------------------------------------------------------- normalize

_GOLD_SAFE_MAP  = {"SAFE": "SAFE", "SUSPICIOUS": "SUSPICIOUS", "MALICIOUS": "MALICIOUS"}
_GOLD_ALIGN_MAP = {"ALIGNED": "ALIGNED", "MISALIGNED": "MISALIGNED"}


def normalize_gold(row: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """Map the reviewed-CSV `final_safe` / `final_align` to canonical gold labels.

    `final_safe` came out of the reviewer as one of {SAFE, SUSPICIOUS, MALICIOUS}
    (or empty). We keep the three-class form so downstream tasks can choose
    a binary collapse (e.g. {SAFE} vs {SUSPICIOUS, MALICIOUS}).
    """
    s = (row.get("final_safe") or "").strip().upper()
    a = (row.get("final_align") or "").strip().upper()
    return _GOLD_SAFE_MAP.get(s), _GOLD_ALIGN_MAP.get(a)


def render_skill_text(rec: Dict[str, Any]) -> str:
    """Combine SKILL.md content (description-first) into one capped string.

    Format:
        <description block>
        \n\n---\n\n
        <remainder of SKILL.md>
    Package files are intentionally NOT inlined here — the SFT eval prompt
    template already operates on description-centric inputs, and inlining
    payloads inflates the text past every reasonable seq budget. Downstream
    tasks that want payload context can read `package_files` from the JSONL.
    """
    desc = ((rec["skill_md"]["description_section"] or {}).get("content")) or rec["first_layer"]["description"] or ""
    body = rec["skill_md"]["non_description"] or ""
    text = (desc.strip() + "\n\n---\n\n" + body.strip()).strip()
    if len(text) > MAX_SKILL_TEXT_CHARS:
        text = text[:MAX_SKILL_TEXT_CHARS]
    return text


# ----------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description="Canonicalize + normalize the SFT challenge set.")
    p.add_argument("--reviewed-csv", type=Path, required=True)
    p.add_argument("--challenge-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="If >0, process only the first N rows (smoke test).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / "challenge_canonical.jsonl"
    out_parquet = args.out_dir / "challenge_eval.parquet"

    with args.reviewed_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[challenge-eval] reviewed rows: {len(rows)}")

    canonical_records: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    by_source: Dict[str, int] = {}
    n_skipped_missing = 0

    for i, row in enumerate(rows, 1):
        sid = row["skill_id"]
        src = (row.get("source") or "").lower() or "unknown"
        by_source[src] = by_source.get(src, 0) + 1

        skill_dir = args.challenge_dir / sid
        if not skill_dir.exists():
            n_skipped_missing += 1
            if n_skipped_missing < 5:
                print(f"  [warn] missing dir for {sid}", file=sys.stderr)
            continue

        try:
            rec = canonicalize(sid, skill_dir)
        except Exception as e:
            print(f"  [warn] canonicalize failed for {sid}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        # Merge the gold-label columns onto the canonical record.
        gold_safe, gold_align = normalize_gold(row)
        rec["gold"] = {"final_safe": gold_safe, "final_align": gold_align}
        rec["scan"] = {
            "static_rule_class":   row.get("static_rule_class"),
            "llm_filter_class":    row.get("llm_filter_class"),
            "alignment_class":     row.get("alignment_class"),
            "behavioral_class":    row.get("behavioral_class"),
            "overall_class":       row.get("overall_class"),
            "bench_classification": row.get("bench_classification"),
            "hsb_tier":            row.get("hsb_tier"),
            "hsb_category":        row.get("hsb_category"),
        }
        rec["source"] = src
        canonical_records.append(rec)

        eval_rows.append({
            "skill_id": sid,
            "skill_text": render_skill_text(rec),
            "overall_class": gold_safe,
            "alignment_class": gold_align,
            "source": src,
            "split": "test",
        })

        if i % 200 == 0:
            print(f"  [{i}/{len(rows)}]  written so far: {len(canonical_records)}", flush=True)

    # JSONL canonical records
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in canonical_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Parquet eval rows. pyarrow is a project dependency, so import here.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(eval_rows)
        pq.write_table(table, out_parquet, compression="zstd")
    except ImportError:
        # Fallback: write CSV next to the parquet path so the run still produces something.
        fallback = out_parquet.with_suffix(".csv")
        with fallback.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["skill_id", "skill_text", "overall_class",
                                              "alignment_class", "source", "split"])
            w.writeheader()
            w.writerows(eval_rows)
        out_parquet = fallback
        print("  [warn] pyarrow not available; wrote CSV instead.")

    print()
    print(f"[challenge-eval] canonical records : {len(canonical_records)}  -> {out_jsonl}")
    print(f"[challenge-eval] eval rows         : {len(eval_rows)}  -> {out_parquet}")
    print(f"[challenge-eval] source breakdown  : {by_source}")
    print(f"[challenge-eval] skipped (missing dir): {n_skipped_missing}")

    # Quick label-distribution summary
    from collections import Counter
    safe_dist = Counter(r["overall_class"] for r in eval_rows)
    align_dist = Counter(r["alignment_class"] for r in eval_rows)
    print(f"[challenge-eval] gold final_safe   : {dict(safe_dist)}")
    print(f"[challenge-eval] gold final_align  : {dict(align_dist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
