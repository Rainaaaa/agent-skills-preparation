#!/usr/bin/env python3
"""Scrub challenge-set skill_ids from the curated CPT / pl_hcl / SFT datasets.

The 1,444 human-reviewed challenge skills are now the gold-label SFT eval set
(see build_sft_challenge.py). To keep that eval set out-of-distribution, any
row in CPT / pl_hcl / SFT that *touches* a challenge skill_id — as the row's
own skill_id OR as an anchor OR as an HCL `source_skill_ids` donor OR as the
SFT classifier candidate — gets dropped.

Outputs are written as NEW dataset versions, side-by-side with the originals:

    full_cpt_v2 -> full_cpt_v3      (stage1 + stage2, filter on skill_id)
    pl_hcl_v1   -> pl_hcl_v2        (stage1 + stage2,
                                     filter on anchor_skill_id OR any source_skill_ids donor)
    sft_v1      -> sft_v2           (filter classifier on skill_id|candidate_skill_id;
                                     filter sft_*.jsonl on metadata.{skill_id,candidate_skill_id})

Each new version dir keeps the same internal structure and writes a fresh
manifest.json + stats.json. The corresponding family registry.json gets a
new version entry appended.

Run
---
    python -m pipeline.scrub_challenge_from_datasets \
        --challenge-csv /N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-scanning/work/skills_sft_challenge/scan_results_reviewed.csv \
        --datasets-root /N/project/AdversarialModeling/datasets/agent_skills/misalignment
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------- shared utils

def load_challenge_ids(csv_path: Path) -> Set[str]:
    ids: Set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = (r.get("skill_id") or "").strip()
            if sid:
                ids.add(sid)
    return ids


def file_sha_size(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {"path": str(p), "size_bytes": 0, "sha256": ""}
    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def write_parquet_filtered(in_path: Path, out_path: Path, mask: pa.Array) -> Dict[str, int]:
    """Apply boolean mask and write zstd parquet. Returns {kept, dropped}."""
    tbl = pq.read_table(in_path)
    kept_tbl = tbl.filter(mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(kept_tbl, out_path, compression="zstd")
    return {"kept": kept_tbl.num_rows, "dropped": tbl.num_rows - kept_tbl.num_rows, "total": tbl.num_rows}


# ---------------------------------------------------------------- per-family scrubbers

def _scrub_skill_id_column(in_path: Path, out_path: Path, bad: Set[str], col: str = "skill_id") -> Dict[str, int]:
    tbl = pq.read_table(in_path)
    if col not in tbl.column_names:
        # Pass through unchanged if the schema lacks our filter column.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, out_path, compression="zstd")
        return {"kept": tbl.num_rows, "dropped": 0, "total": tbl.num_rows}
    col_array = tbl.column(col).to_pylist()
    mask = pa.array([v not in bad for v in col_array], type=pa.bool_())
    return write_parquet_filtered(in_path, out_path, mask)


def _scrub_two_skill_id_columns(in_path: Path, out_path: Path, bad: Set[str],
                                col_a: str, col_b: str) -> Dict[str, int]:
    tbl = pq.read_table(in_path)
    a = tbl.column(col_a).to_pylist()
    b = tbl.column(col_b).to_pylist()
    mask = pa.array([(va not in bad) and (vb not in bad) for va, vb in zip(a, b)], type=pa.bool_())
    return write_parquet_filtered(in_path, out_path, mask)


def _scrub_hcl_pairs(in_path: Path, out_path: Path, bad: Set[str]) -> Dict[str, int]:
    """HCL: drop rows where anchor_skill_id is bad OR any source_skill_ids donor is bad."""
    tbl = pq.read_table(in_path)
    anchors = tbl.column("anchor_skill_id").to_pylist()
    # source_skill_ids is stored as a string repr of a Python list.
    src_col_name = "source_skill_ids" if "source_skill_ids" in tbl.column_names else None
    if src_col_name is None:
        return _scrub_skill_id_column(in_path, out_path, bad, "anchor_skill_id")
    sources = tbl.column(src_col_name).to_pylist()

    keep: List[bool] = []
    for anc, src in zip(anchors, sources):
        if anc in bad:
            keep.append(False); continue
        try:
            donors = ast.literal_eval(src) if isinstance(src, str) else (src or [])
        except (ValueError, SyntaxError):
            donors = []
        if any((isinstance(d, str) and d in bad) for d in donors):
            keep.append(False); continue
        keep.append(True)
    mask = pa.array(keep, type=pa.bool_())
    return write_parquet_filtered(in_path, out_path, mask)


def _scrub_sft_jsonl(in_path: Path, out_path: Path, bad: Set[str]) -> Dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = d.get("metadata") or {}
            sid = meta.get("skill_id")
            cid = meta.get("candidate_skill_id")
            if (sid in bad) or (cid in bad):
                dropped += 1; continue
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    return {"kept": kept, "dropped": dropped, "total": kept + dropped}


# ---------------------------------------------------------------- driver

def scrub_family(in_root: Path, out_root: Path, bad: Set[str], scrubber, glob_pattern: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"files": {}, "totals": {"kept": 0, "dropped": 0, "total": 0}}
    for in_path in sorted(in_root.rglob(glob_pattern)):
        rel = in_path.relative_to(in_root)
        out_path = out_root / rel
        stats = scrubber(in_path, out_path, bad)
        summary["files"][str(rel)] = stats
        for k in ("kept", "dropped", "total"):
            summary["totals"][k] += stats[k]
        print(f"  {rel}  kept={stats['kept']:>9,}  dropped={stats['dropped']:>7,}  total={stats['total']:>9,}", flush=True)
    return summary


def write_manifest(out_dir: Path, family: str, version: str, in_version: str,
                   bad_count: int, totals: Dict[str, int], per_file: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": family,
        "output_version": version,
        "phase_name": "scrub_challenge_from_datasets",
        "input_version": in_version,
        "challenge_skill_ids_excluded": bad_count,
        "totals": totals,
        "per_file": per_file,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "stats.json").write_text(json.dumps({"totals": totals, "per_file": per_file}, indent=2))


def append_to_registry(reg_path: Path, family: str, version: str, in_version: str,
                       out_dir: Path, totals: Dict[str, int], bad_count: int) -> None:
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"dataset_name": family, "versions": []}
    reg["versions"] = [v for v in reg.get("versions", []) if v.get("version") != version]
    reg["versions"].append({
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "derived_from": in_version,
        "scrubbed_challenge_skill_ids": bad_count,
        "totals": totals,
        "output_dir": str(out_dir),
    })
    reg_path.write_text(json.dumps(reg, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description="Scrub challenge skill_ids from CPT / pl_hcl / SFT datasets.")
    p.add_argument("--challenge-csv", type=Path, required=True)
    p.add_argument("--datasets-root", type=Path,
                   default=Path("/N/project/AdversarialModeling/datasets/agent_skills/misalignment"))
    p.add_argument("--in-cpt-version",  default="full_cpt_v2")
    p.add_argument("--out-cpt-version", default="full_cpt_v3")
    p.add_argument("--in-hcl-version",  default="pl_hcl_v1")
    p.add_argument("--out-hcl-version", default="pl_hcl_v2")
    p.add_argument("--in-sft-version",  default="sft_v1")
    p.add_argument("--out-sft-version", default="sft_v2")
    p.add_argument("--skip-cpt", action="store_true")
    p.add_argument("--skip-hcl", action="store_true")
    p.add_argument("--skip-sft", action="store_true")
    args = p.parse_args()

    bad = load_challenge_ids(args.challenge_csv)
    print(f"[scrub] challenge skill_ids: {len(bad):,}")
    print(f"[scrub] datasets root: {args.datasets_root}")

    # ----- full_cpt -----
    if not args.skip_cpt:
        in_root  = args.datasets_root / "full_cpt" / args.in_cpt_version
        out_root = args.datasets_root / "full_cpt" / args.out_cpt_version
        print(f"\n[full_cpt] {args.in_cpt_version} -> {args.out_cpt_version}")
        summary = scrub_family(in_root, out_root, bad, _scrub_skill_id_column, "*.parquet")
        write_manifest(out_root, "full_cpt", args.out_cpt_version, args.in_cpt_version,
                       len(bad), summary["totals"], summary["files"])
        append_to_registry(args.datasets_root / "full_cpt" / "registry.json",
                           "full_cpt", args.out_cpt_version, args.in_cpt_version,
                           out_root, summary["totals"], len(bad))

    # ----- pl_hcl -----
    if not args.skip_hcl:
        in_root  = args.datasets_root / "pl_hcl" / args.in_hcl_version
        out_root = args.datasets_root / "pl_hcl" / args.out_hcl_version
        print(f"\n[pl_hcl] {args.in_hcl_version} -> {args.out_hcl_version}")
        summary = scrub_family(in_root, out_root, bad, _scrub_hcl_pairs, "*.parquet")
        write_manifest(out_root, "pl_hcl", args.out_hcl_version, args.in_hcl_version,
                       len(bad), summary["totals"], summary["files"])
        append_to_registry(args.datasets_root / "pl_hcl" / "registry.json",
                           "pl_hcl", args.out_hcl_version, args.in_hcl_version,
                           out_root, summary["totals"], len(bad))

    # ----- sft -----
    if not args.skip_sft:
        in_root  = args.datasets_root / "sft" / args.in_sft_version
        out_root = args.datasets_root / "sft" / args.out_sft_version
        print(f"\n[sft] {args.in_sft_version} -> {args.out_sft_version}")
        summary: Dict[str, Any] = {"files": {}, "totals": {"kept": 0, "dropped": 0, "total": 0}}

        # Parquet classifier files
        def _sft_classifier(inp: Path, outp: Path, bad_set: Set[str]) -> Dict[str, int]:
            return _scrub_two_skill_id_columns(inp, outp, bad_set, "skill_id", "candidate_skill_id")
        sub = scrub_family(in_root, out_root, bad, _sft_classifier, "classifier_*.parquet")
        summary["files"].update(sub["files"])
        for k in summary["totals"]:
            summary["totals"][k] += sub["totals"][k]
        # JSONL files
        for in_path in sorted(in_root.glob("sft_*.jsonl")):
            rel = in_path.relative_to(in_root)
            out_path = out_root / rel
            stats = _scrub_sft_jsonl(in_path, out_path, bad)
            summary["files"][str(rel)] = stats
            for k in ("kept", "dropped", "total"):
                summary["totals"][k] += stats[k]
            print(f"  {rel}  kept={stats['kept']:>9,}  dropped={stats['dropped']:>7,}  total={stats['total']:>9,}", flush=True)

        write_manifest(out_root, "sft", args.out_sft_version, args.in_sft_version,
                       len(bad), summary["totals"], summary["files"])
        append_to_registry(args.datasets_root / "sft" / "registry.json",
                           "sft", args.out_sft_version, args.in_sft_version,
                           out_root, summary["totals"], len(bad))

    print("\n[scrub] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
