#!/usr/bin/env python3
"""Summary statistics for the curated CPT / pl_hcl / SFT datasets.

Walks each family's version dir, counts rows + label / stage / source
breakdowns where applicable, and prints a tidy report. Designed to be
used both pre-scrub (full_cpt_v2, pl_hcl_v1, sft_v1) and post-scrub
(full_cpt_v3, pl_hcl_v2, sft_v2) so the impact of the challenge-set
removal is visible at a glance.

Run
---
    python -m pipeline.dataset_stats \
        --datasets-root /N/project/AdversarialModeling/datasets/agent_skills/misalignment \
        --cpt-versions full_cpt_v2 full_cpt_v3 \
        --hcl-versions pl_hcl_v1 pl_hcl_v2 \
        --sft-versions sft_v1 sft_v2 sft_challenge_v1
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow.parquet as pq


# ----------------------------------------------------------------- helpers

def _rows_for(p: Path) -> int:
    try:
        return pq.read_metadata(p).num_rows
    except Exception:
        return -1


def _col_counter(p: Path, col: str) -> Counter:
    try:
        tbl = pq.read_table(p, columns=[col])
        return Counter(tbl.column(col).to_pylist())
    except Exception:
        return Counter()


def _jsonl_rows(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


# ----------------------------------------------------------------- family reporters

def report_cpt(version_dir: Path) -> Dict[str, Any]:
    """full_cpt_<v>/stage{1,2}/{train,val,test,unseen}.parquet"""
    out: Dict[str, Any] = {"version": version_dir.name, "stages": {}, "totals": {}}
    grand = Counter()
    for stage_dir in sorted(p for p in version_dir.iterdir() if p.is_dir()):
        stage = {}
        for split in ("train", "val", "test", "unseen"):
            p = stage_dir / f"{split}.parquet"
            stage[split] = _rows_for(p) if p.exists() else 0
            grand[split] += stage[split] if stage[split] > 0 else 0
        out["stages"][stage_dir.name] = stage
    out["totals"] = dict(grand)
    out["grand_total_rows"] = sum(grand.values())
    return out


def report_hcl(version_dir: Path) -> Dict[str, Any]:
    """pl_hcl_<v>/stage{1,2}/{train,val,test,unseen}_{t1,t2,t3a,t3b,t3c}.parquet
    plus the combined {train,val,test,unseen}.parquet"""
    out: Dict[str, Any] = {"version": version_dir.name, "stages": {}, "totals": {}}
    grand = Counter()
    for stage_dir in sorted(p for p in version_dir.iterdir() if p.is_dir()):
        stage: Dict[str, Any] = {}
        for p in sorted(stage_dir.glob("*.parquet")):
            stage[p.name] = _rows_for(p)
            grand["all_files"] += stage[p.name] if stage[p.name] > 0 else 0
        # label distribution across the combined train.parquet (if present)
        for split in ("train", "val", "test"):
            tt = stage_dir / f"{split}.parquet"
            if tt.exists():
                stage[f"{split}_label_dist"] = dict(_col_counter(tt, "label"))
        out["stages"][stage_dir.name] = stage
    out["totals"] = dict(grand)
    return out


def report_sft(version_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"version": version_dir.name, "files": {}, "label_dist": {}}
    for split in ("train", "val", "test"):
        cls = version_dir / f"classifier_{split}.parquet"
        jl  = version_dir / f"sft_{split}.jsonl"
        if cls.exists():
            n = _rows_for(cls)
            out["files"][cls.name] = n
            try:
                out["label_dist"][split] = dict(_col_counter(cls, "label"))
            except Exception:
                pass
        if jl.exists():
            out["files"][jl.name] = _jsonl_rows(jl)
    return out


# ----------------------------------------------------------------- printing

def _print_table(title: str, rows: List[List[str]], headers: List[str]) -> None:
    if not rows:
        return
    print(f"\n== {title} ==")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def print_cpt(rep: Dict[str, Any]) -> None:
    rows = []
    for stage, splits in rep["stages"].items():
        for split, n in splits.items():
            rows.append([rep["version"], stage, split, f"{n:,}"])
    rows.append(["", "", "**total**", f"{rep['grand_total_rows']:,}"])
    _print_table(f"CPT {rep['version']}", rows, ["version", "stage", "split", "rows"])


def print_hcl(rep: Dict[str, Any]) -> None:
    rows = []
    for stage, files in rep["stages"].items():
        for fname, n in files.items():
            if isinstance(n, dict):  # label dist
                rows.append([rep["version"], stage, fname, json.dumps(n)])
            else:
                rows.append([rep["version"], stage, fname, f"{n:,}"])
    rows.append(["", "", "**total_all_files**", f"{rep['totals'].get('all_files', 0):,}"])
    _print_table(f"HCL {rep['version']}", rows, ["version", "stage", "file", "rows / label_dist"])


def print_sft(rep: Dict[str, Any]) -> None:
    rows = []
    for fname, n in rep["files"].items():
        rows.append([rep["version"], fname, f"{n:,}"])
    for split, dist in rep["label_dist"].items():
        rows.append([rep["version"], f"  classifier_{split} label_dist", json.dumps(dist)])
    _print_table(f"SFT {rep['version']}", rows, ["version", "file / metric", "rows / dist"])


# ----------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-root", type=Path,
                   default=Path("/N/project/AdversarialModeling/datasets/agent_skills/misalignment"))
    p.add_argument("--cpt-versions", nargs="*", default=["full_cpt_v2", "full_cpt_v3"])
    p.add_argument("--hcl-versions", nargs="*", default=["pl_hcl_v1", "pl_hcl_v2"])
    p.add_argument("--sft-versions", nargs="*", default=["sft_v1", "sft_v2", "sft_challenge_v1", "challenge"])
    p.add_argument("--json-out", type=Path, default=None,
                   help="Optional path to dump the full structured report as JSON.")
    args = p.parse_args()

    report: Dict[str, Any] = {"cpt": [], "hcl": [], "sft": []}

    for v in args.cpt_versions:
        d = args.datasets_root / "full_cpt" / v
        if not d.exists():
            print(f"[stats] (skip) {d} does not exist")
            continue
        rep = report_cpt(d)
        print_cpt(rep)
        report["cpt"].append(rep)

    for v in args.hcl_versions:
        d = args.datasets_root / "pl_hcl" / v
        if not d.exists():
            print(f"[stats] (skip) {d} does not exist")
            continue
        rep = report_hcl(d)
        print_hcl(rep)
        report["hcl"].append(rep)

    for v in args.sft_versions:
        d = args.datasets_root / "sft" / v
        if not d.exists():
            print(f"[stats] (skip) {d} does not exist")
            continue
        rep = report_sft(d)
        print_sft(rep)
        report["sft"].append(rep)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\n[stats] full report -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
