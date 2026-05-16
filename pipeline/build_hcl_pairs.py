#!/usr/bin/env python3
"""Build the joined HCL pairs file the stages/hcl trainer expects.

The pl_hcl_v2 outputs are *single-view* parquets — one row per (anchor,
rendering-variant) tuple, with the variant kind encoded in the `label`
column ({T1, T2, T3a, T3b, T3c}). The HCL trainer at
agent-skills-training/stages/hcl/data.py wants a *pre-joined* file with
columns:

    anchor_text, pair_text, pair_kind, anchor_stage, pair_stage, label

This script joins each anchor's canonical row (from `<split>.parquet`,
label=positive, M→I layer order) with every same-anchor variant row
(T1 / T2 / T3a / T3b / T3c) to produce one pair row per variant.

  pair_kind mapping (matches stages/hcl/data.py PAIR_KIND_TO_ID):
      T1            -> "positive"     (same skill, I→M reorder)
      T2            -> "swapped"      (metadata swapped from a donor)
      T3a/T3b/T3c   -> "corrupted"    (instruction-layer corruption)

  label is set to 1 for positive pairs, 0 for swapped/corrupted.

Inputs (per stage):
    <pl_hcl_root>/stage{1,2}/<split>.parquet           (canonical anchor)
    <pl_hcl_root>/stage{1,2}/<split>_t1.parquet        (positive variants)
    <pl_hcl_root>/stage{1,2}/<split>_t2.parquet        (swapped negatives)
    <pl_hcl_root>/stage{1,2}/<split>_t3{a,b,c}.parquet (corrupted negatives)

Outputs (per stage):
    <pl_hcl_root>/stage{1,2}/pairs_<split>.parquet

Uses pyarrow streaming so memory stays well under a few GB even on the
large stage-2 T2 file (650K rows × ~30KB text each).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq


PAIR_SPECS: List[tuple[str, str]] = [
    ("t1",  "positive"),
    ("t2",  "swapped"),
    ("t3a", "corrupted"),
    ("t3b", "corrupted"),
    ("t3c", "corrupted"),
]

BATCH_SIZE = 4_000  # rows per write batch — keeps memory under ~500 MB even at 30KB texts


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_anchors(anchor_path: Path) -> Dict[str, str]:
    """Read canonical anchor texts indexed by anchor_skill_id.

    Each anchor appears once per split. Returns {skill_id: text}.
    """
    tbl = pq.read_table(anchor_path, columns=["anchor_skill_id", "text"])
    ids = tbl.column("anchor_skill_id").to_pylist()
    texts = tbl.column("text").to_pylist()
    return dict(zip(ids, texts))


def iter_pair_rows(stage_dir: Path, split: str, anchors: Dict[str, str], stage_name: str):
    """Yield batches of pair-row dicts ready to write to the output parquet."""
    for kind_code, kind_name in PAIR_SPECS:
        path = stage_dir / f"{split}_{kind_code}.parquet"
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        n_emitted = n_skipped = 0
        for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                     columns=["sample_id", "anchor_skill_id", "text",
                                              "sub_strategy", "target_layer",
                                              "source_skill_ids", "pool", "split"]):
            d = batch.to_pydict()
            rows: List[dict] = []
            for i in range(batch.num_rows):
                anchor_sid = d["anchor_skill_id"][i]
                anchor_text = anchors.get(anchor_sid)
                if anchor_text is None:
                    n_skipped += 1
                    continue
                pair_text = d["text"][i]
                rows.append({
                    "sample_id":       _hash(anchor_sid, stage_name, split, kind_code, d["sample_id"][i] or ""),
                    "anchor_skill_id": anchor_sid,
                    "anchor_text":     anchor_text,
                    "pair_text":       pair_text,
                    "pair_kind":       kind_name,
                    "pair_subkind":    kind_code.upper(),       # T1/T2/T3a/T3b/T3c (for slicing)
                    "anchor_stage":    stage_name,
                    "pair_stage":      stage_name,
                    "label":           1 if kind_name == "positive" else 0,
                    "source_skill_ids": d["source_skill_ids"][i],
                    "sub_strategy":    d["sub_strategy"][i],
                    "target_layer":    d["target_layer"][i],
                    "pool":            d["pool"][i],
                    "split":           split,
                })
                n_emitted += 1
            if rows:
                yield rows
        print(f"    {kind_code:>3} kind={kind_name:<9}  emitted={n_emitted:>9,}  skipped(no-anchor)={n_skipped}", flush=True)


def join_split(stage_dir: Path, split: str, stage_name: str, out_path: Path) -> int:
    anchor_path = stage_dir / f"{split}.parquet"
    if not anchor_path.exists():
        print(f"  [skip] no anchor file at {anchor_path}")
        return 0
    print(f"  loading anchors from {anchor_path.name} ...", flush=True)
    anchors = load_anchors(anchor_path)
    print(f"    {len(anchors):,} anchors", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_total = 0
    try:
        for rows in iter_pair_rows(stage_dir, split, anchors, stage_name):
            tbl = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            else:
                # Re-project to ensure schema match across batches (mostly a no-op here).
                tbl = tbl.cast(writer.schema)
            writer.write_table(tbl)
            n_total += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()
    print(f"  -> wrote {n_total:,} pair rows to {out_path}", flush=True)
    return n_total


def main() -> int:
    p = argparse.ArgumentParser(description="Build joined HCL pairs files from single-view pl_hcl outputs.")
    p.add_argument("--pl-hcl-version-dir", type=Path,
                   default=Path("/N/project/AdversarialModeling/datasets/agent_skills/misalignment/pl_hcl/pl_hcl_v2"))
    p.add_argument("--stages", nargs="*", default=["stage1", "stage2"])
    p.add_argument("--splits", nargs="*", default=["train", "val", "test", "unseen"])
    args = p.parse_args()

    summary: Dict[str, Dict[str, int]] = {}
    for stage in args.stages:
        stage_dir = args.pl_hcl_version_dir / stage
        if not stage_dir.exists():
            print(f"[skip] {stage_dir} does not exist")
            continue
        print(f"\n[stage] {stage}")
        summary[stage] = {}
        for split in args.splits:
            out_path = stage_dir / f"pairs_{split}.parquet"
            print(f"\n[{stage}/{split}]")
            n = join_split(stage_dir, split, stage, out_path)
            summary[stage][split] = n

    # Append a manifest entry for this build.
    manifest_path = args.pl_hcl_version_dir / "pairs_manifest.json"
    manifest_path.write_text(json.dumps({
        "phase_name": "build_hcl_pairs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_version_dir": str(args.pl_hcl_version_dir),
        "stages": args.stages,
        "splits": args.splits,
        "row_counts": summary,
        "pair_specs": [{"code": k, "kind": v} for k, v in PAIR_SPECS],
        "notes": (
            "anchor_text = canonical M->I view from <split>.parquet; "
            "pair_text = same-anchor T1/T2/T3a/T3b/T3c view; "
            "pair_kind in {positive, swapped, corrupted}; "
            "label = 1 for positive else 0."
        ),
    }, indent=2))
    print(f"\n[done] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
