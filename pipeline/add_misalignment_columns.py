#!/usr/bin/env python3
"""Add misalignment-task columns to existing SFT classifier parquets.

The misalignment_detection task at
    agent-skills-training/stages/downstream/sft/tasks/misalignment_detection.py
expects rows with:

    skill_text       (str) — the rendered SKILL package contents
    alignment_class  (str) — "ALIGNED" or "MISALIGNED"

The sft_v1/sft_v2 classifier parquets were built for description-matching
SFT (yes/no targets) and lack these columns. This script attaches them
in-place (or via a copy) so the same files become consumable by the
misalignment trainer.

Heuristic mapping for sft_v1/v2 (description-matching paradigm):
    label == 1  (own description matches own evidence) -> ALIGNED
    label == 0  (shifted description doesn't match)    -> MISALIGNED

`skill_text` is set to `evidence_text` (which already holds the rendered
"# Skill Name / # Evidence / # Files" block).

Usage
-----
    python -m pipeline.add_misalignment_columns \
        --target-dir /N/project/AdversarialModeling/datasets/agent_skills/misalignment/sft/sft_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq


CLASSIFIER_GLOB = "classifier_*.parquet"


def _augment(in_path: Path, out_path: Path) -> dict:
    tbl = pq.read_table(in_path)
    cols = tbl.column_names

    # Skip if already augmented.
    have_skill_text = "skill_text" in cols
    have_alignment  = "alignment_class" in cols

    new_cols: List[pa.Array] = []
    new_names: List[str] = []

    if not have_skill_text:
        # Prefer evidence_text; fall back to input_text if not present.
        src = "evidence_text" if "evidence_text" in cols else "input_text" if "input_text" in cols else None
        if src is None:
            raise ValueError(f"{in_path}: no evidence_text or input_text column to derive skill_text from.")
        new_cols.append(tbl.column(src))
        new_names.append("skill_text")

    if not have_alignment:
        if "label" not in cols:
            raise ValueError(f"{in_path}: no label column to derive alignment_class from.")
        # Map label int -> "ALIGNED"/"MISALIGNED" using pyarrow compute.
        labels = tbl.column("label").to_pylist()
        align_arr = pa.array(["ALIGNED" if int(v) == 1 else "MISALIGNED" for v in labels], type=pa.string())
        new_cols.append(align_arr)
        new_names.append("alignment_class")

    if not have_skill_text or not have_alignment:
        # Append the new columns in one go.
        tbl_out = tbl
        for name, arr in zip(new_names, new_cols):
            tbl_out = tbl_out.append_column(name, arr)
    else:
        tbl_out = tbl

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl_out, out_path, compression="zstd")

    return {
        "rows":            tbl_out.num_rows,
        "added_skill_text": not have_skill_text,
        "added_alignment_class": not have_alignment,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Add skill_text + alignment_class columns to SFT classifier parquets.")
    p.add_argument("--target-dir", type=Path, required=True,
                   help="SFT version dir (e.g. .../sft/sft_v2) containing classifier_*.parquet.")
    p.add_argument("--out-dir", type=Path,
                   help="Where to write augmented parquets. Default: in-place (same as --target-dir).")
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir else args.target_dir
    matches = sorted(args.target_dir.glob(CLASSIFIER_GLOB))
    if not matches:
        print(f"[warn] no {CLASSIFIER_GLOB} under {args.target_dir}")
        return 1

    for in_path in matches:
        rel = in_path.relative_to(args.target_dir)
        out_path = out_dir / rel
        stats = _augment(in_path, out_path)
        print(f"  {rel}  rows={stats['rows']:>9,}  "
              f"+skill_text={stats['added_skill_text']}  "
              f"+alignment_class={stats['added_alignment_class']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
