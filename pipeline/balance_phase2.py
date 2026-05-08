#!/usr/bin/env python3
"""Balanced-distribution sampler for the Phase 2 HCL synthesized data.

After `build_phase2_hcl` + `enrich_t3a` + `enrich_t3c` have produced
their per-label parquets, the row counts per type are quite uneven:

    train.parquet            ~  positives             (1 per anchor)
    train_t1.parquet         ~  T1     internal layer permutation        (1 per anchor)
    train_t2.parquet         ~  T2     external layer swap               (4–18 per anchor)
    train_t3a.parquet        ~  T3a    LLM behavior corruption           (1–2 per anchor)
    train_t3b.parquet        ~  T3b    span swap                         (2–4 per anchor)
    train_t3c.parquet        ~  T3c    rule-based hallucinated id        (≤ 2 per anchor)

This module emits **balanced** views so the trainer sees equal counts
per type within each split. It does NOT delete the original per-label
parquets — those remain available for ablations or per-class weighting.

Two views are produced, matching the documented role split:

  T1  → continual pretraining (CPT). Written to:
        full_cpt/<full_cpt_version>/<stage>/<split>_balanced_cpt.parquet
        which is the union of {positives + T1}, capped per type.

  T2/T3 → contrastive learning (HCL). Written to:
        pl_hcl/<output_version>/<stage>/<split>_balanced_contrastive.parquet
        which is the union of {T2, T3a, T3b, T3c}, capped per type to the
        same number of rows per type so no negative type dominates.

Pool/split assignments are inherited from the upstream parquets — this
script only re-shuffles, it never re-assigns. Same skill stays in same
split across stages and across views.

Usage:

    python -m pipeline.balance_phase2 \\
        --pl-hcl-root /path/to/datasets/misalignment/pl_hcl \\
        --full-cpt-root /path/to/datasets/misalignment/full_cpt \\
        --pl-hcl-version pl_hcl_v1 \\
        --full-cpt-version full_cpt_v1 \\
        --splits train,val,test \\
        --target-rows-per-type 0   # 0 = use min(rows per type)

`--target-rows-per-type 0` (default) downsamples each type to the
smallest available count for that split — guaranteed equal counts.
A positive integer caps each type to that many rows (ceil to
`min(rows, target)`); useful for fast smoke runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pipeline._shared import LOGGER, configure_logging, ensure_dir, write_json


CPT_LABELS = ("positive", "t1")
CONTRASTIVE_LABELS = ("t2", "t3a", "t3b", "t3c")


def _parquet_for(stage_dir: Path, split: str, label: str) -> Path:
    """Map (split, label) → on-disk parquet path.

    The positives parquet for a split is `<split>.parquet`; every
    negative type lives in `<split>_<label>.parquet`.
    """
    if label == "positive":
        return stage_dir / f"{split}.parquet"
    return stage_dir / f"{split}_{label}.parquet"


def _read_count(path: Path) -> int:
    """Return the row count of a parquet without materializing it."""
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyarrow is required: pip install pyarrow") from exc
    return pq.ParquetFile(str(path)).metadata.num_rows


def _read_table(path: Path):
    import pyarrow.parquet as pq  # type: ignore
    return pq.read_table(str(path))


def _write_table(path: Path, table) -> None:
    import pyarrow.parquet as pq  # type: ignore
    ensure_dir(path.parent)
    pq.write_table(table, str(path), compression="zstd")


def _balanced_concat(
    paths_per_label: Dict[str, Path],
    target_rows: int,
    seed: int,
) -> Optional[Any]:
    """Read each label's parquet, downsample to `target_rows` rows, concat.

    Returns the unified pyarrow Table (or None if no input had any rows).
    Sampling is deterministic given `seed` so the balanced view is
    reproducible.
    """
    import pyarrow as pa  # type: ignore

    pieces = []
    chosen = {}
    for label, path in paths_per_label.items():
        if not path.exists():
            chosen[label] = 0
            continue
        table = _read_table(path)
        n = table.num_rows
        if n == 0:
            chosen[label] = 0
            continue
        if n > target_rows:
            # Deterministic shuffle then head.
            indices = pa.array(list(range(n)))
            import random as _rnd
            rng = _rnd.Random(f"{seed}|{label}|{path.name}")
            order = list(range(n))
            rng.shuffle(order)
            kept = sorted(order[: target_rows])
            table = table.take(pa.array(kept))
        chosen[label] = table.num_rows
        # Annotate every row with the source label so the trainer can
        # weight or ablate by class without re-deriving from the file name.
        if "balanced_label" not in table.column_names:
            table = table.append_column(
                "balanced_label", pa.array([label] * table.num_rows, type=pa.string())
            )
        pieces.append(table)

    LOGGER.info("[balance] per-label rows kept: %s", chosen)
    if not pieces:
        return None
    return pa.concat_tables(pieces, promote_options="default")


def balance_split(
    *,
    stage: str,
    split: str,
    pl_hcl_stage_dir: Path,
    full_cpt_stage_dir: Path,
    target_rows_per_type: int,
    seed: int,
) -> Dict[str, Dict[str, int]]:
    """Produce both balanced views for one (stage, split). Returns a
    small stats dict suitable for the run-level summary."""
    out: Dict[str, Dict[str, int]] = {}

    # ---- CPT view: positives + T1 -------------------------------------
    # Positives live in the full_cpt stage dir (Phase 1 output); T1
    # lives in pl_hcl with the rest of the negatives. They share the
    # `(stage, split)` partitioning, so we can union them.
    cpt_paths = {
        "positive": full_cpt_stage_dir / f"{split}.parquet",
        "t1":       pl_hcl_stage_dir   / f"{split}_t1.parquet",
    }
    cpt_counts = {label: _read_count(p) for label, p in cpt_paths.items()}
    cpt_target = (
        min(c for c in cpt_counts.values() if c > 0)
        if target_rows_per_type <= 0 and any(cpt_counts.values())
        else target_rows_per_type
    )
    if cpt_target > 0 and any(cpt_counts.values()):
        table = _balanced_concat(cpt_paths, cpt_target, seed)
        if table is not None:
            out_path = full_cpt_stage_dir / f"{split}_balanced_cpt.parquet"
            _write_table(out_path, table)
            LOGGER.info(
                "[balance] CPT %s/%s wrote %d rows to %s",
                stage, split, table.num_rows, out_path,
            )
            out["cpt"] = {"target_per_type": cpt_target, "rows_written": table.num_rows,
                          "input_counts": cpt_counts}
    else:
        LOGGER.info("[balance] CPT %s/%s skipped (no input rows)", stage, split)
        out["cpt"] = {"target_per_type": cpt_target, "rows_written": 0,
                      "input_counts": cpt_counts}

    # ---- Contrastive view: T2 + T3a + T3b + T3c ------------------------
    contr_paths = {label: pl_hcl_stage_dir / f"{split}_{label}.parquet"
                   for label in CONTRASTIVE_LABELS}
    contr_counts = {label: _read_count(p) for label, p in contr_paths.items()}
    nonzero = [c for c in contr_counts.values() if c > 0]
    contr_target = (
        min(nonzero)
        if target_rows_per_type <= 0 and nonzero
        else target_rows_per_type
    )
    if contr_target > 0 and nonzero:
        table = _balanced_concat(contr_paths, contr_target, seed)
        if table is not None:
            out_path = pl_hcl_stage_dir / f"{split}_balanced_contrastive.parquet"
            _write_table(out_path, table)
            LOGGER.info(
                "[balance] HCL %s/%s wrote %d rows to %s",
                stage, split, table.num_rows, out_path,
            )
            out["contrastive"] = {"target_per_type": contr_target,
                                  "rows_written": table.num_rows,
                                  "input_counts": contr_counts}
    else:
        LOGGER.info("[balance] HCL %s/%s skipped (no input rows)", stage, split)
        out["contrastive"] = {"target_per_type": contr_target, "rows_written": 0,
                              "input_counts": contr_counts}

    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pl-hcl-root",     required=True,
                   help="Directory holding pl_hcl/<version>/")
    p.add_argument("--full-cpt-root",   required=True,
                   help="Directory holding full_cpt/<version>/")
    p.add_argument("--pl-hcl-version",  required=True,
                   help="pl_hcl version name (e.g. pl_hcl_v1)")
    p.add_argument("--full-cpt-version", required=True,
                   help="full_cpt version name (e.g. full_cpt_v1)")
    p.add_argument("--splits",  default="train,val,test",
                   help="Comma-separated splits to balance.")
    p.add_argument("--stages",  default="stage1,stage2",
                   help="Comma-separated stages to balance.")
    p.add_argument("--target-rows-per-type", type=int, default=0,
                   help="0 = downsample each type to min(rows). "
                        ">0 = cap each type at this many rows.")
    p.add_argument("--seed",    type=int, default=20260601,
                   help="Sampling seed for deterministic balanced output.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))

    pl_hcl_root  = Path(args.pl_hcl_root).resolve()
    full_cpt_root = Path(args.full_cpt_root).resolve()
    pl_hcl_version_dir  = pl_hcl_root / args.pl_hcl_version
    full_cpt_version_dir = full_cpt_root / args.full_cpt_version

    if not pl_hcl_version_dir.exists():
        LOGGER.error("pl_hcl version dir missing: %s", pl_hcl_version_dir)
        return 1
    if not full_cpt_version_dir.exists():
        LOGGER.error("full_cpt version dir missing: %s", full_cpt_version_dir)
        return 1

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    summary: Dict[str, Any] = {
        "pl_hcl_version": args.pl_hcl_version,
        "full_cpt_version": args.full_cpt_version,
        "target_rows_per_type": args.target_rows_per_type,
        "seed": args.seed,
        "per_stage_split": {},
    }

    for stage in stages:
        pl_hcl_stage = pl_hcl_version_dir / stage
        full_cpt_stage = full_cpt_version_dir / stage
        if not pl_hcl_stage.exists() and not full_cpt_stage.exists():
            LOGGER.warning("[balance] stage %s missing in both roots; skipping", stage)
            continue
        for split in splits:
            entry = balance_split(
                stage=stage,
                split=split,
                pl_hcl_stage_dir=pl_hcl_stage,
                full_cpt_stage_dir=full_cpt_stage,
                target_rows_per_type=args.target_rows_per_type,
                seed=args.seed,
            )
            summary["per_stage_split"][f"{stage}/{split}"] = entry

    summary_path = pl_hcl_version_dir / "stats_balanced.json"
    write_json(summary_path, summary)
    LOGGER.info("[balance] summary written to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
