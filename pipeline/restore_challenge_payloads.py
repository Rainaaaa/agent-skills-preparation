#!/usr/bin/env python3
"""Restore full skill payloads into the SFT challenge folder.

The human-reviewed challenge set at
    /N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-scanning/work/skills_sft_challenge/
contains 1,444 reviewed skills (gold labels in `scan_results_reviewed.csv`).
For the 982 rows with source=`collected`, the per-skill directory typically
holds only `manifest.json` + `SKILL.md` — the reference files / scripts /
docs from the original repo are *missing*.

This restore pulls those payloads from the canonical package store at
    /N/project/AdversarialModeling/agent_skills/packages/<skill_id>/files/
(which is itself a symlink to the raw repo, so files come through with full
fidelity) and copies them into the challenge folder, preserving subdir
structure. Existing files are left untouched (idempotent).

`source=hsb|masb` rows are skipped — they don't live under the BR200
packages root and ship as-is.

Usage
-----
    python -m pipeline.restore_challenge_payloads \
        --reviewed-csv /N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-scanning/work/skills_sft_challenge/scan_results_reviewed.csv \
        --challenge-dir /N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-scanning/work/skills_sft_challenge \
        --packages-root /N/project/AdversarialModeling/agent_skills/packages \
        [--dry-run]

Idempotent: re-running only copies newly-missing files.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple


SKIP_TOPLEVEL = {"manifest.json"}  # SKILL.md already lives in challenge dir as resolved content


def restore_one(sid: str, pkg_root: Path, chal_dir: Path, dry_run: bool) -> Tuple[int, int, int]:
    """Return (copied, skipped_existing, bytes_copied)."""
    src_root = pkg_root / sid / "files"
    try:
        # follow the canonical `files` symlink into raw_repos
        src_root_resolved = src_root.resolve(strict=True)
    except FileNotFoundError:
        return (0, 0, 0)
    if not src_root_resolved.is_dir():
        return (0, 0, 0)

    dst_root = chal_dir / sid
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = skipped = bytes_copied = 0
    for root, _dirs, files in os.walk(src_root_resolved, followlinks=True):
        rel_root = Path(root).relative_to(src_root_resolved)
        for fn in files:
            rel_path = (rel_root / fn).as_posix()
            # Skip top-level SKILL.md / manifest.json — already shipped.
            if rel_root == Path(".") and fn in SKIP_TOPLEVEL:
                continue
            if rel_root == Path(".") and fn in ("SKILL.md", "skill.md"):
                # SKILL.md is already populated in the challenge dir.
                continue

            src = Path(root) / fn
            dst = dst_root / rel_path
            if dst.exists():
                skipped += 1
                continue
            try:
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                bytes_copied += src.stat().st_size
                copied += 1
            except OSError as e:
                print(f"  [warn] copy failed {sid}/{rel_path}: {e}", file=sys.stderr)
    return (copied, skipped, bytes_copied)


def main() -> int:
    p = argparse.ArgumentParser(description="Restore missing skill payloads into the challenge folder.")
    p.add_argument("--reviewed-csv", type=Path, required=True,
                   help="scan_results_reviewed.csv with gold labels (source column required).")
    p.add_argument("--challenge-dir", type=Path, required=True,
                   help="work/skills_sft_challenge/ root — per-skill subdirs live here.")
    p.add_argument("--packages-root", type=Path,
                   default=Path("/N/project/AdversarialModeling/agent_skills/packages"),
                   help="Canonical package store. Each <skill_id>/files/ -> raw_repos symlink.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be copied without touching disk.")
    args = p.parse_args()

    with args.reviewed_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    collected = [r["skill_id"] for r in rows if (r.get("source") or "").lower() == "collected"]
    print(f"[restore] collected rows: {len(collected):>5}  (of {len(rows)} total)")
    print(f"[restore] dry_run={args.dry_run}  packages_root={args.packages_root}")

    n_done = n_with_copies = 0
    tot_copied = tot_skipped = tot_bytes = 0
    for sid in collected:
        copied, skipped, bytes_copied = restore_one(sid, args.packages_root, args.challenge_dir, args.dry_run)
        tot_copied += copied
        tot_skipped += skipped
        tot_bytes += bytes_copied
        if copied:
            n_with_copies += 1
        n_done += 1
        if n_done % 100 == 0:
            print(f"  [{n_done}/{len(collected)}]  copied={tot_copied}  skipped={tot_skipped}  "
                  f"MiB={tot_bytes/1024/1024:.1f}", flush=True)

    print()
    print(f"[restore] done. {n_done} collected skills inspected.")
    print(f"  skills with new files copied : {n_with_copies}")
    print(f"  total files copied           : {tot_copied}")
    print(f"  total files skipped (exist)  : {tot_skipped}")
    print(f"  bytes copied                 : {tot_bytes:,}  ({tot_bytes/1024/1024:.1f} MiB)")
    if args.dry_run:
        print("  (dry-run: no files were written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
