"""Build train_canonical_skill_records.jsonl by filtering out challenge skill_ids.

Streams the BR200 canonical_skill_records.jsonl, drops any record whose
skill_id appears in scan_results_reviewed.csv (the 1,444 challenge skills),
and writes the remainder to inputs/train_canonical_skill_records.jsonl.

Of the 1,444 challenge skills, only 816 are actually present in canonical;
the other 628 (166 collected-not-in-canonical + 263 MASB + 199 HSB) were
never in the BR200 corpus and need no filtering. Expected output count:
~200,858 records (201,674 valid input − 816 challenge overlap).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

SCAN_DIR = Path("/media/volume/skills/AgentSkills-OSS/agent-skills-scanning")
PREP_DIR = Path("/media/volume/skills/AgentSkills-OSS/agent-skills-preparation")
REVIEWED = SCAN_DIR / "work/skills_sft_challenge/scan_results_reviewed.csv"
CANONICAL_FULL = Path("/media/volume/skills/canonical_skill_records.jsonl")
OUT_JSONL = PREP_DIR / "inputs/train_canonical_skill_records.jsonl"


def main() -> int:
    holdout: set[str] = set()
    with REVIEWED.open(newline="") as f:
        for r in csv.DictReader(f):
            holdout.add(r["skill_id"])
    print(f"[train] holdout (challenge) skill_ids: {len(holdout)}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_in = n_out = n_skip = n_err = 0
    with CANONICAL_FULL.open("rb") as fin, OUT_JSONL.open("w") as fout:
        for line in fin:
            n_in += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_err += 1
                continue
            sid = d.get("skill_id")
            if not sid:
                n_err += 1
                continue
            if sid in holdout:
                n_skip += 1
                continue
            fout.write(line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line)
            n_out += 1
            if n_in % 50000 == 0:
                print(f"[train]   in={n_in}  out={n_out}  skip={n_skip}  err={n_err}", flush=True)

    print(f"[train] DONE in {time.time()-t0:.1f}s")
    print(f"[train] in={n_in}  out={n_out}  skipped_challenge={n_skip}  parse_err={n_err}")
    print(f"[train] wrote -> {OUT_JSONL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
