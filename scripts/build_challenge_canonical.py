"""Build challenge_canonical_skill_records.jsonl for the 1,444 reviewed challenge skills.

Hybrid strategy:
  - 816 collected skills present in BR200's canonical_skill_records.jsonl
    → extracted verbatim (preserves full package_files from BR200 canonicalize).
  - 628 not-in-canonical (166 collected + 263 MASB + 199 HSB)
    → local-canonicalized from work/skills_sft_challenge/<skill_id>/.

Both paths produce the same JSONL schema so downstream prep treats them identically.

Per-record schema (matches BR200 canonical_skill_records.jsonl):
  skill_id        : str
  first_layer     : {name, description}              # from SKILL.md frontmatter
  skill_md        : {description_section, non_description}
  package_files   : [{path, content, size_bytes, truncated}, ...]
  derived         : {description_from_metadata, has_explicit_description_heading,
                     used_metadata_description_fallback}

description_section is the body's `## Description` (case-insensitive) block:
  found=True → heading + content captured, that block REMOVED from non_description
  found=False → heading/content null, non_description = full SKILL.md
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

SCAN_DIR = Path("/media/volume/skills/AgentSkills-OSS/agent-skills-scanning")
PREP_DIR = Path("/media/volume/skills/AgentSkills-OSS/agent-skills-preparation")
REVIEWED = SCAN_DIR / "work/skills_sft_challenge/scan_results_reviewed.csv"
SFT_ROOT = SCAN_DIR / "work/skills_sft_challenge"
CANONICAL_FULL = Path("/media/volume/skills/canonical_skill_records.jsonl")
OUT_JSONL = PREP_DIR / "inputs/challenge_canonical_skill_records.jsonl"

# Mirrors what we see in BR200 records (LICENSE.txt at 1467 bytes had truncated=False).
# Prep stage has its own per-file caps, so we keep this generous.
MAX_FILE_BYTES = 200_000

# Files that are NOT package_files (they're consumed for first_layer / housekeeping).
SKIP_FILES = {"SKILL.md", "skill.md", "manifest.json"}

# Case-insensitive: a heading line whose text is exactly "description".
_DESC_HEADING_RE = re.compile(r"^(#+)\s+(description)\s*$", re.IGNORECASE | re.MULTILINE)
_ANY_HEADING_RE = re.compile(r"^#+\s+\S", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body) — body is text minus the ---...--- block."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = m.group(2)
    try:
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            data = {}
    except yaml.YAMLError:
        data = {}
    return data, body


def find_description_section(skill_md: str) -> tuple[dict, str]:
    """Locate a `## Description` block in the full SKILL.md text.

    Returns (description_section_dict, non_description_text). When a
    description heading is found, the heading line through the line
    before the next heading (or EOF) is removed from non_description.
    """
    m = _DESC_HEADING_RE.search(skill_md)
    if not m:
        return ({"found": False, "heading": None, "content": None}, skill_md)

    heading_start = m.start()
    heading_end = m.end()
    heading_text = m.group(2)  # the word "description", as written

    # Find the next heading after this one (any level), to bound the section.
    nxt = _ANY_HEADING_RE.search(skill_md, heading_end)
    section_end = nxt.start() if nxt else len(skill_md)

    content = skill_md[heading_end:section_end].strip("\n")
    # Strip blank lines that may surround the content but preserve internal structure.
    content_stripped = content.strip()

    # non_description: SKILL.md with the [heading_start, section_end) span removed.
    non_description = skill_md[:heading_start] + skill_md[section_end:]
    # Collapse the join point's potential extra blank lines.
    non_description = re.sub(r"\n{3,}", "\n\n", non_description)

    return (
        {"found": True, "heading": heading_text, "content": content_stripped or None},
        non_description,
    )


def collect_package_files(skill_dir: Path) -> list[dict]:
    """Walk the skill dir for additional files (excluding SKILL.md/manifest.json).

    Returns the list of {path, content, size_bytes, truncated} dicts.
    Binary files are skipped (UTF-8 decode failure with `errors="strict"`).
    """
    out: list[dict] = []
    if not skill_dir.exists():
        return out
    for p in sorted(skill_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(skill_dir).as_posix()
        if p.name in SKIP_FILES and "/" not in rel:
            continue  # top-level SKILL.md / manifest.json
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
            # Best-effort: replace bad bytes so we still capture human-readable text.
            content = chunk.decode("utf-8", errors="replace")
        out.append({
            "path": rel,
            "content": content,
            "size_bytes": size,
            "truncated": truncated,
        })
    return out


def local_canonicalize(skill_id: str, skill_dir: Path) -> dict:
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
        "skill_id": skill_id,
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


def load_challenge_set() -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with REVIEWED.open(newline="") as f:
        for r in csv.DictReader(f):
            by_id[r["skill_id"]] = r
    return by_id


def extract_canonical_records(target_ids: set[str]) -> dict[str, dict]:
    """Pull `target_ids` from the BR200 canonical JSONL in one streaming pass."""
    found: dict[str, dict] = {}
    with CANONICAL_FULL.open("rb") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = d.get("skill_id")
            if sid in target_ids and sid not in found:
                found[sid] = d
                if len(found) == len(target_ids):
                    break
    return found


def main() -> int:
    chal = load_challenge_set()
    all_ids = set(chal.keys())
    print(f"[challenge] reviewed rows: {len(chal)}")

    print("[challenge] extracting from BR200 canonical_skill_records.jsonl ...", flush=True)
    t0 = time.time()
    from_canonical = extract_canonical_records(all_ids)
    print(f"[challenge]   pulled {len(from_canonical)} records in {time.time()-t0:.1f}s")

    missing_ids = sorted(all_ids - set(from_canonical))
    print(f"[challenge] local-canonicalizing {len(missing_ids)} skills not in canonical ...", flush=True)

    local_recs: list[dict] = []
    failures: list[tuple[str, str]] = []
    for i, sid in enumerate(missing_ids, 1):
        row = chal[sid]
        # Prefer the package_dir from the reviewed CSV; fall back to SFT_ROOT/sid.
        pdir = Path(row.get("package_dir") or "") if row.get("package_dir") else SFT_ROOT / sid
        if not pdir.exists():
            pdir = SFT_ROOT / sid
        try:
            local_recs.append(local_canonicalize(sid, pdir))
        except Exception as e:
            failures.append((sid, f"{type(e).__name__}: {e}"))
        if i % 100 == 0:
            print(f"[challenge]   {i}/{len(missing_ids)}", flush=True)

    if failures:
        print(f"[challenge] local-canonicalize failures: {len(failures)}", flush=True)
        for sid, msg in failures[:5]:
            print(f"  {sid}: {msg}")

    # Merge: extracted records + locally-built records.
    by_id: dict[str, dict] = {}
    for sid, rec in from_canonical.items():
        by_id[sid] = rec
    for rec in local_recs:
        by_id[rec["skill_id"]] = rec

    # Stable output order: reviewed-CSV order so it's easy to cross-reference.
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with OUT_JSONL.open("w") as f:
        for sid in chal:
            rec = by_id.get(sid)
            if rec is None:
                continue
            f.write(json.dumps(rec) + "\n")
            n_written += 1

    missing_from_output = [sid for sid in chal if sid not in by_id]
    print(f"[challenge] wrote {n_written} records -> {OUT_JSONL}")
    print(f"[challenge] sources: from_canonical={len(from_canonical)}  local={len(local_recs)}  failures={len(failures)}  missing={len(missing_from_output)}")
    if missing_from_output:
        for sid in missing_from_output[:10]:
            print(f"  MISSING: {sid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
