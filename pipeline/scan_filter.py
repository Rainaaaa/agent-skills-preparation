"""Filter canonical skill records against a verdict CSV.

The verdict CSV can come from two sources, in priority order:

  1. **Human-reviewed**  — a hand-curated CSV produced by reviewers.
     This is what production runs should consume. The path is
     configurable so the user can swap auto-scan for human-reviewed
     output the moment review is done, without editing any code.
  2. **Auto-scan**       — `outputs/unified_results.csv` produced by
     agent-skills-scanning's `pipeline.aggregate_results`. Used as a
     placeholder while human review is still in progress.

Either source must use the same column schema:

    skill_id, overall_class, alignment_class    (and optionally other columns)

  - `overall_class`   ∈ {SAFE, SUSPICIOUS, MALICIOUS, ERROR}   (maliciousness pillar)
  - `alignment_class` ∈ {ALIGNED, MISALIGNED, ERROR}            (alignment axis, binary)

Default drop policy (configurable in `config.yaml > filter`):

  drop if `overall_class`   ∈ {MALICIOUS, SUSPICIOUS}
  drop if `alignment_class` ∈ {MISALIGNED}
  keep skills not present in the verdict CSV, with a counter logged

Embedded inside `pipeline.normalize.run_normalize` — the user only has
to point `--scan-results` at the file, or set the env var, or fill in
`filter.scan_results` in config.yaml.

The skills that survive this filter are the ones used to **synthesize
training/val/test data** (T1/T2/T3 negatives in the contrastive layer).
The skills that DON'T survive — those flagged misaligned or malicious —
are kept in a separate audit trail (the path lives in stats.json) so
they can serve as **real-world evaluation data** for downstream tasks.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional, Set

from pipeline._shared import LOGGER


DEFAULT_EXCLUDE_OVERALL = frozenset({"MALICIOUS", "SUSPICIOUS"})
DEFAULT_EXCLUDE_ALIGNMENT = frozenset({"MISALIGNED"})


@dataclass
class FilterPolicy:
    """How to interpret scanning verdicts.

    Two axes are independent: a skill is dropped if EITHER condition fires.
    Set `unscanned_action` to 'drop' to be strict about coverage, 'keep'
    to be charitable (default).
    """

    exclude_overall_classes:   FrozenSet[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_OVERALL)
    exclude_alignment_classes: FrozenSet[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_ALIGNMENT)
    unscanned_action: str = "keep"  # "keep" or "drop"

    @classmethod
    def from_config(cls, filter_cfg: Optional[Dict]) -> "FilterPolicy":
        cfg = filter_cfg or {}
        ovr = cfg.get("exclude_overall_classes")
        aln = cfg.get("exclude_alignment_classes")
        unscanned = (cfg.get("unscanned_action") or "keep").lower()
        return cls(
            exclude_overall_classes=frozenset(
                (s or "").upper() for s in (ovr if ovr is not None else DEFAULT_EXCLUDE_OVERALL)
            ),
            exclude_alignment_classes=frozenset(
                (s or "").upper() for s in (aln if aln is not None else DEFAULT_EXCLUDE_ALIGNMENT)
            ),
            unscanned_action=unscanned if unscanned in ("keep", "drop") else "keep",
        )


@dataclass
class FilterReport:
    total_in_scan_results: int = 0
    excluded_by_overall: int = 0
    excluded_by_alignment: int = 0
    excluded_by_either: int = 0
    kept: int = 0
    unscanned_kept: int = 0
    unscanned_dropped: int = 0
    # The list of dropped skill_ids — useful as a downstream-evaluation
    # corpus (real-world misaligned / malicious cases) once human review
    # is complete. Kept separate from the in-memory keep set so callers
    # can persist it independently.
    excluded_skill_ids: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_in_scan_results":  self.total_in_scan_results,
            "excluded_by_overall":    self.excluded_by_overall,
            "excluded_by_alignment":  self.excluded_by_alignment,
            "excluded_by_either":     self.excluded_by_either,
            "kept":                   self.kept,
            "unscanned_kept":         self.unscanned_kept,
            "unscanned_dropped":      self.unscanned_dropped,
            # Counted separately to keep stats.json scannable; the full
            # list is written next to it as `excluded_skill_ids.txt`.
            "excluded_skill_ids_count": len(self.excluded_skill_ids),
        }


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().upper()


def load_scan_results(unified_csv: Path) -> Dict[str, Dict[str, str]]:
    """Index unified_results.csv by skill_id.

    The CSV's column set is fixed by agent-skills-scanning's aggregator;
    we read just the columns we need and ignore the rest, so future
    scanner additions don't break us.
    """
    if not unified_csv.exists():
        raise FileNotFoundError(f"scan results not found: {unified_csv}")

    out: Dict[str, Dict[str, str]] = {}
    with unified_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("skill_id") or "").strip()
            if not sid:
                continue
            out[sid] = {
                "overall_class":     _norm(row.get("overall_class")),
                "alignment_class":   _norm(row.get("alignment_class")),
                "static_rule_class": _norm(row.get("static_rule_class")),
                "llm_filter_class":  _norm(row.get("llm_filter_class")),
                "behavioral_class":  _norm(row.get("behavioral_class")),
            }
    return out


def build_keep_set(
    candidate_skill_ids: Iterable[str],
    scan_index: Dict[str, Dict[str, str]],
    policy: FilterPolicy,
) -> "tuple[Set[str], FilterReport]":
    """Apply the policy to the candidate skill_ids; return (keep_set, report).

    The report carries the full list of excluded skill_ids so a caller
    can persist them as a downstream-evaluation corpus.
    """
    report = FilterReport(total_in_scan_results=len(scan_index))
    keep: Set[str] = set()

    for sid in candidate_skill_ids:
        verdicts = scan_index.get(sid)
        if verdicts is None:
            if policy.unscanned_action == "drop":
                report.unscanned_dropped += 1
            else:
                report.unscanned_kept += 1
                keep.add(sid)
            continue

        excl_ovr = verdicts["overall_class"] in policy.exclude_overall_classes
        excl_aln = verdicts["alignment_class"] in policy.exclude_alignment_classes

        if excl_ovr:
            report.excluded_by_overall += 1
        if excl_aln:
            report.excluded_by_alignment += 1
        if excl_ovr or excl_aln:
            report.excluded_by_either += 1
            report.excluded_skill_ids.append(sid)
            continue

        keep.add(sid)
        report.kept += 1

    return keep, report


def log_report(report: FilterReport, policy: FilterPolicy) -> None:
    LOGGER.info(
        "[scan_filter] policy: drop_overall=%s drop_alignment=%s unscanned_action=%s",
        sorted(policy.exclude_overall_classes),
        sorted(policy.exclude_alignment_classes),
        policy.unscanned_action,
    )
    LOGGER.info(
        "[scan_filter] excluded_by_overall=%d  excluded_by_alignment=%d  "
        "excluded_by_either=%d  kept=%d  unscanned_kept=%d  unscanned_dropped=%d",
        report.excluded_by_overall,
        report.excluded_by_alignment,
        report.excluded_by_either,
        report.kept,
        report.unscanned_kept,
        report.unscanned_dropped,
    )
