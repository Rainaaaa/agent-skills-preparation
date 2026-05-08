"""Stage A — normalize canonical skill records into a versioned base dataset.

Reads `config.paths.canonical_records` (JSONL produced by the upstream
canonical-export step), applies lightweight cleaning, and writes:

    {prepared_root}/normalized/<version>/normalized_skill_records.jsonl
                                          + .parquet
                                          + manifest.json
                                          + stats.json
                                          + skill_ids.txt

**New (vs. the legacy `data_preparation/`):** if `--scan-results` is
supplied (or `filter.scan_results` is set in config.yaml, or the env var
`AGENTSKILLS_SCAN_RESULTS` is set), the upstream scanning verdicts in
`unified_results.csv` are joined per-skill and skills classified as
malicious or misaligned are dropped before normalization. The drop policy
is configurable under `filter:` in config.yaml; see `scan_filter.py`.

If no scan results are provided the filter is a no-op and the pipeline
runs over the full canonical set, just like before.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pipeline._shared import (
    LOGGER,
    ParquetBatchWriter,
    append_jsonl,
    batched,
    build_outputs_manifest,
    ensure_dir,
    ensure_version_dir,
    finalize_manifest,
    iter_jsonl,
    map_records,
    update_registry,
    write_json,
)
from pipeline.scan_filter import (
    FilterPolicy,
    FilterReport,
    build_keep_set,
    load_scan_results,
    log_report,
)
from pipeline.schema import normalize_record_task


def _resolve_scan_results_path(
    cli_path: Optional[str],
    config: Dict[str, Any],
) -> Optional[Path]:
    """Look in this order: CLI --scan-results → config.filter.scan_results
    → env var AGENTSKILLS_SCAN_RESULTS. Empty / unset → no filter."""
    import os
    if cli_path:
        return Path(cli_path)
    cfg_path = (config.get("filter") or {}).get("scan_results")
    if cfg_path:
        return Path(cfg_path)
    env_path = os.environ.get("AGENTSKILLS_SCAN_RESULTS")
    if env_path:
        return Path(env_path)
    return None


def _maybe_filter_canonical_records(
    canonical_path: Path,
    scan_results_path: Optional[Path],
    config: Dict[str, Any],
) -> "tuple[Iterator[Dict[str, Any]], Optional[FilterReport], Optional[FilterPolicy]]":
    """Yield canonical records, optionally filtered by scan verdicts.

    Two-pass design: load the scan index, then stream-filter the canonical
    JSONL. The scan index is small (~120k entries × few KB = a few MB) so
    holding it in memory is fine; the canonical JSONL can be huge so we
    don't materialize it.
    """
    if scan_results_path is None:
        LOGGER.info("[normalize] no scan-results provided; running without filter.")
        return iter_jsonl(canonical_path), None, None

    if not scan_results_path.exists():
        LOGGER.warning(
            "[normalize] scan-results path does not exist: %s — running without filter.",
            scan_results_path,
        )
        return iter_jsonl(canonical_path), None, None

    policy = FilterPolicy.from_config(config.get("filter"))
    scan_index = load_scan_results(scan_results_path)

    # First pass to collect candidate skill_ids; second pass to yield.
    # We do both via a single in-memory list because the candidate id set
    # is needed up front for build_keep_set's report counts to be correct.
    LOGGER.info("[normalize] loading canonical for filter pass: %s", canonical_path)
    candidates: List[Dict[str, Any]] = list(iter_jsonl(canonical_path))
    candidate_ids = [
        rec.get("skill_id") for rec in candidates if rec.get("skill_id")
    ]
    keep_set, report = build_keep_set(candidate_ids, scan_index, policy)
    log_report(report, policy)

    def _filtered() -> Iterator[Dict[str, Any]]:
        for rec in candidates:
            sid = rec.get("skill_id")
            if sid and sid in keep_set:
                yield rec

    return _filtered(), report, policy


def run_normalize(
    config: Dict[str, Any],
    normalized_version: str,
    workers: int,
    batch_size: int,
    skip_parquet: bool,
    *,
    scan_results: Optional[str] = None,
) -> None:
    canonical_path = Path(config["paths"]["canonical_records"])
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical records file not found: {canonical_path}")

    output_dir = Path(config["paths"]["normalized_root"]) / normalized_version
    ensure_version_dir(output_dir)

    jsonl_path = output_dir / "normalized_skill_records.jsonl"
    parquet_path = output_dir / "normalized_skill_records.parquet"
    skill_ids_path = output_dir / "skill_ids.txt"
    parquet_writer = ParquetBatchWriter(
        parquet_path,
        compression=config["runtime"]["parquet_compression"],
        enabled=not skip_parquet,
    )

    stats: Dict[str, Any] = {
        "missing_name_count": 0,
        "missing_description_count": 0,
        "metadata_fallback_count": 0,
        "explicit_description_heading_count": 0,
        "total_package_files": 0,
        "total_text_chars": 0,
    }
    processed_records = 0
    skill_ids: List[str] = []

    scan_results_path = _resolve_scan_results_path(scan_results, config)
    record_iter, filter_report, filter_policy = _maybe_filter_canonical_records(
        canonical_path, scan_results_path, config
    )

    LOGGER.info("Normalizing canonical records into %s", output_dir)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for batch in batched(record_iter, batch_size):
            normalized_rows = map_records(
                batch,
                normalize_record_task,
                config["runtime"],
                workers,
                executor=executor,
            )
            append_jsonl(jsonl_path, normalized_rows)
            parquet_writer.write_rows(normalized_rows)
            skill_ids.extend(row["skill_id"] for row in normalized_rows)
            processed_records += len(normalized_rows)

            for row in normalized_rows:
                stats["missing_name_count"] += int(not row["name"])
                stats["missing_description_count"] += int(not row["description_text"])
                stats["metadata_fallback_count"] += int(row["used_metadata_description_fallback"])
                stats["explicit_description_heading_count"] += int(row["has_explicit_description_heading"])
                stats["total_package_files"] += int(row["package_files_count"])
                stats["total_text_chars"] += int(row["text_char_count"])

            if processed_records % max(batch_size, 1000) == 0:
                LOGGER.info("Normalized %d records", processed_records)
    finally:
        if executor is not None:
            executor.shutdown()

    parquet_writer.close()
    ensure_dir(skill_ids_path.parent)
    with skill_ids_path.open("w", encoding="utf-8") as handle:
        for skill_id in skill_ids:
            handle.write(skill_id)
            handle.write("\n")

    if processed_records:
        stats["avg_package_files_per_skill"] = stats["total_package_files"] / processed_records
        stats["avg_text_chars_per_skill"] = stats["total_text_chars"] / processed_records
    else:
        stats["avg_package_files_per_skill"] = 0.0
        stats["avg_text_chars_per_skill"] = 0.0

    if filter_report is not None and filter_policy is not None:
        # Persist the audit trail. The excluded list is the seed corpus
        # for downstream real-world evaluation (human review will refine
        # it but not extend it beyond what was flagged here).
        excluded_path = output_dir / "excluded_skill_ids.txt"
        excluded_path.write_text(
            "\n".join(filter_report.excluded_skill_ids) + ("\n" if filter_report.excluded_skill_ids else ""),
            encoding="utf-8",
        )
        stats["scan_filter"] = {
            "enabled": True,
            "scan_results_path": str(scan_results_path) if scan_results_path else "",
            "excluded_skill_ids_path": str(excluded_path),
            "policy": {
                "exclude_overall_classes":   sorted(filter_policy.exclude_overall_classes),
                "exclude_alignment_classes": sorted(filter_policy.exclude_alignment_classes),
                "unscanned_action": filter_policy.unscanned_action,
            },
            "report": filter_report.to_dict(),
        }
    else:
        stats["scan_filter"] = {"enabled": False}

    write_json(output_dir / "stats.json", stats)
    manifest = build_outputs_manifest(
        dataset_name="normalized",
        phase_name="normalize",
        input_path=str(canonical_path),
        output_dir=output_dir,
        normalized_version_used=normalized_version,
        output_version=normalized_version,
        num_input_records=processed_records,
        num_output_samples=processed_records,
        stats=stats,
        config=config,
    )
    finalize_manifest(manifest, output_dir, config, "normalized", normalized_version)
    update_registry(
        Path(config["paths"]["normalized_root"]) / "registry.json",
        "normalized",
        normalized_version,
        manifest,
    )
    LOGGER.info("Normalization complete: %d records", processed_records)
