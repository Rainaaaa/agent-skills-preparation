"""Stage B — build the Phase 1 (Structured CLM CPT) dataset.

Reads `normalized/<version>/normalized_skill_records.jsonl` and emits the
two-stage layout expected by the trainer:

    full_cpt/<output_version>/stage1/{train,val,test,unseen}.parquet
    full_cpt/<output_version>/stage2/{train,val,test,unseen}.parquet

The same skill_id always lands in the same `(pool, split)` across both
stages so we never get cross-stage leakage. Pool ('cpt' / 'unseen') and
split ('train' / 'val' / 'test') use independent seeds; see
`pipeline._shared.assign_phase1_pool_and_split`.

Ported verbatim from the legacy `data_preparation/src/pipeline.py`
(`run_phase1`); only the imports changed for the new package layout.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from pipeline._shared import (
    LOGGER,
    ParquetBatchWriter,
    assign_phase1_pool_and_split,
    batched,
    build_outputs_manifest,
    ensure_dir,
    ensure_version_dir,
    finalize_manifest,
    iter_jsonl,
    map_records,
    require_normalized_input,
    update_registry,
    write_json,
)
from pipeline.schema import build_phase1_samples_task


def run_phase1(
    config: Dict[str, Any],
    normalized_version: str,
    output_version: str,
    workers: int,
    batch_size: int,
    skip_parquet: bool,
) -> None:
    normalized_path = require_normalized_input(config, normalized_version)
    output_dir = Path(config["paths"]["full_cpt_root"]) / output_version
    ensure_version_dir(output_dir)

    stages = ("stage1", "stage2")
    splits = ("train", "val", "test", "unseen")
    for stage in stages:
        ensure_dir(output_dir / stage)

    writers: Dict[Tuple[str, str], ParquetBatchWriter] = {
        (stage, split): ParquetBatchWriter(
            output_dir / stage / f"{split}.parquet",
            compression=config["runtime"]["parquet_compression"],
            enabled=not skip_parquet,
        )
        for stage in stages
        for split in splits
    }

    counts: Dict[Tuple[str, str], int] = {(stage, split): 0 for stage in stages for split in splits}
    section_keys = (
        "metadata_chars", "instruction_chars", "resource_chars", "total_chars",
        "metadata_tokens_est", "instruction_tokens_est", "resource_tokens_est", "total_tokens_est",
    )
    sums: Dict[Tuple[str, str], Dict[str, float]] = {
        (stage, split): {k: 0.0 for k in section_keys} for stage in stages for split in splits
    }
    maxes: Dict[Tuple[str, str], Dict[str, int]] = {
        (stage, split): {k: 0 for k in section_keys} for stage in stages for split in splits
    }
    skills_seen = 0
    skills_emitted_any = 0
    skills_emitted_stage1 = 0
    skills_emitted_stage2 = 0
    skills_dropped_no_fit = 0
    skills_dropped_oversize = 0

    phase1_config = dict(config["runtime"])
    phase1_config.update(config.get("phase1", {}))
    LOGGER.info("Building Phase 1 dataset into %s", output_dir)

    # Pre-filter monster skills before they reach worker IPC. A skill whose
    # combined package payload is many MB cannot fit any stage2 budget anyway,
    # but shipping that 20+ MB blob through 64 worker pipes triggers SIGBUS on
    # /dev/shm-constrained nodes. We use 4x the stage2 char budget as a
    # generous upstream filter.
    stage2_max_tokens = int(phase1_config.get("stage2_max_tokens", 10240))
    parent_filter_max_chars = max(stage2_max_tokens * 16, 256_000)

    def _passes_parent_filter(rec: Dict[str, Any]) -> bool:
        nonlocal skills_dropped_oversize
        approx = max(
            int(rec.get("text_char_count") or 0),
            int(rec.get("package_files_total_bytes") or 0),
        )
        if approx > parent_filter_max_chars:
            skills_dropped_oversize += 1
            return False
        return True

    def _filtered_records() -> Iterator[Dict[str, Any]]:
        for rec in iter_jsonl(normalized_path):
            if _passes_parent_filter(rec):
                yield rec

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for batch in batched(_filtered_records(), batch_size):
            sample_groups = map_records(
                batch,
                build_phase1_samples_task,
                phase1_config,
                workers,
                executor=executor,
            )
            buffered: Dict[Tuple[str, str], List[Dict[str, Any]]] = {key: [] for key in writers}
            for record, samples in zip(batch, sample_groups):
                skills_seen += 1
                if not samples:
                    skills_dropped_no_fit += 1
                    continue
                pool, split = assign_phase1_pool_and_split(record["skill_id"], phase1_config)
                emitted_stages = set()
                for sample in samples:
                    stage = sample["stage"]
                    sample["pool"] = pool
                    sample["split"] = split
                    sample["normalized_version"] = normalized_version
                    sample["output_version"] = output_version
                    key = (stage, split)
                    buffered[key].append(sample)
                    counts[key] += 1
                    for k in section_keys:
                        v = int(sample.get(k, 0) or 0)
                        sums[key][k] += v
                        if v > maxes[key][k]:
                            maxes[key][k] = v
                    emitted_stages.add(stage)
                if "stage1" in emitted_stages:
                    skills_emitted_stage1 += 1
                if "stage2" in emitted_stages:
                    skills_emitted_stage2 += 1
                if emitted_stages:
                    skills_emitted_any += 1
                else:
                    skills_dropped_no_fit += 1
            for key, rows in buffered.items():
                if rows:
                    writers[key].write_rows(rows)
    finally:
        if executor is not None:
            executor.shutdown()

    for writer in writers.values():
        writer.close()

    def _means(stage: str, split: str) -> Dict[str, float]:
        c = counts[(stage, split)]
        if c == 0:
            return {k: 0.0 for k in section_keys}
        return {k: sums[(stage, split)][k] / c for k in section_keys}

    stats = {
        "skills_seen": skills_seen,
        "skills_emitted_any": skills_emitted_any,
        "skills_dropped_no_fit": skills_dropped_no_fit,
        "skills_dropped_oversize": skills_dropped_oversize,
        "parent_filter_max_chars": parent_filter_max_chars,
        "skills_emitted_stage1": skills_emitted_stage1,
        "skills_emitted_stage2": skills_emitted_stage2,
        "rows_per_stage_split": {
            stage: {split: counts[(stage, split)] for split in splits} for stage in stages
        },
        "section_means": {
            stage: {split: _means(stage, split) for split in splits} for stage in stages
        },
        "section_maxes": {
            stage: {split: maxes[(stage, split)] for split in splits} for stage in stages
        },
        "phase1_config": {k: phase1_config.get(k) for k in (
            "stage1_max_tokens", "stage2_max_tokens",
            "cpt_unseen_fraction", "cpt_train_fraction", "cpt_val_fraction", "cpt_test_fraction",
            "pool_seed", "split_seed",
            "max_metadata_chars", "max_instruction_chars",
            "max_resource_per_file_chars", "max_resource_total_chars",
        )},
    }
    write_json(output_dir / "stats.json", stats)
    total_rows = sum(counts.values())
    manifest = build_outputs_manifest(
        dataset_name="full_cpt",
        phase_name="build_phase1",
        input_path=str(normalized_path),
        output_dir=output_dir,
        normalized_version_used=normalized_version,
        output_version=output_version,
        num_input_records=skills_seen,
        num_output_samples=total_rows,
        stats=stats,
        config=config,
    )
    finalize_manifest(manifest, output_dir, config, "full_cpt", output_version)
    update_registry(
        Path(config["paths"]["full_cpt_root"]) / "registry.json",
        "full_cpt",
        output_version,
        manifest,
    )
    LOGGER.info(
        "Phase 1 complete: %d skills seen, %d rows written (stage1=%d, stage2=%d)",
        skills_seen, total_rows, skills_emitted_stage1, skills_emitted_stage2,
    )
