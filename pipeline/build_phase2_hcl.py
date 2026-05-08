"""Phase 2 (PD-HMCL) hierarchical-contrastive negatives builder — Step A.

Builds T1 (internal layer permutation), T2 (external layer swap), and T3b
(span-swap from a same-pool donor) negatives deterministically from the Phase 1
v2 stage outputs. T3a (LLM paraphrase-perturbation) is generated separately by
``enrich_phase2_t3a.py`` and merged in afterward.

Input:
  /N/project/.../full_cpt/<full_cpt_version>/stage{1,2}/{train,val,test,unseen}.parquet

Output:
  /N/project/.../stage_cpt/<output_version>/stage{1,2}/{train,val,test,unseen}.parquet
  /N/project/.../stage_cpt/<output_version>/stats.json
  /N/project/.../stage_cpt/<output_version>/manifest.json

Per-anchor negative counts (defaults from config.json phase2_hcl):
  Stage 1 (M,I; canonical M->I):
    T1 : 1                                         (only non-canonical: I,M)
    T2 : 2 patterns x K=2 donors  = 4
    T3b: 1 layer (I) x K_span=2 donors = 2
  Stage 2 (M,I,R; canonical M->I->R):
    T1 : 5                                         (5 non-canonical orderings)
    T2 : 9 patterns x K=2 donors  = 18
    T3b: 2 layers (I,R) x K_span=2 donors = 4

T2 patterns (stage 2):
  3 single-layer swaps     :  swap_M, swap_I, swap_R
  3 double-layer swaps (B) :  swap_MI_B, swap_MR_B, swap_IR_B    (both donor B)
  3 double-layer swaps (BC):  swap_MI_BC, swap_MR_BC, swap_IR_BC (B and C distinct)

Donor sampling is deterministic per (anchor, pattern, k) using SHA1 over
(seed, anchor_skill_id, pattern_id, k).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

EST_CHARS_PER_TOKEN = 4.0


def _est_tokens(text: str) -> int:
    if not text:
        return 0
    n = len(text)
    return int((n + EST_CHARS_PER_TOKEN - 1) // EST_CHARS_PER_TOKEN)


def _digest(*parts: Any) -> int:
    """Stable 64-bit-ish digest of the parts. Used for deterministic sampling."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


# Section tag patterns (must match Phase 1's schema.py).
_M_INNER = re.compile(r"^<METADATA>\n(.*)\n</METADATA>$", re.DOTALL)
_I_INNER = re.compile(r"^<INSTRUCTION>\n(.*)\n</INSTRUCTION>$", re.DOTALL)
_R_INNER = re.compile(r"^<RESOURCE>\n(.*)\n</RESOURCE>$", re.DOTALL)


def _inner(layer_text: str, layer: str) -> str:
    """Extract the inner content of a tagged layer block."""
    if not layer_text:
        return ""
    rgx = {"M": _M_INNER, "I": _I_INNER, "R": _R_INNER}[layer]
    m = rgx.match(layer_text)
    return m.group(1) if m else layer_text


def _wrap(layer: str, inner: str) -> str:
    """Wrap inner content with the proper layer tags."""
    if not inner:
        return ""
    if layer == "M":
        return f"<METADATA>\n{inner}\n</METADATA>"
    if layer == "I":
        return f"<INSTRUCTION>\n{inner}\n</INSTRUCTION>"
    if layer == "R":
        return f"<RESOURCE>\n{inner}\n</RESOURCE>"
    raise ValueError(f"unknown layer {layer}")


# ---------------------------------------------------------------------------
# Pattern enumerations
# ---------------------------------------------------------------------------

# Stage 1: only M, I. Canonical = (M, I).
STAGE1_T1_ORDERINGS: List[Tuple[str, ...]] = [
    ("I", "M"),
]

# (pattern_id, [(layer, source), ...]) — source is "A" (anchor) or "B"/"C" (donor).
STAGE1_T2_PATTERNS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("swap_M",  [("M", "B"), ("I", "A")]),
    ("swap_I",  [("M", "A"), ("I", "B")]),
]

STAGE1_T3B_LAYERS: List[str] = ["I"]


# Stage 2: M, I, R. Canonical = (M, I, R).
STAGE2_T1_ORDERINGS: List[Tuple[str, ...]] = [
    ("M", "R", "I"),
    ("I", "M", "R"),
    ("I", "R", "M"),
    ("R", "M", "I"),
    ("R", "I", "M"),
]

STAGE2_T2_PATTERNS: List[Tuple[str, List[Tuple[str, str]]]] = [
    # single-layer swaps (one layer from B, others from A)
    ("swap_M",      [("M", "B"), ("I", "A"), ("R", "A")]),
    ("swap_I",      [("M", "A"), ("I", "B"), ("R", "A")]),
    ("swap_R",      [("M", "A"), ("I", "A"), ("R", "B")]),
    # double-layer swaps from same donor B
    ("swap_MI_B",   [("M", "B"), ("I", "B"), ("R", "A")]),
    ("swap_MR_B",   [("M", "B"), ("I", "A"), ("R", "B")]),
    ("swap_IR_B",   [("M", "A"), ("I", "B"), ("R", "B")]),
    # double-layer swaps from two distinct donors B, C
    ("swap_MI_BC",  [("M", "B"), ("I", "C"), ("R", "A")]),
    ("swap_MR_BC",  [("M", "B"), ("I", "A"), ("R", "C")]),
    ("swap_IR_BC",  [("M", "A"), ("I", "B"), ("R", "C")]),
]

STAGE2_T3B_LAYERS: List[str] = ["I", "R"]


def _stage_layer_order(stage: str) -> List[str]:
    return ["M", "I"] if stage == "stage1" else ["M", "I", "R"]


def _stage_t1_orderings(stage: str) -> List[Tuple[str, ...]]:
    return STAGE1_T1_ORDERINGS if stage == "stage1" else STAGE2_T1_ORDERINGS


def _stage_t2_patterns(stage: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
    return STAGE1_T2_PATTERNS if stage == "stage1" else STAGE2_T2_PATTERNS


def _stage_t3b_layers(stage: str) -> List[str]:
    return STAGE1_T3B_LAYERS if stage == "stage1" else STAGE2_T3B_LAYERS


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def _compose_layers(parts: List[str]) -> str:
    """Join non-empty layer-block strings with a single newline (matches Phase 1)."""
    return "\n".join(p for p in parts if p)


def _compose_t1(anchor: Dict[str, str], ordering: Tuple[str, ...]) -> str:
    field_map = {"M": "metadata_text", "I": "instruction_text", "R": "resource_text"}
    return _compose_layers([anchor[field_map[layer]] for layer in ordering])


def _compose_t2(
    anchor: Dict[str, str],
    pattern: List[Tuple[str, str]],
    sources: Dict[str, Dict[str, str]],
) -> str:
    """Compose a T2 negative.

    ``pattern`` lists (layer, source_letter) in canonical order (canonical is
    the order layers appear in the pattern definition). ``sources`` maps each
    source letter ('A','B','C') to a donor row.
    """
    field_map = {"M": "metadata_text", "I": "instruction_text", "R": "resource_text"}
    parts = []
    for layer, src in pattern:
        donor = sources[src]
        parts.append(donor[field_map[layer]])
    return _compose_layers(parts)


def _compose_swap_layer(anchor: Dict[str, str], layer: str, new_layer_text: str, stage: str) -> str:
    """Replace one layer in the anchor with new_layer_text, keeping canonical order."""
    field_map = {"M": "metadata_text", "I": "instruction_text", "R": "resource_text"}
    parts = []
    for ll in _stage_layer_order(stage):
        if ll == layer:
            parts.append(new_layer_text)
        else:
            parts.append(anchor[field_map[ll]])
    return _compose_layers(parts)


# ---------------------------------------------------------------------------
# T3b — span swap
# ---------------------------------------------------------------------------

def _t3b_span_swap_inner(
    anchor_inner: str,
    donor_inner: str,
    fraction: float,
    cap_chars: int,
    min_chars: int,
    rng: random.Random,
) -> Optional[str]:
    """Return anchor_inner with a contiguous span replaced by a same-size span
    from donor_inner. None if not feasible (donor too short or anchor too short).
    """
    n_a = len(anchor_inner)
    n_d = len(donor_inner)
    if n_a == 0 or n_d == 0:
        return None
    span_len = max(min_chars, int(n_a * fraction))
    span_len = min(span_len, cap_chars, n_a, n_d)
    if span_len < min_chars or span_len <= 0:
        return None
    # pick start positions
    a_start = rng.randint(0, n_a - span_len)
    d_start = rng.randint(0, n_d - span_len)
    return anchor_inner[:a_start] + donor_inner[d_start:d_start + span_len] + anchor_inner[a_start + span_len:]


# ---------------------------------------------------------------------------
# Donor pool & sampling
# ---------------------------------------------------------------------------

class DonorPool:
    """All same-(stage, pool) skills, indexed for deterministic donor sampling."""

    def __init__(self, rows: List[Dict[str, Any]]):
        # rows: [{skill_id, metadata_text, instruction_text, resource_text}, ...]
        self.rows = rows
        self.skill_id_to_idx = {r["skill_id"]: i for i, r in enumerate(rows)}
        self.size = len(rows)

    def sample_donors(
        self,
        anchor_skill_id: str,
        pattern_id: str,
        k: int,
        n_needed: int,
        seed: int,
        require_distinct_skill: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return n_needed donors deterministically per (anchor, pattern, k).

        Deterministic walk over a hash-shuffled index, skipping the anchor
        and (optionally) any already-picked donor when n_needed > 1.
        """
        if self.size == 0:
            return []
        order_seed = _digest(seed, anchor_skill_id, pattern_id, k)
        # Walk in a hash-determined order: start position + step that's coprime-ish.
        start = order_seed % self.size
        step = (order_seed >> 32) | 1  # ensure odd; will iterate every position eventually for prime sizes
        if step % self.size == 0:
            step = 1
        picked: List[Dict[str, Any]] = []
        seen_ids: set = set()
        idx = start
        for _ in range(self.size + 1):
            row = self.rows[idx % self.size]
            sid = row["skill_id"]
            if sid != anchor_skill_id and (not require_distinct_skill or sid not in seen_ids):
                picked.append(row)
                seen_ids.add(sid)
                if len(picked) >= n_needed:
                    return picked
            idx += step
        return picked


# ---------------------------------------------------------------------------
# Per-anchor negative generation
# ---------------------------------------------------------------------------

def _make_row(
    *,
    anchor: Dict[str, Any],
    text: str,
    label: str,
    sub_strategy: str,
    target_layer: str,
    source_skill_ids: List[str],
    stage: str,
    output_version: str,
) -> Dict[str, Any]:
    sample_id = hashlib.sha1(
        f"{label}|{sub_strategy}|{anchor['skill_id']}|{text[:128]}".encode("utf-8")
    ).hexdigest()
    return {
        "sample_id": sample_id,
        "anchor_skill_id": anchor["skill_id"],
        "label": label,
        "sub_strategy": sub_strategy,
        "target_layer": target_layer,
        "source_skill_ids": source_skill_ids,
        "text": text,
        "text_chars": len(text),
        "text_tokens_est": _est_tokens(text),
        "pool": anchor["pool"],
        "split": anchor["split"],
        "stage": stage,
        "normalized_version": anchor.get("normalized_version", ""),
        "phase1_output_version": anchor.get("output_version", ""),
        "output_version": output_version,
    }


def build_negatives_for_anchor(
    anchor: Dict[str, Any],
    donors: DonorPool,
    *,
    stage: str,
    cfg: Dict[str, Any],
    output_version: str,
    rng_seed: int,
) -> List[Dict[str, Any]]:
    """Emit positive + T1/T2/T3b negative rows for one anchor."""
    rows: List[Dict[str, Any]] = []
    max_tokens = int(cfg["max_tokens"])
    K = int(cfg["k_donors"])
    K_span = int(cfg["k_span_donors"])
    span_frac = float(cfg["t3b_span_fraction"])
    span_cap = int(cfg["t3b_span_cap_chars"])
    span_min = int(cfg["t3b_span_min_chars"])

    def _emit(text: str, label: str, sub: str, target_layer: str, sources: List[str]) -> None:
        if not text or _est_tokens(text) > max_tokens:
            return
        rows.append(
            _make_row(
                anchor=anchor, text=text, label=label, sub_strategy=sub,
                target_layer=target_layer, source_skill_ids=sources,
                stage=stage, output_version=output_version,
            )
        )

    # Positive (canonical order = the original Phase 1 row's `text` field).
    rows.append(
        _make_row(
            anchor=anchor, text=anchor["text"], label="positive",
            sub_strategy="canonical", target_layer="",
            source_skill_ids=[anchor["skill_id"]],
            stage=stage, output_version=output_version,
        )
    )

    # T1 — internal permutation
    for ordering in _stage_t1_orderings(stage):
        text = _compose_t1(anchor, ordering)
        sub = f"order={'>'.join(ordering)}"
        _emit(text, "T1", sub, "", [anchor["skill_id"]])

    # T2 — external swap, K donors per pattern
    for pattern_id, pattern in _stage_t2_patterns(stage):
        # Determine how many donors needed: 1 (if only B used) or 2 (if B and C).
        n_distinct = 1 if not any(src == "C" for _, src in pattern) else 2
        for k in range(K):
            donors_picked = donors.sample_donors(
                anchor["skill_id"], pattern_id, k, n_distinct, rng_seed,
                require_distinct_skill=True,
            )
            if len(donors_picked) < n_distinct:
                continue
            sources = {"A": anchor, "B": donors_picked[0]}
            if n_distinct >= 2:
                sources["C"] = donors_picked[1]
            text = _compose_t2(anchor, pattern, sources)
            target_layers = sorted({layer for layer, src in pattern if src in ("B", "C")})
            sub = f"{pattern_id}|k={k}|donors={','.join(d['skill_id'] for d in donors_picked)}"
            _emit(
                text, "T2", sub, "+".join(target_layers),
                [anchor["skill_id"]] + [d["skill_id"] for d in donors_picked],
            )

    # T3b — span swap on configured layers, K_span donors per layer
    field_map = {"M": "metadata_text", "I": "instruction_text", "R": "resource_text"}
    for layer in _stage_t3b_layers(stage):
        anchor_layer_full = anchor[field_map[layer]]
        if not anchor_layer_full:
            continue
        anchor_inner = _inner(anchor_layer_full, layer)
        for k in range(K_span):
            donors_picked = donors.sample_donors(
                anchor["skill_id"], f"t3b_{layer}", k, 1, rng_seed,
                require_distinct_skill=True,
            )
            if not donors_picked:
                continue
            donor = donors_picked[0]
            donor_inner = _inner(donor[field_map[layer]], layer)
            # Deterministic span position via per-(anchor,layer,k) seed.
            local_rng = random.Random(_digest(rng_seed, "t3b_span", anchor["skill_id"], layer, k))
            spliced_inner = _t3b_span_swap_inner(
                anchor_inner, donor_inner,
                fraction=span_frac, cap_chars=span_cap, min_chars=span_min, rng=local_rng,
            )
            if spliced_inner is None:
                continue
            new_layer_text = _wrap(layer, spliced_inner)
            text = _compose_swap_layer(anchor, layer, new_layer_text, stage)
            sub = f"layer={layer}|k={k}|donor={donor['skill_id']}|frac={span_frac}"
            _emit(text, "T3b", sub, layer, [anchor["skill_id"], donor["skill_id"]])

    return rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _read_phase1_stage(stage_dir: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore
    rows: List[Dict[str, Any]] = []
    for split in ("train", "val", "test", "unseen"):
        path = stage_dir / f"{split}.parquet"
        if not path.exists():
            LOGGER.warning("Phase 1 input missing: %s", path)
            continue
        t = pq.read_table(path, columns=[
            "sample_id", "skill_id", "stage", "text",
            "metadata_text", "instruction_text", "resource_text",
            "pool", "split", "normalized_version", "output_version",
        ])
        for row in t.to_pylist():
            rows.append(row)
    return rows


def _build_donor_pools(rows: List[Dict[str, Any]]) -> Dict[str, DonorPool]:
    """Bucket rows by pool and build a DonorPool per pool."""
    by_pool: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_pool.setdefault(r["pool"], []).append({
            "skill_id": r["skill_id"],
            "metadata_text": r["metadata_text"],
            "instruction_text": r["instruction_text"],
            "resource_text": r["resource_text"],
        })
    return {pool: DonorPool(rs) for pool, rs in by_pool.items()}


def run_phase2_hcl(
    config: Dict[str, Any],
    full_cpt_version: str,
    output_version: str,
    skip_parquet: bool,
) -> None:
    """Build Phase 2 HCL stage_cpt dataset (T1, T2, T3b only)."""
    full_cpt_root = Path(config["paths"]["full_cpt_root"]) / full_cpt_version
    output_root = Path(config["paths"]["pl_hcl_root"]) / output_version
    if output_root.exists():
        raise FileExistsError(f"Target version already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    phase2_cfg = dict(config.get("runtime", {}))
    phase2_cfg.update(config.get("phase2_hcl", {}))
    rng_seed = int(phase2_cfg.get("seed", 20260601))

    stage_max_tokens = {
        "stage1": int(phase2_cfg.get("stage1_max_tokens", 4096)),
        "stage2": int(phase2_cfg.get("stage2_max_tokens", 10240)),
    }

    overall_stats: Dict[str, Any] = {"per_stage": {}}

    # Lazy import here so that --skip-parquet smoke tests don't need pyarrow.
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    for stage in ("stage1", "stage2"):
        stage_dir_in = full_cpt_root / stage
        if not stage_dir_in.exists():
            LOGGER.warning("Skipping %s: no input at %s", stage, stage_dir_in)
            continue
        stage_dir_out = output_root / stage
        stage_dir_out.mkdir(parents=True, exist_ok=True)

        LOGGER.info("[%s] reading Phase 1 inputs from %s", stage, stage_dir_in)
        anchors = _read_phase1_stage(stage_dir_in)
        LOGGER.info("[%s] %d anchor rows loaded", stage, len(anchors))

        donor_pools = _build_donor_pools(anchors)
        for pool_name, pool in donor_pools.items():
            LOGGER.info("[%s] donor pool %s: %d skills", stage, pool_name, pool.size)

        per_stage_cfg = {
            "max_tokens": stage_max_tokens[stage],
            "k_donors": int(phase2_cfg.get("k_donors", 2)),
            "k_span_donors": int(phase2_cfg.get("k_span_donors", 2)),
            "t3b_span_fraction": float(phase2_cfg.get("t3b_span_fraction", 0.15)),
            "t3b_span_cap_chars": int(phase2_cfg.get("t3b_span_cap_chars", 1500)),
            "t3b_span_min_chars": int(phase2_cfg.get("t3b_span_min_chars", 200)),
        }

        # Per-(split, label) writers — matches the t3a/t3c convention so each
        # negative type lives in its own parquet for easy balanced sampling:
        #   <split>.parquet         — positives only (canonical anchors)
        #   <split>_t1.parquet      — T1 negatives
        #   <split>_t2.parquet      — T2 negatives
        #   <split>_t3b.parquet     — T3b negatives
        # T3a/T3c are produced by their own scripts and live in <split>_t3a/c.parquet.
        SPLITS = ("train", "val", "test", "unseen")
        LABELS = ("positive", "T1", "T2", "T3b")

        def _path_for(split: str, label: str) -> Path:
            if label == "positive":
                return stage_dir_out / f"{split}.parquet"
            return stage_dir_out / f"{split}_{label.lower()}.parquet"

        writers: Dict[Tuple[str, str], Any] = {}
        buffers: Dict[Tuple[str, str], List[Dict[str, Any]]] = {(sp, lb): [] for sp in SPLITS for lb in LABELS}
        BUFFER_FLUSH = 4096
        counts_by_label: Dict[str, Dict[str, int]] = {sp: {lb: 0 for lb in LABELS} for sp in SPLITS}
        rows_per_split: Dict[str, int] = {sp: 0 for sp in SPLITS}

        def _flush(key: Tuple[str, str], force: bool = False) -> None:
            buf = buffers[key]
            if not buf:
                return
            if not force and len(buf) < BUFFER_FLUSH:
                return
            if not skip_parquet:
                table = pa.Table.from_pylist(buf)
                if key not in writers:
                    out_path = _path_for(*key)
                    writers[key] = pq.ParquetWriter(
                        str(out_path), table.schema,
                        compression=config.get("runtime", {}).get("parquet_compression", "zstd"),
                    )
                writers[key].write_table(table)
            buffers[key] = []

        for i, anchor in enumerate(anchors):
            pool_name = anchor["pool"]
            pool = donor_pools[pool_name]
            try:
                rows = build_negatives_for_anchor(
                    anchor, pool,
                    stage=stage,
                    cfg=per_stage_cfg,
                    output_version=output_version,
                    rng_seed=rng_seed,
                )
            except Exception as e:
                LOGGER.error("anchor %s failed: %s", anchor.get("skill_id"), e)
                continue
            split = anchor["split"]
            rows_per_split[split] += len(rows)
            for r in rows:
                lbl = r["label"]
                key = (split, lbl)
                if key not in buffers:
                    # Defensive: if a row has an unexpected label, route to a per-label file
                    # using the same convention; create a buffer slot on the fly.
                    buffers[key] = []
                    counts_by_label[split].setdefault(lbl, 0)
                buffers[key].append(r)
                counts_by_label[split][lbl] = counts_by_label[split].get(lbl, 0) + 1
            if (i + 1) % 5000 == 0:
                LOGGER.info("[%s] %d / %d anchors; rows so far: %s",
                            stage, i + 1, len(anchors),
                            {sp: rows_per_split[sp] for sp in SPLITS})
            for key in list(buffers.keys()):
                _flush(key, force=False)

        for key in list(buffers.keys()):
            _flush(key, force=True)
        for w in writers.values():
            w.close()

        overall_stats["per_stage"][stage] = {
            "anchors": len(anchors),
            "donor_pool_sizes": {p: dp.size for p, dp in donor_pools.items()},
            "rows_per_split": rows_per_split,
            "rows_per_split_per_label": counts_by_label,
            "config": per_stage_cfg,
        }

    # Stats + manifest
    overall_stats["full_cpt_version"] = full_cpt_version
    overall_stats["output_version"] = output_version
    overall_stats["created_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    overall_stats["seed"] = rng_seed
    with (output_root / "stats.json").open("w", encoding="utf-8") as h:
        json.dump(overall_stats, h, ensure_ascii=False, indent=2)

    manifest = {
        "dataset_name": "stage_cpt",
        "phase_name": "build_phase2_hcl",
        "full_cpt_version_used": full_cpt_version,
        "output_version": output_version,
        "stats": overall_stats,
        "created_at_utc": overall_stats["created_at_utc"],
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as h:
        json.dump(manifest, h, ensure_ascii=False, indent=2)

    LOGGER.info("Phase 2 HCL Step A complete: %s", output_root)
