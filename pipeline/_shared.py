"""Shared utilities — IO, YAML config (with env-var interpolation), workers,
parquet writer, manifest/registry/split helpers.

Same API surface as the original `pipeline.py` so the per-stage modules in
this package can call these helpers without each redefining them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


LOGGER = logging.getLogger("agent_skills_preparation")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# JSON / JSONL
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def batched(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def detect_default_workers() -> int:
    for key in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        raw = os.environ.get(key)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                continue
    return max(1, os.cpu_count() or 1)


def map_records(
    records: Sequence[Dict[str, Any]],
    worker_fn: Callable[[Tuple[Dict[str, Any], Dict[str, Any]]], Any],
    worker_config: Dict[str, Any],
    workers: int,
    executor: Optional[ProcessPoolExecutor] = None,
) -> List[Any]:
    """Map `worker_fn` over `records`, optionally in parallel.

    Each task is `(record, worker_config)` so a single per-process payload
    is shared across the pool. If `executor` is provided it's reused (lets
    the caller hold the pool open across batches).
    """
    tasks = [(record, worker_config) for record in records]
    if workers <= 1:
        return [worker_fn(task) for task in tasks]
    if executor is None:
        with ProcessPoolExecutor(max_workers=workers) as local_executor:
            return list(
                local_executor.map(
                    worker_fn,
                    tasks,
                    chunksize=max(1, len(tasks) // (workers * 4) or 1),
                )
            )
    return list(
        executor.map(
            worker_fn,
            tasks,
            chunksize=max(1, len(tasks) // (workers * 4) or 1),
        )
    )


# ---------------------------------------------------------------------------
# Parquet writer (lazy import; gated on pyarrow availability)
# ---------------------------------------------------------------------------

class ParquetBatchWriter:
    """Append rows to a parquet file in batches.

    Lazy-imports pyarrow on first write so a `--skip-parquet` run never
    requires the dep.
    """

    def __init__(self, path: Path, compression: str, enabled: bool) -> None:
        self.path = path
        self.compression = compression
        self.enabled = enabled
        self.writer = None
        self.pa = None
        self.pq = None

    def write_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows or not self.enabled:
            return
        if self.writer is None:
            try:
                import pyarrow as pa  # type: ignore
                import pyarrow.parquet as pq  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "pyarrow is required to write parquet outputs. "
                    "Install requirements.txt or rerun with --skip-parquet."
                ) from exc
            self.pa = pa
            self.pq = pq
            ensure_dir(self.path.parent)
            table = pa.Table.from_pylist(list(rows))
            self.writer = pq.ParquetWriter(
                str(self.path),
                table.schema,
                compression=self.compression,
            )
            self.writer.write_table(table)
            return

        table = self.pa.Table.from_pylist(list(rows))
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


# ---------------------------------------------------------------------------
# YAML/JSON config — env-var ${VAR} / ${VAR:-default} interpolation
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in any string value
    nested inside a dict/list. Non-string leaves pass through."""
    if isinstance(value, str):
        def repl(m: "re.Match[str]") -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")
        return _ENV_VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _derive_dataset_paths(config: Dict[str, Any]) -> None:
    """Mutates `config` in place to add the per-dataset roots derived from
    `paths.prepared_root` so the rest of the pipeline can index them by
    name (`paths.normalized_root`, etc.) without re-deriving each time."""
    paths = config.get("paths") or {}
    prepared_root = paths.get("prepared_root")
    if not prepared_root:
        return
    prepared = Path(prepared_root).expanduser()
    paths["prepared_root"] = str(prepared)
    paths.setdefault("normalized_root", str(prepared / "normalized"))
    paths.setdefault("full_cpt_root",   str(prepared / "full_cpt"))
    paths.setdefault("pl_hcl_root",     str(prepared / "pl_hcl"))
    config["paths"] = paths


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML or JSON config with env-var interpolation in string values.

    Accepts both .yaml/.yml and .json so an old config.json from an older
    deployment still works. New deployments should ship config.yaml.

    Validates required sections + derives per-dataset path roots from
    `paths.prepared_root` so each stage module can find its output dir.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required: pip install pyyaml")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    config = _interpolate_env(data)

    for section in ("project", "paths", "runtime"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    paths = config["paths"]
    for key in ("canonical_records", "prepared_root", "logs_root", "manifests_root"):
        if key not in paths:
            raise ValueError(f"Missing required config path: {key}")

    _derive_dataset_paths(config)
    return config


def summarize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Small JSON-serializable snapshot for embedding in manifests."""
    summary: Dict[str, Any] = {}
    project = config.get("project") or {}
    if isinstance(project, dict):
        summary["project"] = {k: project.get(k) for k in ("name", "description") if k in project}
    runtime = config.get("runtime") or {}
    if isinstance(runtime, dict):
        keep_keys = (
            "split_seed", "train_fraction", "val_fraction", "test_fraction",
            "max_concat_chars", "max_stage_chars", "max_phase1_view_chars",
            "max_package_file_chars", "max_phase3_evidence_chars",
        )
        summary["runtime"] = {k: runtime.get(k) for k in keep_keys if k in runtime}
    return summary


# ---------------------------------------------------------------------------
# Manifest / registry helpers — same shape the legacy pipeline expects
# ---------------------------------------------------------------------------

def update_registry(
    registry_path: Path,
    dataset_name: str,
    version: str,
    manifest: Dict[str, Any],
) -> None:
    if registry_path.exists():
        registry = load_json(registry_path)
    else:
        registry = {"dataset_name": dataset_name, "versions": []}

    filtered_versions = [
        entry for entry in registry.get("versions", []) if entry.get("version") != version
    ]
    filtered_versions.append(
        {
            "version": version,
            "created_at": manifest["created_at"],
            "input_path": manifest["input_path"],
            "output_dir": manifest["output_dir"],
            "normalized_version_used": manifest.get("normalized_version_used"),
            "num_input_records": manifest["num_input_records"],
            "num_output_samples": manifest["num_output_samples"],
        }
    )
    filtered_versions.sort(key=lambda item: item["created_at"])
    registry["versions"] = filtered_versions
    write_json(registry_path, registry)


def write_manifest_copy(
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    dataset_name: str,
    version: str,
) -> None:
    copy_path = Path(config["paths"]["manifests_root"]) / f"{dataset_name}__{version}.json"
    write_json(copy_path, manifest)


def ensure_version_dir(path: Path) -> None:
    """Refuse to overwrite an existing dataset version. Pick a new
    --output-version (or remove the directory yourself) to re-build."""
    if path.exists():
        raise FileExistsError(f"Target version already exists: {path}")
    ensure_dir(path)


def build_outputs_manifest(
    dataset_name: str,
    phase_name: str,
    input_path: str,
    output_dir: Path,
    normalized_version_used: Optional[str],
    output_version: str,
    num_input_records: int,
    num_output_samples: int,
    stats: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "phase_name": phase_name,
        "input_path": input_path,
        "output_dir": str(output_dir),
        "normalized_version_used": normalized_version_used,
        "output_version": output_version,
        "created_at": utc_now(),
        "num_input_records": num_input_records,
        "num_output_samples": num_output_samples,
        "stats": stats,
        "config_summary": summarize_config(config),
    }


def finalize_manifest(
    manifest: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any],
    dataset_name: str,
    version: str,
) -> None:
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            files[path.name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    manifest["files"] = files
    write_json(output_dir / "manifest.json", manifest)
    write_manifest_copy(manifest, config, dataset_name, version)


def require_normalized_input(config: Dict[str, Any], normalized_version: str) -> Path:
    normalized_dir = Path(config["paths"]["normalized_root"]) / normalized_version
    normalized_path = normalized_dir / "normalized_skill_records.jsonl"
    if not normalized_path.exists():
        raise FileNotFoundError(f"Normalized dataset not found: {normalized_path}")
    return normalized_path


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def assign_split(skill_id: str, runtime: Dict[str, Any]) -> str:
    seed = int(runtime["split_seed"])
    train_fraction = float(runtime["train_fraction"])
    val_fraction = float(runtime["val_fraction"])
    digest = hashlib.sha1(f"{seed}:{skill_id}".encode("utf-8")).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    if fraction < train_fraction:
        return "train"
    if fraction < train_fraction + val_fraction:
        return "val"
    return "test"


def assign_phase1_pool_and_split(
    skill_id: str, phase1_cfg: Dict[str, Any]
) -> Tuple[str, str]:
    """Decide pool ('cpt' / 'unseen') and split ('train' / 'val' / 'test' / 'unseen').

    Same skill always gets the same (pool, split) across stage1 and stage2 so
    cross-stage leakage is impossible. Pool and split use independent seeds.
    """
    pool_seed = int(phase1_cfg.get("pool_seed", 20260501))
    split_seed = int(phase1_cfg.get("split_seed", 20260502))
    unseen_fraction = float(phase1_cfg.get("cpt_unseen_fraction", 0.5))
    train_fraction = float(phase1_cfg.get("cpt_train_fraction", 0.8))
    val_fraction = float(phase1_cfg.get("cpt_val_fraction", 0.1))

    pool_h = int(hashlib.sha1(f"{pool_seed}|{skill_id}".encode("utf-8")).hexdigest(), 16)
    pool_p = (pool_h % 1_000_000) / 1_000_000
    if pool_p < unseen_fraction:
        return "unseen", "unseen"

    split_h = int(hashlib.sha1(f"{split_seed}|{skill_id}".encode("utf-8")).hexdigest(), 16)
    split_p = (split_h % 1_000_000) / 1_000_000
    if split_p < train_fraction:
        return "cpt", "train"
    if split_p < train_fraction + val_fraction:
        return "cpt", "val"
    return "cpt", "test"
